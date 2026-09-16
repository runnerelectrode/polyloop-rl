"""The cycle: snapshot -> preflight -> filter -> train -> evaluate -> gate -> promote -> observe.

Each stage reads state.json, writes one artifact, marks itself done. A rerun of the same
cycle skips finished stages (resume). Only train and the two rollout stages touch GPUs; the
budget is checked at every stage edge and an overrun ends the cycle before the next stage,
never in the middle of one (a half-trained adapter is never a candidate).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from polyloop.config import LoopConfig
from polyloop.events import Cycle, LoopStore, now_iso, read_json
from polyloop.harness.rollout import TaskResult, load_tasks, run_rollouts
from polyloop.receipt import holdout_id, paired_stats, write_receipt

STAGES = ["snapshot", "preflight", "filter", "train", "evaluate", "gate", "promote", "observe"]


class CycleAborted(Exception):
    pass


class Runner:
    def __init__(self, cfg: LoopConfig, cycle: Cycle, log=print):
        self.cfg = cfg
        self.cycle = cycle
        self.store: LoopStore = cycle.store
        self.log = log
        self.stage = cfg.stages[0]

    # ---- helpers --------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.log(f"[{self.cycle.id}] {msg}")

    def _gpu_seconds(self) -> float:
        st = self.cycle.state
        return sum(v for v in st.get("gpu_seconds", {}).values())

    def _charge(self, stage: str, seconds: float) -> None:
        st = self.cycle.state
        gs = dict(st.get("gpu_seconds", {}))
        gs[stage] = gs.get(stage, 0.0) + seconds * self.cfg.budget.gpus
        self.cycle.update(gpu_seconds=gs)

    def _check_budget(self, next_stage: str) -> None:
        spent = self._gpu_seconds()
        if spent > self.cfg.budget.max_cycle_gpu_seconds:
            self.cycle.emit("budget.exceeded", spent=spent, cap=self.cfg.budget.max_cycle_gpu_seconds, before=next_stage)
            raise CycleAborted(f"budget exceeded before {next_stage}: {spent:.0f}s GPU > {self.cfg.budget.max_cycle_gpu_seconds}s")

    def _rollout_kwargs(self, **over):
        s, sb = self.stage, self.cfg.sandbox
        kw = dict(base_url=self.cfg.base_url, model=self.cfg.model, renderer=self.cfg.renderer,
                  max_parallel=sb.max_parallel, max_tokens=s.max_tokens, max_turns=s.max_turns,
                  max_trajectory_tokens=s.max_trajectory_tokens, sandbox_timeout=sb.timeout,
                  command_timeout=sb.command_timeout, grader_timeout=sb.grader_timeout,
                  context_window=s.max_trajectory_tokens)
        kw.update(over)
        return kw

    def _run_rollouts(self, stage: str, tasks, sampler_path, k, temperature=1.0, out: Path | None = None) -> list[TaskResult]:
        t0 = time.monotonic()
        results: list[TaskResult] = []

        def on_result(r: TaskResult):
            self.cycle.emit("rollout.task", stage=stage, task=r.task, rewards=r.rewards, error=r.error, seconds=round(r.seconds, 1))
            if out:
                with (out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(r.to_dict()) + "\n")

        results = asyncio.run(run_rollouts(tasks=tasks, sampler_path=sampler_path, k=k, temperature=temperature,
                                           on_result=on_result, trajectories_dir=(out / "trajectories") if out else None,
                                           **self._rollout_kwargs()))
        self._charge(stage, time.monotonic() - t0)
        return results

    # ---- stages ---------------------------------------------------------
    def run(self, upto: str | None = None) -> str:
        if self.cycle.state.get("status") in (None, "failed", "aborted"):
            self.cycle.update(status="running", failed_stage=None, reason=None)
            self.cycle.emit("cycle.resume" if self.cycle.state.get("stages_done") else "cycle.start")
        for name in STAGES:
            if self.cycle.stage_done(name):
                self._say(f"{name}: done earlier, skipping")
            else:
                self._check_budget(name)
                self.cycle.emit("stage.start", stage=name)
                t0 = time.monotonic()
                try:
                    getattr(self, name)()
                except CycleAborted as exc:
                    self.cycle.emit("cycle.aborted", stage=name, reason=str(exc))
                    self.cycle.update(status="aborted", reason=str(exc))
                    self._say(f"aborted at {name}: {exc}")
                    return "aborted"
                except Exception as exc:
                    self.cycle.emit("stage.failed", stage=name, error=f"{type(exc).__name__}: {exc}"[:800])
                    self.cycle.update(status="failed", failed_stage=name)
                    raise
                self.cycle.emit("stage.seconds", stage=name, seconds=round(time.monotonic() - t0, 1))
            if upto == name:
                return "paused"
            if self.cycle.state.get("status") in ("awaiting_approval", "rejected"):
                return self.cycle.state["status"]
        return self.cycle.state.get("status", "done")

    def snapshot(self) -> None:

        pool = load_tasks(self.cfg.tasks)
        holdout = load_tasks(self.cfg.gate.holdout, limit=self.cfg.gate.holdout_limit, seed=self.cfg.gate.holdout_seed)
        hid = holdout_id([t.task_name for t in holdout])
        inc = self.store.incumbent()
        traces = sorted(self._traces_dir().glob("*.jsonl"))
        self.cycle.update(
            trace_files=[str(t) for t in traces],
            pool_size=len(pool), pool_dataset=self.cfg.tasks,
            holdout_names=[t.task_name for t in holdout], holdout_id=hid,
            incumbent=inc, model=self.cfg.model, loop=self.cfg.name, started=now_iso(),
        )
        self.cycle.mark_done("snapshot", pool=len(pool), holdout=len(holdout), holdout_id=hid,
                             incumbent=(inc or {}).get("id", "base"), trace_files=len(traces))

    def preflight(self) -> None:
        problems = []
        st = self.cycle.state
        if st["pool_size"] == 0:
            problems.append("empty task pool")
        if not st["holdout_names"]:
            problems.append("empty holdout")
        if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            problems.append("docker daemon unreachable")
        try:
            import httpx

            r = httpx.get(self.cfg.base_url.rstrip("/") + "/api/v1/get_server_capabilities", timeout=10)
            if r.status_code >= 400:
                problems.append(f"trainer server {r.status_code}")
        except Exception as exc:
            problems.append(f"trainer server unreachable: {exc}")
        free_gb = shutil.disk_usage(Path.home()).free / 1e9
        if free_gb < 50:
            problems.append(f"only {free_gb:.0f} GB free")
        if self.stage.max_steps_off_policy and self.stage.learning_rate > 1e-4 / max(1, self.stage.max_steps_off_policy):
            problems.append("learning rate too high for the staleness bound")
        if problems:
            raise CycleAborted("preflight: " + "; ".join(problems))
        from polyloop.harness.rollout import warm

        t0 = time.monotonic()
        warm(self.cfg.base_url, self.cfg.model, rank=self.stage.lora_rank, log=self._say)
        self._charge("preflight", time.monotonic() - t0)
        self.cycle.mark_done("preflight", checks=["pool", "holdout", "docker", "trainer", "disk", "staleness", "engine-warm"])

    def filter(self) -> None:
        fc = self.cfg.filter
        inc = self.cycle.state.get("incumbent") or {}
        inc_id = inc.get("id", "base")
        stats = self.store.cache("pool_stats")
        under = stats.setdefault(inc_id, {})
        pool = load_tasks(self.cfg.tasks)
        out = self.cycle.subdir("filter")
        import random

        rng = random.Random(self.cycle.id)
        contested = lambda: sorted(t for t, v in under.items() if fc.keep_min < v["pass_rate"] < fc.keep_max)
        rounds = 0
        newly = 0
        while len(contested()) < fc.target_tasks and rounds < fc.max_rounds:
            unmeasured = [t for t in pool if t.task_name not in under]
            if not unmeasured:
                break
            sample = rng.sample(unmeasured, min(fc.pool_sample, len(unmeasured)))
            rounds += 1
            self._say(f"filter round {rounds}: {len(contested())} contested so far; measuring {len(sample)} tasks x {fc.rollouts_per_task} under {inc_id}")
            results = self._run_rollouts("filter", sample, inc.get("sampler_path"), fc.rollouts_per_task, out=out)
            for r in results:
                if r.error is None and r.rewards:
                    under[r.task] = {"pass_rate": r.mean, "n": len(r.rewards), "cycle": self.cycle.id}
                    newly += 1
            self.store.save_cache("pool_stats", stats)
            self._check_budget("filter")
        keep = contested()
        (out / "contested.json").write_text(json.dumps(keep, indent=1))
        self.cycle.update(train_task_names=keep, measured=len(under))
        if len(keep) < fc.min_tasks:
            raise CycleAborted(f"filter: only {len(keep)} contested tasks (< {fc.min_tasks}); measured {len(under)}")
        self.cycle.mark_done("filter", measured=len(under), contested=len(keep), newly_measured=newly, rounds=rounds)

    def _traces_dir(self) -> Path:
        return Path(self.cfg.traces).expanduser() if self.cfg.traces else self.store.root / "traces"

    def _run_worker(self, module: str, spec: dict, log_path: Path) -> None:
        (log_path / "spec.json").write_text(json.dumps(spec, indent=1))
        t0 = time.monotonic()
        with (log_path / "train.log").open("a") as logf:
            proc = subprocess.run([sys.executable, "-m", module, str(log_path / "spec.json")],
                                  stdout=logf, stderr=subprocess.STDOUT)
        self._charge("train", time.monotonic() - t0)
        if proc.returncode != 0:
            raise RuntimeError(f"{module} exited {proc.returncode}; see {log_path / 'train.log'}")

    @staticmethod
    def _last_checkpoint(log_path: Path) -> dict:
        ckpts = [json.loads(l) for l in (log_path / "checkpoints.jsonl").read_text().splitlines() if l.strip()]
        last = [c for c in ckpts if c.get("sampler_path")]
        if not last:
            raise RuntimeError(f"no sampler checkpoint under {log_path}")
        return last[-1]

    @staticmethod
    def _metrics(log_path: Path) -> list[dict]:
        p = log_path / "metrics.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    def train(self) -> None:
        st = self.cycle.state
        inc = st.get("incumbent") or {}
        load_path = inc.get("state_path")
        last = None
        diffs: list[float] = []
        total_steps = 0
        summary = []
        for i, stage in enumerate(self.cfg.stages):
            log_path = self.cycle.subdir(f"train/{i}-{stage.kind}")
            done_before = (log_path / "checkpoints.jsonl").exists() and len(self._metrics(log_path)) >= stage.steps
            if done_before:
                self._say(f"train[{i}] {stage.kind}: {len(self._metrics(log_path))} steps already on disk, reusing")
            elif stage.kind == "rl":
                if not st.get("train_task_names"):
                    self._say(f"train[{i}] rl: no contested tasks, skipped")
                    continue
                spec = {"base_url": self.cfg.base_url, "model": self.cfg.model, "renderer": self.cfg.renderer,
                        "dataset": self.cfg.tasks, "task_names": st["train_task_names"], "stage": stage.__dict__,
                        "sandbox": self.cfg.sandbox.__dict__, "log_path": str(log_path),
                        "load_checkpoint_path": load_path, "logprob_abs_diff_max": self.cfg.gate.logprob_abs_diff_max}
                self._say(f"train[{i}] rl: {len(st['train_task_names'])} tasks, {stage.steps} steps")
                self._run_worker("polyloop.harness.train", spec, log_path)
            elif stage.kind == "opsd" and not done_before:
                from polyloop.harness.trace_rows import write_rows

                traces = sorted(self._traces_dir().glob("*.jsonl"))
                rows_path = log_path / "rows.jsonl"
                n = write_rows(traces, rows_path, max_hint_chars=stage.max_hint_chars)
                self.cycle.emit("opsd.rows", rows=n, trace_files=len(traces))
                if n < stage.min_rows:
                    self._say(f"train[{i}] opsd: {n} rows (< {stage.min_rows}), skipped")
                    continue
                spec = {"base_url": self.cfg.base_url, "model": self.cfg.model, "renderer": self.cfg.renderer,
                        "rows": str(rows_path), "stage": stage.__dict__, "log_path": str(log_path),
                        "load_checkpoint_path": load_path}
                self._say(f"train[{i}] opsd: {n} rows from {len(traces)} trace files, {stage.steps} steps")
                self._run_worker("polyloop.harness.train_opsd", spec, log_path)
            elif not done_before:
                raise CycleAborted(f"train: unknown stage kind {stage.kind!r}")
            last = self._last_checkpoint(log_path)
            load_path = last.get("state_path") or load_path
            metrics = self._metrics(log_path)
            total_steps += len(metrics)
            diffs += [m["optim/rollout_logprobs_abs_diff_mean"] for m in metrics if "optim/rollout_logprobs_abs_diff_mean" in m]
            summary.append({"stage": stage.kind, "steps": len(metrics), "sampler_path": last["sampler_path"]})
        if last is None:
            raise CycleAborted("train: no stage produced a checkpoint (no contested tasks and too few trace rows)")
        candidate = {"id": f"{self.cfg.name}-{self.cycle.id}", "cycle": self.cycle.id,
                     "sampler_path": last["sampler_path"], "state_path": last.get("state_path"),
                     "step": last.get("batch"), "parent": inc.get("id", "base"), "stages": summary}
        self.cycle.update(candidate=candidate, train_steps=total_steps,
                          logprob_abs_diff_max=max(diffs) if diffs else None)
        self.cycle.mark_done("train", steps=total_steps, candidate=candidate["id"], stages=summary,
                             logprob_abs_diff_max=max(diffs) if diffs else None)

    def _eval_policy(self, label: str, policy: dict | None, holdout) -> dict[str, float]:
        g = self.cfg.gate
        pid = (policy or {}).get("id", "base")
        cache = self.store.cache("holdout_cache")
        key = f"{pid}@{self.cycle.state['holdout_id']}@k{g.repeats}@T{g.temperature}"
        if key in cache and label == "incumbent":
            self._say(f"evaluate: incumbent {pid} cached")
            return cache[key]["per_task"]
        out = self.cycle.subdir(f"eval/{label}")
        results = self._run_rollouts(f"evaluate.{label}", holdout, (policy or {}).get("sampler_path"), g.repeats,
                                     temperature=g.temperature, out=out)
        errors = sum(1 for r in results if r.error)
        if errors / max(1, len(results)) > g.max_error_rate:
            raise CycleAborted(f"evaluate: {errors}/{len(results)} {label} rollouts errored")
        per_task = {r.task: r.mean for r in results if r.error is None and r.mean is not None}
        cache[key] = {"policy": pid, "per_task": per_task, "cycle": self.cycle.id, "ts": now_iso()}
        self.store.save_cache("holdout_cache", cache)
        return per_task

    def evaluate(self) -> None:
        st = self.cycle.state
        holdout = load_tasks(self.cfg.gate.holdout, names=st["holdout_names"])
        inc_scores = self._eval_policy("incumbent", st.get("incumbent"), holdout)
        cand_scores = self._eval_policy("candidate", st["candidate"], holdout)
        self.cycle.update(eval_incumbent=inc_scores, eval_candidate=cand_scores)
        self.cycle.mark_done("evaluate", tasks=len(set(inc_scores) & set(cand_scores)))

    def gate(self) -> None:
        g = self.cfg.gate
        st = self.cycle.state
        stats = paired_stats(st["eval_candidate"], st["eval_incumbent"], g.tie_band)
        checks = {
            "paired_delta_above_min": stats["mean_delta"] > g.min_delta,
            "regressions_within_cap": stats["losses"] <= g.max_regressions,
            "logprob_agreement": (st.get("logprob_abs_diff_max") is None) or (st["logprob_abs_diff_max"] <= g.logprob_abs_diff_max),
            "enough_tasks": stats["n_tasks"] >= max(4, g.holdout_limit // 2),
        }
        decision = "promote" if all(checks.values()) else "reject"
        receipt = write_receipt(
            self.cycle.dir / "receipt.json",
            loop=self.cfg.name, cycle=self.cycle.id, ts=now_iso(), model=self.cfg.model,
            candidate=st["candidate"], incumbent=st.get("incumbent") or {"id": "base"},
            holdout={"id": st["holdout_id"], "dataset": g.holdout, "n": len(st["holdout_names"]), "repeats": g.repeats, "temperature": g.temperature},
            train={"tasks": len(st["train_task_names"]), "steps": st.get("train_steps"), "stage": self.stage.__dict__},
            paired=stats, checks=checks, thresholds={"min_delta": g.min_delta, "tie_band": g.tie_band,
                                                     "max_regressions": g.max_regressions, "logprob_abs_diff_max": g.logprob_abs_diff_max},
            logprob_abs_diff_max=st.get("logprob_abs_diff_max"), gpu_seconds=st.get("gpu_seconds", {}),
            decision=decision, promote_mode=self.cfg.promote.mode,
        )
        self.cycle.update(decision=decision, paired=stats, receipt=str(self.cycle.dir / "receipt.json"))
        self.cycle.mark_done("gate", decision=decision, mean_delta=stats["mean_delta"], ci=stats["delta_ci95"],
                             wins=stats["wins"], losses=stats["losses"])
        self._say(f"gate: {decision} (delta {stats['mean_delta']:+.3f}, ci {stats['delta_ci95']}, W/L/T {stats['wins']}/{stats['losses']}/{stats['ties']})")
        if decision == "reject":
            self.cycle.update(status="rejected")

    def promote(self) -> None:
        st = self.cycle.state
        if st.get("decision") != "promote":
            self.cycle.mark_done("promote", action="none", reason="rejected by gate")
            return
        if self.cfg.promote.mode == "auto":
            self.store.set_incumbent(st["candidate"], st["receipt"])
            from polyloop.events import write_json

            write_json(self.store.root / "live.json", {"sampler_path": st["candidate"]["sampler_path"],
                                                       "candidate": st["candidate"]["id"], "promoted_at": now_iso(), "receipt": st["receipt"]})
            notify_proxy(self.cfg.proxy_url, st["candidate"], st["receipt"], log=self._say)
            self.cycle.update(status="promoted")
            self.cycle.mark_done("promote", action="auto", incumbent=st["candidate"]["id"])
        else:
            self.cycle.update(status="awaiting_approval")
            self.cycle.mark_done("promote", action="awaiting_approval", hint=f"polyloop approve {self.cycle.id}")

    def observe(self) -> None:
        # Single-node demo: no serving endpoint to watch. The hook exists so rollback rules
        # (score drop, error rate on the served rung) have a stage to report into.
        self.cycle.mark_done("observe", watched=False)
        if self.cycle.state.get("status") not in ("awaiting_approval", "rejected", "promoted"):
            self.cycle.update(status="done")


def notify_proxy(proxy_url: str | None, candidate: dict, receipt: str | None, log=print) -> None:
    if not proxy_url:
        return
    try:
        import httpx

        r = httpx.post(proxy_url.rstrip("/") + "/admin/promote",
                       json={"sampler_path": candidate["sampler_path"], "candidate": candidate["id"], "receipt": receipt}, timeout=30)
        log(f"proxy live -> {r.json().get('live')}")
    except Exception as exc:  # promotion is recorded either way; the proxy re-reads live.json on restart
        log(f"proxy notify failed: {exc}")


def approve(store: LoopStore, cycle_id: str, log=print) -> dict:
    cycle = store.cycle(cycle_id)
    st = cycle.state
    if st.get("status") != "awaiting_approval":
        raise SystemExit(f"cycle {cycle_id} is {st.get('status')!r}, not awaiting approval")
    store.set_incumbent(st["candidate"], st["receipt"])
    from polyloop.events import write_json

    write_json(store.root / "live.json", {"sampler_path": st["candidate"]["sampler_path"], "candidate": st["candidate"]["id"],
                                          "promoted_at": now_iso(), "receipt": st["receipt"]})
    cycle.update(status="promoted", approved_at=now_iso())
    cycle.emit("promote.approved", candidate=st["candidate"]["id"])
    log(f"promoted {st['candidate']['id']}")
    return st["candidate"]
