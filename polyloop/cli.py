"""polyloop: run and inspect continual-learning cycles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from polyloop.config import load_loop
from polyloop.events import LoopStore


@click.group()
def main():
    """A harness that runs filter -> train -> evaluate -> gate -> promote cycles."""


def _store(loop_path: str):
    cfg = load_loop(loop_path)
    return cfg, LoopStore(cfg.runs_path, cfg.name)


@main.command("run")
@click.option("--loop", "loop_path", required=True, help="Path to loop.yaml")
@click.option("--cycle", "cycle_id", default=None, help="Resume this cycle id instead of starting a new one.")
@click.option("--upto", default=None, type=click.Choice(["snapshot", "preflight", "filter", "train", "evaluate", "gate", "promote", "observe"]),
              help="Stop after this stage.")
def run(loop_path, cycle_id, upto):
    """Run one cycle (resumes finished stages of an existing cycle)."""
    from polyloop.stages import Runner

    cfg, store = _store(loop_path)
    cycle = store.cycle(cycle_id)
    click.echo(f"loop {cfg.name}: cycle {cycle.id} -> {cycle.dir}")
    status = Runner(cfg, cycle, log=click.echo).run(upto=upto)
    click.echo(f"cycle {cycle.id}: {status}")


@main.command("approve")
@click.option("--loop", "loop_path", required=True)
@click.argument("cycle_id")
def approve_cmd(loop_path, cycle_id):
    """Promote a cycle's candidate that is awaiting approval."""
    from polyloop.stages import approve, notify_proxy

    cfg, store = _store(loop_path)
    cand = approve(store, cycle_id, log=click.echo)
    notify_proxy(cfg.proxy_url, cand, store.cycle(cycle_id).state.get("receipt"), log=click.echo)


@main.command("history")
@click.option("--loop", "loop_path", required=True)
def history(loop_path):
    """Cycles, decisions and the lineage of promoted adapters."""
    cfg, store = _store(loop_path)
    inc = store.incumbent()
    click.echo(f"loop {cfg.name}  incumbent: {(inc or {}).get('id', 'base')}")
    for cid in store.cycles():
        st = store.cycle(cid).state
        p = st.get("paired") or {}
        d = f"delta {p['mean_delta']:+.3f} W/L/T {p['wins']}/{p['losses']}/{p['ties']}" if p else ""
        click.echo(f"  {cid}  {st.get('status', '?'):18} stages={len(st.get('stages_done', []))}/8  train_tasks={len(st.get('train_task_names', []))}  {d}")


@main.command("status")
@click.option("--loop", "loop_path", required=True)
@click.argument("cycle_id")
def status(loop_path, cycle_id):
    """Dump one cycle's state."""
    _, store = _store(loop_path)
    click.echo(json.dumps(store.cycle(cycle_id).state, indent=2, sort_keys=True))


@main.command("eval")
@click.option("--loop", "loop_path", required=True)
@click.option("--dataset", default=None, help="Harbor dataset dir or coval://<test_set_id> (default: the loop's holdout).")
@click.option("--limit", type=int, default=None)
@click.option("--sampler-path", default=None, help="tinker:// sampler checkpoint; default = base model")
@click.option("--policy-id", default=None, help="Name for the policy (coval loops: the proxy route and Coval agent it gets); default = base")
@click.option("--repeats", "-k", type=int, default=1)
@click.option("--out", default=None)
def eval_cmd(loop_path, dataset, limit, sampler_path, policy_id, repeats, out):
    """Ad-hoc: score a checkpoint on a task set (baseline numbers)."""
    import asyncio

    from polyloop.harness.rollout import load_tasks, run_rollouts, warm

    cfg, store = _store(loop_path)
    g, s, sb = cfg.gate, cfg.stages[0], cfg.sandbox
    client = None
    if cfg.coval:
        from polyloop.harness.coval import CovalClient

        client = CovalClient.from_env(api_base=cfg.coval.api_base, api_key_env=cfg.coval.api_key_env)
    else:
        warm(cfg.base_url, cfg.model, rank=s.lora_rank, log=click.echo)
    tasks = load_tasks(dataset or g.holdout, limit=limit or g.holdout_limit, seed=g.holdout_seed, coval_client=client)
    out_path = Path(out).expanduser() if out else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    def on_result(r):
        click.echo(f"  {r.task:50} {r.rewards} {r.error or ''} ({r.seconds:.0f}s, turns {r.turns})")
        if out_path and not cfg.coval:
            with (out_path / "results.jsonl").open("a") as f:
                f.write(json.dumps(r.to_dict()) + "\n")

    if cfg.coval:
        from polyloop.harness.coval import run_coval_rollouts

        results = run_coval_rollouts(cfg=cfg, client=client, store=store, tasks=tasks, policy_id=policy_id or "base",
                                     sampler_path=sampler_path, k=repeats, label="eval", out=out_path,
                                     on_result=on_result, log=click.echo)
        ok = [r for r in results if r.error is None and r.mean is not None]
        mean = sum(r.mean for r in ok) / len(ok) if ok else float("nan")
        click.echo(f"{len(ok)}/{len(results)} tasks scored, mean reward {mean:.3f}, errors {len(results) - len(ok)}")
        return
    results = asyncio.run(run_rollouts(
        base_url=cfg.base_url, model=cfg.model, renderer=cfg.renderer, tasks=tasks, sampler_path=sampler_path,
        k=repeats, max_parallel=sb.max_parallel, max_tokens=s.max_tokens, max_turns=s.max_turns,
        max_trajectory_tokens=s.max_trajectory_tokens, sandbox_timeout=sb.timeout, command_timeout=sb.command_timeout,
        grader_timeout=sb.grader_timeout, temperature=g.temperature, context_window=s.max_trajectory_tokens, on_result=on_result,
        trajectories_dir=(out_path / "trajectories") if out_path else None))
    ok = [r for r in results if r.error is None and r.mean is not None]
    mean = sum(r.mean for r in ok) / len(ok) if ok else float("nan")
    click.echo(f"{len(ok)}/{len(results)} tasks scored, mean reward {mean:.3f}, errors {len(results) - len(ok)}")


@main.command("proxy")
@click.option("--loop", "loop_path", required=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8787, show_default=True)
@click.option("--max-tokens", type=int, default=4096, show_default=True)
@click.option("--auth-token", envvar="POLYLOOP_PROXY_TOKEN", default=None,
              help="Bearer token every request (except /healthz) must carry. Set it before exposing the proxy (a coval loop does).")
def proxy_cmd(loop_path, host, port, max_tokens, auth_token):
    """OpenAI/Anthropic-compatible endpoint for your agent; records sessions; serves the live adapter and named policies."""
    from polyloop.proxy import serve

    cfg, _ = _store(loop_path)
    sp = cfg.system_prompt_path
    system_prompt = sp.read_text().strip() if sp and sp.exists() else None
    if sp and not sp.exists():
        raise SystemExit(f"coval.system_prompt not found: {sp}")
    if cfg.coval and not auth_token:
        click.echo("warning: coval loop without --auth-token / POLYLOOP_PROXY_TOKEN; the public tunnel will be open", err=True)
    serve(host, port, base_url=cfg.base_url, model=cfg.model, renderer_name=cfg.renderer,
          runs_dir=cfg.runs_path, loop_name=cfg.name, default_max_tokens=max_tokens,
          auth_token=auth_token, system_prompt=system_prompt)


@main.command("sessions")
@click.option("--tasks", "tasks_dir", required=True, help="Task pool dir (train split).")
@click.option("--n", type=int, default=10)
@click.option("--proxy", default="http://127.0.0.1:8787", show_default=True)
@click.option("--agent-config", default=None, help="mini-swe-agent yaml (default: recipes/pydantic-v2/agent.yaml)")
@click.option("--python", "python_exe", default=sys.executable, help="Python with pydantic v2 + pytest for local sessions.")
@click.option("--mini", "mini_bin", default="mini-swe-agent")
@click.option("--env", "env_class", type=click.Choice(["local", "docker"]), default="local")
@click.option("--parallel", type=int, default=1)
@click.option("--seed", type=int, default=0)
@click.option("--log-dir", default="~/polyloop-runs/sessions")
def sessions_cmd(tasks_dir, n, proxy, agent_config, python_exe, mini_bin, env_class, parallel, seed, log_dir):
    """Run real mini-swe-agent sessions through the proxy on pool tasks (laptop or night driver)."""
    import random

    from polyloop.sessions import run_many

    pool = [d for d in sorted(Path(tasks_dir).expanduser().iterdir()) if (d / "task.toml").exists()]
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n, len(pool)))
    cfg = Path(agent_config).expanduser() if agent_config else Path(__file__).resolve().parent.parent / "recipes" / "pydantic-v2" / "agent.yaml"
    recs = run_many(chosen, parallel, agent_cfg=cfg, proxy=proxy, python=python_exe, mini_bin=mini_bin,
                    log_dir=Path(log_dir).expanduser(), env_class=env_class)
    ok = sum(1 for r in recs if r.get("passed_strict"))
    click.echo(f"{len(recs)} sessions, {ok} passed strict verify, log {Path(log_dir).expanduser()}/sessions.jsonl")


@main.command("warm")
@click.option("--loop", "loop_path", required=True)
def warm_cmd(loop_path):
    """Bring the trainer's sampler engines up (first sample otherwise fails)."""
    from polyloop.harness.rollout import warm

    cfg, _ = _store(loop_path)
    warm(cfg.base_url, cfg.model, rank=cfg.stages[0].lora_rank, log=click.echo)


@main.command("report")
@click.option("--loop", "loop_path", required=True)
@click.option("--out", required=True, help="HTML file to write.")
def report_cmd(loop_path, out):
    """One self-contained HTML page: curve, cycles, receipts."""
    from polyloop.report import build

    cfg, store = _store(loop_path)
    Path(out).expanduser().write_text(build(store, cfg))
    click.echo(f"wrote {out}")


@main.command("ui")
@click.option("--loop", "loop_path", required=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8080, show_default=True)
@click.option("--refresh", type=int, default=5, show_default=True, help="Poll interval in seconds (updates in place, no reload).")
@click.option("--log", "log_path", default=None, help="A log file to tail on the page (e.g. the `polyloop run` output).")
def ui_cmd(loop_path, host, port, refresh, log_path):
    """Live page: current cycle, episodes as they land, training metrics, GPUs, a log tail."""
    from polyloop.ui import serve

    cfg, store = _store(loop_path)
    serve(cfg, store, host, port, refresh, Path(log_path).expanduser() if log_path else None)


@main.group("coval")
def coval():
    """Coval-backed loops: check the wiring, inspect the session ledger."""


@coval.command("check")
@click.option("--loop", "loop_path", required=True)
def coval_check(loop_path):
    """Resolve the agent, persona, test sets and metrics named in loop.yaml; probe the proxy locally and publicly."""
    import httpx

    from polyloop.harness.coval import CovalClient, policy_endpoint, test_set_id

    cfg, store = _store(loop_path)
    if not cfg.coval:
        raise SystemExit("loop.yaml has no `coval:` block")
    cv = cfg.coval
    client = CovalClient.from_env(api_base=cv.api_base, api_key_env=cv.api_key_env)
    agent = client.agent(cv.agent_id)
    click.echo(f"agent    {cv.agent_id}  {agent.get('display_name')!r}  {agent.get('model_type')}  chat_endpoint={ (agent.get('metadata') or {}).get('chat_endpoint') }")
    persona = client.persona(cv.persona_id)
    click.echo(f"persona  {cv.persona_id}  {persona.get('display_name') or persona.get('name')!r}")
    for label, ds in (("pool", cfg.tasks), ("holdout", cfg.gate.holdout)):
        ts = client.test_set(test_set_id(ds))
        click.echo(f"{label:8} {ds}  {ts.get('display_name')!r}  {ts.get('test_case_count')} test cases  type={ts.get('test_set_type')}")
    for m in client.metrics(cv.metric_ids):
        click.echo(f"metric   {m.get('id')}  {m.get('metric_name')!r}  {m.get('metric_type')}")
    agents = store.cache("coval_agents")
    for key, rec in agents.items():
        click.echo(f"policy agent {key.split(':', 1)[1]:24} {rec['agent_id']}  {rec['chat_endpoint']}")
    click.echo(f"live route   {policy_endpoint(cv.public_url, 'live')}")
    ledger = store.root / cv.ledger
    n = sum(1 for l in ledger.read_text().splitlines() if l.strip()) if ledger.exists() else 0
    click.echo(f"ledger   {ledger}  {n} scored sessions")
    for label, url in (("proxy", (cfg.proxy_url or "").rstrip("/") + "/admin/status"), ("public", cv.public_url.rstrip("/") + "/healthz")):
        try:
            from polyloop.stages import _proxy_headers

            r = httpx.get(url, headers=_proxy_headers() if label == "proxy" else {}, timeout=15)
            click.echo(f"{label:8} {url}  {r.status_code}  {r.text[:160]}")
        except Exception as exc:
            click.echo(f"{label:8} {url}  unreachable: {exc}")


@coval.command("seed")
@click.option("--loop", "loop_path", required=True)
@click.option("--test-set", "which", type=click.Choice(["pool", "holdout"]), required=True, help="Which of the loop's two test sets to fill.")
@click.option("--file", "path", required=True, help="JSON list of {input, expected[], description} scenarios.")
def coval_seed(loop_path, which, path):
    """Create the recipe's scenario test cases in a Coval test set (skips inputs that already exist)."""
    from polyloop.harness.coval import CovalClient, seed_test_cases, test_set_id

    cfg, _ = _store(loop_path)
    if not cfg.coval:
        raise SystemExit("loop.yaml has no `coval:` block")
    client = CovalClient.from_env(api_base=cfg.coval.api_base, api_key_env=cfg.coval.api_key_env)
    ts = test_set_id(cfg.tasks if which == "pool" else cfg.gate.holdout)
    cases = json.loads(Path(path).expanduser().read_text())
    ids = seed_test_cases(client, ts, cases)
    click.echo(f"test set {ts}: created {len(ids)} of {len(cases)} test cases ({len(cases) - len(ids)} already present)")


@coval.command("ledger")
@click.option("--loop", "loop_path", required=True)
@click.option("--policy", default=None, help="Only sessions scored under this policy name.")
@click.option("--failed/--all", default=False, help="Only sessions below coval.pass_value.")
@click.option("--limit", type=int, default=20)
def coval_ledger(loop_path, policy, failed, limit):
    """Tail the per-session ledger: reward, metric values and the judge's explanation."""
    cfg, store = _store(loop_path)
    if not cfg.coval:
        raise SystemExit("loop.yaml has no `coval:` block")
    ledger = store.root / cfg.coval.ledger
    if not ledger.exists():
        raise SystemExit(f"no ledger yet at {ledger}")
    recs = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    if policy:
        recs = [r for r in recs if r.get("policy") == policy]
    if failed:
        recs = [r for r in recs if r.get("reward") is not None and r["reward"] < cfg.coval.pass_value]
    for r in recs[-limit:]:
        click.echo(f"{r['ts']}  {r['policy']:20} {str(r.get('task')):14} reward={r.get('reward')}  {r['session']}")
        if r.get("explanation"):
            click.echo("    " + r["explanation"].replace("\n", "\n    ")[:600])


@main.group("tasks")
def tasks():
    """Build task pools."""


@tasks.command("pydantic-v2")
@click.option("--pydantic-repo", required=True, help="Checkout of pydantic at a v2 tag.")
@click.option("--python", "python_exe", required=True, help="Python with the same pydantic version + pytest, used for the filter.")
@click.option("--out", required=True)
@click.option("--limit", type=int, default=None)
@click.option("--skip", default="test_docs.py,test_docs_extraction.py,test_mypy.py,test_plugins.py,test_plugin_loader.py,test_pickle.py,test_migration.py,test_deprecated.py,test_deprecated_fields.py,test_deprecated_validate_arguments.py,test_v1.py,test_exports.py,test_internal.py,test_dunder_all.py,test_meta.py,test_version.py,test_warnings.py")
def tasks_pydantic(pydantic_repo, python_exe, out, limit, skip):
    """pydantic's v2 tests -> v1-idiom migration tasks with a strict-deprecation verifier."""
    from polyloop.tasks.pydantic_v2 import build

    stats = build(Path(pydantic_repo).expanduser(), python_exe, Path(out).expanduser(), limit, set(skip.split(",")))
    click.echo(json.dumps(stats, indent=1))


@tasks.command("split")
@click.option("--pool", required=True, help="Task pool dir (task subdirs).")
@click.option("--out", required=True, help="Output dir; writes <out>/train and <out>/holdout.")
@click.option("--holdout", "n_holdout", type=int, default=40)
@click.option("--by", default="source_file", help="task.toml metadata key to group by (whole groups go to one side).")
@click.option("--seed", type=int, default=0)
def tasks_split(pool, out, n_holdout, by, seed):
    """Split a pool into train/holdout by a grouping key so no source leaks across the split."""
    import random
    import shutil
    import tomllib

    pool_p, out_p = Path(pool).expanduser(), Path(out).expanduser()
    tasks_ = [d for d in sorted(pool_p.iterdir()) if (d / "task.toml").exists()]
    groups: dict[str, list[Path]] = {}
    for d in tasks_:
        meta = tomllib.loads((d / "task.toml").read_text()).get("metadata", {})
        groups.setdefault(str(meta.get(by, d.name)), []).append(d)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    hold, n = [], 0
    for k in keys:
        if n >= n_holdout:
            break
        hold.append(k)
        n += len(groups[k])
    for side, ks in (("holdout", hold), ("train", [k for k in keys if k not in hold])):
        dest = out_p / side
        dest.mkdir(parents=True, exist_ok=True)
        for k in ks:
            for d in groups[k]:
                if not (dest / d.name).exists():
                    shutil.copytree(d, dest / d.name)
    click.echo(json.dumps({"groups": len(keys), "holdout_groups": hold, "holdout_tasks": n,
                           "train_tasks": sum(len(groups[k]) for k in keys if k not in hold)}))


@tasks.command("swesmith")
@click.option("--out", required=True, help="Output dataset dir (Harbor task dirs).")
@click.option("--n", "n_tasks", type=int, default=200)
@click.option("--repos", default=None, help="Comma-separated substrings of image names to keep.")
@click.option("--limit-repos", type=int, default=8, help="If --repos unset: keep the N images with most instances.")
@click.option("--seed", type=int, default=0)
@click.option("--pull/--no-pull", default=True, help="docker pull the images used.")
def tasks_swesmith(out, n_tasks, repos, limit_repos, seed, pull):
    """SWE-smith rows -> Harbor tasks with a fast verifier."""
    from polyloop.tasks.swesmith import build

    info = build(Path(out).expanduser(), n_tasks, limit_repos, repos.split(",") if repos else None, seed, pull)
    click.echo(json.dumps(info))
