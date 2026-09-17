"""Coval as the environment and the verifier.

The pydantic-v2 and swe-mini loops measure a policy by running it in Docker sandboxes against
tests. A voice or chat agent has no tests: its environment is a caller and its verifier is a
judge. Coval (coval.ai) provides both, behind a REST API: a *persona* (the simulated caller), a
*test set* of scenarios, an *agent* (how Coval reaches the thing under test, here the proxy) and
*metrics* (LLM judges, deterministic checks, audio measurements) that score every simulated
conversation. This module maps that onto polyloop's rollout contract:

    task            = one Coval test case (a scenario + expected behaviours)
    K episodes      = one Coval run with iteration_count=K over the chosen test case ids
    reward          = mean of the configured metrics on each simulated conversation
    trace session   = the Coval simulation id (the proxy gets it as X-Session-Id)
    hindsight hint  = the judge's explanation, joined to the traces by session id

Every simulation is a session through `polyloop proxy`, so the tokens the policy produced are
captured the same way as any other traffic, and the OPSD stage trains on them with the judge's
verdict as the hint. Which adapter answers a run is selected by the URL Coval calls: the proxy
serves `/r/policy/<name>/v1/chat/completions` per registered policy, and one Coval agent per
policy (a duplicate of the base agent with that chat_endpoint) is created lazily and cached.

Only the public v1 API is used (https://docs.coval.ai/api-reference/v1/introduction), through
httpx, so no SDK is required. Auth is the X-API-Key header.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from polyloop.events import now_iso

COVAL_PREFIX = "coval://"
POLICY_ROUTE = "/r/policy/{policy}/v1/chat/completions"


def is_coval_dataset(dataset: str) -> bool:
    return str(dataset).startswith(COVAL_PREFIX)


def test_set_id(dataset: str) -> str:
    if not is_coval_dataset(dataset):
        raise ValueError(f"not a coval dataset: {dataset!r}")
    return dataset[len(COVAL_PREFIX):].strip("/")


def policy_name(policy_id: str | None) -> str:
    """A policy id (`base`, `voice-coval-20260917T...`) as a URL path segment."""
    pid = policy_id or "base"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", pid).strip("-") or "base"


def policy_endpoint(public_url: str, name: str) -> str:
    return public_url.rstrip("/") + POLICY_ROUTE.format(policy=name)


@dataclass
class CovalTask:
    """The slice of a Coval test case the loop needs; `task_name` is what every stage keys on."""
    task_name: str                       # the test case id
    test_set_id: str
    input_str: str = ""
    expected_behaviors: list[str] = field(default_factory=list)
    description: str | None = None
    input_type: str | None = None

    @classmethod
    def from_api(cls, rec: dict) -> "CovalTask":
        return cls(task_name=rec["id"], test_set_id=rec.get("test_set_id", ""), input_str=rec.get("input_str") or "",
                   expected_behaviors=list(rec.get("expected_behaviors") or []), description=rec.get("description"),
                   input_type=rec.get("input_type"))


class CovalError(RuntimeError):
    pass


class CovalClient:
    def __init__(self, api_base: str = "https://api.coval.dev/v1", api_key: str | None = None, timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise CovalError("Coval API key missing (set COVAL_API_KEY or coval.api_key_env)")
        self.http = httpx.Client(base_url=api_base.rstrip("/"), headers={"X-API-Key": api_key}, timeout=timeout,
                                 transport=transport)

    @classmethod
    def from_env(cls, api_base: str = "https://api.coval.dev/v1", api_key_env: str = "COVAL_API_KEY", **kw) -> "CovalClient":
        return cls(api_base=os.environ.get("COVAL_API_BASE", api_base), api_key=os.environ.get(api_key_env), **kw)

    # ---- transport ----------------------------------------------------
    def _req(self, method: str, path: str, **kw) -> dict:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            raise CovalError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else {}

    def _paged(self, path: str, key: str, params: dict | None = None, page_size: int = 100) -> list[dict]:
        out: list[dict] = []
        token = None
        while True:
            q = dict(params or {}, page_size=page_size)
            if token:
                q["page_token"] = token
            data = self._req("GET", path, params=q)
            out.extend(data.get(key) or [])
            token = data.get("next_page_token")
            if not token:
                return out

    # ---- resources ----------------------------------------------------
    def test_cases(self, ts_id: str) -> list[CovalTask]:
        recs = self._paged("/test-cases", "test_cases", {"filter": f"test_set_id={ts_id}", "order_by": "create_time"})
        return [CovalTask.from_api(r) for r in recs]

    def test_set(self, ts_id: str) -> dict:
        return self._req("GET", f"/test-sets/{ts_id}").get("test_set", {})

    def agent(self, agent_id: str) -> dict:
        return self._req("GET", f"/agents/{agent_id}").get("agent", {})

    def persona(self, persona_id: str) -> dict:
        return self._req("GET", f"/personas/{persona_id}").get("persona", {})

    def metrics(self, metric_ids: list[str]) -> list[dict]:
        return [self._req("GET", f"/metrics/{m}").get("metric", {}) for m in metric_ids]

    def agent_for_policy(self, base_agent_id: str, name: str, chat_endpoint: str, *, auth_header: str | None,
                         temperature: float, max_tokens: int, loop_name: str) -> str:
        """Duplicate the base agent and point the copy at one policy route on the proxy. The
        request shape is the OpenAI chat body (input_template), the reply is read from
        choices.0.message.content, and the simulation id rides X-Session-Id so the proxy's
        trace for this conversation is keyed by the same id Coval reports the scores under."""
        dup = self._req("POST", f"/agents/{base_agent_id}/duplicate", json={"display_name": f"{loop_name} policy {name}"})
        agent = dup.get("agent") or dup
        agent_id = agent["id"] if "id" in agent else agent["agent_id"]
        base = self.agent(base_agent_id)
        meta = dict(base.get("metadata") or {})
        meta.update(agent_metadata(chat_endpoint, auth_header=auth_header, temperature=temperature, max_tokens=max_tokens))
        self._req("PATCH", f"/agents/{agent_id}", json={"display_name": f"{loop_name} policy {name}", "metadata": meta,
                                                       "customer_agent_id": f"polyloop/{loop_name}/{name}"})
        return agent_id

    def launch_run(self, *, agent_id: str, persona_id: str, ts_id: str, metric_ids: list[str], test_case_ids: list[str],
                   iterations: int, concurrency: int, display_name: str, tags: list[str]) -> dict:
        body = {
            "agent_id": agent_id, "persona_id": persona_id, "test_set_id": ts_id, "metric_ids": metric_ids,
            "options": {"iteration_count": iterations, "concurrency": concurrency, "test_case_ids": test_case_ids},
            "metadata": {"display_name": display_name[:200], "tags": tags[:20], "created_by": "polyloop"},
        }
        return self._req("POST", "/runs", json=body)["run"]

    def run(self, run_id: str) -> dict:
        return self._req("GET", f"/runs/{run_id}")["run"]

    def wait_run(self, run_id: str, *, poll: int, timeout: int, log: Callable[[str], None] = print) -> dict:
        t0 = time.monotonic()
        last = ""
        while True:
            run = self.run(run_id)
            status = run.get("status")
            prog = run.get("progress") or {}
            line = f"coval run {run_id}: {status} {prog.get('completed_test_cases', '?')}/{prog.get('total_test_cases', '?')} done, {prog.get('failed_test_cases', 0)} failed"
            if line != last:
                log(line)
                last = line
            if status in ("COMPLETED", "FAILED", "CANCELLED", "DELETED"):
                return run
            if time.monotonic() - t0 > timeout:
                raise CovalError(f"run {run_id} still {status} after {timeout}s")
            time.sleep(poll)

    def simulations(self, run_id: str) -> list[dict]:
        return self._paged("/conversations/simulated", "simulated_conversations",
                           {"filter": f"run_id={run_id}", "include": "metric_values"}, page_size=200)

    def simulation_metrics(self, simulation_id: str) -> list[dict]:
        return self._paged(f"/conversations/simulated/{simulation_id}/metrics", "metrics", {"view": "BASIC"})


def agent_metadata(chat_endpoint: str, *, auth_header: str | None, temperature: float, max_tokens: int) -> dict:
    """The CHAT-agent metadata keys documented at docs.coval.ai/concepts/agents/connections/chat."""
    meta = {
        "chat_endpoint": chat_endpoint,
        "custom_headers": {"X-Session-Id": "{{simulation_output_id}}", "X-Turn-Type": "main"},
        "input_template": json.dumps({"model": "polyloop", "messages": "{{messages}}", "temperature": temperature,
                                      "max_tokens": max_tokens}).replace('"{{messages}}"', "{{messages}}"),
        "response_message_path": "choices.0.message.content",
        "response_format": "chat_completions",
        "strip_message_timestamps": True,
    }
    if auth_header:
        meta["authorization_header"] = auth_header
    return meta


# ---- tasks -------------------------------------------------------------
def load_coval_tasks(dataset: str, *, client: CovalClient | None = None, limit: int | None = None, seed: int = 0,
                     names: list[str] | None = None) -> list[CovalTask]:
    client = client or CovalClient.from_env()
    tasks = client.test_cases(test_set_id(dataset))
    if names is not None:
        wanted = set(names)
        tasks = [t for t in tasks if t.task_name in wanted]
    if limit and len(tasks) > limit:
        rng = random.Random(seed)
        tasks = sorted(rng.sample(tasks, limit), key=lambda t: t.task_name)
    return tasks


def seed_test_cases(client: CovalClient, ts_id: str, cases: list[dict]) -> list[str]:
    """Create SCENARIO test cases from {"input": str, "expected": [str], "description": str}
    records; returns the new test case ids. Existing cases with the same input are skipped."""
    have = {t.input_str.strip() for t in client.test_cases(ts_id)}
    made = []
    for c in cases:
        if c["input"].strip() in have:
            continue
        body = {"test_set_id": ts_id, "input_str": c["input"], "input_type": c.get("input_type", "SCENARIO"),
                "expected_behaviors": list(c.get("expected", [])), "description": c.get("description", "")}
        r = client._req("POST", "/test-cases", json=body)
        tc = r.get("test_case") or r
        made.append(tc.get("id", "?"))
    return made


# ---- rewards ------------------------------------------------------------
def reward_of(sim: dict, metric_ids: list[str]) -> float | None:
    """Mean of the reward metrics on one simulation; None unless every one of them is a number
    (Coval's list endpoint only carries numeric metric values; a string metric never scores)."""
    vals = sim.get("metric_values") or {}
    got = []
    for m in metric_ids:
        v = vals.get(m)
        if isinstance(v, bool):
            got.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            got.append(float(v))
        else:
            return None
    return sum(got) / len(got) if got else None


def group_simulations(sims: list[dict], metric_ids: list[str]) -> dict[str, dict]:
    """Per test case: rewards of its completed simulations, ids of the sessions, and errors."""
    by_task: dict[str, dict] = {}
    for sim in sims:
        task = sim.get("test_case_id") or "?"
        g = by_task.setdefault(task, {"rewards": [], "sessions": [], "failed": 0, "unscored": 0})
        if sim.get("status") != "COMPLETED":
            g["failed"] += 1
            continue
        r = reward_of(sim, metric_ids)
        if r is None:
            g["unscored"] += 1
            continue
        g["rewards"].append(r)
        g["sessions"].append(sim["simulation_id"])
    return by_task


def judge_explanations(metrics: list[dict], metric_ids: list[str], max_chars: int) -> str | None:
    """Fold the judges' explanations for the reward metrics into one hint block."""
    parts = []
    for m in metrics:
        if m.get("metric_id") not in metric_ids or m.get("status") != "COMPLETED":
            continue
        text = (m.get("explanation") or "").strip()
        if not text:
            continue
        v = m.get("value")
        label = f"{m.get('metric_name') or m['metric_id']} = {v}"
        parts.append(f"{label}: {text}")
    if not parts:
        return None
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars // 2] + "\n...\n" + out[-max_chars // 2:]
    return out


def session_hints_from_ledger(ledger: Path, *, pass_value: float, failures_only: bool) -> dict[str, str]:
    """session id -> hindsight text for the OPSD rows, from the per-session ledger."""
    hints: dict[str, str] = {}
    if not ledger.exists():
        return hints
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        r = rec.get("reward")
        if r is None:
            continue
        passed = r >= pass_value
        if failures_only and passed:
            continue
        verdict = "passed" if passed else "failed"
        text = f"Hindsight from the evaluator: this conversation {verdict} (score {r:.2f})."
        if rec.get("explanation"):
            text += "\n" + rec["explanation"]
        hints[rec["session"]] = text
    return hints


# ---- rollouts -----------------------------------------------------------
def run_coval_rollouts(
    *,
    cfg,                                   # LoopConfig
    client: CovalClient,
    store,                                 # LoopStore (policy->agent cache, session ledger)
    tasks: list[CovalTask],
    policy_id: str | None,
    sampler_path: str | None,
    k: int,
    label: str,
    out: Path | None = None,
    on_result=None,
    log: Callable[[str], None] = print,
):
    """One Coval run per chunk of <=100 test cases, K iterations each; rewards per task.

    Returns TaskResult objects exactly like the sandbox path, so filter, evaluate and the
    receipt do not know which environment produced them."""
    from polyloop.harness.rollout import TaskResult

    cv = cfg.coval
    name = policy_name(policy_id)
    register_policy(cfg.proxy_url, name, sampler_path, log=log)
    agent_id = ensure_policy_agent(cfg, client, store, name, log=log)
    ts_id = tasks[0].test_set_id if tasks and tasks[0].test_set_id else None
    if ts_id is None:
        raise CovalError("tasks carry no test_set_id")
    results: dict[str, TaskResult] = {t.task_name: TaskResult(task=t.task_name) for t in tasks}
    ledger = store.root / cv.ledger
    ids = [t.task_name for t in tasks]
    for chunk_start in range(0, len(ids), 100):
        chunk = ids[chunk_start:chunk_start + 100]
        t0 = time.monotonic()
        run = client.launch_run(agent_id=agent_id, persona_id=cv.persona_id, ts_id=ts_id, metric_ids=cv.metric_ids,
                                test_case_ids=chunk, iterations=k, concurrency=cv.concurrency,
                                display_name=f"{cfg.name} {label} {name}", tags=[f"polyloop", cfg.name, label, name])
        run_id = run["run_id"]
        log(f"coval: launched run {run_id} ({len(chunk)} test cases x {k}) against policy {name} (agent {agent_id})")
        run = client.wait_run(run_id, poll=cv.poll_seconds, timeout=cv.run_timeout, log=log)
        if run.get("status") != "COMPLETED":
            raise CovalError(f"run {run_id} ended {run.get('status')}: {run.get('error') or run.get('error_status')}")
        sims = client.simulations(run_id)
        grouped = group_simulations(sims, cv.metric_ids)
        seconds = time.monotonic() - t0
        # Flush the last turn of every session in the proxy (Coval does not send X-Session-Done).
        for sim in sims:
            finish_session(cfg.proxy_url, sim["simulation_id"])
        explanations: dict[str, str | None] = {}
        for sim in sims:
            if sim.get("status") != "COMPLETED":
                continue
            try:
                explanations[sim["simulation_id"]] = judge_explanations(client.simulation_metrics(sim["simulation_id"]),
                                                                        cv.metric_ids, cv.max_judge_chars)
            except CovalError as exc:
                log(f"coval: explanation fetch failed for {sim['simulation_id']}: {exc}")
                explanations[sim["simulation_id"]] = None
        with ledger.open("a") as lf:
            for sim in sims:
                r = reward_of(sim, cv.metric_ids) if sim.get("status") == "COMPLETED" else None
                rec = {"ts": now_iso(), "session": sim["simulation_id"], "task": sim.get("test_case_id"), "run_id": run_id,
                       "policy": name, "policy_id": policy_id or "base", "label": label, "status": sim.get("status"),
                       "reward": r, "metric_values": sim.get("metric_values") or {},
                       "explanation": explanations.get(sim["simulation_id"])}
                lf.write(json.dumps(rec) + "\n")
                if out:
                    with (out / "rewards.jsonl").open("a") as f:
                        f.write(json.dumps(rec) + "\n")
        for task in chunk:
            res = results[task]
            g = grouped.get(task)
            res.seconds = seconds / max(1, len(chunk))
            if not g or not g["rewards"]:
                res.error = f"no scored simulations (failed={g['failed'] if g else 0}, unscored={g['unscored'] if g else 0})"
            else:
                res.rewards = g["rewards"]
                res.turns = [0] * len(g["rewards"])
                res.tokens = [0] * len(g["rewards"])
                res.stop_reasons = [None] * len(g["rewards"])
            if out:
                with (out / "results.jsonl").open("a") as f:
                    f.write(json.dumps({**res.to_dict(), "sessions": (g or {}).get("sessions", []), "run_id": run_id}) + "\n")
            if on_result:
                on_result(res)
    return [results[t] for t in ids]


def ensure_policy_agent(cfg, client: CovalClient, store, name: str, log=print) -> str:
    cache = store.cache("coval_agents")
    key = f"{cfg.coval.agent_id}:{name}"
    if key in cache:
        return cache[key]["agent_id"]
    endpoint = policy_endpoint(cfg.coval.public_url, name)
    token = os.environ.get("POLYLOOP_PROXY_TOKEN")
    agent_id = client.agent_for_policy(cfg.coval.agent_id, name, endpoint, auth_header=f"Bearer {token}" if token else None,
                                       temperature=cfg.coval.temperature, max_tokens=cfg.coval.max_tokens, loop_name=cfg.name)
    cache[key] = {"agent_id": agent_id, "chat_endpoint": endpoint, "created": now_iso()}
    store.save_cache("coval_agents", cache)
    log(f"coval: agent {agent_id} -> {endpoint}")
    return agent_id


def _proxy_headers() -> dict:
    token = os.environ.get("POLYLOOP_PROXY_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def register_policy(proxy_url: str | None, name: str, sampler_path: str | None, log=print) -> None:
    if not proxy_url:
        raise CovalError("proxy_url must be set in loop.yaml for a coval loop (the proxy serves the policies)")
    r = httpx.post(proxy_url.rstrip("/") + "/admin/policies", json={"name": name, "sampler_path": sampler_path},
                   headers=_proxy_headers(), timeout=600)
    if r.status_code >= 400:
        raise CovalError(f"proxy refused policy {name}: {r.status_code} {r.text[:300]}")
    log(f"proxy: policy {name} -> {sampler_path or 'base model'}")


def finish_session(proxy_url: str | None, session: str) -> None:
    if not proxy_url:
        return
    try:
        httpx.post(proxy_url.rstrip("/") + f"/admin/session/{session}/done", headers=_proxy_headers(), timeout=10)
    except Exception:
        pass
