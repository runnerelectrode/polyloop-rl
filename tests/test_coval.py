import json
from types import SimpleNamespace

import httpx
import pytest

from polyloop.events import LoopStore
from polyloop.harness import coval as cv
from polyloop.harness.coval import (CovalClient, CovalError, CovalTask, agent_metadata, group_simulations, judge_explanations,
                                    load_coval_tasks, policy_endpoint, policy_name, reward_of, run_coval_rollouts,
                                    seed_test_cases, session_hints_from_ledger)


class FakeCoval:
    """Enough of api.coval.dev/v1 for the harness: test cases, agents, runs, simulations, metrics."""

    def __init__(self):
        self.cases = {"tsPOOL": [{"id": f"tc{i}", "test_set_id": "tsPOOL", "input_str": f"scenario {i}", "expected_behaviors": ["x"]} for i in range(5)],
                      "tsHOLD": [{"id": f"h{i}", "test_set_id": "tsHOLD", "input_str": f"hold {i}"} for i in range(3)]}
        self.agents = {"agBASE": {"id": "agBASE", "display_name": "base", "model_type": "MODEL_TYPE_CHAT",
                                  "metadata": {"chat_endpoint": "https://x/r/policy/live/v1/chat/completions", "custom_data": "{}"}}}
        self.runs = {}
        self.sims = {}
        self.launches = []
        self.patches = []
        self.polls = 0
        self.fail_metric_fetch = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "k"
        path, q = request.url.path.replace("/v1", "", 1), dict(request.url.params)
        m = request.method
        if m == "GET" and path == "/test-cases":
            ts = q["filter"].split("=", 1)[1]
            page = int(q.get("page_token") or 0)
            items = self.cases[ts][page * 2:(page + 1) * 2]
            nxt = str(page + 1) if (page + 1) * 2 < len(self.cases[ts]) else ""
            return httpx.Response(200, json={"test_cases": items, "next_page_token": nxt})
        if m == "POST" and path == "/test-cases":
            body = json.loads(request.content)
            rec = {"id": f"new{len(self.cases[body['test_set_id']])}", **body}
            self.cases[body["test_set_id"]].append(rec)
            return httpx.Response(201, json={"test_case": rec})
        if m == "GET" and path.startswith("/agents/"):
            return httpx.Response(200, json={"agent": self.agents[path.split("/")[2]]})
        if m == "POST" and path.endswith("/duplicate"):
            src = self.agents[path.split("/")[2]]
            new = {**json.loads(json.dumps(src)), "id": f"ag{len(self.agents)}"}
            self.agents[new["id"]] = new
            return httpx.Response(201, json={"agent": new})
        if m == "PATCH" and path.startswith("/agents/"):
            body = json.loads(request.content)
            self.agents[path.split("/")[2]].update(body)
            self.patches.append((path.split("/")[2], body))
            return httpx.Response(200, json={"agent": self.agents[path.split("/")[2]]})
        if m == "POST" and path == "/runs":
            body = json.loads(request.content)
            rid = f"run{len(self.runs)}"
            self.launches.append(body)
            self.runs[rid] = {"run_id": rid, "status": "PENDING", "progress": {"total_test_cases": 0, "completed_test_cases": 0}}
            sims = []
            for tc in body["options"]["test_case_ids"]:
                for it in range(body["options"]["iteration_count"]):
                    sid = f"{rid}-{tc}-{it}"
                    score = 1.0 if (tc, it) != ("tc1", 0) else 0.0
                    status = "FAILED" if tc == "tc3" else "COMPLETED"
                    sims.append({"simulation_id": sid, "run_id": rid, "status": status, "test_case_id": tc,
                                 "metric_values": {"mA": score, "mB": 1.0} if status == "COMPLETED" else {}})
            self.sims[rid] = sims
            return httpx.Response(200, json={"run": self.runs[rid]})
        if m == "GET" and path.startswith("/runs/"):
            rid = path.split("/")[2]
            self.polls += 1
            run = self.runs[rid]
            run["status"] = "COMPLETED" if self.polls >= 2 else "IN PROGRESS"
            return httpx.Response(200, json={"run": run})
        if m == "GET" and path == "/conversations/simulated":
            rid = q["filter"].split("=", 1)[1]
            assert q["include"] == "metric_values"
            return httpx.Response(200, json={"simulated_conversations": self.sims[rid], "next_page_token": ""})
        if m == "GET" and path.endswith("/metrics") and "/conversations/simulated/" in path:
            if self.fail_metric_fetch:
                return httpx.Response(500, text="boom")
            sid = path.split("/")[3]
            score = 0.0 if "tc1-0" in sid else 1.0
            return httpx.Response(200, json={"metrics": [
                {"metric_output_id": "o1", "metric_id": "mA", "metric_name": "task completed", "value": score, "status": "COMPLETED",
                 "explanation": "The agent never asked for a phone number." if score == 0 else "All behaviours present."},
                {"metric_output_id": "o2", "metric_id": "mZ", "metric_name": "unrelated", "value": 3, "status": "COMPLETED", "explanation": "ignored"},
                {"metric_output_id": "o3", "metric_id": "mB", "metric_name": "no hallucination", "value": 1.0, "status": "COMPLETED", "explanation": None},
            ], "next_page_token": ""})
        return httpx.Response(404, text=f"no route {m} {path}")


@pytest.fixture
def fake():
    return FakeCoval()


@pytest.fixture
def client(fake):
    return CovalClient(api_base="https://api.test/v1", api_key="k", transport=httpx.MockTransport(fake.handler))


def test_client_requires_key():
    with pytest.raises(CovalError, match="API key"):
        CovalClient(api_key=None)


def test_test_cases_paginate(client):
    tasks = client.test_cases("tsPOOL")
    assert [t.task_name for t in tasks] == ["tc0", "tc1", "tc2", "tc3", "tc4"]
    assert isinstance(tasks[0], CovalTask) and tasks[0].expected_behaviors == ["x"]


def test_load_coval_tasks_limit_and_names(client):
    t = load_coval_tasks("coval://tsPOOL", client=client, limit=3, seed=0)
    assert len(t) == 3 and [x.task_name for x in t] == sorted(x.task_name for x in t)
    t2 = load_coval_tasks("coval://tsPOOL", client=client, limit=3, seed=0)
    assert [x.task_name for x in t] == [x.task_name for x in t2]  # deterministic
    assert [x.task_name for x in load_coval_tasks("coval://tsPOOL", client=client, names=["tc4", "tc0"])] == ["tc0", "tc4"]


def test_reward_and_grouping():
    assert reward_of({"metric_values": {"a": 1, "b": 0.5}}, ["a", "b"]) == 0.75
    assert reward_of({"metric_values": {"a": True}}, ["a"]) == 1.0
    assert reward_of({"metric_values": {"a": 1}}, ["a", "b"]) is None
    assert reward_of({"metric_values": {"a": "yes"}}, ["a"]) is None
    sims = [{"simulation_id": "s1", "status": "COMPLETED", "test_case_id": "t", "metric_values": {"a": 1}},
            {"simulation_id": "s2", "status": "COMPLETED", "test_case_id": "t", "metric_values": {"a": 0}},
            {"simulation_id": "s3", "status": "FAILED", "test_case_id": "t", "metric_values": {}},
            {"simulation_id": "s4", "status": "COMPLETED", "test_case_id": "t", "metric_values": {}}]
    g = group_simulations(sims, ["a"])["t"]
    assert g == {"rewards": [1.0, 0.0], "sessions": ["s1", "s2"], "failed": 1, "unscored": 1}


def test_judge_explanations_only_reward_metrics_and_truncated():
    ms = [{"metric_id": "a", "metric_name": "A", "value": 0, "status": "COMPLETED", "explanation": "x" * 2000},
          {"metric_id": "z", "metric_name": "Z", "value": 1, "status": "COMPLETED", "explanation": "nope"},
          {"metric_id": "b", "metric_name": "B", "value": 1, "status": "IN QUEUE", "explanation": "pending"}]
    out = judge_explanations(ms, ["a", "b"], max_chars=200)
    assert out.startswith("A = 0: xxx") and "..." in out and "nope" not in out and "pending" not in out and len(out) < 220
    assert judge_explanations([], ["a"], 100) is None


def test_session_hints_from_ledger(tmp_path):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [
        {"session": "f", "reward": 0.0, "explanation": "missed the number"},
        {"session": "p", "reward": 1.0, "explanation": "fine"},
        {"session": "n", "reward": None},
    ]) + "\n")
    h = session_hints_from_ledger(ledger, pass_value=1.0, failures_only=True)
    assert set(h) == {"f"} and h["f"] == "Hindsight from the evaluator: this conversation failed (score 0.00).\nmissed the number"
    h2 = session_hints_from_ledger(ledger, pass_value=1.0, failures_only=False)
    assert set(h2) == {"f", "p"} and h2["p"].startswith("Hindsight from the evaluator: this conversation passed")
    assert session_hints_from_ledger(tmp_path / "missing.jsonl", pass_value=1.0, failures_only=True) == {}


def test_policy_names_and_endpoint():
    assert policy_name(None) == "base"
    assert policy_name("voice-coval-20260917T010203Z") == "voice-coval-20260917T010203Z"
    assert policy_name("a/b c") == "a-b-c"
    assert policy_endpoint("https://x.example/", "cand") == "https://x.example/r/policy/cand/v1/chat/completions"


def test_agent_metadata_shape():
    meta = agent_metadata("https://x/r/policy/p/v1/chat/completions", auth_header="Bearer t", temperature=0.7, max_tokens=128)
    tpl = meta["input_template"]
    assert '"messages": {{messages}}' in tpl and '"temperature": 0.7' in tpl and '"max_tokens": 128' in tpl
    assert meta["custom_headers"]["X-Session-Id"] == "{{simulation_output_id}}"
    assert meta["response_message_path"] == "choices.0.message.content"
    assert meta["authorization_header"] == "Bearer t"
    assert "authorization_header" not in agent_metadata("https://x", auth_header=None, temperature=1, max_tokens=1)


def test_seed_skips_existing(client, fake):
    made = seed_test_cases(client, "tsPOOL", [{"input": "scenario 0", "expected": ["x"]}, {"input": "brand new", "expected": ["y"], "description": "d"}])
    assert made == ["new5"]
    assert fake.cases["tsPOOL"][-1]["input_type"] == "SCENARIO"


def _cfg(tmp_path, **over):
    coval = SimpleNamespace(agent_id="agBASE", persona_id="pe", metric_ids=["mA", "mB"], public_url="https://pub.example",
                            concurrency=4, poll_seconds=0, run_timeout=60, temperature=1.0, max_tokens=64,
                            pass_value=1.0, hint_failures_only=True, max_judge_chars=500, ledger="coval_sessions.jsonl")
    for k, v in over.items():
        setattr(coval, k, v)
    return SimpleNamespace(name="voice-coval", coval=coval, proxy_url="http://127.0.0.1:8787")


def test_run_coval_rollouts_end_to_end(tmp_path, client, fake, monkeypatch):
    posted = []
    monkeypatch.setattr(cv.httpx, "post", lambda url, **kw: posted.append((url, kw)) or httpx.Response(200, json={}))
    monkeypatch.setenv("POLYLOOP_PROXY_TOKEN", "tok")
    monkeypatch.setattr(cv.time, "sleep", lambda s: None)
    cfg = _cfg(tmp_path)
    store = LoopStore(tmp_path, cfg.name)
    tasks = load_coval_tasks("coval://tsPOOL", client=client, names=["tc0", "tc1", "tc3"])
    seen = []
    out = tmp_path / "out"
    out.mkdir()
    res = run_coval_rollouts(cfg=cfg, client=client, store=store, tasks=tasks, policy_id="voice-coval-c1", sampler_path="tinker://c1",
                             k=2, label="filter", out=out, on_result=seen.append, log=lambda m: None)
    by = {r.task: r for r in res}
    assert by["tc0"].rewards == [1.0, 1.0] and by["tc0"].error is None
    assert by["tc1"].rewards == [0.5, 1.0]            # (mA=0 + mB=1)/2 on iteration 0
    assert by["tc3"].error and "no scored simulations" in by["tc3"].error and by["tc3"].rewards == []
    assert [r.task for r in seen] == ["tc0", "tc1", "tc3"]
    # proxy: policy registered with the token, every session flushed
    reg = [p for p in posted if p[0].endswith("/admin/policies")]
    assert reg[0][1]["json"] == {"name": "voice-coval-c1", "sampler_path": "tinker://c1"}
    assert reg[0][1]["headers"] == {"Authorization": "Bearer tok"}
    assert sum(1 for p in posted if "/admin/session/" in p[0]) == 6
    # coval: one run against a duplicated agent pointing at the policy route, with the bearer token
    assert len(fake.launches) == 1
    launch = fake.launches[0]
    assert launch["options"] == {"iteration_count": 2, "concurrency": 4, "test_case_ids": ["tc0", "tc1", "tc3"]}
    assert launch["metric_ids"] == ["mA", "mB"] and launch["persona_id"] == "pe" and launch["test_set_id"] == "tsPOOL"
    agent = fake.agents[launch["agent_id"]]
    assert agent["metadata"]["chat_endpoint"] == "https://pub.example/r/policy/voice-coval-c1/v1/chat/completions"
    assert agent["metadata"]["authorization_header"] == "Bearer tok"
    assert agent["metadata"]["custom_data"] == "{}"       # base metadata kept
    assert store.cache("coval_agents")["agBASE:voice-coval-c1"]["agent_id"] == launch["agent_id"]
    # ledger: reward + judge explanation per session; results.jsonl carries the session ids
    ledger = [json.loads(l) for l in (tmp_path / cfg.name / "coval_sessions.jsonl").read_text().splitlines()]
    assert len(ledger) == 6
    failed = [r for r in ledger if r["session"] == "run0-tc1-0"][0]
    assert failed["reward"] == 0.5 and "never asked for a phone number" in failed["explanation"]
    assert failed["explanation"].startswith("task completed = 0.0:")
    assert [r for r in ledger if r["task"] == "tc3"][0]["reward"] is None
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert {r["task"]: r["sessions"] for r in rows}["tc1"] == ["run0-tc1-0", "run0-tc1-1"]
    # second call reuses the cached agent
    run_coval_rollouts(cfg=cfg, client=client, store=store, tasks=tasks[:1], policy_id="voice-coval-c1", sampler_path="tinker://c1",
                       k=1, label="evaluate.candidate", log=lambda m: None)
    assert fake.launches[1]["agent_id"] == launch["agent_id"]
    assert len([a for a in fake.agents if a != "agBASE"]) == 1
    hints = session_hints_from_ledger(tmp_path / cfg.name / "coval_sessions.jsonl", pass_value=1.0, failures_only=True)
    assert set(hints) == {"run0-tc1-0"}


def test_run_requires_proxy_url(tmp_path, client):
    cfg = _cfg(tmp_path)
    cfg.proxy_url = None
    with pytest.raises(CovalError, match="proxy_url"):
        run_coval_rollouts(cfg=cfg, client=client, store=LoopStore(tmp_path, "x"), tasks=client.test_cases("tsPOOL")[:1],
                           policy_id=None, sampler_path=None, k=1, label="eval", log=lambda m: None)


def test_explanation_fetch_failure_is_tolerated(tmp_path, client, fake, monkeypatch):
    monkeypatch.setattr(cv.httpx, "post", lambda url, **kw: httpx.Response(200, json={}))
    monkeypatch.setattr(cv.time, "sleep", lambda s: None)
    fake.fail_metric_fetch = True
    cfg = _cfg(tmp_path)
    res = run_coval_rollouts(cfg=cfg, client=client, store=LoopStore(tmp_path, cfg.name), tasks=client.test_cases("tsPOOL")[:2],
                             policy_id=None, sampler_path=None, k=1, label="eval", log=lambda m: None)
    assert [r.mean for r in res] == [1.0, 0.5]
    ledger = [json.loads(l) for l in (tmp_path / cfg.name / "coval_sessions.jsonl").read_text().splitlines()]
    assert all(r["explanation"] is None for r in ledger)
