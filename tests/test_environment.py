import json
from pathlib import Path

import pytest
import yaml

from polyloop.config import load_loop
from polyloop.environment import load_environment, resolve
from polyloop.events import LoopStore
from polyloop.stages import CycleAborted, Runner


def _loop(tmp_path: Path, **env_options) -> Path:
    cfg = {
        "name": "t", "model": "m", "tasks": "pool", "runs_dir": str(tmp_path / "runs"),
        "environment": {"kind": "tests.fake_env:ScriptedEnvironment", "options": env_options},
        "filter": {"pool_sample": 4, "rollouts_per_task": 2, "min_tasks": 1, "target_tasks": 2, "max_rounds": 2},
        "gate": {"holdout": "holdout", "holdout_limit": 4, "repeats": 2},
    }
    p = tmp_path / "loop.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_resolve_builtin_and_dotted():
    assert resolve("harbor-docker").__name__ == "HarborDockerEnvironment"
    assert resolve("tests.fake_env:ScriptedEnvironment").__name__ == "ScriptedEnvironment"
    with pytest.raises(ValueError):
        resolve("no-such-env")


def test_default_environment_is_harbor_docker(tmp_path):
    p = tmp_path / "loop.yaml"
    p.write_text(yaml.safe_dump({"name": "t", "model": "m", "tasks": "x"}))
    cfg = load_loop(p)
    assert cfg.environment.kind == "harbor-docker"


def test_filter_runs_through_environment(tmp_path):
    rewards = {"base/a": [1, 0], "base/b": [1, 1], "base/c": [0, 1], "base/d": [0, 0]}
    p = _loop(tmp_path, pool=["a", "b", "c", "d"], holdout=["h1", "h2"], rewards=rewards)
    cfg = load_loop(p)
    store = LoopStore(cfg.runs_path, cfg.name)
    cycle = store.cycle(store.new_cycle_id())
    runner = Runner(cfg, cycle, log=lambda m: None)
    assert runner.env.name == "scripted"
    runner.snapshot()
    runner.filter()
    st = cycle.state
    assert sorted(st["train_task_names"]) == ["a", "c"]          # 0 < pass rate < 1
    assert runner.env.calls[0] == {"label": "filter", "policy": "base", "n": 4, "k": 2}
    assert (cycle.subdir("filter") / "results.jsonl").exists()   # the runner, not the env, writes results


def test_evaluate_passes_policy_dicts(tmp_path):
    rewards = {"base/h1": [0, 0], "base/h2": [1, 1], "cand/h1": [1, 1], "cand/h2": [1, 1]}
    p = _loop(tmp_path, pool=["a"], holdout=["h1", "h2"], rewards=rewards)
    cfg = load_loop(p)
    store = LoopStore(cfg.runs_path, cfg.name)
    cycle = store.cycle(store.new_cycle_id())
    runner = Runner(cfg, cycle, log=lambda m: None)
    runner.snapshot()
    cycle.update(candidate={"id": "cand", "sampler_path": "tinker://c"})
    runner.evaluate()
    st = cycle.state
    assert st["eval_incumbent"] == {"h1": 0.0, "h2": 1.0}
    assert st["eval_candidate"] == {"h1": 1.0, "h2": 1.0}
    assert [c["policy"] for c in runner.env.calls] == ["base", "cand"]


def test_preflight_reports_environment_problems(tmp_path):
    p = _loop(tmp_path, pool=[], holdout=[])
    cfg = load_loop(p)
    store = LoopStore(cfg.runs_path, cfg.name)
    cycle = store.cycle(store.new_cycle_id())
    runner = Runner(cfg, cycle, log=lambda m: None)
    runner.snapshot()
    with pytest.raises(CycleAborted, match="empty task pool"):
        runner.preflight()
