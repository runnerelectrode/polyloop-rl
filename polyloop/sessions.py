"""`polyloop sessions`: drive a real coding agent (mini-swe-agent) through the proxy on tasks.

Two uses of the same code path:
- laptop: a few sessions in a local working copy of each task, so the recorded traces are the
  real agent-to-endpoint traffic (the ingest proof);
- night driver on the node: many sessions in parallel in Docker, for trace volume.

Each session gets its own `X-Session-Id` (via litellm `extra_headers`), is closed with
`/admin/session/<id>/done` so the last turn is flushed, and is verified locally afterwards so the
outcome is logged next to the trace (informational; training rewards come from the harness).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

STRICT = ["-W", "error::pydantic.warnings.PydanticDeprecatedSince20"]


def _verify(workdir: Path, python: str) -> bool:
    try:
        r = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *STRICT, "test_app.py"],
                           cwd=workdir, capture_output=True, text=True, timeout=180)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def run_one(task_dir: Path, *, agent_cfg: Path, proxy: str, python: str, mini_bin: str, log_dir: Path,
            env_class: str = "local", image: str = "polyloop-pydantic:latest") -> dict:
    sid = f"{task_dir.name}-{uuid.uuid4().hex[:8]}"
    cfg = yaml.safe_load(agent_cfg.read_text())
    cfg.setdefault("model", {}).setdefault("model_kwargs", {})
    cfg["model"]["model_kwargs"]["api_base"] = proxy.rstrip("/") + "/v1"
    cfg["model"]["model_kwargs"]["extra_headers"] = {"X-Session-Id": sid, "X-Turn-Type": "main"}
    work = Path(tempfile.mkdtemp(prefix="plsess-"))
    for f in ("app.py", "test_app.py"):
        shutil.copy(task_dir / "environment" / f, work / f)
    cfg["environment"] = {"environment_class": env_class, "timeout": 60}
    if env_class == "docker":
        cfg["environment"].update({"image": image, "cwd": "/testbed"})
    else:
        cfg["environment"].update({"cwd": str(work), "env": {"PATH": str(Path(python).parent) + ":" + os.environ.get("PATH", "")}})
    cfg_path = work / "agent.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    task = (task_dir / "instruction.md").read_text()
    log_dir.mkdir(parents=True, exist_ok=True)
    traj = log_dir / f"{sid}.traj.json"
    t0 = time.monotonic()
    proc = subprocess.run([mini_bin, "-y", "-c", str(cfg_path), "-t", task, "-o", str(traj)],
                          cwd=work, capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "MSWEA_COST_TRACKING": "ignore_errors", "OPENAI_API_KEY": "polyloop"})
    seconds = time.monotonic() - t0
    try:
        import httpx

        httpx.post(proxy.rstrip("/") + f"/admin/session/{sid}/done", timeout=10)
    except Exception:
        pass
    passed = _verify(work, python) if env_class == "local" else None
    rec = {"session": sid, "task": task_dir.name, "seconds": round(seconds, 1), "exit": proc.returncode,
           "passed_strict": passed, "workdir": str(work), "stderr_tail": proc.stderr[-400:]}
    with (log_dir / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_many(task_dirs: list[Path], parallel: int, **kw) -> list[dict]:
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        return list(ex.map(lambda d: run_one(d, **kw), task_dirs))
