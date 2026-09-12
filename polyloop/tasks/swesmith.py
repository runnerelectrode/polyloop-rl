"""SWE-smith rows -> Harbor task directories with a verifier that runs in seconds.

Harbor's own swesmith adapter installs uv, swebench and datasets inside the container at
verify time (minutes per episode). The recipe here follows Agent Lightning's evaluate():
the image already holds the bug branch; we restore the FAIL_TO_PASS test files from the
branch's parent commit and run pytest on the F2P nodes plus the PASS_TO_PASS nodes that
live in the same files. .git is moved out of /testbed at build time so the agent cannot
recover the fix from history (reward-hack #1 in the Agent Lightning report).

Task dir layout (Harbor 1.0):
  instruction.md  task.toml  environment/Dockerfile  tests/{test.sh, grade.py, config.json}
"""
from __future__ import annotations

import ast
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

GIT_DIR = "/opt/polyloop_git"

INSTRUCTION = """{problem}

---
The repository is checked out at /testbed and its Python environment is already active.
Fix the issue by editing the source files under /testbed. Do not modify or add tests; the
fix is graded by running the project's existing test suite. When you are done, end your turn.
"""

DOCKERFILE = """FROM {image}
WORKDIR /testbed
ENV PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH
RUN git checkout -q {instance_id} && mv /testbed/.git {git_dir} && git config --global --add safe.directory /testbed
RUN mkdir -p /logs/verifier
"""

TEST_SH = """#!/bin/bash
# polyloop SWE-smith verifier (Agent Lightning evaluate() recipe).
cd /testbed
export PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH
mkdir -p /logs/verifier
python - <<'PY'
import json, subprocess
cfg = json.load(open("/tests/config.json"))
files = sorted({n.split("::", 1)[0] for n in cfg["FAIL_TO_PASS"]})
env = {{"GIT_DIR": "{git_dir}", "GIT_WORK_TREE": "/testbed"}}
subprocess.run(["git", "checkout", "HEAD~1", "--", *files], env={{**__import__("os").environ, **env}}, timeout=120)
PY
XD="-p no:xdist"; python -c 'import xdist' 2>/dev/null && XD="-n4"
NODES=$(python -c 'import json,shlex; c=json.load(open("/tests/config.json")); print(" ".join(shlex.quote(n) for n in c["nodes"]))')
eval timeout {timeout} python -m pytest -rA -p no:cacheprovider $XD $NODES > /logs/test_output.log 2>&1
python /tests/grade.py
"""

GRADE_PY = r'''import json, re
cfg = json.load(open("/tests/config.json"))
out = open("/logs/test_output.log", errors="replace").read()
pat_a = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)", re.M)
pat_b = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)", re.M)
status = {}
for st, node in pat_a.findall(out):
    status[node.split(" - ")[0]] = st
for node, st in pat_b.findall(out):
    status.setdefault(node, st)
ok = {"PASSED", "XFAIL"}
f2p = [n for n in cfg["FAIL_TO_PASS"] if status.get(n) in ok]
p2p = [n for n in cfg["p2p_graded"] if status.get(n) in ok]
resolved = len(f2p) == len(cfg["FAIL_TO_PASS"]) and len(p2p) == len(cfg["p2p_graded"])
open("/logs/verifier/reward.txt", "w").write("1\n" if resolved else "0\n")
json.dump({"resolved": resolved, "f2p": [len(f2p), len(cfg["FAIL_TO_PASS"])],
           "p2p": [len(p2p), len(cfg["p2p_graded"])], "timed_out": "[timed out" in out or out.strip() == ""},
          open("/logs/verifier/report.json", "w"))
print("resolved" if resolved else "unresolved", len(f2p), "/", len(cfg["FAIL_TO_PASS"]), "f2p;", len(p2p), "/", len(cfg["p2p_graded"]), "p2p")
'''

TASK_TOML = """schema_version = "1.0"

[task]
name = "swesmith/{instance_id}"
keywords = ["debugging", "swe-smith", "polyloop"]

[metadata]
benchmark = "SWE-smith"
instance_id = "{instance_id}"
repository = "{repo}"
image = "{image}"
n_f2p = {n_f2p}
n_p2p_graded = {n_p2p}

[verifier]
timeout_sec = {verifier_timeout}

[agent]
timeout_sec = 3600
"""


def _as_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v or "").strip()
    if not s:
        return []
    try:
        return [str(x) for x in json.loads(s)]
    except json.JSONDecodeError:
        return [str(x) for x in ast.literal_eval(s)]


def load_rows(limit_repos: int | None = None, repos: list[str] | None = None, max_tests: int = 200) -> list[dict]:
    from datasets import load_dataset  # heavy import, on purpose local

    ds = load_dataset("SWE-bench/SWE-smith", split="train")
    rows = []
    for r in ds:
        if not (r.get("problem_statement") or "").strip():
            continue
        f2p, p2p = _as_list(r["FAIL_TO_PASS"]), _as_list(r["PASS_TO_PASS"])
        if not f2p or len(f2p) + len(p2p) > max_tests:
            continue
        rows.append({**r, "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p})
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_image[r["image_name"]].append(r)
    if repos:
        keep = {img for img in by_image if any(x in img for x in repos)}
    else:
        ranked = sorted(by_image, key=lambda k: -len(by_image[k]))
        keep = set(ranked[: limit_repos or len(ranked)])
    return [r for img in sorted(keep) for r in by_image[img]]


def write_task(row: dict, out_dir: Path, verifier_timeout: int = 900) -> Path:
    iid = row["instance_id"]
    f2p, p2p = row["FAIL_TO_PASS"], row["PASS_TO_PASS"]
    f2p_files = sorted({n.split("::", 1)[0] for n in f2p})
    p2p_graded = [n for n in p2p if any(n.startswith(f) for f in f2p_files)]
    d = out_dir / iid.lower().replace("/", "__")
    (d / "environment").mkdir(parents=True, exist_ok=True)
    (d / "tests").mkdir(exist_ok=True)
    (d / "instruction.md").write_text(INSTRUCTION.format(problem=row["problem_statement"].strip()))
    (d / "task.toml").write_text(TASK_TOML.format(
        instance_id=iid, repo=row["repo"], image=row["image_name"], n_f2p=len(f2p), n_p2p=len(p2p_graded),
        verifier_timeout=verifier_timeout))
    (d / "environment" / "Dockerfile").write_text(DOCKERFILE.format(image=row["image_name"], instance_id=iid, git_dir=GIT_DIR))
    (d / "tests" / "test.sh").write_text(TEST_SH.format(git_dir=GIT_DIR, timeout=verifier_timeout - 60))
    (d / "tests" / "grade.py").write_text(GRADE_PY)
    (d / "tests" / "config.json").write_text(json.dumps({
        "instance_id": iid, "image_name": row["image_name"], "repo": row["repo"],
        "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p, "p2p_graded": p2p_graded, "nodes": f2p + p2p_graded,
    }, indent=1))
    return d


def build(out_dir: Path, n_tasks: int, limit_repos: int | None, repos: list[str] | None, seed: int,
          pull: bool, max_tests: int = 200, verifier_timeout: int = 900) -> dict:
    rows = load_rows(limit_repos=limit_repos, repos=repos, max_tests=max_tests)
    rng = random.Random(seed)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_image[r["image_name"]].append(r)
    # Round-robin across images so no single repo dominates the pool.
    for lst in by_image.values():
        rng.shuffle(lst)
    chosen: list[dict] = []
    images = sorted(by_image)
    while len(chosen) < n_tasks and any(by_image.values()):
        for img in images:
            if by_image[img] and len(chosen) < n_tasks:
                chosen.append(by_image[img].pop())
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [write_task(r, out_dir, verifier_timeout) for r in chosen]
    used_images = sorted({r["image_name"] for r in chosen})
    (out_dir / "images.txt").write_text("\n".join(used_images) + "\n")
    if pull:
        for img in used_images:
            subprocess.run(["docker", "pull", "-q", img], check=False)
    return {"tasks": len(written), "images": len(used_images), "out_dir": str(out_dir)}
