"""pydantic v1-idiom -> v2 migration tasks, built from pydantic's own v2 test suite.

Each task hoists the model classes of one test function into ``app.py``, rewrites them to
deprecated v1 idioms (``@validator``, ``class Config``, ``.dict()``, ``parse_obj``, ``regex=``,
...), and keeps the test's assertions as ``test_app.py``. The verifier runs pytest with pydantic's
deprecation warnings turned into errors, so a task fails while any v1 idiom remains and passes
once the migration is idiomatic v2 with behavior preserved.

A triple filter keeps only tasks where (a) the original passes strictly, (b) the downgraded
module passes loosely (semantics preserved by the rewrite), and (c) the downgraded module fails
strictly (there is something to migrate).

Task layout (Harbor 1.0): instruction.md, task.toml, environment/Dockerfile,
environment/app.py, environment/test_app.py, tests/test.sh, tests/test_app.py (hidden copy).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

STRICT = "-W error::pydantic.warnings.PydanticDeprecatedSince20"

INSTRUCTION = """`/testbed/app.py` defines pydantic models written against the pydantic v1 API. The project
now runs pydantic v2, and deprecation warnings are treated as errors in CI.

Migrate `app.py` to idiomatic pydantic v2 so that the test suite passes with
`pytest -W error::pydantic.warnings.PydanticDeprecatedSince20`. Preserve behavior: the tests in
`test_app.py` must keep passing unchanged. Only edit `app.py`; do not edit the tests. Run the
tests to check your work. When they pass with deprecations as errors, end your turn.
"""

DOCKERFILE = """FROM polyloop-pydantic:latest
WORKDIR /testbed
COPY app.py test_app.py /testbed/
"""

BASE_IMAGE_DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir "pydantic=={version}" pytest email-validator
WORKDIR /testbed
"""

TEST_SH = """#!/bin/bash
cd /testbed
cp /tests/test_app.py /testbed/test_app.py
mkdir -p /logs/verifier
timeout 120 python -m pytest -q -p no:cacheprovider -W error::pydantic.warnings.PydanticDeprecatedSince20 test_app.py > /logs/test_output.log 2>&1
rc=$?
if [ "$rc" = "0" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
tail -n 5 /logs/test_output.log
"""

TASK_TOML = """schema_version = "1.0"

[task]
name = "pydantic-v2/{name}"
keywords = ["migration", "pydantic", "polyloop"]

[metadata]
benchmark = "polyloop-pydantic-v2"
source_file = "{source_file}"
source_test = "{source_test}"
idioms = {idioms}

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 1200
"""

# ---- v2 -> v1 idiom rewrite (mechanical, behavior-preserving under pydantic v2) -------------

def _downgrade(src: str) -> tuple[str, list[str]]:
    """Return (rewritten source, idioms introduced). Only rules with exact v2 equivalents."""
    idioms: list[str] = []
    out = src

    def sub(pattern, repl, name, flags=0):
        nonlocal out
        new, n = re.subn(pattern, repl, out, flags=flags)
        if n:
            idioms.append(name)
            out = new

    # method / classmethod renames
    sub(r"\.model_dump_json\(", ".json(", "json()")
    sub(r"\.model_dump\(", ".dict(", "dict()")
    sub(r"\.model_validate_json\(", ".parse_raw(", "parse_raw()")
    sub(r"\.model_validate\(", ".parse_obj(", "parse_obj()")
    sub(r"\.model_copy\(", ".copy(", "copy()")
    sub(r"\.model_json_schema\(", ".schema(", "schema()")
    sub(r"\.model_construct\(", ".construct(", "construct()")
    sub(r"\.model_fields\b", ".__fields__", "__fields__")
    sub(r"\.model_fields_set\b", ".__fields_set__", "__fields_set__")
    # Field(pattern=...) -> Field(regex=...)
    sub(r"\bpattern=", "regex=", "Field(regex=)")
    # field_validator -> validator (drop @classmethod that follows it)
    sub(r"@field_validator\(([^)]*?),\s*mode=['\"]before['\"]\)\s*\n(\s*)@classmethod\n",
        r"@validator(\1, pre=True)\n", "validator(pre=True)")
    sub(r"@field_validator\(([^)]*?),\s*mode=['\"]after['\"]\)\s*\n(\s*)@classmethod\n", r"@validator(\1)\n", "validator")
    sub(r"@field_validator\(([^)]*?)\)\s*\n(\s*)@classmethod\n", r"@validator(\1)\n", "validator")
    # model_validator(mode="before") + classmethod -> root_validator(pre=True)
    sub(r"@model_validator\(mode=['\"]before['\"]\)\s*\n(\s*)@classmethod\n",
        r"@root_validator(pre=True)\n", "root_validator(pre=True)")
    # model_config = ConfigDict(k=v, ...) -> class Config: k = v  (single-line only)
    def cfg(m):
        indent = m.group(1)
        body = m.group(2)
        pairs = [p.strip() for p in body.split(",") if p.strip()]
        lines = []
        for p in pairs:
            k, _, v = p.partition("=")
            k = k.strip()
            k = {"from_attributes": "orm_mode", "populate_by_name": "allow_population_by_field_name",
                 "str_strip_whitespace": "anystr_strip_whitespace", "str_to_lower": "anystr_lower",
                 "validate_default": "validate_all", "json_schema_extra": "schema_extra"}.get(k, k)
            lines.append(f"{indent}    {k} = {v.strip()}")
        return f"{indent}class Config:\n" + "\n".join(lines) + "\n"
    new, n = re.subn(r"^([ \t]*)model_config\s*=\s*ConfigDict\(\s*([^()]*?)\s*,?\s*\)[ \t]*\n", cfg, out, flags=re.M | re.S)
    if n:
        idioms.append("class Config")
        out = new
    # imports: make sure v1 names are importable from pydantic (they are, with deprecation)
    needed = []
    if "@validator(" in out and not re.search(r"\bvalidator\b", src):
        needed.append("validator")
    if "@root_validator(" in out and not re.search(r"\broot_validator\b", src):
        needed.append("root_validator")
    if needed:
        out = re.sub(r"^(from pydantic import )(.*)$", lambda m: f"{m.group(1)}{m.group(2)}, {', '.join(needed)}",
                     out, count=1, flags=re.M)
    return out, sorted(set(idioms))


# ---- extraction -----------------------------------------------------------------------------

@dataclass
class Candidate:
    source_file: str
    test_name: str
    app_src: str
    test_src: str


def _segment(src: str, node: ast.AST) -> str:
    """Source of a nested node with its own indentation removed (first line is at col 0 already)."""
    lines = src.splitlines()[node.lineno - 1 : node.end_lineno]
    if not lines:
        return ""
    col = node.col_offset
    out = [lines[0][col:] if lines[0][:col].strip() == "" else lines[0].lstrip()]
    for ln in lines[1:]:
        out.append(ln[col:] if ln[:col].strip() == "" else ln.lstrip())
    return "\n".join(out)


def _module_imports(tree: ast.Module, src: str) -> list[str]:
    lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node)
            if seg and "from .conftest" not in seg and "import conftest" not in seg:
                lines.append(seg)
    return lines


def _module_defs(tree: ast.Module, src: str) -> dict[str, tuple[ast.AST, str]]:
    """Module-level helper definitions a hoisted class may reference (name -> (node, source))."""
    defs: dict[str, tuple[ast.AST, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("test_"):
            defs[node.name] = (node, ast.get_source_segment(src, node) or "")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = (node, ast.get_source_segment(src, node) or "")
    return defs


def _closure(names: set[str], defs: dict[str, tuple[ast.AST, str]]) -> list[str] | None:
    """Transitively collect helper sources for `names`; None if a helper is a pytest fixture or too odd."""
    seen: list[str] = []
    todo = [n for n in names if n in defs]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        node, src_seg = defs[n]
        if isinstance(node, ast.FunctionDef) and any("fixture" in ast.dump(d) for d in node.decorator_list):
            return None
        seen.append(n)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in defs and sub.id not in seen:
                todo.append(sub.id)
    # emit in module order
    order = {name: i for i, name in enumerate(defs)}
    return [defs[n][1] for n in sorted(seen, key=lambda x: order[x])]


def extract_candidates(test_file: Path) -> list[Candidate]:
    src = test_file.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    imports = _module_imports(tree, src)
    defs = _module_defs(tree, src)
    out: list[Candidate] = []
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
            continue
        decos = [ast.unparse(d) for d in fn.decorator_list]
        if any(not d.startswith("pytest.mark.parametrize") for d in decos):
            continue
        params = {a.arg for a in fn.args.args}
        if fn.args.args and not decos:
            continue  # fixtures, not parametrize
        classes = [s for s in fn.body if isinstance(s, ast.ClassDef)]
        rest = [s for s in fn.body if not isinstance(s, ast.ClassDef)]
        if not classes or not rest:
            continue
        class_src = "\n\n".join(_segment(src, c) for c in classes)
        if "BaseModel" not in class_src:
            continue
        used = {n.id for c in classes for n in ast.walk(c) if isinstance(n, ast.Name)}
        helpers = _closure(used & set(defs), defs)
        if helpers is None:
            continue
        # helpers referenced by the test body must also travel (they live in app.py, imported via *)
        body_used = {n.id for s in rest for n in ast.walk(s) if isinstance(n, ast.Name)}
        body_helpers = _closure(body_used & set(defs), defs)
        if body_helpers is None:
            continue
        for h in body_helpers:
            if h not in helpers:
                helpers.append(h)
        body_src = "\n".join(_segment(src, s) for s in rest)
        if any(k in body_src for k in ("monkeypatch", "tmp_path", "capsys", "recwarn", "request.")):
            continue
        if params and not (used & params) is False and (used & params):
            continue  # hoisted classes must not depend on parametrize arguments
        helper_src = ("\n\n".join(helpers) + "\n\n\n") if helpers else ""
        app = "\n".join(imports) + "\n\n\n" + helper_src + class_src + "\n"
        if decos:
            fn2 = ast.FunctionDef(name=fn.name, args=fn.args, body=rest, decorator_list=fn.decorator_list, returns=None, type_params=[])
            ast.fix_missing_locations(fn2)
            test = "\n".join(imports) + "\nfrom app import *  # noqa\n\n\n" + ast.unparse(fn2) + "\n"
        else:
            test = "\n".join(imports) + "\nfrom app import *  # noqa\n\n\ndef " + fn.name + "():\n" + textwrap.indent(body_src, "    ") + "\n"
        out.append(Candidate(test_file.name, fn.name, app, test))
    return out


def _run_pytest(python: str, workdir: Path, strict: bool) -> bool:
    cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x"]
    if strict:
        cmd += STRICT.split()
    cmd.append("test_app.py")
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def triple_filter(python: str, cand: Candidate) -> tuple[str, list[str]] | None:
    down, idioms = _downgrade(cand.app_src)
    if not idioms:
        return None
    try:
        ast.parse(down)
    except SyntaxError:
        return None
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        (wd / "test_app.py").write_text(cand.test_src)
        (wd / "app.py").write_text(cand.app_src)
        if not _run_pytest(python, wd, strict=True):
            return None
        (wd / "app.py").write_text(down)
        if not _run_pytest(python, wd, strict=False):
            return None
        if _run_pytest(python, wd, strict=True):
            return None
    return down, idioms


def write_task(out_dir: Path, cand: Candidate, down: str, idioms: list[str]) -> Path:
    name = f"{cand.source_file[:-3]}__{cand.test_name}"
    d = out_dir / name
    (d / "environment").mkdir(parents=True, exist_ok=True)
    (d / "tests").mkdir(exist_ok=True)
    (d / "instruction.md").write_text(INSTRUCTION)
    (d / "task.toml").write_text(TASK_TOML.format(name=name, source_file=cand.source_file, source_test=cand.test_name,
                                                 idioms=json.dumps(idioms)))
    (d / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (d / "environment" / "app.py").write_text(down)
    (d / "environment" / "test_app.py").write_text(cand.test_src)
    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "tests" / "test_app.py").write_text(cand.test_src)
    (d / "solution").mkdir(exist_ok=True)
    (d / "solution" / "app.py").write_text(cand.app_src)
    return d


def build(pydantic_repo: Path, python: str, out_dir: Path, limit: int | None, skip_files: set[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Dockerfile.base").write_text(BASE_IMAGE_DOCKERFILE.format(version="2.11.7"))
    files = sorted((pydantic_repo / "tests").glob("test_*.py"))
    stats = {"files": 0, "candidates": 0, "kept": 0, "by_file": {}}
    kept = 0
    for f in files:
        if f.name in skip_files:
            continue
        stats["files"] += 1
        cands = extract_candidates(f)
        stats["candidates"] += len(cands)
        n = 0
        for c in cands:
            res = triple_filter(python, c)
            if res is None:
                continue
            write_task(out_dir, c, *res)
            n += 1
            kept += 1
            if limit and kept >= limit:
                break
        if n:
            stats["by_file"][f.name] = n
        if limit and kept >= limit:
            break
    stats["kept"] = kept
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=1))
    return stats
