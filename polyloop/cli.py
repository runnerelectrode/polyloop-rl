"""polyloop: run and inspect continual-learning cycles."""
from __future__ import annotations

import json
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
@click.option("--dataset", default=None, help="Harbor dataset dir (default: the loop's holdout).")
@click.option("--limit", type=int, default=None)
@click.option("--sampler-path", default=None, help="tinker:// sampler checkpoint; default = base model")
@click.option("--repeats", "-k", type=int, default=1)
@click.option("--out", default=None)
def eval_cmd(loop_path, dataset, limit, sampler_path, repeats, out):
    """Ad-hoc: score a checkpoint on a task set (baseline numbers)."""
    import asyncio

    from polyloop.harness.rollout import load_tasks, run_rollouts, warm

    cfg, _ = _store(loop_path)
    g, s, sb = cfg.gate, cfg.stages[0], cfg.sandbox
    warm(cfg.base_url, cfg.model, rank=s.lora_rank, log=click.echo)
    tasks = load_tasks(dataset or g.holdout, limit=limit or g.holdout_limit, seed=g.holdout_seed)
    out_path = Path(out).expanduser() if out else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    def on_result(r):
        click.echo(f"  {r.task:50} {r.rewards} {r.error or ''} ({r.seconds:.0f}s, turns {r.turns})")
        if out_path:
            with (out_path / "results.jsonl").open("a") as f:
                f.write(json.dumps(r.to_dict()) + "\n")

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
def proxy_cmd(loop_path, host, port, max_tokens):
    """OpenAI/Anthropic-compatible endpoint for your coding agent; records sessions; serves the live adapter."""
    from polyloop.proxy import serve

    cfg, _ = _store(loop_path)
    serve(host, port, base_url=cfg.base_url, model=cfg.model, renderer_name=cfg.renderer,
          runs_dir=cfg.runs_path, loop_name=cfg.name, default_max_tokens=max_tokens)


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
