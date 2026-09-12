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
    from polyloop.stages import approve

    _, store = _store(loop_path)
    approve(store, cycle_id, log=click.echo)


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

    from polyloop.harness.rollout import load_tasks, run_rollouts

    cfg, _ = _store(loop_path)
    g, s, sb = cfg.gate, cfg.stages[0], cfg.sandbox
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
        grader_timeout=sb.grader_timeout, temperature=g.temperature, context_window=s.max_trajectory_tokens, on_result=on_result))
    ok = [r for r in results if r.error is None and r.mean is not None]
    mean = sum(r.mean for r in ok) / len(ok) if ok else float("nan")
    click.echo(f"{len(ok)}/{len(results)} tasks scored, mean reward {mean:.3f}, errors {len(results) - len(ok)}")


@main.group("tasks")
def tasks():
    """Build task pools."""


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
