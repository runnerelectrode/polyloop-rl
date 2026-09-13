"""Train stage worker: one cookbook Harbor RL run, driven by a JSON spec, in its own process.

Mirrors `rlcli train harbor` (same builders, same Docker sandbox, same token bridge, same
logprob guard) but is built here so the loop can pass what the CLI does not expose yet:
the incumbent adapter to resume from, the grader timeout and the trajectory token cap.

    python -m polyloop.harness.train spec.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    from rlcli.compat import as_loss_fn
    from rlcli.harbor_tito import BridgingHarborDatasetBuilder
    from rlcli.sandbox_docker import local_docker_sandbox_factory
    from rlcli.tokenizer_lock import register_locked_tokenizer
    from rlcli import logprob_guard
    from tinker_cookbook.rl import train as rl_train

    from polyloop.harness.rollout import load_tasks, prepare_env, renderer_for

    prepare_env(spec["base_url"])
    register_locked_tokenizer(spec["model"])
    tasks = load_tasks(spec["dataset"], names=spec.get("task_names"))
    if not tasks:
        print("no tasks", file=sys.stderr)
        return 2
    renderer = renderer_for(spec["model"], spec.get("renderer"))
    st = spec["stage"]
    builder = BridgingHarborDatasetBuilder(
        tasks=tasks, batch_size=st["groups_per_batch"], group_size=st["group_size"],
        model_name=spec["model"], renderer_name=renderer, max_turns=st["max_turns"],
        sandbox_timeout=spec["sandbox"]["timeout"], command_timeout=spec["sandbox"]["command_timeout"],
        grader_timeout=spec["sandbox"]["grader_timeout"], max_trajectory_tokens=st["max_trajectory_tokens"],
        sandbox_factory=local_docker_sandbox_factory,
    )
    kwargs = dict(
        learning_rate=st["learning_rate"], dataset_builder=builder, model_name=spec["model"],
        recipe_name="polyloop_rl", max_tokens=st["max_tokens"], log_path=spec["log_path"],
        renderer_name=renderer, lora_rank=st["lora_rank"], save_every=st["save_every"], eval_every=0,
        base_url=spec["base_url"], loss_fn=as_loss_fn(st["loss"]), max_steps=st["steps"],
    )
    if spec.get("load_checkpoint_path"):
        kwargs["load_checkpoint_path"] = spec["load_checkpoint_path"]
    if st.get("max_steps_off_policy"):
        kwargs["async_config"] = rl_train.AsyncConfig(
            max_steps_off_policy=st["max_steps_off_policy"], groups_per_batch=st["groups_per_batch"])
    config = rl_train.Config(**kwargs)
    guarded = logprob_guard.install(threshold=spec.get("logprob_abs_diff_max", 0.05))
    print(f"[polyloop.train] {len(tasks)} tasks, {st['steps']} steps, logprob guard={'on' if guarded else 'OFF'}", flush=True)
    asyncio.run(rl_train.main(config))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
