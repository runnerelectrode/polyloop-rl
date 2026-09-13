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


def _tolerate_logprob_length_mismatch(rl_train) -> None:
    """The cookbook's sample-vs-train KL metric indexes the server's returned per-token
    logprobs with the datum's action mask. On long packed episodes SkyRL's fused LM-head
    path has returned fewer logprobs than tokens (seen: 14189 for a 16071-token datum), which
    raises IndexError and kills the run after a successful optimizer step. The metric is
    diagnostic only (the loss is computed server-side), so skip it and count the mismatch
    instead of dying. Tracked as a server-side issue to fix upstream."""
    import logging

    original = rl_train.compute_kl_sample_train
    log = logging.getLogger("polyloop.train")

    def tolerant(data_D, training_logprobs_D):
        try:
            return original(data_D, training_logprobs_D)
        except (IndexError, RuntimeError, ValueError) as exc:
            keep = []
            for i, (d, lp) in enumerate(zip(data_D, training_logprobs_D)):
                try:
                    original([d], [lp])
                    keep.append(i)
                except (IndexError, RuntimeError, ValueError):
                    pass
            bad = len(data_D) - len(keep)
            log.warning("logprob length mismatch on %d/%d datums (%s); KL metric computed on the rest", bad, len(data_D), exc)
            out = dict(original([data_D[i] for i in keep], [training_logprobs_D[i] for i in keep])) if keep else {}
            out["polyloop/logprob_len_mismatch"] = float(bad)
            return out

    rl_train.compute_kl_sample_train = tolerant


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
    st = spec["stage"]
    # The cookbook dataset is one pass over its task list, groups_per_batch groups per step.
    # Tile the contested tasks so the run has exactly `steps` batches (fresh rollouts each
    # time a task recurs; a few passes over a small contested set is normal for GRPO).
    import random

    want = st["steps"] * st["groups_per_batch"]
    rng = random.Random(spec.get("seed", 0))
    tiled = []
    while len(tiled) < want:
        chunk = list(tasks)
        rng.shuffle(chunk)
        tiled.extend(chunk)
    tasks = tiled[:want]
    renderer = renderer_for(spec["model"], spec.get("renderer"))
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
    _tolerate_logprob_length_mismatch(rl_train)
    guarded = logprob_guard.install(threshold=spec.get("logprob_abs_diff_max", 0.05))
    print(f"[polyloop.train] {len(tasks)} tasks, {st['steps']} steps, logprob guard={'on' if guarded else 'OFF'}", flush=True)
    asyncio.run(rl_train.main(config))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
