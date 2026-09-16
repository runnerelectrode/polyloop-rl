"""OPSD train worker: hinted on-policy self-distillation on rows built from traces.

Mirrors `rlcli train opsd` (same JsonlPromptDatasetBuilder, HintedTeacher and cookbook loop)
but is driven by a JSON spec so the cycle can pass `load_checkpoint_path` (the incumbent or the
preceding RL stage's state) and a fixed log directory. Runs in its own process.

    python -m polyloop.harness.train_opsd spec.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    from tinker_cookbook.distillation import train_on_policy
    from tinker_cookbook.distillation.datasets import DistillationDatasetConfig
    from rlcli import opsd as opsd_lib
    from rlcli.tokenizer_lock import register_locked_tokenizer

    from polyloop.harness.rollout import prepare_env, renderer_for

    prepare_env(spec["base_url"])
    register_locked_tokenizer(spec["model"])
    rows = opsd_lib.load_prompt_rows(spec["rows"])
    st = spec["stage"]
    renderer = renderer_for(spec["model"], spec.get("renderer"))
    teacher_config = opsd_lib.teacher_config_for(spec["model"], st.get("teacher"))
    builder = opsd_lib.JsonlPromptDatasetBuilder(
        file_path=spec["rows"], groups_per_batch=st["groups_per_batch"], group_size=st["group_size"],
        model_name_for_tokenizer=spec["model"], renderer_name=renderer, teacher_hint=st.get("teacher_hint"),
    )
    kwargs = dict(
        learning_rate=st["learning_rate"],
        dataset_configs=[DistillationDatasetConfig(dataset_builder=builder, teacher_config=teacher_config,
                                                   groups_per_batch=st["groups_per_batch"])],
        model_name=spec["model"], recipe_name="polyloop_opsd", max_tokens=st["max_tokens"],
        log_path=spec["log_path"], renderer_name=renderer, lora_rank=st["lora_rank"],
        kl_penalty_coef=st.get("kl_penalty_coef", 1.0), kl_discount_factor=st.get("kl_discount_factor", 0.0),
        save_every=st.get("save_every", 2), eval_every=0, base_url=spec["base_url"], loss_fn="importance_sampling",
        max_steps=st["steps"],
    )
    if spec.get("load_checkpoint_path"):
        kwargs["load_checkpoint_path"] = spec["load_checkpoint_path"]
    config = train_on_policy.Config(**kwargs)
    print(f"[polyloop.opsd] {len(rows)} rows, {st['steps']} steps, teacher={opsd_lib.describe_teacher(teacher_config, st.get('teacher_hint'))}", flush=True)
    asyncio.run(opsd_lib.main(config, teacher_hint=st.get("teacher_hint")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
