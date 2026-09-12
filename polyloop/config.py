"""loop.yaml -> LoopConfig. Everything the controller needs is declared here, nothing is
inferred at runtime: the held-out split, the thresholds, the budget and the promote mode are
owned by this file, not by whatever proposes changes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    kind: str = "rl"                     # rl | (sft, opsd reserved)
    harness: str = "cookbook-bash"       # the agent loop that produces episodes
    group_size: int = 4
    groups_per_batch: int = 4
    steps: int = 8
    max_steps_off_policy: int = 0
    learning_rate: float = 1e-5
    max_tokens: int = 4096               # per generation
    max_turns: int = 30
    max_trajectory_tokens: int = 32 * 1024
    lora_rank: int = 32
    loss: str = "importance_sampling"
    save_every: int = 4


@dataclass
class SandboxConfig:
    backend: str = "docker"
    timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 900
    max_parallel: int = 16


@dataclass
class FilterConfig:
    pool_sample: int = 32                # tasks measured per cycle under the incumbent
    rollouts_per_task: int = 4
    keep_min: float = 0.0                # keep tasks whose pass rate is in (keep_min, keep_max)
    keep_max: float = 1.0
    min_tasks: int = 4                   # refuse to train on fewer contested tasks


@dataclass
class TriggerConfig:
    cron: str | None = None
    new_tasks_min: int = 0
    cooldown_hours: float = 0.0


@dataclass
class BudgetConfig:
    gpus: int = 2
    max_cycle_gpu_seconds: int = 8 * 3600 * 2
    max_stage_seconds: int = 6 * 3600


@dataclass
class GateConfig:
    holdout: str = "swebench-verified@1.0/swebench-verified"   # dataset dir under ~/.cache/harbor/tasks
    holdout_limit: int = 50
    holdout_seed: int = 0
    repeats: int = 2
    temperature: float = 1.0
    min_delta: float = 0.01              # paired mean delta the candidate must beat
    tie_band: float = 0.01               # per-task drop within this band is a tie, not a regression
    max_regressions: int = 2
    logprob_abs_diff_max: float = 0.05   # trainer-vs-sampler agreement over the cycle's steps
    max_error_rate: float = 0.2          # rollout error share that invalidates an eval


@dataclass
class PromoteConfig:
    mode: str = "approve"                # approve | auto


@dataclass
class LoopConfig:
    name: str
    model: str
    tasks: str                            # train pool: dataset dir under ~/.cache/harbor/tasks or absolute
    base_url: str = "http://127.0.0.1:8000"
    runs_dir: str = "~/polyloop-runs"
    renderer: str | None = None
    stages: list[StageConfig] = field(default_factory=lambda: [StageConfig()])
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    promote: PromoteConfig = field(default_factory=PromoteConfig)
    replay_fraction: float = 0.05
    source_path: str | None = None

    @property
    def runs_path(self) -> Path:
        return Path(self.runs_dir).expanduser()


def _build(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


def load_loop(path: str | Path) -> LoopConfig:
    p = Path(path).expanduser()
    raw = yaml.safe_load(p.read_text()) or {}
    stages = [_build(StageConfig, s) for s in raw.pop("stages", [])] or [StageConfig()]
    cfg = LoopConfig(
        name=raw.pop("name"),
        model=raw.pop("model"),
        tasks=raw.pop("tasks"),
        base_url=raw.pop("base_url", "http://127.0.0.1:8000"),
        runs_dir=raw.pop("runs_dir", "~/polyloop-runs"),
        renderer=raw.pop("renderer", None),
        stages=stages,
        sandbox=_build(SandboxConfig, raw.pop("sandbox", None)),
        filter=_build(FilterConfig, raw.pop("filter", None)),
        trigger=_build(TriggerConfig, raw.pop("trigger", None)),
        budget=_build(BudgetConfig, raw.pop("budget", None)),
        gate=_build(GateConfig, raw.pop("gate", None)),
        promote=_build(PromoteConfig, raw.pop("promote", None)),
        replay_fraction=raw.pop("replay_fraction", 0.05),
        source_path=str(p),
    )
    if raw:
        raise ValueError(f"loop.yaml: unknown top-level keys {sorted(raw)}")
    return cfg
