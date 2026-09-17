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
    # opsd-only
    teacher: str | None = None            # None = same weights (self-distillation)
    teacher_hint: str | None = None       # run-wide hint; rows carry their own from traces
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    max_hint_chars: int = 2000
    min_rows: int = 8


@dataclass
class SandboxConfig:
    backend: str = "docker"
    timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 900
    max_parallel: int = 16


@dataclass
class FilterConfig:
    pool_sample: int = 32                # tasks measured per round under the incumbent
    rollouts_per_task: int = 4
    keep_min: float = 0.0                # keep tasks whose pass rate is in (keep_min, keep_max)
    keep_max: float = 1.0
    min_tasks: int = 4                   # refuse to train on fewer contested tasks
    target_tasks: int = 16               # keep measuring rounds until this many are contested (or pool exhausted)
    max_rounds: int = 4


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
class CovalConfig:
    """Coval (coval.ai) as the environment and the verifier: Coval's simulated user talks to the
    policy through the proxy, Coval's metrics score every conversation, polyloop reads the
    scores back as rewards. `tasks` and `gate.holdout` are then `coval://<test_set_id>`."""
    agent_id: str                         # the CHAT agent in Coval whose chat_endpoint is the proxy (base config)
    persona_id: str                       # the simulated caller
    metric_ids: list[str]                 # reward per conversation = mean of these numeric metrics (binary judges -> 0/1)
    public_url: str                       # HTTPS URL Coval reaches the proxy at (a tunnel; Coval refuses private IPs)
    api_base: str = "https://api.coval.dev/v1"
    api_key_env: str = "COVAL_API_KEY"
    concurrency: int = 8                  # simulations Coval runs in parallel per run
    poll_seconds: int = 15
    run_timeout: int = 3600               # seconds to wait for one run before the stage errors
    temperature: float = 1.0              # sampling params the per-policy agent sends in its OpenAI request
    max_tokens: int = 256
    system_prompt: str | None = None      # file (relative to loop.yaml) the proxy prepends on policy routes
    pass_value: float = 1.0               # reward >= this counts as passed; failures get the judge hint
    hint_failures_only: bool = True       # append the judge explanation only to failed sessions' rows
    max_judge_chars: int = 1200
    ledger: str = "coval_sessions.jsonl"  # per-session rewards + explanations, under runs/<loop>/


@dataclass
class LoopConfig:
    name: str
    model: str
    tasks: str                            # train pool: dataset dir under ~/.cache/harbor/tasks or absolute
    base_url: str = "http://127.0.0.1:8000"
    runs_dir: str = "~/polyloop-runs"
    renderer: str | None = None
    proxy_url: str | None = None          # polyloop proxy; promote posts the winner here
    traces: str | None = None             # dir of proxy trace JSONL (default <runs>/<loop>/traces)
    stages: list[StageConfig] = field(default_factory=lambda: [StageConfig()])
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    promote: PromoteConfig = field(default_factory=PromoteConfig)
    coval: CovalConfig | None = None      # set -> rollouts go through Coval instead of Docker sandboxes
    replay_fraction: float = 0.05
    source_path: str | None = None

    @property
    def runs_path(self) -> Path:
        return Path(self.runs_dir).expanduser()

    @property
    def system_prompt_path(self) -> Path | None:
        if not (self.coval and self.coval.system_prompt):
            return None
        p = Path(self.coval.system_prompt).expanduser()
        if not p.is_absolute() and self.source_path:
            p = Path(self.source_path).parent / p
        return p


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
    coval_raw = raw.pop("coval", None)
    cfg = LoopConfig(
        name=raw.pop("name"),
        model=raw.pop("model"),
        tasks=raw.pop("tasks"),
        base_url=raw.pop("base_url", "http://127.0.0.1:8000"),
        runs_dir=raw.pop("runs_dir", "~/polyloop-runs"),
        renderer=raw.pop("renderer", None),
        proxy_url=raw.pop("proxy_url", None),
        traces=raw.pop("traces", None),
        stages=stages,
        sandbox=_build(SandboxConfig, raw.pop("sandbox", None)),
        filter=_build(FilterConfig, raw.pop("filter", None)),
        trigger=_build(TriggerConfig, raw.pop("trigger", None)),
        budget=_build(BudgetConfig, raw.pop("budget", None)),
        gate=_build(GateConfig, raw.pop("gate", None)),
        promote=_build(PromoteConfig, raw.pop("promote", None)),
        coval=_build(CovalConfig, coval_raw) if coval_raw is not None else None,
        replay_fraction=raw.pop("replay_fraction", 0.05),
        source_path=str(p),
    )
    if raw:
        raise ValueError(f"loop.yaml: unknown top-level keys {sorted(raw)}")
    if cfg.coval:
        for label, ds in (("tasks", cfg.tasks), ("gate.holdout", cfg.gate.holdout)):
            if not ds.startswith("coval://"):
                raise ValueError(f"loop.yaml: with a `coval:` block, {label} must be coval://<test_set_id>, got {ds!r}")
        if cfg.tasks == cfg.gate.holdout:
            raise ValueError("loop.yaml: tasks and gate.holdout name the same Coval test set; the holdout must never be trained on")
        for st in cfg.stages:
            if st.kind == "rl":
                raise ValueError("loop.yaml: an `rl` stage needs Docker sandboxes; a coval loop trains with `opsd` (see recipes/voice-coval/program.md)")
    return cfg
