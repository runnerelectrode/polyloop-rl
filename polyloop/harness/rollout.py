"""Run K episodes per task for one policy checkpoint and return per-task rewards.

Used by the filter stage (pass rate under the incumbent) and the evaluate stage (candidate
vs incumbent on the held-out split). Same env, sandbox and token bridge as the train stage,
so what is measured is what is trained: the cookbook's Harbor bash-tool loop, one Docker
container per episode, reward from tests/test.sh.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task: str
    rewards: list[float] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    stop_reasons: list[str | None] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    @property
    def mean(self) -> float | None:
        return sum(self.rewards) / len(self.rewards) if self.rewards else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean"] = self.mean
        return d


def prepare_env(base_url: str) -> None:
    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
    os.environ["TINKER_BASE_URL"] = base_url


def renderer_for(model: str, renderer: str | None) -> str:
    if renderer:
        return renderer
    from tinker_cookbook import model_info

    return model_info.get_recommended_renderer_name(model)


def warm(base_url: str, model: str, rank: int = 32, timeout: float = 2400.0, log=print) -> str:
    """Bring the trainer's inference engines up. SkyRL creates them lazily on the first
    sampling-related call from a training run, so base-model sampling before any run fails
    with "inference engine not ready". One LoRA model + one sampler save is the wake-up call
    (same sequence the hosted tier uses at boot). Returns the sampler path, ignorable."""
    import tinker

    prepare_env(base_url)
    t0 = time.monotonic()
    service = tinker.ServiceClient(base_url=base_url)
    tc = service.create_lora_training_client(base_model=model, rank=rank)
    fut = tc.save_weights_for_sampler(name="polyloop-warm")
    resp = fut.result(timeout=timeout) if "timeout" in fut.result.__code__.co_varnames else fut.result()
    path = getattr(resp, "path", None)
    log(f"engine warm in {time.monotonic() - t0:.0f}s ({path})")
    # The engines live only while a session is alive: when the warming process exits, SkyRL
    # expires the session, unloads the model and tears the inference engines down. Long-lived
    # callers (the proxy) keep the returned client referenced for their whole lifetime.
    warm.last_client = tc  # type: ignore[attr-defined]
    return path or ""


def load_tasks(dataset: str, limit: int | None = None, seed: int = 0, names: list[str] | None = None, coval_client=None):
    from polyloop.harness.coval import is_coval_dataset

    if is_coval_dataset(dataset):
        # coval://<test_set_id>: the tasks are Coval test cases; the environment is Coval's
        # simulated caller, not a sandbox. Same shape (task_name) for every stage.
        from polyloop.harness.coval import load_coval_tasks

        return load_coval_tasks(dataset, client=coval_client, limit=limit, seed=seed, names=names)
    from tinker_cookbook.recipes.harbor_rl.harbor_env import load_harbor_tasks

    path = Path(dataset).expanduser()
    tasks = load_harbor_tasks(str(path.resolve()) if path.is_dir() else dataset)
    if names is not None:
        wanted = set(names)
        tasks = [t for t in tasks if t.task_name in wanted]
    if limit and len(tasks) > limit:
        import random

        rng = random.Random(seed)
        tasks = sorted(rng.sample(tasks, limit), key=lambda t: t.task_name)
    return tasks


async def run_rollouts(
    *,
    base_url: str,
    model: str,
    renderer: str | None,
    tasks,
    sampler_path: str | None,
    k: int,
    max_parallel: int,
    max_tokens: int,
    max_turns: int,
    max_trajectory_tokens: int,
    sandbox_timeout: int,
    command_timeout: int,
    grader_timeout: int,
    temperature: float = 1.0,
    context_window: int | None = None,
    on_result=None,
    trajectories_dir: Path | None = None,
) -> list[TaskResult]:
    import tinker
    from rlcli.harbor_tito import BridgingHarborDatasetBuilder
    from rlcli.sandbox_docker import local_docker_sandbox_factory
    from rlcli.tokenizer_lock import register_locked_tokenizer
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.rl.rollouts import do_group_rollout

    prepare_env(base_url)
    register_locked_tokenizer(model)
    service = tinker.ServiceClient(base_url=base_url)
    if sampler_path:
        sampling = service.create_sampling_client(model_path=sampler_path)
    else:
        sampling = service.create_sampling_client(base_model=model)
    policy = TinkerTokenCompleter(sampling, max_tokens=max_tokens, temperature=temperature,
                                  context_window=context_window)
    builder = BridgingHarborDatasetBuilder(
        tasks=tasks, batch_size=max(1, len(tasks)), group_size=k, model_name=model,
        renderer_name=renderer_for(model, renderer), max_turns=max_turns,
        sandbox_timeout=sandbox_timeout, command_timeout=command_timeout, grader_timeout=grader_timeout,
        max_trajectory_tokens=max_trajectory_tokens, sandbox_factory=local_docker_sandbox_factory,
    )
    groups = builder._make_env_group_builders(k)
    sem = asyncio.Semaphore(max_parallel)
    if trajectories_dir is not None:
        # Every episode as an ATIF trajectory (Harbor's format) with token ids: the audit
        # trail a receipt points at, and what a proposing agent reads to diagnose failures.
        from rlcli import atif

        trajectories_dir.mkdir(parents=True, exist_ok=True)
        atif.configure(atif.FileSink(str(trajectories_dir)))

    async def one(egb) -> TaskResult:
        res = TaskResult(task=egb.task.task_name)
        t0 = time.monotonic()
        async with sem:
            try:
                tg = await do_group_rollout(egb, policy)
                res.rewards = [float(r) for r in tg.get_total_rewards()]
                for traj in tg.trajectories_G:
                    res.turns.append(len(traj.transitions))
                    res.tokens.append(sum(len(t.ac.tokens) for t in traj.transitions))
                    res.stop_reasons.append(getattr(traj, "stop_reason", None))
            except Exception as exc:  # one bad task must not sink the batch
                logger.exception("rollout failed for %s", res.task)
                res.error = f"{type(exc).__name__}: {exc}"[:500]
        res.seconds = time.monotonic() - t0
        if on_result:
            on_result(res)
        return res

    try:
        return await asyncio.gather(*(one(g) for g in groups))
    finally:
        if trajectories_dir is not None:
            from rlcli import atif

            atif.configure(None)
