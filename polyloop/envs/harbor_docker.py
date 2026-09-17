"""The default environment: Harbor tasks run by the tinker-cookbook bash-tool loop in local Docker
sandboxes, scored by each task's verifier. This is what the coding-agent recipes use."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from polyloop.environment import BaseEnvironment, Policy
from polyloop.harness.rollout import TaskResult, load_tasks, run_rollouts


class HarborDockerEnvironment(BaseEnvironment):
    name = "harbor-docker"
    needs_engine_warm = True

    def load_tasks(self, dataset: str, *, limit: int | None = None, seed: int = 0, names: list[str] | None = None) -> list[Any]:
        return load_tasks(dataset, limit=limit, seed=seed, names=names)

    def _rollout_kwargs(self, **over):
        s, sb = self.cfg.stages[0], self.cfg.sandbox
        kw = dict(base_url=self.cfg.base_url, model=self.cfg.model, renderer=self.cfg.renderer,
                  max_parallel=sb.max_parallel, max_tokens=s.max_tokens, max_turns=s.max_turns,
                  max_trajectory_tokens=s.max_trajectory_tokens, sandbox_timeout=sb.timeout,
                  command_timeout=sb.command_timeout, grader_timeout=sb.grader_timeout,
                  context_window=s.max_trajectory_tokens)
        kw.update(over)
        return kw

    def run_rollouts(self, *, label: str, tasks: list[Any], policy: Policy, k: int, temperature: float,
                     out: Path | None, on_result: Callable[[TaskResult], None]) -> list[TaskResult]:
        return asyncio.run(run_rollouts(tasks=tasks, sampler_path=(policy or {}).get("sampler_path"), k=k,
                                        temperature=temperature, on_result=on_result,
                                        trajectories_dir=(out / "trajectories") if out else None,
                                        **self._rollout_kwargs()))

    def preflight(self) -> list[str]:
        if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            return ["docker daemon unreachable"]
        return []
