"""The environment interface: where episodes come from and how they are scored.

The cycle runner never decides how a task is run. It asks the loop's environment to load the
task pool and the held-out split, to run K episodes of each task under a policy and return
rewards, to check its own prerequisites in preflight, and (optionally) to contribute per-session
hindsight hints and exclusions for the OPSD stage. The controller keeps what it owns: the
split, the thresholds, the budget, the receipt, the lineage.

Implementations live anywhere. `loop.yaml` names one:

    environment:
      kind: harbor-docker            # a registered name ...
      # kind: polyvoice.envs.coval:CovalEnvironment   # ... or module:Class
      options: {}                    # passed to the constructor as keyword arguments

Registered names come from the built-in table below and from the `polyloop.environments`
entry-point group, so an external package can add one without touching this repo.
"""
from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from polyloop.harness.rollout import TaskResult

Policy = dict  # {"id": str, "sampler_path": str | None, "state_path": str | None}

BUILTIN = {
    "harbor-docker": "polyloop.envs.harbor_docker:HarborDockerEnvironment",
}


@runtime_checkable
class Environment(Protocol):
    name: str
    needs_engine_warm: bool

    def load_tasks(self, dataset: str, *, limit: int | None = None, seed: int = 0,
                   names: list[str] | None = None) -> list[Any]:
        """Tasks are opaque to the controller except for `.task_name`."""

    def run_rollouts(self, *, label: str, tasks: list[Any], policy: Policy, k: int, temperature: float,
                     out: Path | None, on_result: Callable[[TaskResult], None]) -> list[TaskResult]:
        """K episodes per task under `policy`; call `on_result` as each task finishes."""

    def preflight(self) -> list[str]:
        """Config-only checks; a non-empty list aborts the cycle before anything is spent."""

    def session_hints(self) -> dict[str, str]:
        """Hindsight text per proxy session id, joined into OPSD rows (verdicts, judge explanations)."""

    def excluded_sessions(self) -> set[str]:
        """Proxy sessions that must never be trained on (held-out evaluations, probes)."""


class BaseEnvironment:
    """Defaults for the optional parts of the protocol."""

    name = "base"
    needs_engine_warm = True

    def __init__(self, cfg, store, log=print, **options):
        self.cfg = cfg
        self.store = store
        self.log = log
        self.options = options

    def preflight(self) -> list[str]:
        return []

    def session_hints(self) -> dict[str, str]:
        return {}

    def excluded_sessions(self) -> set[str]:
        return set()


def _registered() -> dict[str, str]:
    table = dict(BUILTIN)
    try:
        for ep in entry_points(group="polyloop.environments"):
            table.setdefault(ep.name, ep.value)
    except Exception:  # pragma: no cover - metadata oddities on exotic installs
        pass
    return table


def resolve(kind: str) -> type:
    """`kind` is a registered name or `module:Class` / `module.Class`."""
    target = _registered().get(kind, kind)
    if ":" in target:
        mod, _, attr = target.partition(":")
    else:
        mod, _, attr = target.rpartition(".")
    if not mod or not attr:
        raise ValueError(f"environment {kind!r}: not registered and not a module:Class path (known: {sorted(_registered())})")
    try:
        cls = getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"environment {kind!r}: cannot import {target}: {exc}") from exc
    return cls


def load_environment(cfg, store, log=print) -> Environment:
    ec = cfg.environment
    cls = resolve(ec.kind)
    env = cls(cfg, store, log=log, **(ec.options or {}))
    if not isinstance(env, Environment):
        raise TypeError(f"environment {ec.kind!r}: {cls.__name__} does not implement the Environment protocol")
    return env
