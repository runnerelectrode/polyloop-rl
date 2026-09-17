"""A scripted environment: tasks are names, rewards come from a table. Used to drive the cycle
runner without a trainer server, Docker, or a GPU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polyloop.environment import BaseEnvironment
from polyloop.harness.rollout import TaskResult


@dataclass
class Task:
    task_name: str


class ScriptedEnvironment(BaseEnvironment):
    name = "scripted"
    needs_engine_warm = False

    def __init__(self, cfg, store, log=print, *, pool=(), holdout=(), rewards=None, hints=None, excluded=()):
        super().__init__(cfg, store, log=log)
        self.pool, self.holdout = list(pool), list(holdout)
        self.rewards = rewards or {}          # "policy_id/task" -> list[float]
        self.hints = dict(hints or {})
        self.excluded = set(excluded)
        self.calls: list[dict] = []

    def load_tasks(self, dataset, *, limit=None, seed=0, names=None):
        src = self.holdout if dataset == "holdout" else self.pool
        if names:
            return [Task(n) for n in names]
        return [Task(n) for n in src[: limit or len(src)]]

    def run_rollouts(self, *, label, tasks, policy, k, temperature, out, on_result):
        self.calls.append({"label": label, "policy": policy.get("id", "base"), "n": len(tasks), "k": k})
        res = []
        for t in tasks:
            r = TaskResult(task=t.task_name, rewards=list(self.rewards.get(f"{policy.get('id', 'base')}/{t.task_name}", [0.0] * k))[:k])
            on_result(r)
            res.append(r)
        return res

    def session_hints(self):
        return dict(self.hints)

    def excluded_sessions(self):
        return set(self.excluded)
