"""promotion_receipt.v1: the one document a promotion decision leaves behind."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(s) / len(s))
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def paired_stats(cand: dict[str, float], inc: dict[str, float], tie_band: float) -> dict:
    common = sorted(set(cand) & set(inc))
    deltas = [cand[t] - inc[t] for t in common]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    lo, hi = bootstrap_ci(deltas)
    wins = sum(1 for d in deltas if d > tie_band)
    losses = sum(1 for d in deltas if d < -tie_band)
    return {
        "n_tasks": len(common),
        "candidate_mean": sum(cand[t] for t in common) / len(common) if common else None,
        "incumbent_mean": sum(inc[t] for t in common) / len(common) if common else None,
        "mean_delta": mean_delta,
        "delta_ci95": [lo, hi],
        "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
    }


def holdout_id(task_names: list[str]) -> str:
    h = hashlib.sha256("\n".join(sorted(task_names)).encode()).hexdigest()[:16]
    return f"holdout-{len(task_names)}-{h}"


def write_receipt(path: Path, **fields) -> dict:
    doc = {"schema": "promotion_receipt.v1", **fields}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return doc
