"""Tolerate logprob-length mismatches in the cookbook's sample-vs-train KL metric.

The metric indexes the server's returned per-token logprobs with the datum's mask. SkyRL
returns exactly `len(token_weights)` logprobs, so a mismatch means the datum's mask and its
response disagree (seen on long packed episodes). The metric is diagnostic; the loss is
computed server-side. Skip the offending datums and count them rather than kill a run that
has already taken its optimizer steps.
"""
from __future__ import annotations

import logging

log = logging.getLogger("polyloop.kl_guard")


def _tolerant(original):
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

    tolerant._polyloop_guard = True  # type: ignore[attr-defined]
    return tolerant


def install() -> list[str]:
    """Patch every module that holds a reference to compute_kl_sample_train."""
    patched = []
    from tinker_cookbook.rl import metrics as m

    if not getattr(m.compute_kl_sample_train, "_polyloop_guard", False):
        m.compute_kl_sample_train = _tolerant(m.compute_kl_sample_train)
        patched.append("rl.metrics")
    for modname in ("tinker_cookbook.rl.train", "tinker_cookbook.distillation.train_on_policy", "tinker_cookbook.distillation.sdft"):
        try:
            mod = __import__(modname, fromlist=["compute_kl_sample_train"])
        except ImportError:
            continue
        fn = getattr(mod, "compute_kl_sample_train", None)
        if fn is not None and not getattr(fn, "_polyloop_guard", False):
            mod.compute_kl_sample_train = _tolerant(fn)
            patched.append(modname)
    return patched
