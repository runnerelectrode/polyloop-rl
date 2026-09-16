# Results

## pydantic-v2 migration, Qwen3.5-4B, cycle 1 (2026-09-16)

The first complete unattended cycle of the loop on a real coding-agent workload. Every stage ran
(snapshot → preflight → filter → train → evaluate → gate); the gate **rejected** the candidate, which is
the correct outcome for the numbers below. Raw artifacts: `docs/results/pydantic-v2-4b-cycle1/`
(receipt, events, state, per-stage metrics, report page).

**Setup**
- Task: hoisted pydantic 2.11.7 test modules downgraded to v1 idioms; the agent must migrate `app.py`
  so the hidden `test_app.py` passes under `-W error::pydantic.warnings.PydanticDeprecatedSince20`.
  Pool 67 tasks, split by source test file: 41 train / 24 holdout.
- Agent: mini-swe-agent (text mode) through `polyloop proxy` on the node; 24 sessions in Docker (all exit 0)
  plus 6 laptop sessions → 345 recorded assistant turns in `traces/2026-09-16.jsonl`.
- Model: Qwen/Qwen3.5-4B, LoRA rank 32, SkyRL Tinker server (Megatron), 4×A6000 48 GB (Lambda, $4.36/h).
- Train: RL stage 3 steps (group 4, 8 groups/batch, 13 contested tasks from the filter) chained into an
  OPSD stage 2 steps (16 groups × 2, hinted teacher from the next-state of each trace turn, KL coef 1.0).
- Gate: paired holdout of 24 tasks × 4 repeats, temperature 1.0, 95% paired bootstrap CI,
  min_delta 0.10, tie 0.01, max_regressions 2.

**Numbers**

| stage | value |
|---|---|
| RL reward per step | 0.594 → 0.844 → 0.719 |
| OPSD teacher KL per step | 0.102 → 0.056 |
| incumbent (base 4B) holdout mean | 0.844 |
| candidate holdout mean | 0.875 |
| paired delta | +0.031, CI95 [−0.042, +0.104] |
| wins / losses / ties (tasks) | 6 / 4 / 14 |
| checks | enough_tasks ✓, logprob_agreement ✓, paired_delta_above_min ✗, regressions_within_cap ✗ |
| GPU-seconds | preflight 93, filter 1190, train 3413, evaluate 856 + 920 (total ≈ 6470 of a 12600 budget) |
| wall clock | cycle 73 min; node 149 min total ≈ $10.8 |

**Reading**
- The loop works end to end: traces were ingested from a real agent, turned into hinted OPSD rows, trained
  on top of the RL checkpoint, evaluated paired against the incumbent, and refused promotion with a receipt.
- The task pool is too easy for the gate to show a promotion: the base model already solves 84% of the holdout,
  so a +10 point paired delta is nearly impossible and the CI on 24 tasks is ±7 points. A pool with base pass
  rate around 40–60% (more v1 idioms per task, or a different migration) is the next thing to build.
- `polyloop/logprob_len_mismatch` counts every datum in both stages: the cookbook KL-to-sampler metric receives a
  mask longer than the returned logprobs. It is metric-only (guarded), but the root cause is still open.
- Not implemented yet: pausing the sampler around `save_weights_for_sampler` (the H100 timeout seen earlier
  was avoided here by capping generate concurrency at 8 per engine).
