# polyloop-rl design

Status: draft, 2026-09-12. The controller is code; the researcher is an agent that only proposes.

## The loop

A **Loop** is a per-tenant policy: which traces and tasks feed which training stages, when a
cycle fires, what it may spend, what a candidate must beat, and how it earns traffic.

```yaml
name: swe-mini
model_id: swe-mini-v                # adapter lineage; each cycle publishes swe-mini-v{n}
sources:
  tasks: swe-smith-6k               # Harbor-format tasks with verifiers
  traces: swe-mini-prod             # optional: judged episodes from production
stages:
  - kind: rl
    harness: mini-swe-agent
    group_size: 4
    groups_per_batch: 8
    steps: 40
    max_steps_off_policy: 2
trigger:
  any_of:
    - cron: "0 3 * * *"
    - new_traces: {min: 200, contested_min: 60}
  cooldown: 6h
budget:
  gpu_seconds_per_day: 28800
  max_cycle_gpu_seconds: 14400
replay:
  fraction: 0.05                    # on-policy replay of prior cycles' high-reward rollouts
gate:
  holdout: swe-bench-verified-484   # fixed split, never trained on
  repeats: 3
  min_delta: 0.01                   # paired delta tie band
  regression_suite: prior-cycles    # tasks that must not drop
  kl_to_base_max: 0.15
promote:
  mode: approve                     # approve | auto
  ladder: [shadow:0.10:30m, canary:0.05:2h, canary:0.25:6h, full]
  rollback_on: {score_drop: 0.05, error_rate: 0.02}
```

## A cycle

Eight stages, each writing one artifact the next stage reads. Only train and evaluate touch GPUs.

1. **snapshot** — dataset as of the trigger time; replay slice selected; snapshot id recorded.
2. **preflight** — config-only checks in seconds: dataset non-empty, contested count above the
   trigger's minimum, base model and renderer match the lineage, learning rate consistent with
   the staleness bound, budget remaining above the forecast. Refuses before any GPU is allocated.
3. **filter** — keep tasks with 0 < pass rate < 1 under the current adapter (uniform groups have
   zero advantage). Recorded outcomes first; re-roll only tasks with fewer than G samples.
4. **train** — the loop's stages, chained; a checkpoint id passes between them.
5. **evaluate** — candidate and incumbent on the same held-out tasks with K repeats; keep the
   per-task paired delta; run the regression suite; compute KL of candidate to base on the
   new snapshot; run the trainer-vs-sampler logprob agreement check.
6. **gate** — promote only if paired delta > min_delta, no regression-suite task drops more than
   the tie band, KL-to-base under the cap, agreement check passes. Write a promotion receipt.
7. **promote** — weight rewrites on the serving endpoint: shadow, canary, full; incumbent stays
   resident; each rung has a dwell time and a rollback rule.
8. **observe** — rollback rules watched; a rollback marks the checkpoint rejected and feeds the
   failing slice back to the next trigger as high-priority data.

## The researcher (agent)

An autoresearch-style recipe: `program.md` + `loop.yaml` + editable files declared in a manifest.
The agent may propose: failure clusters, new eval cases and synthesized tasks (validated by
deterministic checks before entering any split), config changes, harness changes, diagnoses.
Proposals are keyed to a plan hash. Nothing the agent writes changes what is measured: the
held-out split, the judge version, the gate thresholds and the ledger are owned by code and
are not runtime-writable.

## Forgetting controls (defaults)

- Frozen teacher per cycle for self-distillation stages (never a fast EMA).
- 5% on-policy replay from prior cycles' own high-reward rollouts.
- KL-to-base measured on the new snapshot, capped by the gate.
- Learning rate lowered with staleness (max stable LR ~ 1/staleness).

## Scheduling (later, hosted)

Postgres queue, least-attained-service per account, budget forecast, waves that amortize a
GPU boot across tenants, preemption only at stage edges, warm-pool controller. Not part of the
single-node demo.

## Sources

Design doc and research reports: see `docs/PRECEDENTS.md`.
