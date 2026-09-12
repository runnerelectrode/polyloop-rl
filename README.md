# polyloop-rl

A harness that keeps an agent policy learning after deployment. It runs **cycles**: take the
current adapter, find the tasks it half-solves, train on them, score the result against the
incumbent on a fixed held-out split, write a receipt, and promote only if the receipt says so.

The controller is code. An agent (Claude, Codex, a script) can propose tasks, config changes and
diagnoses through a recipe, but it cannot touch what is measured: the held-out split, the
thresholds, the budget and the lineage are declared in `loop.yaml` and owned by the controller.

Status: **v0, under construction (September 2026).** The cycle runner, stages, receipts, task
builder and CLI are here; the first demo cycles are running on one Lambda 2xH100 node. Numbers
land in `docs/RESULTS.md` when they exist. Nothing here is a product yet.

## What a cycle does

```
snapshot   pin the task pool, the held-out split (by id) and the incumbent adapter
preflight  config-only checks: pool, holdout, docker, trainer server, disk, staleness bound
filter     measure pass rate under the incumbent; keep tasks with 0 < pass < 1 (zero advantage otherwise)
train      cookbook Harbor RL, LoRA, token-in/token-out, resumed from the incumbent adapter
evaluate   candidate and incumbent on the same held-out tasks, K repeats, paired per task
gate       paired delta > min_delta, regressions <= cap, trainer-vs-sampler logprob agreement  -> receipt.json
promote    approve (default: a human runs `polyloop approve <cycle>`) or auto
observe    hook for rollback rules once an endpoint serves the adapter (not in the single-node demo)
```

Every cycle is a directory of append-only events plus the artifacts each stage wrote, so a cycle
can be resumed, audited, or replayed. The lineage of promoted adapters is one JSON file.

## The demo: a coding agent that gets better nightly

- Model: `Qwen/Qwen3.5-9B` with a LoRA adapter (rank 32), trained with SkyRL through the Tinker API.
- Agent: the tinker-cookbook Harbor bash-tool loop, one Docker container per episode.
- Train pool: SWE-smith instances converted to Harbor tasks with a verifier that runs in seconds
  (`polyloop tasks swesmith`). The `.git` directory is moved out of `/testbed` at build time so
  the agent cannot recover the fix from history.
- Held-out: a fixed, seeded subset of SWE-bench Verified (Harbor `swebench-verified@1.0`), never trained on.
- Hardware: one Lambda `gpu_2x_h100_sxm5` node. GPU0 trains, GPU1 samples, the CPUs run sandboxes.

Why a 9B model: the published receipt for this exact setup is Qwen3.5-9B trained on SWE-smith
going from 41.8% to 56.4% on SWE-bench Verified (Agent Lightning, arXiv:2608.17528). The loop's
product is the curve across cycles and the receipts behind each promotion, not the absolute score;
a 9B model gives the largest, cheapest delta that fits a 2-GPU node with 28k-token episodes.

## Install (on the GPU node)

```bash
git clone https://github.com/runnerelectrode/polyloop-rl && cd polyloop-rl
uv venv ~/venvs/polyloop --python 3.12
uv pip install --python ~/venvs/polyloop/bin/python -e ".[run]" \
  "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@f46eddde86e5397138917516a6c69d2ecbf538b1"

# trainer + sampler: SkyRL's Tinker-compatible server, one GPU each
rlcli serve start --base-model Qwen/Qwen3.5-9B --backend megatron --gpus 1 --tp 1 --max-model-len 32768 \
  --backend-config '{"trainer.placement.colocate_all": false}'

# task pool and held-out split
polyloop tasks swesmith --out ~/polyloop-tasks/swesmith-pool --n 200 --limit-repos 8
uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified@1.0
```

## Run

```bash
polyloop eval --loop recipes/swe-mini/loop.yaml --limit 20          # baseline of the base model on the holdout
polyloop run  --loop recipes/swe-mini/loop.yaml                      # one full cycle
polyloop history --loop recipes/swe-mini/loop.yaml                   # cycles, deltas, lineage
polyloop approve --loop recipes/swe-mini/loop.yaml <cycle-id>        # promote a gated candidate
polyloop run  --loop recipes/swe-mini/loop.yaml --cycle <cycle-id>   # resume a cycle at its next stage
```

`loop.yaml` (see `recipes/swe-mini/`) declares the stage, the sandbox limits, the filter, the budget,
the gate and the promote mode. `program.md` next to it says what a proposing agent may change.

## What it builds on

| Piece | Where it comes from |
|---|---|
| Trainer + sampler with a Tinker API, LoRA, multi-adapter | [SkyRL](https://github.com/NovaSky-AI/SkyRL) via [rlcli](https://github.com/polygramme/rlcli) |
| Harbor bash-tool RL loop, async off-policy bound | [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) `recipes/harbor_rl` |
| Token-in/token-out bridge, Docker sandbox, logprob agreement guard | rlcli (`tito_bridge`, `sandbox_docker`, `logprob_guard`) |
| Task format and the SWE-bench Verified tasks | [Harbor](https://github.com/harbor-framework/harbor) |
| Verifier recipe and reward-hack closures for SWE-smith | [Agent Lightning](https://github.com/microsoft/agent-lightning) `examples/swe_smith` |
| Pass-rate filter (0 < pass < 1) | [Lego-RL](https://github.com/LegoX/Lego-RL) preflight |
| Attempt contract, event log, "agent proposes, code measures" | [OpenRL autoresearch](https://github.com/gke-labs/open-rl) |
| Budget overrun = no candidate, never a partial one | Baseten's [rlm](https://github.com/basetenlabs/rlm) fork |
| Receipt shape | Understudy `promotion_receipt.v1` |

`docs/DESIGN.md` has the full design; `docs/PRECEDENTS.md` lists what was surveyed and why none of it
already runs this loop end to end.

## Layout

```
polyloop/
  config.py        loop.yaml -> LoopConfig
  events.py        per-cycle events.jsonl + state.json, lineage.json
  stages.py        the eight stages and the resume logic
  receipt.py       paired stats, bootstrap CI, promotion_receipt.v1
  harness/
    rollout.py     K episodes per task for one checkpoint (filter + evaluate)
    train.py       train worker: cookbook RL run from a JSON spec, in its own process
  tasks/
    swesmith.py    SWE-smith rows -> Harbor tasks with a fast verifier
  cli.py           polyloop run | approve | history | status | eval | tasks
recipes/swe-mini/  loop.yaml + program.md
docs/              DESIGN.md, PRECEDENTS.md
```

## License

Apache-2.0.
