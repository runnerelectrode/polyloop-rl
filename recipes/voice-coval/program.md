# voice-coval: what the proposer may do

You are the proposer for the `voice-coval` loop. The controller (`polyloop run`) owns the two
Coval test sets, the reward metric ids, the gate thresholds, the budget and the lineage; you do
not edit those.

You may propose, as pull requests against this recipe:

- new scenarios for the pool (`scenarios-pool.json`, then `polyloop coval seed --test-set pool`),
  with the pass-rate filter deciding whether they are contested enough to train on. A scenario
  is a caller intent plus the behaviours the judge checks; write the behaviours the way the
  metric prompts read them;
- changes to the `opsd` stage (group size, steps, learning rate, hint length, `min_rows`) with the
  receipt of the cycle that motivated them;
- changes to `system.md`, which the proxy prepends on every simulated conversation. A prompt
  change is a policy change: it goes through the same gate as an adapter, so ship it with the
  cycle's receipt;
- diagnoses of rejected cycles: `polyloop coval ledger --failed` lists the sessions the judge
  failed and why; `cycles/<id>/eval/*/rewards.jsonl` has the per-scenario scores of both policies
  and `train/*/rows.jsonl` the rows the student trained on. Cluster the explanations, say what
  to change.

You may not: change `gate.*`, `promote.*`, `coval.metric_ids`, the test set ids, or write to
`lineage.json`. Adding a metric to the reward changes what is measured; that is a controller
change, made in a separate PR with the holdout re-baselined.

## Next stage to build (not in this recipe yet)

GRPO on the captured sessions: the ledger already pairs every session with its reward and the
proxy holds the token-exact turns, so the group is the K simulations of one scenario and the
advantage is the reward minus the scenario mean. That stage needs a trainer that consumes logged
tokens instead of sampling in a sandbox (`polyloop/harness/train.py` samples live). Until then
the loop learns from hinted self-distillation only.
