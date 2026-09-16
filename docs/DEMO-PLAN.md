# Demo plan: train your own coding model from your agent's traces, nightly, gated

Status: research done 2026-09-15, sources verified 2026-09-16, build not started. This is the concrete answer to "what do we
need to ship the OPSD harness so it gives an OpenClaw-RL-like demo on SWE tasks".

## The claim the demo makes

Your coding agent runs an open model (Qwen3.5-9B) through polyloop's proxy while it works on
your repos. Every night polyloop turns the day's sessions into training data, trains a candidate
adapter with hybrid GRPO + hinted on-policy self-distillation, scores candidate vs incumbent on a
held-out task set in sandboxes, gates with paired statistics, writes a receipt, and promotes the
winner into the adapter the agent uses tomorrow. After N nights: a held-out curve with
confidence intervals, one receipt per promotion, before/after transcripts, cost per cycle.

Nobody ships this end to end (survey in PRECEDENTS.md). OpenClaw-RL has ingest + hinted OPD but
no gate and is stale since May; Whitney does it as a human service; ART, Unsloth, Modal, Castform,
prime-rl stop at the training job.

## What is needed, and what is not

| Need | Verdict | Detail |
|---|---|---|
| Traces | **Yes, token-exact** | Raw sampled token ids + logprobs per turn, captured at the proxy, never re-rendered. Session id and main/side turn headers (OpenClaw-RL's two headers). The next-state (tool output, test output, or next user message) is paired to each assistant turn. Compaction and pruning off in the agent. |
| Labeled dataset | **No** | OPSD needs no labels. GRPO needs a reward, see verifier. |
| Verifier | **Where available** | Repo tests run in the sandbox are the hard reward. Where no tests exist, a frozen copy of the base checkpoint acts as a judge on (assistant turn, next state) and emits a ternary verdict plus a 1-3 sentence hindsight hint (OpenClaw-RL judges with the frozen initial weights, not the live policy). A turn with no positive hint is dropped; the final turn of a session is submitted without a next state rather than dropped (OpenClaw-RL's actual rule). |
| Held-out eval tasks | **Yes, non-negotiable** | A trace-only eval cannot support a gated claim. Use a task set with verifiers that the trace pool never touches. |
| Judge for trace-derived eval | Secondary | Prefix replay against the human's later edit, and pairwise vs the logged response with swapped order. Cheap, biased; supporting evidence only. |

## Method (from the research receipts)

- **Capture**: OpenAI-compatible facade in front of the sampler that stitches turns on raw tokens
  (ART `prefix_cache` / Prime renderers bridge design; rlcli `tito_bridge` already does the
  bridge). OpenClaw-RL's own proxy re-tokenizes text and is not token-exact; do not copy it.
  Qwen3.5: thinking off and raw-token stitching, because re-rendering drops thinking blocks.
- **Hint**: frozen base checkpoint as judge, prompt = (assistant turn t, next state t+1 with its
  role), output `\boxed{1|-1}` + `[HINT]...`; 3 votes (OpenClaw-RL's OPD-only launcher uses 1),
  keep the longest positive hint, else drop the sample. Hint appended to the last user message
  for the teacher only; the student never sees it.
- **Loss**: per-token advantage `A_t = log pi_T(a_t | s+h) - log pi_old(a_t | s)`, clipped
  (C = 1 to 2), PPO-clipped surrogate; hybrid `L = L_GRPO + L_OPD` (OpenClaw-RL: hybrid 10.3 vs
  GRPO 14.1 vs OPD-only 29.7 sessions-to-success; OPD alone is 2-3x slower). Overlap-guided hint
  selection where 3 candidate hints exist. KL-to-reference 0; LoRA rank 64 (OpenClaw-RL's
  combine launcher; its plain RL launchers use rank 16).
- **Rollouts**: OPD needs 1 rollout per prompt; GRPO on verifier tasks uses groups of 8.
- **Forgetting**: two controls with separate receipts. Replay of generic chat data during training
  (Thinking Machines: IFEval 85 to 45 at 100% new data, 79 with a 70/30 mix) and, when a cycle
  regresses, a distillation pass from the frozen base model on chat prompts (that is what restored
  IFEval to 83 in their post). SDFT reports a -0.1 to -1.5 point prior-capability drop vs -5.3 to
  -12.1 for SFT. Gate on KL-to-base measured on new-task inputs (RL's Razor, R^2 0.71) plus a small
  IFEval / LiveCodeBench slice.

## Evaluation and gate

- **Held-out set**: SWE-rebench monthly split (48 to 110 fresh tasks per month, prebuilt images)
  plus a fixed 100-task SWE-bench Verified subset for the headline. Repo-level separation between
  the trace pool and the held-out set is OUR job: neither dataset card guarantees it, so build an
  exclusion list by repo, order the pool by task hash, and plant canary strings. Do not reuse
  OpenClaw-RL's SWE recipe or data: its swe-rl track trains on SWE-bench Verified by default.
- **Repeats**: K = 4 per task per policy. With n = 100 and K = 4 the 95% CI half-width is about
  7.6 pp (7.1 to 8.3 depending on question-level variance); certifying +3 pp at K = 4 needs roughly
  1,100 to 1,600 tasks (Miller's paired formula). So a single nightly gate cannot certify +3 pp. The gate therefore uses: paired mean delta with tie band, regression count cap,
  KL-to-base cap, and accumulates evidence across cycles (the curve, not one night).
- **Promotion**: candidate replaces incumbent only when delta > tie band, regressions <= cap,
  KL-to-base <= cap, logprob agreement passes. Receipt (promotion_receipt.v1) records the paired
  table, CI, regression list, cost, and the trace snapshot hash the candidate trained on.

## Build steps in polyloop

1. **ingest**: proxy with session/turn headers, raw-token stitching, next-state pairing, episode
   store (ATIF + token ids). Reuse rlcli `tito_bridge` and `capture`; port OpenClaw-RL's header
   contract and drop rule.
2. **hint**: judge stage on the sampler; PRM verdict + hint per trainable turn; overlap selection.
3. **learn**: hybrid stage on the SkyRL Tinker server. Extend rlcli's `HintedTeacher` from
   single-turn prompts to per-turn prefixes of coding episodes; combine with the existing GRPO
   stage on verifier tasks; replay slice.
4. **evaluate + gate**: SWE-rebench monthly + Verified-100 in Docker sandboxes, K = 4, paired
   bootstrap, tie band, regression count, KL-to-base; receipt.
5. **promote**: adapter switch on the sampler the proxy fronts, so the agent uses the winner at
   once; rollback rule.
6. **report**: curve with CI per cycle, cost per cycle, two before/after transcripts, receipts.

Open blocker carried over: SkyRL sampler `sample_tokens` timeout after the second weight sync
(see polyloop-rl build notes); options are lower sampler concurrency after sync, restart between
batches, async off-policy K = 1, or an ART/Miles train backend.

## Receipts this rests on

OpenClaw-RL tech report 2603.10165 and repo (hint prompt, loss, headers, numbers); Hindsight Hint
Distillation 2605.11556 (SWE-bench Verified 43.0 to 51.2 with gold-patch hints + SFT); SDPO
2601.20802 (LiveCodeBench 48.8 vs GRPO 41.2 with test feedback as teacher context); SDFT
2601.19897 (prior-capability drop -0.1 to -1.5 vs SFT -5.3 to -12.1); Thinking Machines on-policy
distillation post (its 9-30x cost claim is vs SFT and off-policy distillation, not vs RL); Whitney
OPSD case studies; Miller /
Anthropic paired-eval statistics 2411.00640; SWE-rebench and SWE-bench-Live; RL's Razor
2509.04259; Retaining by Doing 2510.18874; Cursor real-time RL post; Agent Lightning 2608.17528
(no seeds or error bars reported).
