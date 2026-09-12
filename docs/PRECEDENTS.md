# Precedents (research, 2026-09-12)

Question: does any paper, product, or open-source repo implement an end-to-end automated
continual-learning loop for LLM policies (trigger, snapshot, train, gate, promote/rollback,
multi-tenant scheduling, repeat)? Answer: no. The parts exist; the combination does not.

Closest:

- **Cursor real-time RL** (blog, Mar 2026): shipped, fully automated, ~5 h checkpoint-to-prod
  for Composer, ~2 h for Tab, eval-suite gate. One full-parameter model, no published rollback.
- **Trajectory AI**: traces in, continuous LoRA training, eval-suite gate, multi-LoRA (8
  adapters, 2.81x). Human approval mandatory; no canary or auto-rollback.
- **Pioneer Agent** (arXiv 2604.09791): agent orchestrates trace analysis, data synthesis and
  Tinker retrains; gate = accuracy >= 0.96 and <= 2 regressions on current AND previous held-out
  sets; rollback on drop; replay 10–20%. SFT-only; checkpoint rollback, not traffic split.
- **AReaL 2.0** position paper (arXiv 2607.01120): describes every part (evolution control plane,
  replay-first eval, shadow/canary, rollback first-class); implements only the weight-update loop.
- **CLaaS** (arXiv 2606.05559): buffer-triggered online LoRA RL behind a chat API, hot-reload;
  no snapshot, gate or rollback.
- **OpenClaw-RL** (5.7k stars): live traffic -> judge -> train -> serve, unattended and ungated.
- **OpenRL autoresearch** (gke-labs/open-rl): attempt contract (`program.md`, `autoresearch.toml`,
  `run_attempt.py`): command, editable files, one metric, keep or reset. The cycle in miniature.
- **understudy-agent-tools**: `promotion_receipt.v1` schema: decision requires fresh-holdout
  evidence, serving parity, and a demotion trigger. Schema only.

Singletons: Osmosis = only claimed auto-trigger; Adaptive ML `auto_deploy` = only auto-promote;
Uber Michelangelo = only auto-rollback (classic ML); AWS AgentCore = trace-driven A/B with p<0.05
for prompts/config; TensorZero archived Jun 2026.

## Recipe vs code

Every autoresearch-style loop that met a real optimizer grew code around the agent (allowed
files and timeouts, anti-thrash hooks, plan-hash approval and append-only ledger, hidden
harnesses after metric hacks). Scheduling, snapshots, sandboxes and lineage are never
agent-done anywhere. Across repeated post-training campaigns only rule-based schedules improved
(arXiv 2606.21089); ResearchGym agents beat baseline in 1/15 runs. Hence: controller in code,
agent as proposer.

## Demo evidence

- Agent Lightning v1.0 (arXiv 2608.17528): Qwen3.5-9B + mini-swe-agent, SWE-smith 6K, GRPO,
  SWE-bench Verified 41.8 -> 56.4 (full-parameter, 4x B200).
- SWE-bench Verified run-to-run noise ~ +-0.7 over 3 runs; infra noise ~1.5 pts; 24% of
  instances have weak tests (report the Epoch 484-instance filtered set too).
- Terminal-Bench 2.1: Qwen3.5-9B + mini-swe-agent 16.1 +-3.7 -> 28.8 +-1.4 after RL; 5 trials.
- Credibility = multi-cycle curve + error bars + cost per cycle + reproducible held-out.
