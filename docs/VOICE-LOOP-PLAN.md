# Voice agent loop with Coval as the gate

Research date: 2026-09-17. Sources are Coval's docs and generated SDK (read at code level), a live
read-only probe of the Coval tenant, Daily/Pipecat's PhoneLLM and PhoneBench pages, and the papers
and repos cited inline. Anything not verified is marked.

## The claim we want to demo

A self-hosted voice agent's LLM gets better from its own calls, nightly, with a receipt: production
call traces are scored, the scores become training signal, a candidate adapter is trained in a
sandbox, and Coval simulations decide whether it ships. Nobody publishes this loop for voice
today. The closest public pieces are Daily's PhoneLLM (an SFT of Nemotron 3 Nano on undisclosed
phone data, scored on a closed judge benchmark), PolyAI's Dialog-RSN-1 (SFT + RFT, undisclosed
data and reward), Phonely's LoRA fine-tunes on Groq (99.2% vs GPT-4o 94.7% on an undisclosed
eval), and MUA-RL (RL with an LLM-simulated user, binary DB-state reward, open). None of them are
gated, none are nightly, none publish lineage.

## What is trainable and what is not

The trainable part is the text LLM inside a cascaded pipeline (STT → LLM → TTS). STT, TTS and turn
detection stay fixed (Pipecat's smart-turn is the model for how a trained voice component is
released: weights, data and script on HF). τ-Voice's finding justifies the split: 79 to 90% of
voice-agent failures are agent behaviour, not audio.

Model choice:

| model | why / why not |
|---|---|
| Qwen3.5-4B / 9B | already runs in our stack (SkyRL Megatron LoRA); good enough for the demo; not what the voice market uses |
| Nemotron 3 Nano 30B-A3B | what PhoneLLM is built on; NVIDIA Open Model License; vLLM LoRA support for NemotronH; hybrid Mamba-2 MoE, so SkyRL's Megatron LoRA path for it is UNVERIFIED (must smoke-test before committing) |
| Nemotron-Flash 1B/3B | exists (arXiv 2511.18890, Nov 2025) but CC-BY-NC-4.0, no tool-calling docs, no voice use anywhere; not usable |
| Qwen 3.8 27B | top of Pipecat's open voice-readiness benchmark (98.2% at 649 ms); a later step, not the demo |

Recommendation: demo on Qwen3.5-9B, then port the recipe to Nemotron 3 Nano once the Megatron
smoke test passes. "PhoneLLM-class" is the pitch, and PhoneLLM's own base scored 28.6 on
PhoneBench before SFT, so the headroom on Nano is real.

## Where Coval fits (and where it must not)

Coval is three things for the loop, verified against the API:

1. **Production scoring.** `POST /v1/conversations:submit` takes transcript + audio; an org-wide
   policy decides which metrics score every call; results come back per conversation with judge
   explanations. This is the reward source for the offline (OPSD) path.
2. **Held-out gate.** A run = agent × persona × test set × mutations. Register the candidate as a
   mutation of the base agent (a different `chat_endpoint` or a header the proxy maps to an
   adapter), launch one run with both, and read `results.metrics[m].values[]` keyed by
   `simulation_output_id`. Each simulation carries `test_case_id`, so polyloop can compute its
   usual paired delta and bootstrap CI per test case × iteration.
3. **Pre-production tests.** Scheduled runs, run templates, the GitHub Action (launch + poll, no
   threshold of its own), monitors on run completion, agent versions with revert. That is the
   "test the agent before production" flow the user wants, and it needs no code from us beyond
   the agent registration.

What Coval must not be: the training rollout engine. Every simulation is a real-time
conversation billed in simulation minutes (Starter: $100/mo for 100 min, $0.40/min overage,
5 concurrent; Growth: $500/mo for 1,000 min, 25 concurrent). API caps: `iteration_count` ≤ 50,
`concurrency` ≤ 100, chat billing per minute UNVERIFIED (the pricing page only lists voice
minutes). A single RL step of 8 tasks × group 4 is 32 conversations; at 2 min each that is 64
simulation minutes, about $26 on Starter overage and 13 min of wall clock at 5 concurrent.
Training needs thousands of rollouts per cycle. They have to run in our sandbox.

## The sandbox: "voice traces" as text rollouts

Verified prior art says text-mode rollouts with a persona LLM and injected ASR noise are the
standard cheap approximation:

- Pipecat Evals `user.modality: text` and LiveKit Agents `session.run(user_input=...)` run the
  unchanged agent in-process with user turns bypassing STT and TTS skipped.
- τ²-bench ships `AgentGymEnv` (Gymnasium interface since v0.2.1): text observations, tool or text
  actions, configurable user LLM, and a reward that is a DB-state diff plus required actions plus
  NL assertions. Its telecom domain gives the user simulator its own tools (dual control).
  τ-Voice keeps the same tasks and grader byte-identical, so a text-trained policy can be
  re-scored in audio later.
- MUA-RL trained Qwen3-8B/14B/32B this way (GPT-4o as the user, reward = 1 only on DB-state
  success, dialogue-content constraints dropped during RL).
- ASR-noise injection for dialogue training: TOD-DA (word and phoneme level), Speak & Spell
  (keyword-targeted phonetic confusions), MEDSAGE (few-shot LLM as the noise model from real
  STT pairs), arXiv 2401.02297 (fit the error distribution from your own STT).

So the sandbox for one rollout is: task spec (scenario, persona prompt, mock tools, expected DB
state or expected behaviors) → persona LLM plays the caller, its turns pass through an ASR-noise
injector fitted to our STT → the policy answers through the polyloop proxy (token-exact) → mock
tools mutate a small DB → grader returns a verifiable reward. No audio. One rollout costs a few
seconds of sampler time plus persona-LLM tokens.

Known traps, with the mitigation each source proposes:

- Simulated callers "bend over backwards to make the conversation succeed" (Coval's CEO, Apr
  2026); calibrate the simulator against production transcripts before RL (arXiv 2605.26403)
  and discard rollouts where the simulator itself erred (EVA-Bench's regeneration step).
- Reward-relevant perturbations hurt far more than observation noise (ToolRL-DR): randomize tool
  failures and state transitions, not just ASR noise.
- Judge-only rewards get gamed (ART·E's exploration bonus, the 2606.03238 taxonomy). Keep the
  primary reward verifiable (DB state, tool arguments, escalation discipline) and use judges only
  as penalties or at the gate.

## Three signal sources, one loop

```
production calls ──submit──▶ Coval conversation metrics ──▶ rewards + transcripts ──▶ OPSD rows (next-state hint)
synthetic tasks  ──────────▶ sandbox persona env (text, ASR noise, mock tools, DB grader) ──▶ RL groups
held-out test set ─────────▶ Coval run: base vs candidate mutation, K iterations ──▶ paired receipt ──▶ promote
```

Cycle stages map onto the existing harness unchanged: snapshot pins the test set version and
the Coval metric versions; filter measures pass rate in the sandbox; train runs rl then opsd;
evaluate launches the Coval run; gate reads it; promote bumps the agent version on Coval and
`live.json` on the proxy.

## Synthetic dataset

Coval's Test Set Generator and its conversation-to-test-case conversion exist only in the UI
(no API). Build it client-side, as the SDK example already does:

1. Pull flagged production conversations (`GET /v1/conversations`), keep the ones a metric
   failed, and write TRANSCRIPT test cases with `expected_behaviors` (must-always / must-never).
2. Generate SCENARIO cases from a task description with parameters (`{"name": [...]}` style) for
   the sandbox, in τ²-bench task format so the DB grader applies.
3. Split by scenario family into train (sandbox) and held-out (Coval), never by instance.

Open phone-conversation data for bootstrapping: SpokenWOZ (5.7k dialogues, 8 domains), plus the
HF call-center transcript sets (licences unverified). PhoneData and PhoneBench are not released.

## Build list in polyloop-rl

1. `polyloop/coval.py`: register agent and mutations, launch run, poll or webhook, fetch
   rewards joined by `simulation_output_id`, fetch transcripts, submit production conversations.
2. Proxy: accept Coval's chat contract (`{sessionId, messages}` → OpenAI, `status: ended`),
   honor `X-Coval-Simulation-Id` as the session id, map a policy header to an adapter so one
   endpoint serves base and candidate.
3. `polyloop/envs/persona.py`: the text sandbox above; start by wrapping τ²-bench `AgentGymEnv`
   (retail + telecom) and adding the ASR-noise injector and persona prompts copied from Coval's
   seven default personas.
4. `evaluate.coval` stage: one run, two mutations, paired receipt from run results.
5. `recipes/voice-booking/loop.yaml` + `docs/RESULTS.md` entry.

Verification items before building on Nemotron: SkyRL Megatron LoRA on NemotronH; Coval chat
simulation billing; `job_complete` webhook payload (undocumented, poll instead).

## Costs for a first demo

| item | estimate |
|---|---|
| Sandbox RL + OPSD, 9B, one cycle | same as the pydantic cycle: about 2 h on 4×A6000 or 1 h on 2×H100, $10 to $15 |
| Coval gate: 24 held-out cases × 2 policies × 2 iterations at ~2 min | ~190 simulation minutes; exceeds Starter's 100, about $36 overage; fits Growth |
| Persona LLM tokens for ~1,000 sandbox rollouts | small (a 4B/9B persona on the same sampler, or an API model) |

The Coval account today has one Coval-managed OpenAI Realtime starter agent, seven default
personas, and no test sets, metrics, runs or conversations. Everything above starts from zero.
