# voice-coval: a voice agent's LLM that learns from Coval evals, nightly

The pydantic-v2 loop measures a coding policy with tests in a sandbox. A voice agent has no
tests: what it has is callers and a judge. This recipe swaps both into polyloop without
changing the cycle:

| polyloop stage | pydantic-v2 | voice-coval |
|---|---|---|
| task | a repo with a hidden test | a Coval **test case**: a caller scenario plus expected behaviours |
| environment | Docker sandbox, bash tool | Coval's simulated caller (**persona**) talking to the proxy |
| verifier | pytest with deprecations as errors | Coval **metrics** (LLM judges) on the transcript; reward = their mean |
| traces | mini-swe-agent through the proxy | every simulated conversation, through the same proxy |
| hindsight hint | the next tool output | the caller's next line **plus the judge's explanation** of why the call failed |
| train | GRPO in sandboxes, then OPSD | OPSD on the captured calls (GRPO on logged tokens is the next stage, see program.md) |
| gate | paired delta on held-out repos | paired delta on a held-out Coval test set, K repeats per scenario |
| promote | proxy serves the adapter | proxy serves the adapter; the voice pipeline never changes |

The policy is the language model inside the voice pipeline (speech-to-text, LLM, text-to-speech).
The recipe pins **`nvidia/Nemotron-Flash-3B-Instruct`**, NVIDIA's hybrid Mamba/attention model
built for the latency budget of a spoken turn. The loop is text-in, text-out at the LLM boundary:
Coval's chat connection drives the proxy directly for the nightly cycle, and a real voice stack
(Pipecat, LiveKit, a Twilio ConversationRelay handler) points its LLM at the same proxy URL for
production traffic, so both kinds of sessions land in the same trace files.

## Serving Nemotron Flash: read this first

Nemotron Flash is a custom architecture (`model_type: nemotron_flash`, `trust_remote_code`,
CC-BY-NC-4.0). As of 2026-09-17:

- **vLLM does not implement it** (no `NemotronFlashForCausalLM` in vLLM's supported models).
  polyloop's existing node stack is SkyRL's Tinker server with a vLLM sampler, so `polyloop warm`
  and the proxy fail on this model id until one of these lands: a vLLM model implementation, or a
  Tinker-compatible server whose sampler is TensorRT-LLM (which has an AutoDeploy factory for
  Nemotron Flash) or NeMo Automodel (which trains it, with LoRA recipes for the 1B checkpoint).
- The trainer side is the same story: SkyRL's Megatron backend has no bridge for the hybrid
  layers. NeMo Automodel does (`nemotron_flash_1b_squad_peft.yaml`).

The recipe is written against the Tinker API polyloop already speaks, so it runs unchanged the
day a Tinker server serves the model. To run the loop **tonight** on the current node, change
`model:` to a chat model the SkyRL server already serves with LoRA (the pydantic loops use
`Qwen/Qwen3.5-4B` with `renderer: qwen3_5_disable_thinking`); nothing else in the recipe depends
on the model. Nemotron Flash ships no chat template (its documented prompt is
`User: ...\nAssistant:`), which is why the recipe names the cookbook's `role_colon` renderer.

## Setup

1. **Coval account and key.** Create an API key in the Coval dashboard and export it:
   `export COVAL_API_KEY=...`. Everything below uses the public v1 API
   (https://docs.coval.ai/api-reference/v1/introduction); no SDK is needed.
2. **Expose the proxy.** Coval calls the agent over HTTPS and refuses private addresses. On the
   node, start the proxy with a token and open a tunnel:

   ```bash
   export POLYLOOP_PROXY_TOKEN=$(openssl rand -hex 16)
   polyloop proxy --loop recipes/voice-coval/loop.yaml --port 8787          # reads the token from the env
   cloudflared tunnel --url http://127.0.0.1:8787                            # prints https://<name>.trycloudflare.com
   ```

   Put the printed URL in `coval.public_url`. The proxy now answers `/healthz` publicly and
   requires `Authorization: Bearer <token>` on everything else.
3. **Base agent in Coval.** Create a `MODEL_TYPE_CHAT` agent (dashboard or `POST /agents`) and
   paste its id into `coval.agent_id`. Its chat endpoint can point at the live route,
   `<public_url>/r/policy/live/v1/chat/completions`; polyloop duplicates this agent once per
   policy it evaluates (`base`, each candidate, each incumbent) and sets the copy's endpoint to
   that policy's route, the OpenAI request template, `X-Session-Id: {{simulation_output_id}}`
   and the bearer token. The copies are cached in `runs/<loop>/coval_agents.json`.
4. **Persona.** Pick or create the simulated caller and paste its id into `coval.persona_id`.
5. **Metrics.** Create binary LLM-judge metrics (dashboard or `POST /metrics`) that read the
   test case's expected behaviours, for example "Did the agent do everything listed in the
   expected behaviours?" and "Did the agent state any fact, price or availability not in its
   instructions?" (inverted so 1 = good). List their ids under `coval.metric_ids`. The reward is
   their mean, so keep them 0/1 or 0..1.
6. **Test sets.** Create two SCENARIO test sets in Coval, paste their ids into `tasks:` and
   `gate.holdout:`, then load the recipe's scenarios:

   ```bash
   L=recipes/voice-coval/loop.yaml
   polyloop coval seed --loop $L --test-set pool    --file recipes/voice-coval/scenarios-pool.json
   polyloop coval seed --loop $L --test-set holdout --file recipes/voice-coval/scenarios-holdout.json
   polyloop coval check --loop $L      # resolves every id, probes the proxy locally and through the tunnel
   ```

   The holdout is never trained on; `load_loop` refuses a recipe whose two ids are equal.

## Run

```bash
L=recipes/voice-coval/loop.yaml
polyloop eval --loop $L --limit 8 -k 2                 # baseline: base model on 8 holdout scenarios x 2
polyloop run  --loop $L                                # one cycle: snapshot ... gate
polyloop coval ledger --loop $L --failed               # what the judge said about the failed calls
polyloop approve --loop $L <cycle-id>                  # the proxy's live route switches to the winner
```

A cycle costs, on Coval's side, `filter.pool_sample x filter.rollouts_per_task` conversations per
filter round plus `2 x gate.holdout_limit x gate.repeats` for the gate (the incumbent's holdout
scores are cached across cycles, so it is half that after the first). With the defaults: 60 for
the filter and 192 for the first gate.

## What the stages do here

- **snapshot** pins the pool and holdout test case ids (the holdout id is a hash of them), the
  incumbent and the trace files.
- **preflight** resolves the agent, persona, both test sets and every metric through the API,
  checks that `coval.temperature` equals `gate.temperature`, that the proxy answers locally and
  through the tunnel, then warms the sampler. No simulation is launched.
- **filter** runs the incumbent on a sample of pool scenarios (one Coval run, K iterations each),
  keeps the scenarios it half-solves, and writes every session's reward, metric values and
  judge explanation to `runs/<loop>/coval_sessions.jsonl` (the ledger).
- **train** builds OPSD rows from the traces. A row's hint is the caller's next line and, for
  sessions the judge failed, the judge's explanation; the teacher sees the hint, the student
  does not. Sessions from production traffic have no judge line and train on the caller's
  reply alone.
- **evaluate** registers the candidate and the incumbent as proxy policies, gets (or creates)
  a Coval agent per policy, runs both on the same holdout scenarios with K iterations, and
  reads the scores back per scenario.
- **gate / promote / observe** are unchanged: paired delta, regression cap, logprob agreement,
  receipt, then `live.json` and `/admin/promote`.

## Files

```
loop.yaml               the loop; every REPLACE_ is an id from your Coval workspace
system.md               the receptionist prompt the proxy prepends on policy routes
scenarios-pool.json     16 caller scenarios with expected behaviours (train pool)
scenarios-holdout.json  8 scenarios the gate scores on; never trained on
program.md              what a proposing agent may change, and the next stage to build
```
