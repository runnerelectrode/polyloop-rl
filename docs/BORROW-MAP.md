# Borrow map (code read 2026-09-15)

All sources Apache-2.0. Local scratch clones: open-rl, miles, verl, dynamo, OpenClaw-RL, ART, SkyRL@9719b4f.

| polyloop stage | Borrow | Where | Notes |
|---|---|---|---|
| ingest: headers | `X-Session-Id`, `X-Turn-Type: main\|side`, `X-Session-Done` | OpenClaw-RL `extensions/rl-training-headers/index.ts` (55 lines); server read in `openclaw-opd/openclaw_opd_api_server.py:326-360` | For Claude Code / Codex / OpenCode use Dynamo's header normalization table (`docs/.../session-ids.mdx`, `lib/llm/src/protocols/common/extensions.rs`) |
| ingest: next-state pairing + drop rule | turn t's next state = `messages[-1]` of turn t+1 in the same session; on session done, turns without a next state are dropped | `openclaw_opd_api_server.py:702-819`, `_flush_pending_record`, `_maybe_submit_ready_samples` (~120 lines) | Do NOT copy its tokenization: it re-templates and re-tokenizes text (`apply_chat_template`, 769-777) |
| ingest: raw-token stitching facade | radix trie keyed on rendered ids, swaps in the raw sampled prefix, `token_id:<id>` logprob wire format | ART `src/art/tinker/server.py:362-421`, `prefix_cache.py` (208 lines), `_normalize_qwen3_dot_messages` (76-118) | Add a per-session key; replace `sample_async` with our Tinker sampler. Ours: rlcli `tito_bridge.py:201-238`, `capture.py:72-137` |
| ingest: mask alignment | `align_response_metadata`, `merge_assistant_tokens`, `render_delta_token_id` | verl `verl/utils/tokenizer/continuous_token.py` (180, 197, 243, 365, 562) | Borrow the semantics for the episode store; keep our bridge |
| hint judge | `build_hint_judge_messages`, `parse_judge_result` (`\boxed{±1}` + `[HINT_START]..[HINT_END]`), `select_best_hint`, `append_hint_to_messages`, `majority_vote` | OpenClaw-RL `openclaw-tinker/scorers.py` (~250 lines, Tinker-native) + `openclaw-combine/prm_teacher_postprocess.py` (overlap selection) | Run the judge on our sampler, same adapter |
| learn: datum + advantage | `adv = w_opd*(teacher_lp − student_lp)*mask + w_rl*reward*mask` → `tinker.Datum` | OpenClaw-RL `openclaw-tinker/data_formatter.py:149-186`; clip in `hint_opd_loss.py:395-411`; PPO surrogate `combine_loss.py` | Emit via cookbook `train_on_policy` with `loss_fn="ppo"`, or call `forward_backward` directly (`openclaw-tinker/trainer.py:131-170`) |
| learn: teacher logprobs | use Tinker `forward` on the same LoRA with the hinted prefix + raw completion | SkyRL `api.py:1129` → `skyrl_train_backend.py:927-985` | Exact (trainer weights). rlcli `HintedTeacher` (`opsd.py:145-183`) already splices raw ids; extend its prefix key to per-turn coding prefixes and add the assistant-span mask (cookbook `sdft._extract_completion_tokens`) |
| learn: replay | per-datum teacher clients | cookbook `train_on_policy.incorporate_kl_penalty` (54-130) | Frozen base client for replay datums |
| gate | ours (`polyloop/stages.py`) | + open-rl `run_attempt.py` idioms: `ensure_attempt_is_new` (dedupe by commit), `run_logged` timeout status, `event_log.append_ui_events` | Judge alternative: ART `rewards/ruler.py` |
| promote | `load_lora_adapter(name, path)` in place, sampling routes by `model=<lora_name>` | SkyRL `vllm_server_actor.py:378-420`, `remote_inference_client.py:1272-1330`, naming `workers/worker.py:1345-1362` | Add `/admin/promote` in the proxy: load winner as `live`, bump cache salt (`_weight_version`, l.236), record receipt hash. Leases so evals are never swapped mid-flight: ART `adapter_leases.py` + `local/adapter_leases.py` (~100 lines) |
| scheduler | ours | Miles `rollout/fully_async_rollout.py:108-200` for the K=1 async pattern (`--max-weight-staleness`) | Miles forbids multi-LoRA with fully-async |

## The sampler crash, diagnosed from code

"RPC call to sample_tokens timed out" is vLLM 0.26's multiproc-executor RPC timeout: a GPU worker hung after the weight sync. On our node the path was Megatron + LoRA with `merge_lora=False`, which is the in-place `load_lora_adapter` branch in `worker_dispatch.py:648-700` and has no pause at all, while up to `SKYRL_GENERATE_CONCURRENCY_PER_ENGINE` (default 512) requests were in flight with ~10k-token prompts. Miles pauses generation (retract, requeue, flush cache, bump version); ART never syncs into running requests (new adapter name per step, or CAS-guarded slot with leases).

Fix to try first: `SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=8`, and call `pause_generation` / `resume_generation` (`remote_inference_client.py:1015-1020`) around `save_weights_for_sampler`. Also check per-datum `all_token_weights` length: `forward` returns lists padded to `max_response_len`, which matches the "fewer logprobs than tokens" issue.

## Topology

Laptop: coding agent only (OpenAI base URL over an SSH tunnel; compaction and thinking off).
GPU node (2xH100): proxy (CPU), sampler serving `model=live`, trainer, Docker sandboxes, nightly cycle. Promote = adapter switch on the sampler; the agent needs no change.
