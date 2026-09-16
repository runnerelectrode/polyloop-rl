#!/bin/bash
# Bring a fresh 2xH100 node (Ubuntu 22.04, CUDA 12.8, Docker) to a state where the pydantic-v2 loop
# can run. Idempotent. Run as the node user; logs to ~/logs.
#
#   scp scripts/node_up.sh node:~/ && ssh node 'bash ~/node_up.sh'
#
# Afterwards, in tmux on the node:
#   ~/serve.sh                      # SkyRL Tinker server (trainer GPU0 + sampler GPU1)
#   polyloop proxy --loop ...       # OpenAI-compatible endpoint on :8787 for the laptop agent
#   polyloop ui --loop ... --log ~/logs/cycle.log
#   polyloop run --loop ...
set -euo pipefail
exec > >(tee -a ~/logs/node_up.log) 2>&1 || true
mkdir -p ~/logs ~/work ~/polyloop-tasks
echo "== node_up $(date -u)"
sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs jq htop tmux build-essential ffmpeg >/dev/null
python3 -m pip install -q --user uv huggingface_hub
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
sudo usermod -aG docker "$USER" || true

cd ~/work
[ -d SkyRL ] || { git clone -q https://github.com/NovaSky-AI/SkyRL && (cd SkyRL && git checkout -q 9719b4f74ae9cbb6ec022a8d67c1e8a835b52c7d); }
[ -d tinker-cookbook ] || { git clone -q https://github.com/thinking-machines-lab/tinker-cookbook && (cd tinker-cookbook && git checkout -q f46eddd); }
[ -d rlcli ] || git clone -q https://github.com/polygramme/rlcli
[ -d polyloop-rl ] || git clone -q https://github.com/runnerelectrode/polyloop-rl
(cd polyloop-rl && git pull -q)

uv venv -q ~/venvs/rlcli --python 3.12 || true
uv pip install -q --python ~/venvs/rlcli/bin/python -e "./rlcli[train]" "tinker-cookbook @ file://$HOME/work/tinker-cookbook" harbor datasets -e ./polyloop-rl
echo "== venv ok: $(~/venvs/rlcli/bin/python -c 'import polyloop, rlcli, tinker_cookbook; print("polyloop", polyloop.__version__)')"

# SkyRL env for the server: sync once (megatron extra), drop torchcodec (cu13 wheel breaks vLLM import on cu128)
cd ~/work/SkyRL && rm -f uv.lock && RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 MAX_JOBS=16 uv sync -q --extra tinker --extra megatron
uv pip uninstall -q --python ~/work/SkyRL/.venv/bin/python torchcodec 2>/dev/null || true
uv pip install -q --python ~/work/SkyRL/.venv/bin/python peft
echo "== skyrl env ok"

# Server launcher (Qwen3.5-9B Megatron LoRA, trainer + sampler split, 28k-episode memory settings, sampler concurrency capped)
cat > ~/serve.sh <<'EOF'
#!/bin/bash
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export RLCLI_SKYRL_SOURCE=$HOME/work/SkyRL RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RLCLI_SKIP_SYNC=1
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=2400
export SKYRL_GENERATE_CONCURRENCY_PER_ENGINE=${SKYRL_GENERATE_CONCURRENCY_PER_ENGINE:-8}
exec ~/venvs/rlcli/bin/rlcli serve start --base-model Qwen/Qwen3.5-9B --backend megatron --gpus 1 --tp 1 --max-model-len 32768 \
  --backend-config '{"trainer.placement.colocate_all": false, "trainer.policy.megatron_config.lora_config.merge_lora": false, "trainer.policy.model.lora.max_loras": 4, "trainer.policy.model.lora.max_cpu_loras": 8, "trainer.policy.language_model_only": true, "trainer.logprobs_chunk_size": 1024, "trainer.fused_lm_head_logprob": true, "trainer.micro_train_batch_size_per_gpu": 1, "trainer.micro_forward_batch_size_per_gpu": 1, "trainer.max_tokens_per_microbatch": 16384, "generator.inference_engine.enforce_eager": false, "generator.inference_engine.engine_init_kwargs": {"enforce_eager": false, "max_model_len": 32768, "enable_prefix_caching": true}}' \
  --wait 3600
EOF
chmod +x ~/serve.sh

# Task images: the pydantic base image (tasks are FROM it)
if [ -f ~/polyloop-tasks/pydantic-v2/Dockerfile.base ]; then
  (cd ~/polyloop-tasks/pydantic-v2 && docker build -q -t polyloop-pydantic:latest -f Dockerfile.base . && echo "== base image ok")
else
  echo "== NOTE: rsync the task pool to ~/polyloop-tasks/pydantic-v2 (Dockerfile.base, train/, holdout/) then build the base image"
fi
echo "== node_up done $(date -u)"
