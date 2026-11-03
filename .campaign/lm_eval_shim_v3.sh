#!/usr/bin/env bash
# Host shim: run lm_eval inside the pinned ROCm 10 image, with the patched
# source tree on PYTHONPATH and the prebuilt lm_eval venv.
#
# Caches: inductor/Triton caches are mounted from the host so repeated evals do
# not recompile the same graphs in every container.
#
# VLLM_*/HSA_*/AITER_* variables are forwarded so the equivalence gate can run
# the same kill-switch baseline the benchmark uses.
set -e
IMG=vllm/vllm-openai-rocm:nightly-rocm100
CACHES="$HOME/qwen38/caches"
mkdir -p "$CACHES/inductor" "$CACHES/triton"

FORWARD=()
while IFS= read -r kv; do
  FORWARD+=(-e "$kv")
done < <(env | grep -E "^((VLLM|HSA|AITER|TORCH|SGLANG|NCCL|HIP)_|AMDGCN_|AMD_LOG_LEVEL=)" || true)

exec docker run --rm --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  -e PYTHONPATH=/workspace/vllm \
  -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
  -v /home/amatveev/work/vllm:/workspace/vllm \
  -v /home/amatveev/work/lmeval:/venv \
  -v /home/amatveev/amalgamator:/home/amatveev/amalgamator \
  -v "$CACHES:/cache" \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  "${FORWARD[@]}" \
  --entrypoint /venv/bin/lm_eval "$IMG" "$@"
