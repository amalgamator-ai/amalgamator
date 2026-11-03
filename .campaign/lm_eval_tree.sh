#!/usr/bin/env bash
# lm_eval against a chosen vLLM tree, in the pinned ROCm 10 image.
#
# The accuracy gate for a PR compares two trees (baseline vs patched) with the
# same task, shots and sampling, so the tree is the only difference. Caches are
# mounted from the host so both arms reuse compiled graphs.
#
# usage: lm_eval_tree.sh <vllm-tree> <output-json> [extra lm_eval args...]
set -euo pipefail
TREE="${1:?usage: lm_eval_tree.sh <vllm-tree> <output-json> [args...]}"
OUT="${2:?usage: lm_eval_tree.sh <vllm-tree> <output-json> [args...]}"
shift 2

source "$HOME/qwen38/images.env"
IMG="$VLLM_IMAGE"
CACHES="$HOME/qwen38/caches"
mkdir -p "$CACHES/inductor" "$CACHES/triton"

exec docker run --rm --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  -e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
  -v "$TREE":/workspace/vllm \
  -v /home/amatveev/work/lmeval:/venv \
  -v "$CACHES":/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v /home/amatveev/qwen38/lmeval_out:/out \
  --entrypoint /venv/bin/lm_eval "$IMG" \
  --model vllm \
  --model_args "pretrained=Qwen/Qwen3.5-35B-A3B-FP8,trust_remote_code=True,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.85,dtype=bfloat16" \
  --tasks gsm8k --num_fewshot 5 --batch_size 64 \
  --output_path "/out/$OUT" \
  "$@"
