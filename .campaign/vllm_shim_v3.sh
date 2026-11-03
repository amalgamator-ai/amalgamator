#!/usr/bin/env bash
# Host shim: run vLLM inside the pinned ROCm 10 image, importing the patched
# source tree at /workspace/vllm (compiled extensions installed in-tree).
#
# Cache mounts: torch.compile and Triton build their caches inside the container
# by default, so every fresh container pays ~4-5 min of inductor compilation and
# Triton JIT before serving. Mounting a host cache dir makes that one-time per
# image/model combination. The AITER JIT cache is left alone: it lives inside
# site-packages next to prebuilt kernels, and overlaying it would shadow them.
#
# VLLM_*/HSA_*/AITER_* variables from the calling shell are forwarded into the
# container, because the runtime gate drives its baseline arm with an env-var
# kill switch written as a `KILL_SWITCH=1 vllm serve ...` prefix -- that prefix
# binds to this shim process, so without forwarding it would never reach the
# server and the baseline would silently equal the patched run.
#
# The amalgamator tree is mounted at its host path so the bench agent can point
# --result-dir at a runtime logs directory and have the JSON appear on the host.
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
  -v /home/amatveev/amalgamator:/home/amatveev/amalgamator \
  -v "$CACHES:/cache" \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  "${FORWARD[@]}" \
  --entrypoint vllm "$IMG" "$@"
