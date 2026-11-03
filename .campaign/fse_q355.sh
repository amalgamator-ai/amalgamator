#!/usr/bin/env bash
# Does the fused-shared-expert path (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS)
# actually help this model, on the P1 baseline?
#
# The cross-framework plan ranks it second (1,218 us/step = 7.5% of the stock
# block). Before spending hours of codegen on it, measure whether the existing
# switch produces that gain here — and whether the FP8 MoE even passes the
# compatibility check. Two arms, same pinned image, same mounted tree, same
# cell; the only difference is the env var.
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
TREE=/home/amatveev/work/vllm_e0d
LOG=~/qwen38/logs/fse_q355.log
: > "$LOG"

CE=(-e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf -v "$TREE":/workspace/vllm)
CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

echo "FSE START $(date -u +%H:%M:%S)" >> "$LOG"

arm() {  # <label> <fse-value>
  local label=$1 fse=$2
  local name="vllm_fse_${label}_${PORT}"
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$name" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    "${CE[@]}" -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS="$fse" \
    "$IMG" \
    --model "$MODEL" --served-model-name "$SERVED" \
    "${CS[@]}" --max-num-seqs 1 >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 30 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 | grep -iE 'fused shared expert|FSE|is ENABLED' | tail -3 | cut -c1-170 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "fse_${label}" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

arm off 0
arm on  1

echo "FSE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
