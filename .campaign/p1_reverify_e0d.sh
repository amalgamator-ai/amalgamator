#!/usr/bin/env bash
# Re-verify P1 on the current main (e0d00e4b8) rather than on 2b009fc6d, which
# is where the original numbers were taken. Same pinned image, same cell, same
# bench; the only difference is the tree.
#   1. stock e0d00e4b8            -> ROCM_ATTN (Triton paged attention)
#   2. e0d00e4b8 + P1             -> ROCM_AITER_UNIFIED_ATTN at max_num_seqs<=8
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
LOG=~/qwen38/logs/p1_reverify_e0d.log
: > "$LOG"

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

arm() {  # <label> <tree> <max_num_seqs>
  local label=$1 tree=$2 mns=$3
  local name="vllm_p1v_${label}_${PORT}"
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$name" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    -e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1 \
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf -v "$tree":/workspace/vllm \
    "$IMG" \
    --model "$MODEL" --served-model-name "$SERVED" \
    "${CS[@]}" --max-num-seqs "$mns" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 30 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 | grep -E 'is ENABLED|Using .* backend out of potential' | tail -2 | cut -c1-150 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "p1v_${label}" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "p1v_${label}_r2" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

echo "P1V START $(date -u +%H:%M:%S)" >> "$LOG"
arm stock /home/amatveev/work/vllm_e0d_stock 8
arm p1    /home/amatveev/work/vllm_e0d       8
echo "P1V_DONE $(date -u +%H:%M:%S)" >> "$LOG"
