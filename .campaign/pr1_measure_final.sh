#!/usr/bin/env bash
# Final numbers for PR 1 (prefer AITER backends over ROCM_ATTN at small batch).
#
# Four arms, each a separate tree, same pinned image, same flags:
#   1  stock e0d00e4b8                  Qwen3.5-35B   -> ROCM_ATTN
#   2  stock + PR1                       Qwen3.5-35B   -> AITER FA
#   3  stock e0d00e4b8                  Qwen3.8-27B   -> ROCM_ATTN
#   4  stock + PR1                       Qwen3.8-27B   -> AITER FA
#
# max_num_seqs is 8 on both sides so the gate is the only difference.
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
PORT=8100
IMG="$VLLM_IMAGE"
LOG=~/qwen38/logs/pr1_measure.log
: > "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -n 9; then
  echo "another GPU driver holds the lock - refusing to run" >> "$LOG"; exit 1
fi

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

arm() {  # <label> <model> <tree> <c1|c1c8>
  local label=$1 model=$2 tree=$3 cells=$4
  local name="vllm_pr1f_${label}_${PORT}"
  export MODEL="$model" SERVED="$model"
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$name" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    -e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1 \
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf -v "$tree":/workspace/vllm \
    "$IMG" \
    --model "$model" --served-model-name "$SERVED" \
    "${CS[@]}" --max-num-seqs 8 >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 25 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 \
    | grep -E 'Trying AITER attention|Using .* backend \(|is ENABLED' \
    | tail -2 | cut -c1-160 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "pr1f_${label}_c1" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "pr1f_${label}_c1_r2" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  if [ "$cells" = "c1c8" ]; then
    ~/qwen38/scripts/bench_cell_m.sh "pr1f_${label}_c8" "$PORT" 8192 512 8 32 >> "$LOG" 2>&1
  fi
  docker rm -f "$name" >/dev/null 2>&1
}

Q35=Qwen/Qwen3.5-35B-A3B-FP8
Q38=Qwen/Qwen3.8-27B-FP8
STOCK=/home/amatveev/work/vllm_e0d_stock
PATCHED=/home/amatveev/work/vllm_e0d

echo "PR1F START $(date -u +%H:%M:%S)" >> "$LOG"
arm q35_stock   "$Q35" "$STOCK"   c1c8
arm q35_patched "$Q35" "$PATCHED" c1c8
arm q38_stock   "$Q38" "$STOCK"   c1
arm q38_patched "$Q38" "$PATCHED" c1
echo "PR1F_DONE $(date -u +%H:%M:%S)" >> "$LOG"
