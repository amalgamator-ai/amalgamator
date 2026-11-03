#!/usr/bin/env bash
# Verify the shipped condition at both ends: head_size 256 with a large
# max_num_seqs (where the previous batch-gated version would have kept
# ROCM_ATTN), and a small one. The frontend should now select AITER FA in both.
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
TREE=/home/amatveev/work/vllm_e0d
LOG=~/qwen38/logs/pr1_final_check.log
: > "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 7200 9; then
  echo "no GPU lock in 2h" >> "$LOG"; exit 1
fi

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

arm() {  # <label> <max_num_seqs>
  local label=$1 mns=$2
  local name="vllm_pr1fin_${label}_${PORT}"
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$name" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    -e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1 \
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf -v "$TREE":/workspace/vllm \
    "$IMG" \
    --model "$MODEL" --served-model-name "$SERVED" \
    "${CS[@]}" --max-num-seqs "$mns" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 | grep -E 'Trying AITER attention|Using .* backend' | tail -2 \
    | cut -c1-160 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "pr1fin_${label}" "$PORT" 8192 $3 "$mns" "$4" >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

echo "PR1FIN START $(date -u +%H:%M:%S)" >> "$LOG"
arm mns64 64 512 128
arm mns1 1 1024 8
echo "PR1FIN_DONE $(date -u +%H:%M:%S)" >> "$LOG"
