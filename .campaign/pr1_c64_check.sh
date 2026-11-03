#!/usr/bin/env bash
# Does the small-batch gate earn its place?
#
# The gate exists because AITER unified was measured slower than ROCM_ATTN at
# c64. The PR now prefers AITER FA, so the same question has to be asked of FA:
# if it holds up at high concurrency too, the gate is unnecessary and the change
# can be unconditional; if it regresses, this is the evidence for the gate.
#
# Both arms run the patched tree at max_num_seqs=64 (above the gate, so the
# default arm selects ROCM_ATTN); the other arm forces ROCM_AITER_FA.
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
TREE=/home/amatveev/work/vllm_e0d
LOG=~/qwen38/logs/pr1_c64.log
: > "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -n 9; then
  echo "GPU lock held elsewhere - aborting" >> "$LOG"; exit 1
fi

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

arm() {  # <label> ; rest: extra server args
  local label=$1; shift
  local name="vllm_pr1c64_${label}_${PORT}"
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
    "${CS[@]}" --max-num-seqs 64 "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 | grep -E 'Using .* backend \(|Trying AITER' | tail -2 \
    | cut -c1-150 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "pr1c64_${label}" "$PORT" 8192 512 64 128 >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

echo "C64 START $(date -u +%H:%M:%S)" >> "$LOG"
arm rocm_attn
arm aiter_fa --attention-backend ROCM_AITER_FA
echo "C64_DONE $(date -u +%H:%M:%S)" >> "$LOG"
