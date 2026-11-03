#!/usr/bin/env bash
# Smoke check on the rebased branches: does each one boot and select the
# behaviour it claims, on top of the new main?
#
# The image's compiled extensions are from the older build, so this is not a
# measurement of record; it is a check that the rebased Python still starts an
# engine and reaches the code path the PR changes.
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
LOG=~/qwen38/logs/rebased_boot.log
: > "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
flock -w 1800 9 || { echo "no GPU lock" >> "$LOG"; exit 1; }

boot() {  # <label> <tree> <mns>
  local label=$1 tree=$2 mns=$3
  local name="vllm_rb_${label}_${PORT}"
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
    --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 \
    --max-model-len 16384 --max-num-seqs "$mns" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 1800 "$name" >> "$LOG" 2>&1; then
    echo "BOOT FAILED $label" >> "$LOG"
    docker logs --tail 25 "$name" 2>&1 | tail -20 >> "$LOG"
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "BOOT OK $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker logs "$name" 2>&1 \
    | grep -E 'Trying AITER attention|fused shared experts are ENABLED|Using .* backend \(' \
    | tail -2 | cut -c1-165 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "rb_${label}" "$PORT" 2048 128 1 4 >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

echo "RB START $(date -u +%H:%M:%S)" >> "$LOG"
boot pr1 /home/amatveev/work/vllm_e0d 1
boot pr2 /home/amatveev/work/vllm_pr_fse 1
echo "RB_DONE $(date -u +%H:%M:%S)" >> "$LOG"
