#!/usr/bin/env bash
# Two cells the PR bodies still quote from the older base:
#   - Qwen3.5-35B at c64 (the "high concurrency is not worse" cell of PR 1)
#   - Qwen3.8-27B at c1, stock arm (the pre-PR-1 number; the patched arm comes
#     from the candidate sweep on the same tree)
set -u
cd "$HOME/qwen38" || exit 1
PORT=8100
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
LOG=~/qwen38/logs/qwen38_extra.log
: > "$LOG"

run() {  # <label> <tree> <model> <max-num-seqs> <extra serve flags...>
  local label=$1 tree=$2 model=$3 mns=$4; shift 4
  local name="vllm_x_${label}_${PORT}"
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
    --model "$MODEL" --served-model-name "$SERVED" \
    --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 \
    --max-model-len 65536 --max-num-seqs "$mns" "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 25 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "READY $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker logs "$name" 2>&1 | grep -E 'Trying AITER attention' | tail -1 | cut -c1-140 >> "$LOG"
  case "$label" in
    *c64*)
      ~/qwen38/scripts/bench_cell_m.sh "x_${label}"    "$PORT" 8192 512 64 128 >> "$LOG" 2>&1
      ~/qwen38/scripts/bench_cell_m.sh "x_${label}_r2" "$PORT" 8192 512 64 128 >> "$LOG" 2>&1
      ;;
    *)
      ~/qwen38/scripts/bench_cell_m.sh "x_${label}"    "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
      ~/qwen38/scripts/bench_cell_m.sh "x_${label}_r2" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
      ;;
  esac
  echo "ARM DONE $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker rm -f "$name" >/dev/null 2>&1
}

echo "EXTRA START $(date -u +%H:%M:%S)" >> "$LOG"
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done

run q355_c64_stock /home/amatveev/work/wt76_stock Qwen/Qwen3.5-35B-A3B-FP8 64
run q355_c64_pr1   /home/amatveev/work/wt76_pr1   Qwen/Qwen3.5-35B-A3B-FP8 64
run q38_c1_stock   /home/amatveev/work/wt76_stock Qwen/Qwen3.8-27B-FP8     8
echo "EXTRA_DONE $(date -u +%H:%M:%S)" >> "$LOG"
