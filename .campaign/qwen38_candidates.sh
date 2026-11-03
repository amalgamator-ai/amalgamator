#!/usr/bin/env bash
# Qwen3.8-27B on one MI300X: which remaining lever actually pays?
#
# Base tree is main-ish (76c6f3f2a9) + the attention backend reordering (PR 1),
# because that is what the model runs on once PR 1 lands; measuring candidates
# against a base that still has the 16 ms/step Triton attention would hide them.
#
# Arms, one server at a time:
#   base        no extra flags
#   mtp1        speculative decode, MTP draft layer, 1 token
#   mtp1_fp8kv  mtp1 + fp8 KV cache
#   fp8kv       fp8 KV cache only
#
# Cell: 8192 in / 1024 out, concurrency 1 (the decode cell PR 1 fixed), plus a
# c8 cell for whatever arm wins, if time allows.
set -u
cd "$HOME/qwen38" || exit 1
export MODEL=Qwen/Qwen3.8-27B-FP8
export SERVED=Qwen/Qwen3.8-27B-FP8
PORT=8100
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
TREE_MAIN=/home/amatveev/work/wt76_pr1
LOG=~/qwen38/logs/qwen38_candidates.log
: > "$LOG"

SPEC1='{"method":"mtp","num_speculative_tokens":1}'

arm() {  # <label> <tree> <extra flags...>
  local label=$1 tree=$2; shift 2
  local name="vllm_q38_${label}_${PORT}"
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
    --max-model-len 32768 --max-num-seqs 8 "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 25 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "READY $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker logs "$name" 2>&1 \
    | grep -iE 'Trying AITER attention|speculative|draft model|kv cache dtype|Using .* backend' \
    | tail -4 | cut -c1-165 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "q38_${label}_c1"    "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "q38_${label}_c1_r2" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  grep -E 'Mean TPOT|Request throughput' "$LOG" | tail -2 >> "$LOG"
  echo "ARM DONE $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker rm -f "$name" >/dev/null 2>&1
}

echo "Q38 START $(date -u +%H:%M:%S)" >> "$LOG"

# Wait for the PR re-measurement (or any other driver) to release the GPU.
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 14400 9; then
  echo "no GPU lock in 4h - aborting" >> "$LOG"; exit 1
fi

for _ in $(seq 1 60); do
  others=$(docker ps --format '{{.Names}}' | wc -l)
  [ "$others" -eq 0 ] && break
  echo "waiting for containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >> "$LOG"
  sleep 30
done
[ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] || {
  echo "GPU still busy - aborting" >> "$LOG"; exit 1; }

arm base       "$TREE_MAIN"
arm mtp1       "$TREE_MAIN" --speculative-config "$SPEC1"
arm mtp1_fp8kv "$TREE_MAIN" --speculative-config "$SPEC1" --kv-cache-dtype fp8
arm fp8kv      "$TREE_MAIN" --kv-cache-dtype fp8
echo "Q38_DONE $(date -u +%H:%M:%S)" >> "$LOG"
