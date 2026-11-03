#!/usr/bin/env bash
# What is left after P1 (attention selector) and P2 (fused shared experts)?
#
# The residual plan's second-ranked item is "match SGLang's FULL-attention
# backend and KV dtype" (ROCM_AITER_FA + fp8 KV, 8.33% of the block). Both are
# reachable from the command line, so this probes them on the P1+P2 tree before
# any code is written:
#
#   A  AITER unified + bf16 KV     (today's behaviour after P1+P2)
#   B  AITER unified + fp8 KV      (KV dtype only)
#   C  ROCM_AITER_FA + bf16 KV     (backend only — does it even validate for a
#                                   hybrid model whose attention block is 1056?)
#   D  ROCM_AITER_FA + fp8 KV      (the configuration SGLang effectively runs)
#
# An arm that cannot start is a result too: it tells us the remaining gap is
# blocked on the hybrid KV page-size coupling rather than on a missing knob.
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
TREE=/home/amatveev/work/vllm_p1_fse   # e0d00e4b8 + P1 + P2
LOG=~/qwen38/logs/pr3_probe.log
: > "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -n 9; then
  echo "another GPU driver holds the lock - refusing to run" >> "$LOG"; exit 1
fi

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

arm() {  # <label> ; rest: extra server args
  local label=$1; shift
  local name="vllm_pr3_${label}_${PORT}"
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
    "${CS[@]}" --max-num-seqs 8 "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 900 "$name" >> "$LOG" 2>&1; then
    echo "ARM $label DID NOT START" >> "$LOG"
    docker logs --tail 25 "$name" 2>&1 | grep -iE 'error|not support|assert|raise|backend' \
      | tail -8 | cut -c1-200 >> "$LOG"
    docker rm -f "$name" >/dev/null 2>&1
    return 1
  fi
  docker logs "$name" 2>&1 \
    | grep -E 'Using .* backend out of potential|is ENABLED|attention block size' \
    | tail -3 | cut -c1-150 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "pr3_${label}_c1" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "pr3_${label}_c8" "$PORT" 8192 512 8 32 >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

echo "PR3 PROBE START $(date -u +%H:%M:%S)" >> "$LOG"
arm unified_bf16
arm unified_fp8kv  --kv-cache-dtype fp8_e4m3
arm aiterfa_bf16   --attention-backend ROCM_AITER_FA
arm aiterfa_fp8kv  --attention-backend ROCM_AITER_FA --kv-cache-dtype fp8_e4m3
echo "PR3_PROBE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
