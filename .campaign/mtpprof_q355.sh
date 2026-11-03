#!/usr/bin/env bash
# Capture torch traces of the c64 cell in both arms (MTP and no spec decode) so
# the MTP collapse can be attributed to kernels instead of guessed at from
# end-to-end latency. The two traces are the input pair for a single-trace +
# cross-commit analysis that produces a plan for the fix.
# Waits for the cause run to release the GPU first.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG=vllm/vllm-openai-rocm:nightly-rocm100
LOG=~/qwen38/logs/mtpprof_q355.log
ROOT=~/qwen38/traces
: > "$LOG"
mkdir -p "$ROOT"

PROF='{"profiler": "torch", "torch_profiler_dir": "/tmp/vllm_traces", "delay_iterations": 120, "max_iterations": 20, "ignore_frontend": true}'
M3='{"method":"mtp","num_speculative_tokens":3}'

arm() {  # <tag> [extra server args...]
  local tag=$1; shift
  local name="vllm_mtpprof_${tag}_${PORT}"
  local out="$ROOT/$tag"
  mkdir -p "$out"; rm -f "$out"/*.json 2>/dev/null
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$name" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    -e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1 \
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
    -v /home/amatveev/work/vllm:/workspace/vllm \
    -v "$out":/tmp/vllm_traces \
    "$IMG" \
    --model "$MODEL" --served-model-name "$SERVED" \
    --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536 \
    --max-num-seqs 64 --profiler-config "$PROF" "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $tag" >> "$LOG"; docker logs --tail 30 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "READY $tag $(date -u +%H:%M:%S)" >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "mtpprof_${tag}_c64" 8100 8192 128 64 128 >> "$LOG" 2>&1
  sleep 30
  docker rm -f "$name" >/dev/null 2>&1
  ls -la "$out" >> "$LOG" 2>&1
}

echo "MTPPROF WAIT $(date -u +%H:%M:%S)" >> "$LOG"
for _ in $(seq 1 300); do
  grep -qE '^MTPCAUSE_DONE' ~/qwen38/logs/mtpcause_q355.log && break
  sleep 60
done
grep -qE '^MTPCAUSE_DONE' ~/qwen38/logs/mtpcause_q355.log || {
  echo "MTPCAUSE never finished - aborting" >> "$LOG"; exit 1; }

echo "MTPPROF START $(date -u +%H:%M:%S)" >> "$LOG"
arm mtp    --speculative-config "$M3"
arm nospec
echo "MTPPROF_DONE $(date -u +%H:%M:%S)" >> "$LOG"
