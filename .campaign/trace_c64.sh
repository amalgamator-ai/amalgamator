#!/usr/bin/env bash
# Capture c64 torch traces (no-spec and MTP) for the MTP collapse analysis.
#
# The previous attempt force-removed the container 30 s after the benchmark and
# lost both traces: vLLM exports the Kineto trace from TorchProfilerWrapper.stop(),
# which runs during engine shutdown, so `docker rm -f` (SIGKILL) discards it.
# Here the server is stopped with SIGTERM and given time to flush, and the file
# is checked before the arm is declared done.
#
# The no-spec container is reused if it is already up on the port.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG=vllm/vllm-openai-rocm:nightly-rocm100
LOG=~/qwen38/logs/trace_c64.log
ROOT=~/qwen38/traces
: > "$LOG"
mkdir -p "$ROOT/nospec" "$ROOT/mtp"

PROF='{"profiler": "torch", "torch_profiler_dir": "/tmp/vllm_traces", "delay_iterations": 60, "max_iterations": 20, "ignore_frontend": true}'
M3='{"method":"mtp","num_speculative_tokens":3}'

boot() {  # <tag> [extra serve args...]
  local tag=$1; shift
  local name="vllm_trace_${tag}_${PORT}"
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo "reusing running $name (waiting for health)" >> "$LOG"
    ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1 || return 1
    echo "READY $tag $(date -u +%H:%M:%S) (reused)" >> "$LOG"
    return 0
  fi
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$name" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    -e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1 \
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
    -v /home/amatveev/work/vllm:/workspace/vllm \
    -v "$ROOT/$tag":/tmp/vllm_traces \
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
}

finish() {  # <tag> <container-name>
  local tag=$1 name=$2
  ~/qwen38/scripts/bench_cell_m.sh "trace_${tag}_c64" "$PORT" 8192 128 64 128 >> "$LOG" 2>&1
  echo "BENCH DONE $tag $(date -u +%H:%M:%S) - stopping server to flush the trace" >> "$LOG"
  # SIGTERM, long grace: the export runs during engine shutdown.
  docker stop -t 300 "$name" >> "$LOG" 2>&1
  for _ in $(seq 1 60); do
    ls "$ROOT/$tag"/*.json >/dev/null 2>&1 && break
    sleep 5
  done
  docker rm -f "$name" >/dev/null 2>&1
  if ls "$ROOT/$tag"/*.json >/dev/null 2>&1; then
    echo "TRACE OK $tag: $(ls -la "$ROOT/$tag"/*.json | tr '\n' ' ')" >> "$LOG"
  else
    echo "TRACE MISSING for $tag" >> "$LOG"
  fi
  ls -la "$ROOT/$tag" >> "$LOG" 2>&1
}

echo "TRACE START $(date -u +%H:%M:%S)" >> "$LOG"

have_trace() { ls "$ROOT/$1"/*.json >/dev/null 2>&1; }

# arm 1: no spec decode
if have_trace nospec; then
  echo "nospec trace already present - skipping arm" >> "$LOG"
else
  if docker ps --format '{{.Names}}' | grep -qx "vllm_mtpprof_nospec_${PORT}"; then
    docker rename "vllm_mtpprof_nospec_${PORT}" "vllm_trace_nospec_${PORT}" >> "$LOG" 2>&1
  fi
  boot nospec && finish nospec "vllm_trace_nospec_${PORT}"
fi

# arm 2: MTP
if have_trace mtp; then
  echo "mtp trace already present - skipping arm" >> "$LOG"
else
  boot mtp --speculative-config "$M3" && finish mtp "vllm_trace_mtp_${PORT}"
fi

echo "TRACE_C64_DONE $(date -u +%H:%M:%S)" >> "$LOG"
# analyze_mtp.sh waits on this marker.
echo "MTPPROF_DONE $(date -u +%H:%M:%S)" >> ~/qwen38/logs/mtpprof_q355.log
