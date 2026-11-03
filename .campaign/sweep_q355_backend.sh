#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8: does the Qwen3.8 P1 finding reproduce?
# vLLM default backend (ROCM_ATTN) vs explicit ROCM_AITER_UNIFIED_ATTN, no spec.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=q355sweep
S=/home/amatveev/qwen38/scripts
PORT=8100
NAME=vllm_q355sweep_${PORT}

arm() {
  local label=$1; shift
  docker ps --format "{{.Names}}" | grep -E "^vllm_q355sweep_" | xargs -r docker rm -f >/dev/null 2>&1
  sleep 5
  $S/serve_vllm_m.sh off "$PORT" "$@" || { echo "SERVE FAIL $label"; return 1; }
  if ! $S/wait_ready.sh "$PORT" 2400 "$NAME"; then
    echo "NOT READY $label"; docker logs --tail 50 "$NAME" 2>&1 | tail -25
    docker rm -f "$NAME" >/dev/null 2>&1; return 1
  fi
  docker logs "$NAME" 2>&1 | grep -E "Using [A-Z_]+ backend out of potential|is ENABLED" | tail -2
  $S/bench_cell_m.sh "${label}_isl8192_osl1024_c1"  "$PORT" 8192 1024 1 8   > ~/qwen38/logs/${label}_c1.log  2>&1
  $S/bench_cell_m.sh "${label}_isl8192_osl512_c8"   "$PORT" 8192 512  8 32  > ~/qwen38/logs/${label}_c8.log  2>&1
  $S/bench_cell_m.sh "${label}_isl8192_osl512_c64"  "$PORT" 8192 512  64 128 > ~/qwen38/logs/${label}_c64.log 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1
  sleep 5
  echo "ARM DONE $label $(date -u +%H:%M:%S)"
}

arm q355sweep_def
arm q355sweep_aiter --attention-backend ROCM_AITER_UNIFIED_ATTN
echo "SWEEP_Q355_DONE $(date -u +%H:%M:%S)"
