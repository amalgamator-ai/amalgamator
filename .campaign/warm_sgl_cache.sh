#!/bin/bash
# Warm one SGLang boot, then extract its AITER JIT cache to the host so future
# containers mount a populated directory (JIT recompilation costs ~14 min/boot).
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=q355cache
PORT=31001
NAME=sglang_q355cache_${PORT}
CACHE=/home/amatveev/qwen38/caches/aiter-sglang
LOG=~/qwen38/logs/sgl_cache_warm.log

echo "WARM START $(date -u +%H:%M:%S)" > "$LOG"
~/qwen38/scripts/serve_sglang_m.sh off "$PORT" >> "$LOG" 2>&1 || { echo "SERVE FAIL" >> "$LOG"; exit 1; }
if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
  echo "NOT READY" >> "$LOG"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
echo "READY $(date -u +%H:%M:%S)" >> "$LOG"
docker exec "$NAME" bash -c "du -sh /sgl-workspace/aiter/aiter/jit/build" >> "$LOG" 2>&1
mkdir -p "$CACHE"
docker cp "$NAME:/sgl-workspace/aiter/aiter/jit/build/." "$CACHE/" >> "$LOG" 2>&1
du -sh "$CACHE" >> "$LOG" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "CACHE_EXTRACTED $(date -u +%H:%M:%S)" >> "$LOG"
