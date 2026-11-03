#!/bin/bash
# Validate the mounted SGLang AITER JIT cache: boot must be much faster than the
# ~14 min cold build, and the server must actually serve.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=q355cachechk
PORT=31002
NAME=sglang_q355cachechk_${PORT}
LOG=~/qwen38/logs/sgl_cache_validate.log

START=$(date +%s)
echo "BOOT START $(date -u +%H:%M:%S)" > "$LOG"
~/qwen38/scripts/serve_sglang_m.sh off "$PORT" >> "$LOG" 2>&1 || { echo "SERVE FAIL" >> "$LOG"; exit 1; }
if ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
  END=$(date +%s)
  echo "READY in $((END-START)) s $(date -u +%H:%M:%S)" >> "$LOG"
  curl -s --max-time 30 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-35B-A3B-FP8","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}' \
    >> "$LOG" 2>&1
else
  echo "NOT READY after $(( $(date +%s) - START )) s" >> "$LOG"
  docker logs --tail 30 "$NAME" >> "$LOG" 2>&1
fi
docker rm -f "$NAME" >/dev/null 2>&1
echo "VALIDATE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
