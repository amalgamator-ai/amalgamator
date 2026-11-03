#!/usr/bin/env bash
# SGLang reference for the same cell, so the Qwen3.8 numbers are compared
# against a same-week measurement rather than the 2026-10-31 one.
set -u
cd "$HOME/qwen38" || exit 1
IMG="lmsysorg/sglang-rocm@sha256:6aeadaeeb357dd0d36730f3200784c4aece97dd744e99b49d40571d4c796c9dd"
MODEL=Qwen/Qwen3.8-27B-FP8
PORT=8100
LOG=~/qwen38/logs/qwen38_sgl.log
: > "$LOG"

echo "SGL START $(date -u +%H:%M:%S)" >> "$LOG"
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done

name="sgl_q38_${PORT}"
docker rm -f "$name" >/dev/null 2>&1
sleep 5
docker run -d --name "$name" --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  -e SGLANG_USE_AITER=1 -e HF_HOME=/hf \
  -v /home/amatveev/qwen38/hf:/hf \
  "$IMG" \
  --model-path "$MODEL" --served-model-name "$MODEL" \
  --port "$PORT" --tp 1 --trust-remote-code \
  --attention-backend aiter --kv-cache-dtype fp8_e4m3 --page-size 16 \
  --mem-fraction-static 0.85 --chunked-prefill-size 16384 \
  --disable-radix-cache >/dev/null
if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
  echo "SGL NOT READY" >> "$LOG"; docker logs --tail 25 "$name" >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1; echo "SGL_DONE $(date -u +%H:%M:%S)" >> "$LOG"; exit 1
fi
echo "SGL READY $(date -u +%H:%M:%S)" >> "$LOG"
for r in 1 2; do
  ~/qwen38/scripts/bench_cell_m.sh "q38_sgl_r${r}" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
done
docker rm -f "$name" >/dev/null 2>&1
echo "SGL_DONE $(date -u +%H:%M:%S)" >> "$LOG"
