#!/usr/bin/env bash
# MTP at c8: the depth/acceptance behaviour changes with batch (on Qwen3.5 MTP
# collapsed past c8), so the recommendation needs one c8 point.
set -u
cd "$HOME/qwen38" || exit 1
export MODEL=Qwen/Qwen3.8-27B-FP8
export SERVED=Qwen/Qwen3.8-27B-FP8
PORT=8100
TREE=/home/amatveev/work/wt76_pr1
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
LOG=~/qwen38/logs/qwen38_mtpc8.log
: > "$LOG"

echo "C8 START $(date -u +%H:%M:%S)" >> "$LOG"
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done

name="vllm_q38c8_${PORT}"
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
  --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
  --no-enable-prefix-caching --gpu-memory-utilization 0.90 \
  --max-model-len 32768 --max-num-seqs 8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' >/dev/null
if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
  echo "NOT READY" >> "$LOG"; docker rm -f "$name" >/dev/null 2>&1; exit 1
fi
echo "READY $(date -u +%H:%M:%S)" >> "$LOG"
for r in 1 2; do
  ~/qwen38/scripts/bench_cell_m.sh "q38mtp_c8_r${r}" "$PORT" 8192 1024 8 32 >> "$LOG" 2>&1
done
grep -A2 'Acceptance rate' "$LOG" | tail -3 >> "$LOG"
docker rm -f "$name" >/dev/null 2>&1
echo "C8_DONE $(date -u +%H:%M:%S)" >> "$LOG"
