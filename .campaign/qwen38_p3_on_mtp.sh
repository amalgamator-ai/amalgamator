#!/usr/bin/env bash
# Qwen3.8-27B with MTP, with and without the ROCm fused QK-norm kernel.
# MTP is the configuration the report recommends, so the code change has to be
# measured on top of it, not only against stock decode.
set -u
cd "$HOME/qwen38" || exit 1
export MODEL=Qwen/Qwen3.8-27B-FP8
export SERVED=Qwen/Qwen3.8-27B-FP8
PORT=8100
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
SPEC1='{"method":"mtp","num_speculative_tokens":1}'
LOG=~/qwen38/logs/qwen38_p3mtp.log
: > "$LOG"

arm() {  # <label> <tree>
  local label=$1 tree=$2
  local name="vllm_q38pm_${label}_${PORT}"
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
    --max-model-len 32768 --max-num-seqs 8 \
    --speculative-config "$SPEC1" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 25 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "READY $label $(date -u +%H:%M:%S)" >> "$LOG"
  for r in 1 2 3; do
    ~/qwen38/scripts/bench_cell_m.sh "q38pm_${label}_c1_r${r}" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  done
  echo "ARM DONE $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker rm -f "$name" >/dev/null 2>&1
}

echo "P3MTP START $(date -u +%H:%M:%S)" >> "$LOG"
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done

arm mtp    /home/amatveev/work/wt76_pr1
arm mtp_p3 /home/amatveev/work/wt76_p3
echo "P3MTP_DONE $(date -u +%H:%M:%S)" >> "$LOG"
