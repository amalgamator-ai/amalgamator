#!/usr/bin/env bash
# Is the greedy divergence between the base and P3 arms caused by the patch, or
# by run-to-run nondeterminism? Boot the same base tree again and ask it the
# same question twice, once in-server and once across servers.
set -u
cd "$HOME/qwen38" || exit 1
export MODEL=Qwen/Qwen3.8-27B-FP8
export SERVED=Qwen/Qwen3.8-27B-FP8
PORT=8100
TREE=/home/amatveev/work/wt76_pr1
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
LOG=~/qwen38/logs/qwen38_det.log
: > "$LOG"

echo "DET START $(date -u +%H:%M:%S)" >> "$LOG"
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done

name="vllm_q38det_${PORT}"
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
  --max-model-len 32768 --max-num-seqs 8 >/dev/null
if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
  echo "NOT READY" >> "$LOG"; docker rm -f "$name" >/dev/null 2>&1; exit 1
fi
echo "READY $(date -u +%H:%M:%S)" >> "$LOG"

req() {  # <outfile>
  curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$SERVED"'","prompt":"The capital of France is","max_tokens":64,"temperature":0,"seed":1234}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' > "$1"
}
req ~/qwen38/results/det_base_a.txt
req ~/qwen38/results/det_base_b.txt
if diff -q ~/qwen38/results/det_base_a.txt ~/qwen38/results/det_base_b.txt >/dev/null; then
  echo "IN-SERVER IDENTICAL" >> "$LOG"
else
  echo "IN-SERVER DIFFERS" >> "$LOG"
fi
if diff -q ~/qwen38/results/det_base_a.txt ~/qwen38/results/q38_p3_base_text.txt >/dev/null; then
  echo "ACROSS BOOTS IDENTICAL" >> "$LOG"
else
  echo "ACROSS BOOTS DIFFERS" >> "$LOG"
fi
docker rm -f "$name" >/dev/null 2>&1
echo "DET_DONE $(date -u +%H:%M:%S)" >> "$LOG"
