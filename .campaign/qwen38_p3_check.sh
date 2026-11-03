#!/usr/bin/env bash
# Qwen3.8-27B: does enabling vLLM's fused QK-norm+RoPE+gate Triton kernel on
# ROCm change speed or output?
#
# Two arms, same tree except that one line:
#   base  wt76_pr1                       (main-ish + attention reordering)
#   p3    wt76_pr1 + the ROCm guard widen for fused_qk_rmsnorm_rope_gate
#
# Each arm: greedy generation of a fixed prompt (output equivalence), then the
# 8K/c1 decode cell twice. Same prompt and seed on both arms.
set -u
cd "$HOME/qwen38" || exit 1
export MODEL=Qwen/Qwen3.8-27B-FP8
export SERVED=Qwen/Qwen3.8-27B-FP8
PORT=8100
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
LOG=~/qwen38/logs/qwen38_p3.log
: > "$LOG"

arm() {  # <label> <tree>
  local label=$1 tree=$2
  local name="vllm_q38p3_${label}_${PORT}"
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
    --max-model-len 32768 --max-num-seqs 8 >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 25 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "READY $label $(date -u +%H:%M:%S)" >> "$LOG"

  # Output equivalence: greedy, fixed prompt, no sampling.
  curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$SERVED"'","prompt":"The capital of France is","max_tokens":64,"temperature":0,"seed":1234}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' \
    > "$HOME/qwen38/results/q38_p3_${label}_text.txt" 2>>"$LOG"
  echo "TEXTLEN $label $(wc -c < "$HOME/qwen38/results/q38_p3_${label}_text.txt")" >> "$LOG"

  ~/qwen38/scripts/bench_cell_m.sh "q38p3_${label}_c1"    "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "q38p3_${label}_c1_r2" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  echo "ARM DONE $label $(date -u +%H:%M:%S)" >> "$LOG"
  docker rm -f "$name" >/dev/null 2>&1
}

echo "P3 START $(date -u +%H:%M:%S)" >> "$LOG"
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done

arm base /home/amatveev/work/wt76_pr1
arm p3   /home/amatveev/work/wt76_p3
if diff -q "$HOME/qwen38/results/q38_p3_base_text.txt" \
           "$HOME/qwen38/results/q38_p3_p3_text.txt" >/dev/null 2>&1; then
  echo "OUTPUT IDENTICAL" >> "$LOG"
else
  echo "OUTPUT DIFFERS" >> "$LOG"
  diff "$HOME/qwen38/results/q38_p3_base_text.txt" \
       "$HOME/qwen38/results/q38_p3_p3_text.txt" | head -20 >> "$LOG"
fi
echo "P3_DONE $(date -u +%H:%M:%S)" >> "$LOG"
