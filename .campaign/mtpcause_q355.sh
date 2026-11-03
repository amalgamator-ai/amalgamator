#!/usr/bin/env bash
# Discriminate the two candidate causes of the MTP collapse at high concurrency:
#   D  eager execution (no CUDA graphs at all)
#      -> if this is FAST, the cost is graph padding/replay, not the model work
#   E  server max_num_seqs 16 with client concurrency 64
#      -> if this is FAST, the cost tracks the captured graph window, not the
#         number of in-flight requests
# Waits for the shape sweep to finish so the two never share the GPU.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
NAME=vllm_mtpcause_${PORT}
IMG=vllm/vllm-openai-rocm:nightly-rocm100
LOG=~/qwen38/logs/mtpcause_q355.log
: > "$LOG"

CE=(-e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf
    -v /home/amatveev/work/vllm:/workspace/vllm)
CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

echo "MTPCAUSE WAIT $(date -u +%H:%M:%S)" >> "$LOG"
for _ in $(seq 1 240); do
  grep -qE '^MTPSHAPE_DONE' ~/qwen38/logs/mtpshape_q355.log && break
  sleep 60
done
grep -qE '^MTPSHAPE_DONE' ~/qwen38/logs/mtpshape_q355.log || {
  echo "MTPSHAPE never finished - aborting" >> "$LOG"; exit 1; }

start() {  # <label> <mns> [extra server args...]
  local label=$1 mns=$2; shift 2
  docker rm -f "$NAME" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$NAME" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined "${CE[@]}" "$IMG" \
    --model "$MODEL" --served-model-name "$SERVED" \
    "${CS[@]}" --max-num-seqs "$mns" "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"
    docker logs --tail 30 "$NAME" >> "$LOG" 2>&1
    docker rm -f "$NAME" >/dev/null 2>&1
    return 1
  fi
  docker logs "$NAME" 2>&1 \
    | grep -oE "cudagraph_mode': [^,]*|max_cudagraph_capture_size': [0-9]+" \
    | head -2 >> "$LOG"
  docker logs "$NAME" 2>&1 | grep -c "Capturing CUDA graphs (FULL)" >> "$LOG"
}

cell() { ~/qwen38/scripts/bench_cell_m.sh "$@" >> "$LOG" 2>&1; }

M3='{"method":"mtp","num_speculative_tokens":3}'

echo "MTPCAUSE START $(date -u +%H:%M:%S)" >> "$LOG"

start d 64 --speculative-config "$M3" --enforce-eager && \
  cell mtpcause_c64_n3_eager 8100 8192 512 64 128

start e 16 --speculative-config "$M3" && \
  cell mtpcause_c64_n3_mns16 8100 8192 512 64 128

docker rm -f "$NAME" >/dev/null 2>&1
echo "MTPCAUSE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
