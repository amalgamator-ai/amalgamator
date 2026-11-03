#!/usr/bin/env bash
# Follow-ups from the first candidate round:
#   m1 MTP@c1 = 3.64 ms (beats SGLang 4.55); m3 fp8 KV@c1 = 4.59 ms (+13%).
# Now map the combinations and find where MTP's c64 collapse begins.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=comb
PORT=8100
NAME=vllm_comb_${PORT}
IMG=vllm/vllm-openai-rocm:nightly-rocm100
LOG=~/qwen38/logs/comb_q355.log
: > "$LOG"

CE=(-e PYTHONPATH=/workspace/vllm -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf -v /home/amatveev/work/vllm:/workspace/vllm)
CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

start() {
  local label=$1 mns=$2
  shift 2
  docker rm -f "$NAME" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$NAME" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    "${CE[@]}" "$IMG" \
    --model Qwen/Qwen3.5-35B-A3B-FP8 --served-model-name Qwen/Qwen3.5-35B-A3B-FP8 \
    "${CS[@]}" --max-num-seqs "$mns" "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"
    docker logs --tail 20 "$NAME" >> "$LOG" 2>&1
    return 1
  fi
  docker logs "$NAME" 2>&1 | grep -E "is ENABLED|Using [A-Z_]+ backend" | tail -2 >> "$LOG"
}

cell() {
  ~/qwen38/scripts/bench_cell_m.sh "$1" "$PORT" "$2" "$3" "$4" "$5" >> "$LOG" 2>&1
}

MTP='{"method":"mtp","num_speculative_tokens":3}'
echo "COMB START $(date -u +%H:%M:%S)" >> "$LOG"

start n1 8 --kv-cache-dtype fp8_e4m3 && cell comb_kvfp8_isl8192_osl512_c8 8192 512 8 32
start n2 1 --kv-cache-dtype fp8_e4m3 --speculative-config "$MTP" && cell comb_kvfp8mtp_isl8192_osl1024_c1 8192 1024 1 8
start n3 8 --speculative-config "$MTP" && cell comb_mtp_isl8192_osl512_c8 8192 512 8 32
docker rm -f "$NAME" >/dev/null 2>&1
echo "COMB_DONE $(date -u +%H:%M:%S)" >> "$LOG"
