#!/usr/bin/env bash
# Candidate improvements beyond the plan, all measured on top of P1.
# Written as explicit blocks (no shared helper) so each arm's flags are visible.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=cand
PORT=8100
NAME=vllm_cand_${PORT}
LOG=~/qwen38/logs/cand_q355.log
IMG=vllm/vllm-openai-rocm:nightly-rocm100
AITER_FMOE=/usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv
: > "$LOG"

COMMON_ENV=(-e PYTHONPATH=/workspace/vllm -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1
            -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf -v /home/amatveev/work/vllm:/workspace/vllm)
COMMON_SRV=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
            --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

start() {  # <label> <max_num_seqs> ; rest of args: extra server args
  local label=$1 mns=$2; shift 2
  docker rm -f "$NAME" >/dev/null 2>&1
  sleep 5
  docker run -d --name "$NAME" --network host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined \
    "${COMMON_ENV[@]}" "${EXTRA_ENV[@]}" \
    "$IMG" \
    --model Qwen/Qwen3.5-35B-A3B-FP8 --served-model-name Qwen/Qwen3.5-35B-A3B-FP8 \
    "${COMMON_SRV[@]}" --max-num-seqs "$mns" "$@" >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 20 "$NAME" >> "$LOG" 2>&1
    docker rm -f "$NAME" >/dev/null 2>&1; return 1
  fi
  docker logs "$NAME" 2>&1 | grep -E "is ENABLED|Using [A-Z_]+ backend out of potential|speculative" | tail -3 >> "$LOG"
}

cell() { ~/qwen38/scripts/bench_cell_m.sh "$1" "$PORT" "$2" "$3" "$4" "$5" >> "$LOG" 2>&1; }

echo "CAND START $(date -u +%H:%M:%S)" >> "$LOG"

# m1 ------------------------------------------------------------- MTP spec, c1
EXTRA_ENV=()
start m1 1 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  && cell cand_mtp_isl8192_osl1024_c1 8192 1024 1 8

# m2 ------------------------------------------------------------ MTP spec, c64
start m2 64 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  && cell cand_mtp_isl8192_osl512_c64 8192 512 64 128

# m3 ------------------------------------------------------------ fp8 KV, c1
start m3 1 --kv-cache-dtype fp8_e4m3 \
  && cell cand_kvfp8_isl8192_osl1024_c1 8192 1024 1 8

# m4 ------------------------------------------- AITER tuned FMOE configs, c1
EXTRA_ENV=(-e AITER_CONFIG_FMOE="$AITER_FMOE")
start m4 1 && cell cand_fmoe_isl8192_osl1024_c1 8192 1024 1 8

docker rm -f "$NAME" >/dev/null 2>&1
echo "CAND_DONE $(date -u +%H:%M:%S)" >> "$LOG"
