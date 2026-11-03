#!/usr/bin/env bash
# Locate the MTP spec-decode collapse on Qwen3.5-35B-A3B (P1 already in the tree).
# Known points at 8192 in / 512 out: c8 = 9.85 ms, c64 = 360.16 ms (no-spec c64 = 49.70).
# Sweep c16/c32/c64 at num_speculative_tokens=3, then test n=1 and fp8 KV at c64.
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
NAME=vllm_mtpshape_${PORT}
IMG=vllm/vllm-openai-rocm:nightly-rocm100
LOG=~/qwen38/logs/mtpshape_q355.log
: > "$LOG"

CE=(-e PYTHONPATH=/workspace/vllm -e PYTHONDONTWRITEBYTECODE=1
    -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1
    -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf
    -v /home/amatveev/work/vllm:/workspace/vllm)
CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

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
    | grep -E "is ENABLED|speculative_config=SpeculativeConfig|kv_cache_dtype=" \
    | head -2 | cut -c1-160 >> "$LOG"
}

cell() { ~/qwen38/scripts/bench_cell_m.sh "$@" >> "$LOG" 2>&1; }

M3='{"method":"mtp","num_speculative_tokens":3}'
M1='{"method":"mtp","num_speculative_tokens":1}'

echo "MTPSHAPE START $(date -u +%H:%M:%S)" >> "$LOG"

# A: locate the cliff at n=3
start a 64 --speculative-config "$M3" && {
  cell mtpshape_c16_n3 8100 8192 512 16 32
  cell mtpshape_c32_n3 8100 8192 512 32 64
  cell mtpshape_c64_n3 8100 8192 512 64 128
}

# B: does a single speculative token avoid it?
start b 64 --speculative-config "$M1" && \
  cell mtpshape_c64_n1 8100 8192 512 64 128

# C: fp8 KV (the c8 winner) with n=3 at c64
start c 64 --kv-cache-dtype fp8_e4m3 --speculative-config "$M3" && \
  cell mtpshape_c64_n3kvfp8 8100 8192 512 64 128

docker rm -f "$NAME" >/dev/null 2>&1
echo "MTPSHAPE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
