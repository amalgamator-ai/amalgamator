#!/usr/bin/env bash
# Post-P1 position of vLLM on Qwen3.5-35B-A3B: with the selector fix in the tree,
# bench the cells the gate covers (mns<=8: c1, c8) and the cell it excludes (c64).
set -u
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=p1q355
PORT=8100
NAME=vllm_p1q355_${PORT}
LOG=~/qwen38/logs/p1_q355_measure.log
: > "$LOG"

ARM=""
run() {  # <max_num_seqs> <cell-label> <isl> <osl> <conc> <np>
  local mns=$1 label=$2 isl=$3 osl=$4 conc=$5 np=$6
  if [ "$ARM" != "$mns" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1; sleep 5
    docker run -d --name "$NAME" --network host --ipc=host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
      --security-opt seccomp=unconfined \
      -e PYTHONPATH=/workspace/vllm \
      -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
      -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
      -v /home/amatveev/work/vllm:/workspace/vllm \
      vllm/vllm-openai-rocm:nightly-rocm100 \
      --model Qwen/Qwen3.5-35B-A3B-FP8 --served-model-name Qwen/Qwen3.5-35B-A3B-FP8 \
      --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
      --no-enable-prefix-caching --gpu-memory-utilization 0.90 \
      --max-model-len 65536 --max-num-seqs "$mns" >/dev/null
    if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
      echo "NOT READY mns=$mns" >> "$LOG"; return 1
    fi
    docker logs "$NAME" 2>&1 | grep -E "is ENABLED|Using [A-Z_]+ backend out of potential" | tail -2 >> "$LOG"
    ARM=$mns
  fi
  ~/qwen38/scripts/bench_cell_m.sh "$label" "$PORT" "$isl" "$osl" "$conc" "$np" >> "$LOG" 2>&1
}

echo "P1 MEASURE START $(date -u +%H:%M:%S)" >> "$LOG"
run 8  p1q355_isl8192_osl1024_c1 8192 1024 1 8
run 8  p1q355_isl8192_osl512_c8  8192 512  8 32
run 64 p1q355_isl8192_osl512_c64 8192 512  64 128
docker rm -f "$NAME" >/dev/null 2>&1
echo "P1_MEASURE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
