#!/usr/bin/env bash
# PR-2 measurement: the fused-shared-expert default, measured twice.
#
#   standalone  stock e0d00e4b8      vs  stock + FSE default      (proves the
#                                                                    patch is
#                                                                    independent
#                                                                    of P1)
#   stacked     e0d00e4b8 + P1       vs  + P1 + FSE default       (the number
#                                                                    the PR
#                                                                    quotes,
#                                                                    because P1
#                                                                    lands
#                                                                    first)
#
# Each arm is a separate tree, so no env override is involved: the difference
# between the two trees is the patch itself. Both cells per arm, two cells:
# c1 (decode) and c8 (throughput).
set -u
cd "$HOME/qwen38" || exit 1
source "$HOME/qwen38/images.env"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="$VLLM_IMAGE"
LOG=~/qwen38/logs/pr2_measure.log
: > "$LOG"

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

arm() {  # <label> <tree>
  local label=$1 tree=$2
  local name="vllm_pr2_${label}_${PORT}"
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
    "${CS[@]}" --max-num-seqs 8 >/dev/null
  if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$name" >> "$LOG" 2>&1; then
    echo "NOT READY $label" >> "$LOG"; docker logs --tail 30 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  docker logs "$name" 2>&1 | grep -iE 'fused shared expert|is ENABLED' | tail -2 | cut -c1-150 >> "$LOG"
  ~/qwen38/scripts/bench_cell_m.sh "pr2_${label}_c1" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "pr2_${label}_c1_r2" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "pr2_${label}_c8" "$PORT" 8192 512 8 32 >> "$LOG" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
}

echo "PR2 START $(date -u +%H:%M:%S)" >> "$LOG"

# Exactly one GPU driver at a time. Two copies of this script (a relaunch while
# a previous one was still alive) started arms on two containers at once; both
# engines fought for VRAM and both died in init, leaving arms with no result.
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -n 9; then
  echo "another GPU driver holds $HOME/qwen38/.gpu.lock - refusing to run" >> "$LOG"
  exit 1
fi

# One server at a time on this GPU: a second arm starting while another
# container holds 90% of VRAM makes the new engine die in init, silently
# producing an arm with no result.
for _ in $(seq 1 120); do
  others=$(docker ps --format '{{.Names}}' | grep -v "^vllm_pr2_" | wc -l)
  [ "$others" -eq 0 ] && break
  echo "waiting for other containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >> "$LOG"
  sleep 30
done
if [ "$(docker ps --format '{{.Names}}' | grep -v '^vllm_pr2_' | wc -l)" -ne 0 ]; then
  echo "GPU still busy with another container - aborting" >> "$LOG"; exit 1
fi

arm stock     /home/amatveev/work/vllm_e0d_stock   # e0d00e4b8
arm stock_fse /home/amatveev/work/vllm_pr_fse      # e0d00e4b8 + FSE default
arm p1        /home/amatveev/work/vllm_e0d         # e0d00e4b8 + P1
arm p1_fse    /home/amatveev/work/vllm_p1_fse      # e0d00e4b8 + P1 + FSE default
echo "PR2_DONE $(date -u +%H:%M:%S)" >> "$LOG"
