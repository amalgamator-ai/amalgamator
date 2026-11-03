#!/usr/bin/env bash
# Re-measure both PRs on a near-current base, in ROCm 10 docker.
#
# Base is the vLLM commit inside the 2026-11-02 nightly (76c6f3f2a9), 51 commits
# behind main (3a228c1fe). Four arms, each a separate worktree, same image and
# flags, so the patch is the only difference:
#
#   stock  76c6f3f2a9
#   pr1    76c6f3f2a9 + attention backend ordering
#   pr2    76c6f3f2a9 + fused shared experts default
#   both   76c6f3f2a9 + both
#
# pr1/both are the paired comparison for PR 1; pr2/both is PR 2 standalone, and
# pr1-stock / both-pr1 give PR 2 stacked on PR 1.
set -u
cd "$HOME/qwen38" || exit 1
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
PORT=8100
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
LOG=~/qwen38/logs/pr_rebase_measure.log
: > "$LOG"

CS=(--port "$PORT" --tensor-parallel-size 1 --trust-remote-code
    --no-enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 65536)

# Engagement discipline: an arm that does not show the expected code path in
# its own log is a broken arm, not a slow one.
expect() {  # <arm> <container>
  local arm=$1 name=$2
  case "$arm" in
    stock) want="none" ;;
    pr1)   want="aiter_priority" ;;
    pr2)   want="fse" ;;
    both)  want="aiter_priority,fse" ;;
  esac
  local logs; logs=$(docker logs "$name" 2>&1)
  if [ "$want" = none ]; then
    echo "$logs" | grep -qE 'Trying AITER attention backends|fused shared experts are ENABLED' \
      && { echo "ENGAGE FAIL $arm: stock shows a patched path" >> "$LOG"; return 1; }
  else
    if [[ "$want" == *aiter_priority* ]]; then
      echo "$logs" | grep -q 'Trying AITER attention backends' \
        || { echo "ENGAGE FAIL $arm: no AITER priority line" >> "$LOG"; return 1; }
    fi
    if [[ "$want" == *fse* ]]; then
      echo "$logs" | grep -q 'AITER fused shared experts are ENABLED' \
        || { echo "ENGAGE FAIL $arm: no FSE enabled line" >> "$LOG"; return 1; }
    fi
  fi
  echo "$logs" | grep -E 'Trying AITER attention|fused shared experts are ENABLED' \
    | tail -2 | cut -c1-165 >> "$LOG"
  return 0
}

arm() {  # <arm> <tree>
  local arm=$1 tree=$2
  local name="vllm_rb2_${arm}_${PORT}"
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
    echo "NOT READY $arm" >> "$LOG"; docker logs --tail 30 "$name" >> "$LOG" 2>&1
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  echo "READY $arm $(date -u +%H:%M:%S)" >> "$LOG"
  if ! expect "$arm" "$name"; then
    echo "ARM REJECTED $arm" >> "$LOG"
    docker logs --tail 20 "$name" 2>&1 | tail -15 >> "$LOG"
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi
  ~/qwen38/scripts/bench_cell_m.sh "rb2_${arm}_c1"    "$PORT" 8192 1024 1 8  >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "rb2_${arm}_c1_r2" "$PORT" 8192 1024 1 8  >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "rb2_${arm}_c8"    "$PORT" 8192 1024 8 32 >> "$LOG" 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "rb2_${arm}_c8_r2" "$PORT" 8192 1024 8 32 >> "$LOG" 2>&1
  echo "ARM DONE $arm $(date -u +%H:%M:%S)" >> "$LOG"
  docker rm -f "$name" >/dev/null 2>&1
}

echo "RB2 START $(date -u +%H:%M:%S)" >> "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -n 9; then
  echo "another GPU driver holds $HOME/qwen38/.gpu.lock - refusing to run" >> "$LOG"
  exit 1
fi

for _ in $(seq 1 120); do
  others=$(docker ps --format '{{.Names}}' | grep -v "^vllm_rb2_" | wc -l)
  [ "$others" -eq 0 ] && break
  echo "waiting for other containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')" >> "$LOG"
  sleep 30
done
if [ "$(docker ps --format '{{.Names}}' | grep -v '^vllm_rb2_' | wc -l)" -ne 0 ]; then
  echo "GPU still busy with another container - aborting" >> "$LOG"; exit 1
fi

arm stock /home/amatveev/work/wt76_stock
arm pr1   /home/amatveev/work/wt76_pr1
arm pr2   /home/amatveev/work/wt76_pr2
arm both  /home/amatveev/work/wt76_both
echo "RB2_DONE $(date -u +%H:%M:%S)" >> "$LOG"
