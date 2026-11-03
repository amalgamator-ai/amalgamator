#!/usr/bin/env bash
# Re-run both PR test suites on the branches as pushed (rebased onto current
# upstream main), not on the previous base.
set -u
BOX_REPO=/home/amatveev/work/vllm
IMG="vllm/vllm-openai-rocm@sha256:4c92e167b9b4739e9c7181ae384fafc2b8e6da0de9ddef7c4829ce0d581b12eb"
LOG=~/qwen38/logs/pr_tests_rebased.log
: > "$LOG"

cd "$BOX_REPO" || exit 1
git fetch -q https://github.com/vllm-project/vllm.git main || { echo "fetch failed" >> "$LOG"; exit 1; }
BASE=$(git rev-parse FETCH_HEAD)
echo "BASE $BASE $(date -u +%H:%M:%S)" >> "$LOG"

for arm in pr1 pr2; do
  wt="$HOME/work/wtchk2_$arm"
  git worktree remove --force "$wt" >/dev/null 2>&1; rm -rf "$wt"
  git worktree add -f "$wt" "$BASE" --detach >/dev/null 2>&1 || { echo "worktree $arm failed" >> "$LOG"; continue; }
  if [ "$arm" = pr1 ]; then
    git -C "$wt" apply --3way ~/qwen38/pr_patches/PR1_attention_selector.patch || { echo "apply $arm failed" >> "$LOG"; continue; }
  else
    git -C "$wt" apply --3way ~/qwen38/pr_patches/PR2_fse_default.patch || { echo "apply $arm failed" >> "$LOG"; continue; }
  fi
  bash ~/qwen38/sync_vllm_ext.sh "$wt" "$IMG" >/dev/null 2>&1
  if [ "$arm" = pr1 ]; then
    tests="/workspace/vllm/tests/platforms/test_rocm_aiter_backend_priority.py"
  else
    tests="/workspace/vllm/tests/model_executor/layers/test_fused_shared_expert_default.py"
  fi
  echo "== $arm ($tests)" >> "$LOG"
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined -e VLLM_ROCM_USE_AITER=1 --entrypoint python3 -e PYTHONPATH=/workspace/vllm \
    -e PYTHONDONTWRITEBYTECODE=1 -v "$wt":/workspace/vllm -w /workspace/vllm "$IMG" \
    -m pytest --noconftest -q "$tests" 2>&1 | tail -4 >> "$LOG"
done
echo "TESTS_DONE $(date -u +%H:%M:%S)" >> "$LOG"
