#!/usr/bin/env bash
# Accuracy gate for PR-2: gsm8k, 5-shot, same sampling, two trees.
#   A  e0d00e4b8 + P1              (baseline)
#   B  e0d00e4b8 + P1 + FSE default (patched)
# Waits for the perf A/B to release the GPU and takes the same lock, so the two
# never overlap.
set -u
cd "$HOME/qwen38" || exit 1
LOG=~/qwen38/logs/pr2_lm_eval.log
: > "$LOG"
mkdir -p ~/qwen38/lmeval_out

echo "LMEVAL WAIT $(date -u +%H:%M:%S)" >> "$LOG"
for _ in $(seq 1 240); do
  grep -qE '^PR2_DONE' ~/qwen38/logs/pr2_measure.log && break
  sleep 60
done
grep -qE '^PR2_DONE' ~/qwen38/logs/pr2_measure.log || {
  echo "PR2 measurement never finished - aborting" >> "$LOG"; exit 1; }

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -n 9; then
  echo "GPU lock held by another driver - aborting" >> "$LOG"; exit 1
fi

for arm in baseline:/home/amatveev/work/vllm_e0d patched:/home/amatveev/work/vllm_p1_fse; do
  name=${arm%%:*}; tree=${arm##*:}
  echo "LMEVAL $name START $(date -u +%H:%M:%S) tree=$tree" >> "$LOG"
  bash ~/qwen38/lm_eval_tree.sh "$tree" "pr2_${name}" >> "$LOG" 2>&1
  echo "LMEVAL $name RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
done
echo "PR2_LMEVAL_DONE $(date -u +%H:%M:%S)" >> "$LOG"
