#!/usr/bin/env bash
# Accuracy runs, queued behind the perf measurements:
#   1  PR1 tree (AITER FA preferred)      gsm8k 5-shot   -- does the backend
#                                                          change anything?
#   2  PR2 baseline (P1) rep 2            gsm8k 5-shot   -- second sample, so
#   3  PR2 patched (P1 + FSE) rep 2       gsm8k 5-shot      the comparison is
#                                                             not one draw
set -u
cd "$HOME/qwen38" || exit 1
LOG=~/qwen38/logs/pr_lm_eval_queue.log
: > "$LOG"

for _ in $(seq 1 240); do
  grep -qE '^PR1F_DONE' ~/qwen38/logs/pr1_measure.log &&
    grep -qE '^C64_DONE' ~/qwen38/logs/pr1_c64.log && break
  sleep 60
done
grep -qE '^C64_DONE' ~/qwen38/logs/pr1_c64.log || {
  echo "perf measurements never finished - aborting" >> "$LOG"; exit 1; }

# Wait for the lock rather than failing: the runs are serialised on purpose.
exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 7200 9; then
  echo "could not get the GPU lock in 2h - aborting" >> "$LOG"; exit 1
fi

run() {  # <name> <tree>
  echo "LMEVAL $1 START $(date -u +%H:%M:%S) $2" >> "$LOG"
  bash ~/qwen38/lm_eval_tree.sh "$2" "$1" >> "$LOG" 2>&1
  echo "LMEVAL $1 RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
}

# The P1 tree is both "PR1 applied" and "PR2 baseline", so one run serves
# both PRs: it checks the FA backend's output and gives the FSE comparison a
# second baseline sample.
run pr1_fa_p1_base /home/amatveev/work/vllm_e0d
run pr2_fse_r2     /home/amatveev/work/vllm_p1_fse

grep -E 'flexible-extract|strict-match' "$LOG" | tail -6 >> "$LOG"
echo "PR_LMEVAL_QUEUE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
