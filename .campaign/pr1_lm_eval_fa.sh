#!/usr/bin/env bash
# Accuracy on the path PR 1 actually ships.
#
# The two gsm8k runs so far were taken before the condition changed to head
# size, when lm_eval's default max_num_seqs kept the old order, so they measured
# ROCM_ATTN and serve as the baseline. This run uses the final tree, where
# head_size=256 selects AITER FA regardless of batch size.
set -u
cd "$HOME/qwen38" || exit 1
LOG=~/qwen38/logs/pr1_lm_eval_fa.log
: > "$LOG"

for _ in $(seq 1 120); do
  grep -qE '^PR1FIN_DONE' ~/qwen38/logs/pr1_final_check.log && break
  sleep 60
done

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 3600 9; then
  echo "no GPU lock in 1h - aborting" >> "$LOG"; exit 1
fi

echo "LMEVAL fa START $(date -u +%H:%M:%S)" >> "$LOG"
bash ~/qwen38/lm_eval_tree.sh /home/amatveev/work/vllm_e0d pr1_fa >> "$LOG" 2>&1
echo "LMEVAL fa RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
grep -E 'flexible-extract|strict-match' "$LOG" | tail -2 >> "$LOG"
echo "PR1_LMEVAL_FA_DONE $(date -u +%H:%M:%S)" >> "$LOG"
