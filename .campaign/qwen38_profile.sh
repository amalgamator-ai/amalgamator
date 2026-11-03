#!/usr/bin/env bash
# Fresh trace of Qwen3.8-27B at 8K/c1 on the post-PR1 tree, run through the
# amalgamator profile pipeline (source-mounted, pinned ROCm 10 image).
#
# The 2026-10-31 analysis for this model predates the attention fix, so its
# profile is 97% Triton paged attention and says nothing about what is left.
# This produces a trace of the code the model actually runs now.
set -u
cd "$HOME/amalgamator" || exit 1
LOG=~/qwen38/logs/qwen38_profile.log
: > "$LOG"

exec 9>"$HOME/qwen38/.gpu.lock"
if ! flock -w 21600 9; then
  echo "no GPU lock in 6h - aborting" >> "$LOG"; exit 1
fi
for _ in $(seq 1 60); do
  [ "$(docker ps --format '{{.Names}}' | wc -l)" -eq 0 ] && break
  sleep 30
done
echo "PROFILE START $(date -u +%H:%M:%S)" >> "$LOG"

python3 -m llm.auto_profile_server_client.pipeline \
  --config llm/auto_profile_server_client/test_configs/qwen38_post_p1.json \
  >> "$LOG" 2>&1
rc=$?
echo "PROFILE RC=$rc $(date -u +%H:%M:%S)" >> "$LOG"
find llm/auto_profile_server_client/results/qwen38_post_p1 -name '*.trace.json.gz' \
  | head -5 >> "$LOG"
echo "PROFILE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
