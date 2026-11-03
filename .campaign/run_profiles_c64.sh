#!/bin/bash
# Capture c64 traces for both spec-decode arms through auto_profile, which arms
# the profiler with /start_profile and /stop_profile instead of relying on the
# server-side iteration schedule. The hand-rolled attempts produced no trace
# files even with a graceful stop, so this uses the pipeline that is known to
# produce traces for this model.
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"
LOG=~/qwen38/logs/profiles_c64.log
echo "PROFILES START $(date -u +%H:%M:%S)" > "$LOG"

for arm in nospec mtp; do
  echo "ARM $arm START $(date -u +%H:%M:%S)" >> "$LOG"
  .venv/bin/python -m llm.auto_profile_server_client \
    --config "llm/auto_profile_server_client/test_configs/qwen35_${arm}_c64.json" \
    >> "$LOG" 2>&1
  echo "ARM $arm RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
done

find llm/auto_profile_server_client/results/qwen35_mtp_c64 \
     llm/auto_profile_server_client/results/qwen35_nospec_c64 \
     -name '*trace.json*' -printf '%s %p\n' >> "$LOG" 2>&1
echo "PROFILES_DONE $(date -u +%H:%M:%S)" >> "$LOG"
