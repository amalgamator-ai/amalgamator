#!/bin/bash
# Re-capture the c64 no-spec trace after fixing _compute_profile_iterations:
# the old delay (8,320 steps for a ~260-step run) meant the profiler window
# never opened, so the first attempt produced no trace file.
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"
LOG=~/qwen38/logs/profile_nospec_c64.log
echo "NOSPEC START $(date -u +%H:%M:%S)" > "$LOG"
bash ~/qwen38/preflight_images.sh >> "$LOG" 2>&1 || {
  echo "IMAGE PREFLIGHT FAILED" >> "$LOG"; exit 1; }
rm -rf llm/auto_profile_server_client/results/qwen35_nospec_c64
.venv/bin/python -m llm.auto_profile_server_client \
  --config llm/auto_profile_server_client/test_configs/qwen35_nospec_c64.json \
  >> "$LOG" 2>&1
rc=$?
echo "NOSPEC RC=$rc $(date -u +%H:%M:%S)" >> "$LOG"
find llm/auto_profile_server_client/results/qwen35_nospec_c64 \
  -name '*trace*' -printf '%s %p\n' >> "$LOG" 2>&1
echo "NOSPEC_DONE $(date -u +%H:%M:%S)" >> "$LOG"
