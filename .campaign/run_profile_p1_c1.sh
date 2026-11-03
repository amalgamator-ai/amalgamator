#!/bin/bash
# Trace vLLM with P1 applied at the reference cell (8K in / 1K out, c1), from a
# mounted tree that matches the pinned image build (e0d00e4b8 + P1). This is the
# residual baseline: the stock trace is dominated by the 1.06 ms Triton paged
# attention kernel, so a plan written from it re-proposes P1's work.
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"
LOG=~/qwen38/logs/profile_p1_c1.log
echo "P1PROF START $(date -u +%H:%M:%S)" > "$LOG"
# The config pins both images by digest; this re-checks that they are ROCm 10.
bash ~/qwen38/preflight_images.sh >> "$LOG" 2>&1 || {
  echo "IMAGE PREFLIGHT FAILED" >> "$LOG"; exit 1; }
rm -rf llm/auto_profile_server_client/results/qwen35_p1_c1
.venv/bin/python -m llm.auto_profile_server_client \
  --config llm/auto_profile_server_client/test_configs/qwen35_p1_c1.json \
  >> "$LOG" 2>&1
rc=$?
echo "P1PROF RC=$rc $(date -u +%H:%M:%S)" >> "$LOG"
find llm/auto_profile_server_client/results/qwen35_p1_c1 \
  -name '*trace*' -printf '%s %p\n' >> "$LOG" 2>&1
echo "P1PROF_DONE $(date -u +%H:%M:%S)" >> "$LOG"
