#!/bin/bash
# Drive the Qwen3.5-35B-A3B campaign after the profile finishes:
#   1. trace analyses (LLM-bound, no GPU)
#   2. 2x2x6 matrix (GPU-bound) in parallel
#   3. backend sweep (does the P1 head_dim-256 gap reproduce?)
set -u
LOG=~/qwen38/logs/chain_q355.log
echo "CHAIN START $(date -u +%H:%M:%S)" > "$LOG"

# 1. wait for the profile marker
for i in $(seq 1 240); do
  grep -qE "^PROFILE_Q355_DONE" ~/qwen38/logs/auto_profile_q355.log && break
  sleep 30
done
grep -qE "^PROFILE_Q355_DONE" ~/qwen38/logs/auto_profile_q355.log \
  || { echo "PROFILE TIMEOUT - not starting" >> "$LOG"; exit 1; }
echo "profile done, GPU release wait $(date -u +%H:%M:%S)" >> "$LOG"
sleep 30

# 2. analyses in the background (they do not touch the GPU)
nohup ~/qwen38/analyze_q355.sh > /dev/null 2>&1 &
echo "analyses launched $(date -u +%H:%M:%S)" >> "$LOG"

# 3. matrix (GPU)
~/qwen38/scripts/run_matrix_q355.sh >> "$LOG" 2>&1
echo "MATRIX_DONE $(date -u +%H:%M:%S)" >> "$LOG"

# 4. backend sweep
~/qwen38/scripts/sweep_q355_backend.sh >> "$LOG" 2>&1
echo "SWEEP_DONE $(date -u +%H:%M:%S)" >> "$LOG"

echo "CHAIN_Q355_DONE $(date -u +%H:%M:%S)" >> "$LOG"
