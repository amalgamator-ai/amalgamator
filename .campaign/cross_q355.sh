#!/bin/bash
# Wait for both single-trace analyses, then run cross-trace matching + compare +
# improvement plan. One iteration per stage (see configs/*_config.py).
# The agent section is deliberately absent: make_agent_config falls back to the
# checked-in llm/auto_analyze/configs/claude_config.json (DeepSeek V4.1, 1M ctx).
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"

LOG=~/qwen38/logs/cross_q355.log
ROOT=llm/auto_analyze/results/results_analyze_qwen35_35b_a3b_mi300x_decode8k/test_results/test_1
echo "CROSS WAIT START $(date -u +%H:%M:%S)" > "$LOG"

for _ in $(seq 1 360); do
  grep -qE "^ANALYZE_Q355_DONE" ~/qwen38/logs/analyze_q355.log && break
  sleep 60
done
if ! grep -qE "^ANALYZE_Q355_DONE" ~/qwen38/logs/analyze_q355.log; then
  echo "ANALYSIS TIMEOUT - not starting cross-trace" >> "$LOG"
  exit 1
fi

for fw in vllm sglang; do
  if [ ! -f "$ROOT/$fw/result_metadata.json" ]; then
    echo "MISSING $fw result_metadata.json - not starting cross-trace" >> "$LOG"
    exit 1
  fi
done

cat > "$ROOT/cross_trace_config.json" <<EOF
{
    "traces": [
        {"result_dir": "/home/amatveev/amalgamator/$ROOT/vllm"},
        {"result_dir": "/home/amatveev/amalgamator/$ROOT/sglang"}
    ],
    "target_trace_id": 0,
    "output_dir": "/home/amatveev/amalgamator/$ROOT/cross_trace",
    "make_improvement_plan": true,
    "num_matching_iterations": 1,
    "num_compare_iterations": 1,
    "num_improvement_plan_iterations": 1,
    "thinking-mode": "deep"
}
EOF

echo "CROSS START $(date -u +%H:%M:%S)" >> "$LOG"
.venv/bin/python -m llm.auto_analyze.run_cross_trace \
  --config "$ROOT/cross_trace_config.json" \
  --agent-config llm/auto_analyze/configs/deepseek_v41_config.json >> "$LOG" 2>&1
rc=$?
echo "CROSS_RC $rc $(date -u +%H:%M:%S)" >> "$LOG"
if [ "$rc" -eq 0 ]; then
  echo "CROSS_Q355_DONE $(date -u +%H:%M:%S)" >> "$LOG"
else
  echo "CROSS_Q355_FAILED $(date -u +%H:%M:%S)" >> "$LOG"
fi
