#!/bin/bash
# Residual analysis: single-trace analysis of the P1-applied vLLM trace, then a
# cross-trace against the existing SGLang trace at the same cell. The resulting
# improvement plan is what isolates the optimizations that are NOT the attention
# selector, i.e. the candidates for their own PRs on top of P1.
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"

LOG=~/qwen38/logs/analyze_p1.log
ROOT=llm/auto_analyze/results/results_analyze_qwen35_35b_p1_c1/test_results/test_1
PROF=llm/auto_profile_server_client/results/qwen35_p1_c1
SGL=llm/auto_analyze/results/results_analyze_qwen35_35b_a3b_mi300x_decode8k/test_results/test_1/sglang
SRC=/home/amatveev/work/vllm_e0d
MODEL=Qwen/Qwen3.5-35B-A3B-FP8
AGENT=llm/auto_analyze/configs/deepseek_v41_config.json
echo "ANALYZE_P1 START $(date -u +%H:%M:%S)" > "$LOG"

for _ in $(seq 1 240); do
  grep -qE '^P1PROF_DONE' ~/qwen38/logs/profile_p1_c1.log && break
  sleep 60
done
grep -qE '^P1PROF_DONE' ~/qwen38/logs/profile_p1_c1.log || {
  echo "P1PROF never finished - aborting" >> "$LOG"; exit 1; }

if [ -n "$(git -C "$SRC" status --porcelain | grep -v '^??')" ]; then
  echo "source tree is dirty - commit or stash before analysis" >> "$LOG"
  git -C "$SRC" status --short >> "$LOG" 2>&1
  exit 1
fi

armdir=$(find "$PROF/vllm" -maxdepth 1 -mindepth 1 -type d | head -1)
trace=$(ls "$armdir"/profiling/*trace.json.gz 2>/dev/null | head -1)
if [ -z "$trace" ]; then
  echo "MISSING TRACE under $armdir/profiling" >> "$LOG"; exit 1
fi
echo "TRACE $trace" >> "$LOG"

mkdir -p "$ROOT"
.venv/bin/python -m llm.auto_analyze.create_single_trace_config \
  --model "$MODEL" \
  --gpu-type "AMD Instinct MI300X" \
  --batch-size-range 1 \
  --prefill-size-range 8192 \
  --output-size-range 1024 \
  --trace-file "$trace" \
  --run-log-file "$armdir/profiling/server.log" \
  --clean-source-code-path "$SRC" \
  --commit-id HEAD \
  --framework-name vllm \
  --analyze-output-dir "$ROOT/vllm" \
  --output-config-file "$ROOT/single_trace_config_vllm" \
  --enable-single-trace-perf-analysis >> "$LOG" 2>&1

echo "SINGLE START $(date -u +%H:%M:%S)" >> "$LOG"
.venv/bin/python -m llm.auto_analyze.run_single_trace \
  --config "$ROOT/single_trace_config_vllm.json" \
  --agent-config "$AGENT" >> "$LOG" 2>&1
echo "SINGLE RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
[ -f "$ROOT/vllm/result_metadata.json" ] || {
  echo "single-trace analysis incomplete - not starting cross-trace" >> "$LOG"; exit 1; }

cat > "$ROOT/cross_trace_config.json" <<EOF
{
    "traces": [
        {"result_dir": "/home/amatveev/amalgamator/$ROOT/vllm"},
        {"result_dir": "/home/amatveev/amalgamator/$SGL"}
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
  --agent-config "$AGENT" >> "$LOG" 2>&1
echo "CROSS RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
echo "ANALYZE_P1_DONE $(date -u +%H:%M:%S)" >> "$LOG"
