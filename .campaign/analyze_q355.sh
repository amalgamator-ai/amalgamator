#!/bin/bash
# Create single-trace configs for the Qwen3.5-35B-A3B profile results and run
# both trace analyses sequentially.
#
# Layout note: auto_profile writes per-framework run dirs one level deeper than
# the framework name -- results/<model>/vllm/vllm-<tag>/ and results/<model>/sgl/sgl-<tag>/.
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"

ROOT=llm/auto_analyze/results/results_analyze_qwen35_35b_a3b_mi300x_decode8k/test_results/test_1
P=llm/auto_profile_server_client/results/qwen35_35b_a3b_mi300x_decode8k
MODEL=Qwen/Qwen3.5-35B-A3B-FP8
VLLM_COMMIT=2b009fc6d1493a9a1ad19f4d0c7230b7e6b7e63a
SGL_COMMIT=297e54fd105e974cdf401e5b4d20253a22c16f90

mkdir -p "$ROOT"
echo "ANALYZE START $(date -u +%H:%M:%S)" > ~/qwen38/logs/analyze_q355.log

gen() {  # <framework> <framework-dir> <source-tree> <commit>
  local fw=$1 fwdir=$2 src=$3 commit=$4
  local armdir trace
  armdir=$(find "$P/$fwdir" -maxdepth 1 -mindepth 1 -type d | sort | head -1)
  if [ -z "$armdir" ]; then
    echo "MISSING ARM DIR for $fw (looked under $P/$fwdir)" | tee -a ~/qwen38/logs/analyze_q355.log
    return 1
  fi
  trace=$(ls "$armdir"/profiling/*trace.json.gz 2>/dev/null | head -1)
  if [ -z "$trace" ]; then
    echo "MISSING TRACE for $fw under $armdir/profiling" | tee -a ~/qwen38/logs/analyze_q355.log
    return 1
  fi
  echo "GEN $fw: armdir=$armdir" | tee -a ~/qwen38/logs/analyze_q355.log
  .venv/bin/python -m llm.auto_analyze.create_single_trace_config \
    --model "$MODEL" \
    --gpu-type "AMD Instinct MI300X" \
    --batch-size-range 1 \
    --prefill-size-range 8192 \
    --output-size-range 1024 \
    --trace-file "$trace" \
    --run-log-file "$armdir/profiling/server.log" \
    --clean-source-code-path "$src" \
    --commit-id "$commit" \
    --framework-name "$fw" \
    --analyze-output-dir "$ROOT/$fw" \
    --output-config-file "$ROOT/single_trace_config_$fw" \
    --enable-single-trace-perf-analysis >> ~/qwen38/logs/analyze_q355.log 2>&1
}

gen vllm   vllm /home/amatveev/work/vllm   "$VLLM_COMMIT"
gen sglang sgl  /home/amatveev/work/sglang "$SGL_COMMIT"

for fw in vllm sglang; do
  if [ -f "$ROOT/$fw/result_metadata.json" ]; then
    echo "SKIP single_trace $fw (analysis already complete)" | tee -a ~/qwen38/logs/analyze_q355.log
    continue
  fi
  if [ ! -f "$ROOT/single_trace_config_$fw.json" ]; then
    echo "SKIP single_trace $fw (no config)" | tee -a ~/qwen38/logs/analyze_q355.log
    continue
  fi
  echo "RUN single_trace $fw $(date -u +%H:%M:%S)" | tee -a ~/qwen38/logs/analyze_q355.log
  .venv/bin/python -m llm.auto_analyze.run_single_trace \
    --config "$ROOT/single_trace_config_$fw.json" >> ~/qwen38/logs/analyze_q355.log 2>&1
  echo "SINGLE_${fw}_DONE $? $(date -u +%H:%M:%S)" | tee -a ~/qwen38/logs/analyze_q355.log
done
echo "ANALYZE_Q355_DONE $(date -u +%H:%M:%S)" | tee -a ~/qwen38/logs/analyze_q355.log
