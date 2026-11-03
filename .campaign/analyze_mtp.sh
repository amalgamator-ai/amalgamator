#!/bin/bash
# Analyse the two profiled c64 traces (MTP vs no spec decode) and produce a
# cross-trace improvement plan that targets the MTP arm. Both single-trace
# analyses are CPU-only and run in parallel; the cross-trace runs after both.
#
# Both traces come from the same pinned image build, and both arms run with
# max_num_seqs=64, where the P1 small-batch gate is inactive — so the pair is
# consistent even though the trees differ.
set -u
cd "$HOME/amalgamator" || exit 1
source "$HOME/amalgamator_env.sh"

LOG=~/qwen38/logs/analyze_mtp.log
ROOT=llm/auto_analyze/results/results_analyze_qwen35_35b_mtp_c64/test_results/test_1
MTP_PROF=llm/auto_profile_server_client/results/qwen35_mtp_c64
NOSPEC_PROF=llm/auto_profile_server_client/results/qwen35_nospec_c64
MODEL=Qwen/Qwen3.5-35B-A3B-FP8
SRC=/home/amatveev/work/vllm_e0d
AGENT=llm/auto_analyze/configs/deepseek_v41_config.json
echo "ANALYZE_MTP START $(date -u +%H:%M:%S)" > "$LOG"

for _ in $(seq 1 240); do
  grep -qE '^NOSPEC_DONE' ~/qwen38/logs/profile_nospec_c64.log && break
  sleep 60
done
grep -qE '^NOSPEC_DONE' ~/qwen38/logs/profile_nospec_c64.log || {
  echo "nospec profile never finished - aborting" >> "$LOG"; exit 1; }

if [ -n "$(git -C "$SRC" status --porcelain | grep -v '^??')" ]; then
  echo "source tree is dirty - commit or stash before the analysis" >> "$LOG"
  git -C "$SRC" status --short >> "$LOG" 2>&1
  exit 1
fi

mkdir -p "$ROOT"

gen() {  # <tag> <profile-results-dir>
  local tag=$1 prof=$2
  local armdir trace
  armdir=$(find "$prof/vllm" -maxdepth 1 -mindepth 1 -type d | head -1)
  trace=$(ls "$armdir"/profiling/*trace.json.gz 2>/dev/null | head -1)
  if [ -z "$trace" ]; then
    echo "MISSING TRACE for $tag under $armdir/profiling" >> "$LOG"; return 1
  fi
  echo "GEN $tag trace=$trace" >> "$LOG"
  .venv/bin/python -m llm.auto_analyze.create_single_trace_config \
    --model "$MODEL" \
    --gpu-type "AMD Instinct MI300X" \
    --batch-size-range 64 \
    --prefill-size-range 8192 \
    --output-size-range 128 \
    --trace-file "$trace" \
    --run-log-file "$armdir/profiling/server.log" \
    --clean-source-code-path "$SRC" \
    --commit-id HEAD \
    --framework-name vllm \
    --analyze-output-dir "$ROOT/$tag" \
    --output-config-file "$ROOT/single_trace_config_$tag" \
    --enable-single-trace-perf-analysis >> "$LOG" 2>&1
}

gen mtp "$MTP_PROF"
gen nospec "$NOSPEC_PROF"
for tag in mtp nospec; do
  [ -f "$ROOT/single_trace_config_$tag.json" ] || {
    echo "no config for $tag - aborting" >> "$LOG"; exit 1; }
done

echo "SINGLE START $(date -u +%H:%M:%S)" >> "$LOG"
for tag in mtp nospec; do
  nohup .venv/bin/python -m llm.auto_analyze.run_single_trace \
    --config "$ROOT/single_trace_config_$tag.json" \
    --agent-config "$AGENT" >> "$LOG.$tag" 2>&1 &
  echo "launched single_trace $tag pid $!" >> "$LOG"
done

for _ in $(seq 1 600); do
  n=$(ls "$ROOT"/mtp/result_metadata.json "$ROOT"/nospec/result_metadata.json 2>/dev/null | wc -l)
  [ "$n" = "2" ] && break
  sleep 60
done
[ "$(ls "$ROOT"/mtp/result_metadata.json "$ROOT"/nospec/result_metadata.json 2>/dev/null | wc -l)" = "2" ] || {
  echo "SINGLE ANALYSES DID NOT BOTH COMPLETE - no cross-trace" >> "$LOG"; exit 1; }
echo "SINGLE DONE $(date -u +%H:%M:%S)" >> "$LOG"

cat > "$ROOT/cross_trace_config.json" <<EOF
{
    "traces": [
        {"result_dir": "/home/amatveev/amalgamator/$ROOT/mtp"},
        {"result_dir": "/home/amatveev/amalgamator/$ROOT/nospec"}
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
echo "ANALYZE_MTP_DONE $(date -u +%H:%M:%S)" >> "$LOG"
