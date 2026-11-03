#!/usr/bin/env bash
# After the post-PR-1 profile finishes, build the single-trace config for the new
# trace and run the amalgamator's single-trace analysis on it.
#
# The old analysis for this model measured a step that was 97% Triton paged
# attention; that kernel is what PR 1 replaced, so the plan has to be re-derived
# from a trace of the current code.
set -u
cd "$HOME/amalgamator" || exit 1
LOG=~/qwen38/logs/qwen38_analyze.log
: > "$LOG"
RES=llm/auto_profile_server_client/results/qwen38_post_p1
OUT=llm/auto_analyze/results/results_analyze_qwen38_post_p1
mkdir -p "$OUT"

echo "ANALYZE START $(date -u +%H:%M:%S)" >> "$LOG"

# The profile writes the trace during engine shutdown; wait for a non-empty one.
for _ in $(seq 1 120); do
  trace=$(find "$RES" -name '*.trace.json.gz' -size +1M 2>/dev/null | head -1)
  [ -n "$trace" ] && break
  sleep 30
done
if [ -z "${trace:-}" ]; then
  echo "no trace found under $RES" >> "$LOG"; exit 1
fi
echo "TRACE $trace" >> "$LOG"

python3 ~/qwen38/gen_single_trace_config.py \
  "$RES" /home/amatveev/work/wt76_pr1 76c6f3f2a9 \
  "$OUT/single_trace_config.json" >> "$LOG" 2>&1

bash -lc "set -a; source ~/amalgamator_env.sh; set +a; \
  PYTHONPATH=. python3 llm/auto_analyze/run_single_trace.py \
  --config $OUT/single_trace_config.json" >> "$LOG" 2>&1
echo "ANALYZE RC=$? $(date -u +%H:%M:%S)" >> "$LOG"
ls "$OUT" >> "$LOG"
echo "ANALYZE_DONE $(date -u +%H:%M:%S)" >> "$LOG"
