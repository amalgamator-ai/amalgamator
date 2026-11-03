#!/usr/bin/env bash
# H2 test on Qwen3.5-35B-A3B (run one mode per invocation):
#   h2_q355.sh p1        tree = P1 only (AITER unified at small batch)
#   h2_q355.sh p1p2      tree = P1 + P2 patch (fused gated RMSNorm for quantized)
#   h2_q355.sh p1p2kill  tree = P1 + P2, with the P2 kill switch set
# Each mode: one 8192/1024 c1 benched cell.
set -u
MODE="${1:?p1|p1p2|p1p2kill}"
export MODEL=Qwen/Qwen3.5-35B-A3B-FP8
export SERVED=Qwen/Qwen3.5-35B-A3B-FP8
export TAG=h2q355
PORT=8100
NAME=vllm_h2q355_${PORT}
S=/home/amatveev/qwen38/scripts
LOG=~/qwen38/logs/h2_q355_${MODE}.log
VLLM=/home/amatveev/work/vllm
P1=~/qwen38/patches/p1_aiter_unified_small_batch.patch
P2=~/amalgamator/llm/auto_code_gen/results/qwen38_27b_mi300x_decode8k_p2/code_gen/V2_review.patch
: > "$LOG"

echo "MODE $MODE START $(date -u +%H:%M:%S)" >> "$LOG"

cd "$VLLM" || exit 1
git checkout -- . && git clean -fd >> "$LOG" 2>&1
git apply "$P1" || { echo "P1 APPLY FAILED" >> "$LOG"; exit 1; }
if [ "$MODE" != "p1" ]; then
  git apply "$P2" || { echo "P2 APPLY FAILED" >> "$LOG"; exit 1; }
fi
echo "--- tree ---" >> "$LOG"; git diff --stat >> "$LOG" 2>&1

ENVARGS=()
[ "$MODE" = "p1p2kill" ] && ENVARGS=(VLLM_DISABLE_ROCM_FUSED_GATED_RMSNORM=1)

docker ps --format "{{.Names}}" | grep -E "^vllm_h2q355_" | xargs -r docker rm -f >/dev/null 2>&1
sleep 5
env "${ENVARGS[@]}" $S/serve_vllm_m.sh off "$PORT" >> "$LOG" 2>&1
if ! $S/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
  echo "NOT READY" >> "$LOG"; docker logs --tail 30 "$NAME" >> "$LOG" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
docker logs "$NAME" 2>&1 | grep -E "is ENABLED|is DISABLED|Using [A-Z_]+ backend out of potential" | tail -3 >> "$LOG"
$S/bench_cell_m.sh "h2q355_${MODE}_isl8192_osl1024_c1" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "MODE $MODE DONE $(date -u +%H:%M:%S)" >> "$LOG"
