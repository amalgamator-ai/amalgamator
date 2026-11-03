#!/usr/bin/env bash
# H2 test, corrected: the first attempt ran with the server's DEFAULT max_num_seqs
# (256), so P1's <=8 gate never fired and every arm measured the stock backend
# (15.06 ms). This version pins --max-num-seqs 1 and passes env vars explicitly.
#
#   h2_q355b.sh p1        tree = P1 only
#   h2_q355b.sh p1p2      tree = P1 + P2 patch
#   h2_q355b.sh p1p2kill  tree = P1 + P2 with VLLM_DISABLE_ROCM_FUSED_GATED_RMSNORM=1
set -u
MODE="${1:?p1|p1p2|p1p2kill}"
PORT=8100
NAME="h2b_vllm_${PORT}"
OUT=/home/amatveev/amatveev_h2
LOG=~/qwen38/logs/h2b_q355_${MODE}.log
VLLM=/home/amatveev/work/vllm
P1=~/qwen38/patches/p1_aiter_unified_small_batch.patch
P2=~/amalgamator/llm/auto_code_gen/results/qwen38_27b_mi300x_decode8k_p2/code_gen/V2_review.patch
mkdir -p "$OUT"
: > "$LOG"

cd "$VLLM" || exit 1
git checkout -- . >/dev/null 2>&1; git clean -fd >/dev/null 2>&1
git apply "$P1" || { echo "P1 APPLY FAILED" >> "$LOG"; exit 1; }
[ "$MODE" != "p1" ] && { git apply "$P2" || { echo "P2 APPLY FAILED" >> "$LOG"; exit 1; }; }
echo "--- tree ($MODE) ---" >> "$LOG"; git diff --stat >> "$LOG" 2>&1

ENVARGS=()
[ "$MODE" = "p1p2kill" ] && ENVARGS=(-e VLLM_DISABLE_ROCM_FUSED_GATED_RMSNORM=1)

docker rm -f "$NAME" >/dev/null 2>&1
sleep 5
docker run -d --name "$NAME" --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  -e PYTHONPATH=/workspace/vllm \
  -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
  -v /home/amatveev/work/vllm:/workspace/vllm \
  "${ENVARGS[@]}" \
  vllm/vllm-openai-rocm:nightly-rocm100 \
  --model Qwen/Qwen3.5-35B-A3B-FP8 --served-model-name Qwen/Qwen3.5-35B-A3B-FP8 \
  --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
  --no-enable-prefix-caching --gpu-memory-utilization 0.90 \
  --max-model-len 65536 --max-num-seqs 1 >/dev/null

if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME" >> "$LOG" 2>&1; then
  echo "NOT READY" >> "$LOG"; docker logs --tail 30 "$NAME" >> "$LOG" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
docker logs "$NAME" 2>&1 | grep -E "is ENABLED|is DISABLED|Using [A-Z_]+ backend out of potential" | tail -4 >> "$LOG"

export MODEL=Qwen/Qwen3.5-35B-A3B-FP8 SERVED=Qwen/Qwen3.5-35B-A3B-FP8 TAG=h2b
~/qwen38/scripts/bench_cell_m.sh "h2b_${MODE}_isl8192_osl1024_c1" "$PORT" 8192 1024 1 8 >> "$LOG" 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
echo "MODE $MODE DONE $(date -u +%H:%M:%S)" >> "$LOG"
