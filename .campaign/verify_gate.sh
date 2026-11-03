#!/usr/bin/env bash
# Feature-gate verification: serve the patched tree with a chosen max_num_seqs
# and bench the cells that matter for that gate state.
#
#   verify_gate.sh <on|off> <max_num_seqs>
#
#   on  = patched selection active (no kill switch)
#   off = kill switch VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH=1
#
# The patch only changes behaviour when the engine is configured with
# max_num_seqs <= 8, so:
#   max_num_seqs <= 8  -> bench c1 and c8 (feature active in the 'on' arm)
#   max_num_seqs  > 8  -> bench c64      (feature inactive in BOTH arms; the two
#                                        arms must match exactly, proving the
#                                        high-concurrency path is untouched)
set -u
ARM="${1:?on|off}"
MNS="${2:?max_num_seqs}"
PORT=8123
NAME="verify_vllm_${PORT}"
# Keep the arm in its own variable: TAG is rewritten for result labels below, and
# an arm comparison keyed on TAG after that rewrite silently runs both arms with
# the feature enabled (observed once -- the kill-switch arm reported ENABLED).
export MODEL=Qwen/Qwen3.8-27B-FP8 SERVED=q38 TAG="verify_${ARM}_mns${MNS}"

docker rm -f "$NAME" >/dev/null 2>&1
sleep 5
ENVARGS=()
[ "$ARM" = "off" ] && ENVARGS=(-e VLLM_DISABLE_AITER_UNIFIED_SMALL_BATCH=1)

echo "[$ARM mns=$MNS] starting $(date -u +%H:%M:%S)"
docker run -d --name "$NAME" --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  -e PYTHONPATH=/workspace/vllm \
  -e VLLM_ROCM_USE_AITER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HF_HOME=/hf -v /home/amatveev/qwen38/hf:/hf \
  -v /home/amatveev/work/vllm:/workspace/vllm \
  -v /home/amatveev/qwen38/caches:/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  "${ENVARGS[@]}" \
  vllm/vllm-openai-rocm:nightly-rocm100 \
  --model Qwen/Qwen3.8-27B-FP8 --served-model-name q38 --port "$PORT" \
  --tensor-parallel-size 1 --no-enable-prefix-caching --trust-remote-code \
  --gpu-memory-utilization 0.90 --max-model-len 65536 --max-num-seqs "$MNS" >/dev/null

if ! ~/qwen38/scripts/wait_ready.sh "$PORT" 2400 "$NAME"; then
  echo "[$TAG mns=$MNS] NOT READY"; docker logs --tail 40 "$NAME" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi
echo "[$TAG mns=$MNS] ready $(date -u +%H:%M:%S)"
docker logs "$NAME" 2>&1 | grep -E "is (ENABLED|DISABLED)|Using [A-Z_]+ backend out of potential" | tail -2

L="verify_mns${MNS}_${TAG}"
if [ "$MNS" -le 8 ]; then
  ~/qwen38/scripts/bench_cell_m.sh "${L}_isl8192_osl1024_c1" "$PORT" 8192 1024 1 8  > ~/qwen38/logs/${L}_c1.log 2>&1
  ~/qwen38/scripts/bench_cell_m.sh "${L}_isl8192_osl512_c8"  "$PORT" 8192 512  8 32  > ~/qwen38/logs/${L}_c8.log 2>&1
else
  ~/qwen38/scripts/bench_cell_m.sh "${L}_isl8192_osl512_c64" "$PORT" 8192 512  64 128 > ~/qwen38/logs/${L}_c64.log 2>&1
fi
echo "[$TAG mns=$MNS] done $(date -u +%H:%M:%S)"
docker rm -f "$NAME" >/dev/null 2>&1
sleep 10
