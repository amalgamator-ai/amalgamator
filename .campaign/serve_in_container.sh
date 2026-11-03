#!/usr/bin/env bash
# Run a serving command inside the pinned ROCm 10 image, with a source tree
# mounted so the container serves the tree under test instead of its own build.
#
# The codegen runtime gate compares a baseline arm and a patched arm; both must
# run in the identical environment, and both must actually contain the patch.
# Letting the agent assemble `docker run` by hand is how a floating tag and a
# guessed mount sneak in.
#
# Env:
#   CODEGEN_TREE    host path of the vLLM tree to mount (default: the P1 base)
#   CONTAINER_NAME  container name (default: vllm_codegen)
# usage:
#   serve_in_container.sh "vllm serve MODEL --port 8123 ..."
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for envfile in "$HERE/images.env" "$HERE/../images.env"; do
  [ -f "$envfile" ] && { source "$envfile"; break; }
done
[ -n "${VLLM_IMAGE:-}" ] || { echo "images.env not found next to $0" >&2; exit 1; }

TREE="${CODEGEN_TREE:-/home/amatveev/work/vllm_e0d}"
NAME="${CONTAINER_NAME:-vllm_codegen}"
HF="${HF_DIR:-/home/amatveev/qwen38/hf}"
cmd="${1:?usage: serve_in_container.sh \"<serving command>\"}"

[ -d "$TREE/vllm" ] || { echo "no vllm tree at $TREE" >&2; exit 1; }

exec docker run --rm --name "$NAME" --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined \
  -e PYTHONPATH=/workspace/vllm \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HF_HOME=/hf \
  -v "$HF":/hf \
  -v "$TREE":/workspace/vllm \
  --entrypoint bash "$VLLM_IMAGE" -c "$cmd"
