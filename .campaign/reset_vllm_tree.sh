#!/bin/bash
# Reset the vLLM target tree to HEAD before a new codegen campaign step, so the
# next run's patch is standalone. Build dirs and installed extensions are
# git-ignored, so a plain checkout+clean (no -x) keeps them.
#
# Guarded on purpose: an unguarded `git checkout -- .` that lands in the wrong
# repository is destructive, so this refuses to run anywhere but the target tree.
set -eu

VLLM_DIR=/home/amatveev/work/vllm

if [ ! -f "$VLLM_DIR/vllm/platforms/rocm.py" ]; then
    echo "refusing: $VLLM_DIR does not look like the vLLM source tree" >&2
    exit 1
fi

cd "$VLLM_DIR"

echo "--- before ---"
git status --porcelain | head -20

git checkout -- .
git clean -fd

echo "--- after ---"
git status --porcelain | head -20
git log --oneline -1
