#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# smoke_compile.sh — smoke test for the shared venv_compile.sh code path.
#
# Runs the same venv setup + cmake build that run_iteration.sh and
# run_diagnostic.sh use, then verifies `import vllm` works.
#
# Expected env vars:
#   OUT      — per-job output directory
#   VLLM_DIR — vllm source tree on the workspace PVC
#   HF_HOME  — HuggingFace cache directory

export UV_CACHE_DIR=/tmp/uv-cache
export TORCH_CUDA_ARCH_LIST="10.0"
export FLASHINFER_JIT_DIR=/tmp/flashinfer
export FLASHINFER_WORKSPACE_DIR=/tmp/flashinfer
export VLLM_CACHE_ROOT=/tmp/vllm-cache

mkdir -p "$OUT"
touch "$OUT/build.log" "$OUT/runner.log"
echo "[smoke] out=$OUT vllm=$VLLM_DIR HF_HOME=$HF_HOME" | tee $OUT/runner.log

BUILD_FAIL_STATUS="SMOKE_BUILD_FAILED"
STATUS_FILE="$OUT/smoke_status.txt"

# === INCLUDE: venv_compile.sh ===

# ── verify import ─────────────────────────────────────────────────────
echo "[smoke] verifying import vllm..." | tee -a $OUT/runner.log
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tee -a $OUT/runner.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "SMOKE_IMPORT_FAILED" > $OUT/smoke_status.txt
    echo "[smoke] FAILED — import vllm failed" | tee -a $OUT/runner.log
    exit 1
fi

echo "SMOKE_OK" > $OUT/smoke_status.txt
echo "[smoke] OK" | tee -a $OUT/runner.log
