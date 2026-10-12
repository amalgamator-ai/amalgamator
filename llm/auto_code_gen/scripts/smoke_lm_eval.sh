#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# smoke_lm_eval.sh — smoke test for lm_eval in the pod-local venv.
#
# Runs the same venv setup as run_iteration.sh, installs lm-eval[api]
# the same way, then verifies the full import chain works:
#   import lm_eval
#   import vllm
#   import transformers (which may pull in torchvision)
#
# This catches dependency conflicts (torch/torchvision ABI mismatch,
# missing operators, etc.) before a real benchmark wastes GPU time.
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
echo "[smoke-lm-eval] out=$OUT vllm=$VLLM_DIR" | tee $OUT/runner.log

BUILD_FAIL_STATUS="SMOKE_BUILD_FAILED"
STATUS_FILE="$OUT/smoke_status.txt"

# === INCLUDE: venv_compile.sh ===

# ── install lm-eval (same steps as run_iteration.sh) ─────────────────
echo "[smoke-lm-eval] installing lm-eval[api]..." | tee -a $OUT/runner.log
uv pip install "lm-eval[api]" 2>&1 | tee -a $OUT/build.log

# ── verify imports ───────────────────────────────────────────────────
echo "[smoke-lm-eval] verifying imports..." | tee -a $OUT/runner.log

python3 -c "
import sys

failures = []

# 1. vllm
try:
    import vllm
    print('OK  import vllm ({})'.format(vllm.__version__))
except Exception as e:
    failures.append('import vllm: {}'.format(e))
    print('FAIL import vllm: {}'.format(e))

# 2. lm_eval
try:
    import lm_eval
    print('OK  import lm_eval ({})'.format(lm_eval.__version__))
except Exception as e:
    failures.append('import lm_eval: {}'.format(e))
    print('FAIL import lm_eval: {}'.format(e))

# 3. transformers (pulls in torchvision on some versions)
try:
    import transformers
    print('OK  import transformers ({})'.format(transformers.__version__))
except Exception as e:
    failures.append('import transformers: {}'.format(e))
    print('FAIL import transformers: {}'.format(e))

# 4. torchvision (the actual failure point for the nms operator crash)
try:
    import torchvision
    print('OK  import torchvision ({})'.format(torchvision.__version__))
except Exception as e:
    failures.append('import torchvision: {}'.format(e))
    print('FAIL import torchvision: {}'.format(e))

# 5. torch version check
try:
    import torch
    tv_version = getattr(torchvision, '__version__', 'N/A') if 'torchvision' in dir() else 'not imported'
    print('INFO torch={} torchvision={}'.format(torch.__version__, tv_version))
except Exception:
    pass

if failures:
    print('\\n{} import(s) FAILED:'.format(len(failures)))
    for f in failures:
        print('  - {}'.format(f))
    sys.exit(1)
else:
    print('\\nAll imports OK')
" 2>&1 | tee -a $OUT/runner.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "SMOKE_LM_EVAL_IMPORT_FAILED" > $OUT/smoke_status.txt
    echo "[smoke-lm-eval] FAILED — imports failed" | tee -a $OUT/runner.log
    exit 1
fi

echo "SMOKE_OK" > $OUT/smoke_status.txt
echo "[smoke-lm-eval] OK" | tee -a $OUT/runner.log
