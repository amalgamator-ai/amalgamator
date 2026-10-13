# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

set -x

EXTRA_PREPARE_CMDS=(
    "pip install pandas"
)
EXTRA_RUN_FLAGS="
    --compilation_config.pass_config.fuse_allreduce_rms true \
"

# MoE
export VLLM_USE_FLASHINFER_MOE_FP4=1


set +x
