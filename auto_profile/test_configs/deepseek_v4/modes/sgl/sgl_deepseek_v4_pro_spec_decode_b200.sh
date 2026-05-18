set -x

EXTRA_PREPARE_CMDS=(
    "pip install --upgrade transformers"
    "python -m pip install distro"
)

EXTRA_RUN_FLAGS="
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algo EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 8192 \
    --disable-flashinfer-autotune \
    --swa-full-tokens-ratio 0.1 \
    --mem-fraction-static 0.90 \
"

export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

set +x