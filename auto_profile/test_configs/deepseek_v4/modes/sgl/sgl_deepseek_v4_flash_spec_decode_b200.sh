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
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --swa-full-tokens-ratio 0.1 \
"

set +x