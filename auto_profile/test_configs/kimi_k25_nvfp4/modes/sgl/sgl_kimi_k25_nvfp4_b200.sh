set -x

EXTRA_PREPARE_CMDS=(
    "pip install --upgrade transformers"
    "python -m pip install distro"
)

EXTRA_RUN_FLAGS="
    --mem-fraction-static=0.9 \
    --kv-cache-dtype fp8_e4m3 \
"

set +x