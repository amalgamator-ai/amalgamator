set -x

EXTRA_PREPARE_CMDS=()

EXTRA_RUN_FLAGS=" \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --no-enable-flashinfer-autotune \
    --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2} \
"

#--attention-config {\"use_fp4_indexer_cache\":true} \
#--attention_config.use_fp4_indexer_cache=True \

set +x