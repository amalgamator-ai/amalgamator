set -x

# MoE
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND="latency" # "throughput"

set +x