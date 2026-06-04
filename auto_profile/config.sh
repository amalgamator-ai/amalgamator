AUTO_PROFILE_DIR="auto_profile"
INFRA_CONFIG="${AUTO_PROFILE_DIR}/test_configs/infra_config.json"

verify_cwd() {
    local cwd
    cwd="$(pwd)"

    local missing=0

    if [[ ! -d "$cwd/${AUTO_PROFILE_DIR}" ]]; then
        if [[ ! -d "$cwd/$dir" ]]; then
            echo "Error: Required directory '$dir' not found in $cwd" >&2
            missing=1
        fi
    fi

    if [[ $missing -ne 0 ]]; then
        return 1
    fi

    return 0
}

# Dirs
if ! verify_cwd; then
  exit 1
fi
BASE_DIR="$(pwd)"

DATASETS_DIR="datasets"
TRACES_DIR="traces"

# GPU Types
B200="b200"
H200="h200"

# Docker
DOCKER_PORT=30000
DOCKER_HF_HUB_CACHE="/app/hf_hub_cache"
DOCKER_HF_HOME="/app/hf_home"
DOCKER_BASE_DIR="/app/ai_auto_analyze"

# Framework names
SGL="sgl"
VLLM="vllm"
TRT="trt"

# Docker dirs
DOCKER_AUTO_PROFILE_DIR="${DOCKER_BASE_DIR}/${AUTO_PROFILE_DIR}"

# General test configs
NUM_WARMUPS=2
NUM_WAVES=4
NUM_TRACE_ITERS=50

NSYS_DEFAULT_FLAGS=" \
  -t cuda,nvtx \
  -c cudaProfilerApi \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
"

