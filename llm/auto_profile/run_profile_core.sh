#!/bin/bash

RUN_CONFIG="${1:?Usage: $0 <run_config.json>}"

source llm/auto_profile/utils.sh

if [[ ! -f "$INFRA_CONFIG" ]]; then
    echo "Error: Infra config file not found: $INFRA_CONFIG" >&2
    exit 1
fi

if [[ ! -f "$RUN_CONFIG" ]]; then
    echo "Error: Run config file not found: $RUN_CONFIG" >&2
    exit 1
fi

load_run_config "$RUN_CONFIG"

host_results_dir="${AUTO_PROFILE_DIR}/${RESULTS_DIR}"
log_info "Pre-creating results directories at: ${host_results_dir}"
mkdir -p "${host_results_dir}/${VLLM}" "${host_results_dir}/${SGL}" "${host_results_dir}/${TRT}"

if is_framework_enabled sgl; then
    echo "RUN SGL DOCKER"
    echo "==============="
    time ./llm/auto_profile/sgl/sgl_run_docker.sh "$RUN_CONFIG"
fi

if is_framework_enabled vllm; then
    echo "RUN VLLM DOCKER"
    echo "==============="
    time ./llm/auto_profile/vllm/vllm_run_docker.sh "$RUN_CONFIG"
fi

if is_framework_enabled trt; then
    echo "RUN TRT DOCKER"
    echo "==============="
    time ./llm/auto_profile/trt/trt_run_docker.sh "$RUN_CONFIG"
fi
