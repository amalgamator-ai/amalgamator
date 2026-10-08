#!/bin/bash

RUN_CONFIG="${1:?Usage: $0 <run_config.json>}"

source llm/auto_profile/utils.sh
load_run_config "$RUN_CONFIG"

nsys_flags=""
if is_trace_enabled sgl; then
    nsys_dir=$(find_nsys_dir)
    nsys_flags="-v ${nsys_dir}:/nsys:ro -e PATH=/nsys/bin:${PATH}"
fi

run_docker ${SGL} ${SGL_DOCKER_IMAGE} "${nsys_flags}" "${RUN_CONFIG}"
