#!/bin/bash

RUN_CONFIG="${1:?Usage: $0 <run_config.json>}"

source auto_profile/utils.sh
load_run_config "$RUN_CONFIG"

clean_dir_contents ${SGL_DOCKER_RESULTS_DIR}
write_run_metadata ${SGL_DOCKER_RESULTS_DIR} ${SGL_DOCKER_IMAGE}
write_framework_commit ${SGL_DOCKER_RESULTS_DIR} ${SGL}

log_info "Run profiles:"

for p in "${PROFILES[@]}"; do
    model=$p
    gpu_ids=${PROFILE_GPU_IDS[$p]}
    input_len=${PROFILE_INPUT_LENS[$p]}
    output_len=${PROFILE_OUTPUT_LENS[$p]}
    log_info "  Profiling:"
    log_info "    model      = ${model}"
    log_info "    gpu_ids    = ${gpu_ids}"
    log_info "    input_len  = ${input_len}"
    log_info "    output_len = ${output_len}"

    num_gpus=$(echo "${gpu_ids}" | awk -F',' '{print NF}')
    model_dirname=$(echo "${model}" | sed 's/\//__/g')

    model_dir=${SGL_DOCKER_RESULTS_DIR}/${model_dirname}
    create_dir_if_missing ${model_dir}
    
    for concurrency in ${PROFILE_CONCURRENCIES}; do
        (
            set -euo pipefail
            
            log_info "      Run concurrency = ${concurrency}"
            
            ((num_requests = concurrency * NUM_WAVES))

            EXTRA_PREPARE_CMDS=()
            EXTRA_RUN_FLAGS=""

            # Set extra env vars
            mode="none"
            if [[ -n "${PROFILE_SGL_MODES[$p]}" \
                && "${PROFILE_SGL_MODES[$p]}" != "none" ]]; then
                mode=${PROFILE_SGL_MODES[$p]}
                log_info "Set SGL MODE = ${mode}"
                source "${RUN_CONFIG_DIR}/modes/${SGL}/${mode}"
            fi
            mode_name="${mode%.*}"

            # Generate test ID
            test_name="$(
                make_test_name \
                    ${SGL} \
                    ${model_dirname} \
                    ${num_gpus} \
                    ${concurrency} \
                    ${input_len} \
                    ${output_len} \
                    ${mode_name}
                )"
            
            # Create test dir (for outputs)
            test_dir="$(
                make_test_dirname \
                    ${model_dir} \
                    ${test_name}
                )"
            create_dir_if_missing ${test_dir}
            
            # Create output filenames
            result_filename="$(
                make_result_filename \
                    ${test_dir} \
                    ${test_name}
                )"
            
            run_prepare_filename="$(
                make_prepare_log_filename \
                    ${test_dir} \
                    ${test_name}
                )"
            
            run_log_filename="$(
                make_run_log_filename \
                    ${test_dir} \
                    ${test_name}
                )"
            
            run_log_profile_filename="$(
                make_run_log_profile_filename \
                    ${test_dir} \
                    ${test_name}
                )"
            
            # This will speedup capture for low batch sizes
            CUDA_GRAPH_FLAG=""
            if (( concurrency < 64 )); then
                CUDA_GRAPH_FLAG="--cuda-graph-max-bs=64"
            fi

            # Set profile env vars
            profile_flag=""
            profile_acts=""
            profile_stage=""
            profile_prefix=""
            if is_trace_enabled sgl; then
                log_info "SGL profile is enabled."
                
                # Enable profile
                profile_flag="--profile"

                # Use NSYS
                profile_acts="--profile-activities CUDA_PROFILER"

                # Set profile stage
                profile_stage="--profile-stage=decode" # Can be either all, prefill or decode
                
                trace_file_prefix="$(
                    make_trace_file_prefix \
                        ${test_dir} \
                        ${test_name}
                    )"

                profile_prefix="nsys profile ${NSYS_DEFAULT_FLAGS} -o ${trace_file_prefix}"
            fi

            
            # Increase watchdog timeout to 30min to avoid timeout 
            # due to long 10-20min DeepGEMM compilations
            DIST_TIMEOUT_FLAG="--dist-timeout 1800"

            # Run main cmd
            run_cmd="python -m sglang.bench_one_batch \
                --model-path ${model} \
                --tp ${num_gpus} \
                --batch-size ${concurrency} \
                --input-len ${input_len} \
                --output-len ${output_len} \
                --result-filename ${result_filename} \
                --trust-remote-code \
                ${DIST_TIMEOUT_FLAG} \
                ${CUDA_GRAPH_FLAG} \
                ${EXTRA_RUN_FLAGS}"
                

            # Run prepare cmds
            if [[ ${#EXTRA_PREPARE_CMDS[@]} -gt 0 ]]; then
                log_info "RUN PREPARE"
                for prep_cmd in "${EXTRA_PREPARE_CMDS[@]}"; do
                    log_info "  Running: ${prep_cmd}"
                    run_and_log ${run_prepare_filename} \
                        bash -c "${prep_cmd}"
                done
            fi
            
            # Run main cmd
            log_info "RUN NORMAL"
            run_and_log ${run_log_filename} \
                env CUDA_VISIBLE_DEVICES=${gpu_ids} \
                ${run_cmd}
            
            if is_trace_enabled sgl; then
                log_info "RUN PROFILE"
                run_and_log ${run_log_profile_filename} \
                    env CUDA_VISIBLE_DEVICES=${gpu_ids} \
                    ${profile_prefix} ${run_cmd} $profile_flag $profile_acts $profile_stage

                # Convert nsys binary file to SQLite format (python can read it)
                nsys export \
                    --type=sqlite \
                    --output="${trace_file_prefix}.sqlite" \
                    ${trace_file_prefix}.nsys-rep
            fi  
        )    
    done
done

