#!/bin/bash

RUN_CONFIG="${1:?Usage: $0 <run_config.json>}"

source auto_profile/utils.sh
load_run_config "$RUN_CONFIG"

clean_dir_contents ${VLLM_DOCKER_RESULTS_DIR}
write_run_metadata ${VLLM_DOCKER_RESULTS_DIR} ${VLLM_DOCKER_IMAGE}
write_framework_commit ${VLLM_DOCKER_RESULTS_DIR} ${VLLM}

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

    model_dir=${VLLM_DOCKER_RESULTS_DIR}/${model_dirname}
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
            if [[ -n "${PROFILE_VLLM_MODES[$p]}" \
                && "${PROFILE_VLLM_MODES[$p]}" != "none" ]]; then
                mode=${PROFILE_VLLM_MODES[$p]}
                log_info "Set VLLM MODE = ${mode}"
                source "${RUN_CONFIG_DIR}/modes/${VLLM}/${mode}"
            fi
            mode_name="${mode%.*}"

            # Set attn backend
            attn_backend=""
            if [[ -v VLLM_ATTN_BACKEND && -n "$VLLM_ATTN_BACKEND" ]]; then
                log_info "Set attn backend to ${VLLM_ATTN_BACKEND}"
                attn_backend="--attention-backend ${VLLM_ATTN_BACKEND}"
            fi

            # Generate test ID
            test_name="$(
                make_test_name \
                    ${VLLM} \
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
            
            # Set profile vars (if enabled)
            profile_flag=""
            profile_json=""
            profile_prefix=""
            if is_trace_enabled vllm; then
                log_info "VLLM profile is enabled."
                
                num_warmups=0
                start_iter=$(calc_start_iter ${num_warmups} ${NUM_WAVES} ${output_len})

                printf -v profiler_config_json \
                    '{"profiler":"cuda","delay_iterations":%d,"max_iterations":%d}' \
                    "$start_iter" "$NUM_TRACE_ITERS"

                profile_flag="--profile"
                profile_json="--profiler-config ${profiler_config_json}"
                
                trace_file_prefix="$(
                    make_trace_file_prefix \
                        ${test_dir} \
                        ${test_name}
                    )"
                
                profile_prefix="nsys profile ${NSYS_DEFAULT_FLAGS} -o ${trace_file_prefix}"
            fi

            # TODO: [AlexM] Fine-tune more
            max_model_len=$(( 1024 * 128 ))

            # Run
            run_cmd="vllm bench throughput \
                --async-engine \
                --backend vllm \
                --model ${model} \
                ${attn_backend} \
                --tensor-parallel-size ${num_gpus} \
                --dataset-name random \
                --random-input-len ${input_len} \
                --random-output-len ${output_len} \
                --max-num-seqs ${concurrency} \
                --num-prompts ${num_requests} \
                --max-model-len ${max_model_len} \
                --output-json ${result_filename} \
                --trust-remote-code \
                --async-scheduling \
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
            
            if is_trace_enabled vllm; then
                log_info "RUN PROFILE"
                run_and_log ${run_log_profile_filename} \
                    env CUDA_VISIBLE_DEVICES=${gpu_ids} \
                    ${profile_prefix} ${run_cmd} ${profile_flag} ${profile_json}

                # Convert nsys binary file to SQLite format (python can read it)
                nsys export \
                    --type=sqlite \
                    --output="${trace_file_prefix}.sqlite" \
                    ${trace_file_prefix}.nsys-rep
            fi
        )
    done
done

