#!/bin/bash

source "$(dirname "$0")/../auto_profile/utils.sh"

PRS_DIR="/home/alexm-redhat/code/ai_auto_analyze/auto_code_gen/prs"

# Run
source "$(dirname "$0")/../env.sh"
python -m auto_code_gen.run_investigate_issue \
    --frameworks sglang vllm \
    --framework-code-trace-files $PRS_DIR/sglang_code_trace.txt $PRS_DIR/vllm_code_trace.txt \
    --code-port-plan-file $PRS_DIR/code_port_plan_V4_fixed_from_sglang_to_vllm.txt \
    --test-plan-file $PRS_DIR/test_plan_from_sglang_to_vllm.txt \
    --code-port-plan-review-evolution-file $PRS_DIR/code_port_plan_V4_total_review_evolution_from_sglang_to_vllm.txt \
    --code-pr-info-file $PRS_DIR/code_gen_V3_PR_INFO_from_sglang_to_vllm.txt \
    --code-pr-file $PRS_DIR/code_gen_V3_PR_from_sglang_to_vllm.patch \
    --code-pr-review-evolution-file $PRS_DIR/code_gen_V3_PR_TOTAL_REVIEW_EVOLUTION_from_sglang_to_vllm.txt \
    --issue-desc-file $PRS_DIR/issue_1.txt \
    --output-file-prefix issue_1_research
        
