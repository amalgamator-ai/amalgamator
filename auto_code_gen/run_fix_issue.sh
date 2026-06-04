#!/bin/bash

source "$(dirname "$0")/../auto_profile/utils.sh"

PRS_DIR="/home/alexm-redhat/code/ai_auto_analyze/auto_code_gen/prs"
ISSUE_DIR="issue_1"

# Run
source "$(dirname "$0")/../env.sh"
python -m auto_code_gen.run_fix_issue \
    --high_level_code_plan_file ${PRS_DIR}/high_level_code_plan.txt \
    --prs_dir ${PRS_DIR} \
    --issue_to_fix_file ${PRS_DIR}/$ISSUE_DIR/issue_to_fix_run_log.txt \
    --issue_cwd ${PRS_DIR}/$ISSUE_DIR \
