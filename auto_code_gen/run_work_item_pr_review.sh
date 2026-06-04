#!/bin/bash

source "$(dirname "$0")/../auto_profile/utils.sh"

# Run
source "$(dirname "$0")/../env.sh"
python -m auto_code_gen.run_work_items \
    --code-gen-dir /home/alexm-redhat/code/ai_auto_analyze/auto_code_gen/prs_step_2 \
    --work-items-file /home/alexm-redhat/code/ai_auto_analyze/auto_code_gen/prs_step_2/work_item_pr_review.txt \

