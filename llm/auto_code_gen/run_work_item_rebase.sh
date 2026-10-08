#!/bin/bash

source "$(dirname "$0")/../auto_profile/utils.sh"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Run
python -m llm.auto_code_gen.run_work_items \
    --code-gen-dir "$PROJECT_ROOT"/llm/auto_code_gen/prs_step_1 \
    --work-items-file "$PROJECT_ROOT"/llm/auto_code_gen/prs_step_1/work_item_rebase.txt \

