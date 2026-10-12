#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

source "$(dirname "$0")/../auto_profile/utils.sh"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

PRS_DIR=""$PROJECT_ROOT"/llm/auto_code_gen/prs"
ISSUE_DIR="issue_1"

# Run
python -m llm.auto_code_gen.run_fix_issue \
    --high_level_code_plan_file ${PRS_DIR}/high_level_code_plan.txt \
    --prs_dir ${PRS_DIR} \
    --issue_to_fix_file ${PRS_DIR}/$ISSUE_DIR/issue_to_fix_run_log.txt \
    --issue_cwd ${PRS_DIR}/$ISSUE_DIR \
