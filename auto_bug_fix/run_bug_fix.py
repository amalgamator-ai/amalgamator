"""Main orchestrator for the auto_bug_fix pipeline.

Usage:
    Edit ``auto_bug_fix/bug_fix_config.py`` to configure your project, then run:

        python -m auto_bug_fix.run_bug_fix

The pipeline steps are:
    1. Reset target branch (git reset + clean)
    2. Trace what the fix commit touches on the source branch
    3. Port tests from the fix commit (if config.port_tests is True)
    4. Plan how to apply the fix on the target branch (N review iterations)
    5. Validate test coverage; plan supplemental tests
    6. Generate the fix patch and apply it (M review iterations)
    7. Autonomous build-test-fix loop until clean or retry limit hit
"""

import sys
import time
import asyncio

from common.utils import Tee, clear_repo
from common.claude_utils import ClaudeConfig, claude_run

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_TestPortPrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
    gen_TestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
    gen_RunAndFixPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PORT_MANIFEST_FILE,
    RUN_AND_FIX_FAILURE_FILE,
)

LOG_FILE = "__run_log_bug_fix.txt"
NUM_PLAN_REVIEWS = 4
NUM_CODE_REVIEWS = 3


def gen_prompts(
    claude_config: ClaudeConfig | None = None,
    config: BugFixConfig | None = None,
) -> list:
    if claude_config is None:
        from auto_bug_fix.bug_fix_config import claude_config as _default
        claude_config = _default
    if config is None:
        from auto_bug_fix.bug_fix_config import bug_fix_config as _default
        config = _default

    clear_repo_cmd = {"cmd": 'clear_repo("{}")'.format(config.repo_path)}
    context = create_context_str(claude_config, config)
    branches = [config.source_branch, config.target_branch]

    # Step 2: Code trace on source branch
    code_trace_prompt = gen_CodeTracePrompt(context=context, source_branch=config.source_branch)
    branch_code_trace_files = [code_trace_prompt.output_file]

    # Step 3: Port tests from fix commit
    test_port_prompt = gen_TestPortPrompt(
        context=context,
        source_fix_commit=config.source_fix_commit,
        target_branch=config.target_branch,
    )

    # Step 4: CodePortPlan iterations
    code_port_plan_and_review_prompts = []
    previous_attempt_file = ""
    for i in range(NUM_PLAN_REVIEWS):
        plan_file = "{}_V{}_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        review_file = "{}_V{}_review_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        fixed_file = "{}_V{}_fixed_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        evolution_file = "{}_V{}_total_review_evolution_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )

        plan_prompt = gen_CodePortPlanPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            disallowed_modules=config.disallowed_modules,
            previous_attempt_file=previous_attempt_file,
            output_file=plan_file,
        )
        review_prompt = gen_ReviewCodePortPlanPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            code_port_plan_file=plan_file,
            output_review_file=review_file,
            output_fixed_file=fixed_file,
            output_total_review_summary_file=evolution_file,
        )
        previous_attempt_file = fixed_file
        code_port_plan_and_review_prompts.append(plan_prompt.prompt())
        code_port_plan_and_review_prompts.append(review_prompt.prompt())

    final_plan_file = previous_attempt_file

    # Step 5: Test plan
    test_plan_prompt = gen_TestPlanPrompt(
        context=context,
        branches=branches,
        branch_code_trace_files=branch_code_trace_files,
        code_port_plan_file=final_plan_file,
        test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
    )

    # Step 6: CodeGen iterations
    code_gen_and_review_prompts = []
    previous_code_gen_file = ""
    for i in range(NUM_CODE_REVIEWS):
        pr_info_file = "{}_V{}_PR_INFO_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_file = "{}_V{}_PR_from_{}_to_{}.patch".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_review_file = "{}_V{}_PR_REVIEW_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_fixed_file = "{}_V{}_PR_FIXED_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )
        pr_evolution_file = "{}_V{}_PR_TOTAL_REVIEW_EVOLUTION_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, branches[0], branches[1]
        )

        gen_prompt = gen_CodeGenPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            code_port_plan_file=final_plan_file,
            test_plan_file=test_plan_prompt.output_file,
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            previous_attempt_file=previous_code_gen_file,
            output_info_file=pr_info_file,
            output_pr_file=pr_file,
        )
        review_prompt = gen_ReviewCodeGenPrompt(
            context=context,
            branches=branches,
            branch_code_trace_files=branch_code_trace_files,
            code_port_plan_file=final_plan_file,
            test_plan_file=test_plan_prompt.output_file,
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            code_pr_info_file=pr_info_file,
            code_pr_file=pr_file,
            output_review_file=pr_review_file,
            output_fixed_file=pr_fixed_file,
            output_total_review_summary_file=pr_evolution_file,
        )
        previous_code_gen_file = pr_fixed_file
        code_gen_and_review_prompts.append(clear_repo_cmd)
        code_gen_and_review_prompts.append(gen_prompt.prompt())
        code_gen_and_review_prompts.append(review_prompt.prompt())

    # Step 7: RunAndFix
    run_and_fix_prompt = gen_RunAndFixPrompt(
        context=context,
        build_command=config.build_command,
        test_command=config.test_command,
        build_dir=config.build_dir,
        max_build_test_retries=config.max_build_test_retries,
    )

    # Assemble
    prompts: list = []
    prompts.append(clear_repo_cmd)
    prompts.append(code_trace_prompt.prompt())
    if config.port_tests:
        prompts.append(test_port_prompt.prompt())
    prompts.extend(code_port_plan_and_review_prompts)
    prompts.append(test_plan_prompt.prompt())
    prompts.extend(code_gen_and_review_prompts)
    prompts.append(run_and_fix_prompt.prompt())

    return prompts


if __name__ == "__main__":
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    from auto_bug_fix.bug_fix_config import claude_config, bug_fix_config

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(claude_config, bug_fix_config)))
    duration_time = time.time() - start_time
    print("FINISHED ALL: total_duration = {}".format(duration_time))
