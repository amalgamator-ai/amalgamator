import os
import sys
import time
import argparse
import asyncio

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run

from auto_code_gen.code_gen_configs import CodeGenConfig

from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
    gen_TestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PLAN_PREFIX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI code generation pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    return parser.parse_args()


def gen_prompts(code_gen_config, claude_config):
    clear_target_dir_cmd = {
        "cmd": 'utils.clear_vllm_source_tree("{}")'.format(
            code_gen_config.source_code_dir
        )
    }

    context = create_context_str(claude_config, code_gen_config)

    source_fw = code_gen_config.source_framework
    target_fw = code_gen_config.target_framework

    source_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=source_fw)
    target_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=target_fw)

    framework_code_trace_files = [
        source_code_trace_prompt.output_file,
        target_code_trace_prompt.output_file,
    ]

    num_code_port_plan_iterations = code_gen_config.num_code_port_plan_iterations
    code_port_plan_and_review_prompts = []
    prev_output_file = None
    prev_output_summary_file = None
    for i in range(num_code_port_plan_iterations):
        # Step 1: Generate
        code_port_plan_prompt = gen_CodePortPlanPrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            code_port_disallowed_modules=code_gen_config.code_port_disallowed_modules,
            output_file="{}_V{}.txt".format(CODE_PORT_PLAN_FILE_PREFIX, i + 1),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=i + 1,
        )

        # Step 2: Review and fix
        review_code_port_plan_prompt = gen_ReviewCodePortPlanPrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            input_file=code_port_plan_prompt.output_file,
            input_summary_file=code_port_plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1
            ),
            iteration=i + 1,
        )

        # Feed back for next iteration
        prev_output_file = review_code_port_plan_prompt.output_file
        prev_output_summary_file = review_code_port_plan_prompt.output_summary_file

        code_port_plan_and_review_prompts.append(code_port_plan_prompt.prompt())
        code_port_plan_and_review_prompts.append(review_code_port_plan_prompt.prompt())

    final_code_port_plan_file = prev_output_file

    num_test_plan_iterations = code_gen_config.num_test_plan_iterations
    test_plan_prompts = []
    prev_output_file = None
    prev_output_summary_file = None
    for i in range(num_test_plan_iterations):
        test_plan_prompt = gen_TestPlanPrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=final_code_port_plan_file,
            output_file="{}_V{}.txt".format(TEST_PLAN_PREFIX, i + 1),
            output_summary_file="{}_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, i + 1
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=i + 1,
        )
        prev_output_file = test_plan_prompt.output_file
        prev_output_summary_file = test_plan_prompt.output_summary_file
        test_plan_prompts.append(test_plan_prompt.prompt())

    final_test_plan_file = prev_output_file

    num_code_gen_iterations = code_gen_config.num_code_gen_iterations

    code_gen_and_review_prompts = []
    prev_output_patch_file = None
    prev_output_summary_file = None

    for i in range(num_code_gen_iterations):
        code_gen_prompt = gen_CodeGenPrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=final_code_port_plan_file,
            test_plan_file=final_test_plan_file,
            output_patch_file="{}_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, i + 1
            ),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, i + 1
            ),
            prev_output_patch_file=prev_output_patch_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=i + 1,
        )

        code_review_prompt = gen_ReviewCodeGenPrompt(
            context=context,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=final_code_port_plan_file,
            test_plan_file=final_test_plan_file,
            input_patch_file=code_gen_prompt.output_patch_file,
            input_summary_file=code_gen_prompt.output_summary_file,
            output_patch_file="{}_review_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, i + 1
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, i + 1
            ),
            iteration=i + 1,
        )
        prev_output_patch_file = code_review_prompt.output_patch_file
        prev_output_summary_file = code_review_prompt.output_summary_file

        code_gen_and_review_prompts.append(clear_target_dir_cmd)
        code_gen_and_review_prompts.append(code_gen_prompt.prompt())
        code_gen_and_review_prompts.append(code_review_prompt.prompt())

    prompts = []

    prompts.append(clear_target_dir_cmd)
    prompts.append(source_code_trace_prompt.prompt())
    prompts.append(target_code_trace_prompt.prompt())
    prompts.extend(code_port_plan_and_review_prompts)
    prompts.extend(test_plan_prompts)
    prompts.extend(code_gen_and_review_prompts)

    return prompts


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()

    setup_logging("code_gen")

    os.makedirs(code_gen_config.output_dir, exist_ok=True)
    safe_clean_dir(code_gen_config.output_dir)

    print("Preparing source code branches...")
    code_gen_config.prepare_branches()

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(code_gen_config, claude_config)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
