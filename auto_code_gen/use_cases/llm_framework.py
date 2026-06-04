import os

from auto_code_gen.use_cases.base import UseCase
from common.claude_utils import PipelineStep, claude_run
from common.utils import clear_vllm_source_tree

from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
    gen_TestPlanPrompt,
    gen_ReviewTestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
    TEST_PLAN_PREFIX,
)


# ---------------------------------------------------------------------------
# Prompt template constants -- moved from ClassVar defaults on the 7 prompt
# dataclasses in code_gen_prompts.py so they live with the use-case.
# ---------------------------------------------------------------------------

LLM_CODE_TRACE_TEMPLATE = """

{context}

<definitions>
<framework>
{framework}
</framework>
</definitions>

<instructions>
The goal of this task is to detect the <code_trace> inside <framework> that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>). Think hard for this task and follow these guidelines:
- Analyze and understand in-detail the improvement plan step <plan_step> by inspecting all relevant files and data.
- Detect the <code_trace> inside <framework> that is relevant for improvement plan step <plan_step>.
    - Find all of the trace from higher levels to lower levels of the code
    - Go deep, into the kernel source codes if needed, including the C/C++/CUDA source codes of the kernel, and the associated C/C++/python wrappers, including any third-party libraries. Do not miss anything.
    - I.e find all of the relevant code pieces in the code that are relevant here and activated
- For the detected <code_trace>, perform a detailed analysis of how exactly the <code_trace> executes for the following iteration modes:
    - decode-only runs
    - prefill-only runs
    - a mix of prefill and decode runs
- Document each execution mode with its details, step-by-step from the high-level point of the <code_trace> to the lower-level point. Provide all critical details, which includes:
    - For each function in the trace, provide its goal, info, inputs (+shapes), and outputs (+shapes)
    - For the lowest level functions, inspect their source code in detail, and understand their inputs/outputs, including shapes and function assumptions.
    - Document classes/objects and their relations
    - Be professional, clear and consice, while documenting the trace gradually, in an incremental way from top to bottom.
- Document how cuda graphs affect each execution mode as well and how it is handled in the codebase
</instructions>

<output>
- Dump results to <output_dir>/{output_file}
</output>

"""

LLM_CODE_PORT_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<disallowed_modules>
{disallowed_modules}
</disallowed_modules>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
</definition_explanations>

<instructions>
The goal of this task is to provide a high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside "target" framework by porting <code_trace> parts from the "source" framework to "target" framework. Think hard for this task and follow these guidelines:
- Analyze and understand in-detail the <code_trace> of the "source" framework from <code_trace_files>.
- Analyze and understand in-detail the <code_trace> of the "target" framework from <code_trace_files>.
- Compare in-detail these 2 code traces step-by-step by traversing their code pieces via the associated call-chains.
- Detect <code_trace> parts of the "source" framework to port to "target" framework. Maximize the amount of code parts that are ported AS IS with MINIMAL necessary changes. Take into account the execution modes (and their effects on inputs/outputs/shapes and call-chains):
    - decode-only
    - prefill-only
    - mixed prefill and decode
    - For each execution mode above, trace fully input/outputs/shapes over various sizes of tokens, including edge cases, to see that everything looks good and makes total sense in terms of correctness.
- The goal of the porting process is to COPY-PASTE from the "source" framework AS MUCH AS POSSIBLE. Keep things simple, and add only minimal code integration points that are actually required.
- Ensure maximal size parts are ported so that the integration points for each ported part are minimal in terms of code changes inside the "target" framework. I.e. prefer to port code as much as possible and not invent new code (to avoid potential new bugs).

- For each ported code part, ensure minimal changes to the part and minimal changes to the "target" framework.
- For each ported code part, if code adjustments are necessary due to "target" framework constraints, then try best to minimize these code adjustments as much as possible.
- For each ported code part, provide step-by-step integration coding details into "target" framework. Make sure to take care of decode-only, prefill-only and mix execution modes correctly.
- For each ported code part, provide porting idea documentation, with what is ported, why, what is unchanged, and what is changed, and why. Be clear, concise and professional.
- For any code part in "target" framework that is used here, ensure it is used correctly with the ported code parts.
    - Verify inputs/outputs for all execution modes: decode, prefill, and mix
    - Verify all constraints and dependencies

- For the porting process, if a faster kernel from the "source" framework is the reason for the speedups, then strictly follow these guidelines to copy-paste (vendor) the faster kernel to the "target" framework:
    - DO NOT import or use any of the <disallowed_modules>
    - Find faster kernel source codes on all levels:
        - The full C/C++/CUDA source code with the associated compilation/flags process.
        - C/C++/CUDA wrappers that are to propagate calls to the kernel.
    - Analyze and understand in-depth how kernels are implemented in the "target" framework and reuse this logic for the porting process of the faster kernel.
    - Copy-paste, vedor and re-implement this kernel in the "target" framework fully, including the C/C++/CUDA source code, compilation process, and its C/C++ and python wrappers. Make sure to copy-paste AS IS as much as possible to keep similar logic to the "source" framework.
    - If there is a similar kernel in the "target" framework already, then DO NOT REUSE IT. Instead, vendor and copy-paste the implementation as described above, since it is highly unlikely that the kernels are the same unless you can check the source codes and compare line by line.

- Ensure cuda graphs are handled properly for all execution modes
- Ensure the end-to-end multi-step plan is coherent, bug-free and works for all execution modes:
    - Sanity and verify shapes of inputs/outputs
    - Verify API usage is fully correct and coherent
    - Verify the lowest level parts are used correctly
    - Inspect and trace all of the necessary source code points that are sensitive.

- Ensure the new code in the "target" framework will run by default for the case that we care about.

- Provide risk analysis and potential sensitive breaking points

</instructions>

<output>
- Dump the high-level multi-step coding plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the work done to generate the coding plan in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

LLM_REVIEW_CODE_PORT_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <input_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the high-level multi-step coding plan in <input_file> (iteration {iteration}). Do the following and think hard:
- Understand in-detail the plan in <input_file>, and restate the plan step-by-step.
- Review the plan in-detail for execution modes (1) decode-only, (2) prefill-only, and (3) mixed.
    - Fully trace each execution mode with simulated propagation of inputs/outputs/shapes, and verify all shapes are correct with respect to function APIs.
    - If a low-level (or third party) kernel API is invoked, then verify all input parameters and assumptions of this kernel API. Go deep and analyze kernel's source code as well, by fetching it from whether it is located.
- Review that the plan fully implements the related improvement plan step, and the default execution triggers it.
- Review plan has correct memory management of CPU/GPU buffers in general and with respect to cuda graphs.
- Review proper scheduling constraints and separation between prefill/decode/mixed (if needed)
- Review misc issues:
    - incorrect assumptions
    - missing steps
    - bad ordering or sequencing
    - ambiguity or vagueness
    - missing edge cases
    - architectural risks
    - hidden dependencies

- For each issue found, document:
    - The affected part of the plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected plan with all issues fixed.
</instructions>

<output>
- Dump the corrected plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed to the code port plan, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""

LLM_TEST_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
</definition_explanations>

<instructions>
The goal of this task is to generate a multi-step testing plan for the implementation described in <code_port_plan_file> (iteration {iteration}). Do the following and think hard:
- Analyze and understand in-detail the coding plan in <code_port_plan_file>. Ensure to have a full understanding of the multi-step process, intended behaviors, features and implementation details.
- Analyze and understand in-detail the tests in "source" framework that are relevant for the <code_port_plan_file>, and the associated <code_trace> call-chains, params etc...
- Analyze and understand in-detail the tests in "target" framework that are relevant for the <code_port_plan_file> and the associated <code_trace> call-chains, params etc...
- For both frameworks, understand test structures, test code reuse patterns, baselines used, and main ideas used to implement these tests. Dive deep into details to get a good understanding of how testing works.
- Plan a sequence of unit tests to verify small and large changes. Ensure to cover execution modes: decode-only, prefill-only and mix. Also cover cuda graphs, and any input/output tensor behavior that is needed.
    - Provide FULL coverage.
    - Provide edge case testing.
    - And anything else that is critical to check for correctness and speed gains.
- Plan a sequence of end-to-end tests for <code_port_plan_file> that test the whole change from <code_port_plan_file> end-to-end.
    - For the logical part of the transformer block that was modified, create this WHOLE part in the test with relevant classes/objects/functions, and test it end-to-end for correctness and speed gains. Ensure to compare vs known baseline.
    - Make sure to have a test that actually runs a model (maybe a smaller version of <model>) and verifies it works.
- Plan a sequence of tests to verify the expected performance gains:
    - If a kernel was modified, then provide a kernel-level test to verify it is faster than before, with proper inputs/outputs and comparison vs known baseline.
    - Otherwise, plan the relevant minimal test to verify speed gains exist vs a known baseline.
</instructions>

<output>
- Dump the planned tests to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the test plan generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

LLM_REVIEW_TEST_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <input_file> is a file that describes the multi-step testing plan for the implementation in <code_port_plan_file>.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the multi-step testing plan in <input_file> (iteration {iteration}). Do the following and think hard:
- Understand in-detail the test plan in <input_file>, and restate the plan step-by-step.
- Verify the test plan covers all execution modes: decode-only, prefill-only, and mixed.
- Verify the test plan covers cuda graphs behavior.
- Verify the test plan has proper unit tests for each ported/modified component.
- Verify the test plan has end-to-end tests that exercise the full change.
- Verify the test plan has performance tests that measure the expected speedup.
- Review misc issues:
    - missing test coverage
    - incorrect test assumptions
    - missing edge cases
    - tests that would not actually verify correctness
    - tests that would not detect regressions
    - missing baselines for comparison

- For each issue found, document:
    - The affected part of the test plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected test plan with all issues fixed.
</instructions>

<output>
- Dump the corrected test plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""

LLM_CODE_GEN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <test_plan_file> is a file that describes the high-level multi-step testing plan for the implementation in <code_port_plan_file>.
</definition_explanations>

<instructions>
The goal of this task is to generate a code patch for the "target" framework that implements the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file> (iteration {iteration}). Do the following and think hard:
- Analyze and understand in-detail the coding plan in <code_port_plan_file>
- Analyze and understand in-detail the testing plan in <test_plan_file>
- Implement the plans in <code_port_plan_file> and <test_plan_file> EXACTLY as they described:
    - Follow strictly the code and test plans as described, DO NOT DIVERGE.
    - If there is new (ported/vendored) C/C++/CUDA kernel code, then proceed on implementing the porting as planned, execute on it fully even it is really complex.
    - Recompile the "target" when there are changes to C/C++/CUDA codes. Ensure re-compilation succeeds, else iterate and fix until full success. Do not skip this step.
    - For recompile, USE INCREMENTAL compilation for the "target" by running the following commands (based on https://docs.vllm.ai/en/latest/contributing/incremental_build/):
        1. python tools/generate_cmake_presets.py --force-overwrite
        2. cmake --preset release
        3. cmake --build --preset release --target install
    - For recompile, use all available CPUs to make the process as fast as possible.
    - Avoid recompilation from scratch, and instead use the incremental compilation process described above.
    - Make sure to take into account all execution modes and verify their <code_trace> inputs/outputs/shapes for decode-only, prefill-only and mixed.
- Apply the new code patch to "target" framework inside <target_source_code_dir>, and if needed, fully re-compile the codebase.
- IMPORTANT: ALL code modifications (source code, tests, build files) MUST be made exclusively inside <target_source_code_dir>. Do NOT modify any files outside of this directory.
- Run all of the tests, do NOT SKIP anything, and make sure ALL TESTS ARE RUNNING AND PASSING (with the fully recompiled codebase if needed). If there are failures, then fix, revisit, rewrite, and re-run everything again.
- Ensure the code, both main code and test code, is written professionally, clearly, well-documented, and well-formatted, while taking into account how code is written in the "target" framework.
- Review your work critically and fix issues

Feature Toggle and Startup Verification:
- The generated code MUST include the following two mechanisms:
    1. STARTUP LOG: Add a log message that prints during "target" framework startup (e.g., during model initialization or engine startup) when the new feature is active. The message must follow the format:
       `[AutoPerf] Plan step <plan_step>: <short_description_of_feature> is ENABLED`
       This log must appear in stdout/stderr so it can be detected in runtime logs.
    2. ENVIRONMENT VARIABLE DISABLE SWITCH: Add an environment variable named `VLLM_DISABLE_AUTO_PERF_PLAN_<plan_step>` (e.g., `VLLM_DISABLE_AUTO_PERF_PLAN_2`) that, when set to "1", disables the new feature entirely and falls back to the original code paths. When disabled, print:
       `[AutoPerf] Plan step <plan_step>: <short_description_of_feature> is DISABLED (env override)`
    - The env var check must happen early in the relevant code path (e.g., during model/layer initialization).
    - When the env var is NOT set or set to anything other than "1", the new feature runs by default.
</instructions>

<output>
- Dump the code patch to <output_dir>/{output_patch_file} (it must have both code and its tests inside)
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the code patch and tests generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
    - The exact environment variable name used to disable the new feature (e.g., VLLM_DISABLE_AUTO_PERF_PLAN_2)
    - The exact startup log message that confirms the feature is enabled
    - The exact startup log message that confirms the feature is disabled
</output>

"""

LLM_REVIEW_CODE_GEN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<input_patch_file>
{input_patch_file}
</input_patch_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <test_plan_file> is a file that describes the high-level multi-step testing plan for the implementation in <code_port_plan_file>.
- <input_patch_file> is the code patch for <target_framework> that implements the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file>.
- <input_summary_file> is a file that summarizes the work done to generate <input_patch_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the code patch in <input_patch_file> (iteration {iteration}). Do the following and think hard:
- Understand in-detail the code in <input_patch_file>, and restate the coding process step-by-step.
- Review the code in-detail for execution modes (1) decode-only, (2) prefill-only, and (3) mixed.
    - Fully trace each execution mode with simulated propagation of inputs/outputs/shapes, and verify all shapes are correct with respect to function APIs.
    - If a low-level (or third party) kernel API is invoked, then verify all input parameters and assumptions of this kernel API. Go deep and analyze kernel's source code as well, by fetching it from whether it is located.
- Review that the code fully implements the related improvement plan step, and the default execution triggers it.
- Review plan has correct memory management of CPU/GPU buffers in general and with respect to cuda graphs.
- Review proper scheduling constraints and separation between prefill/decode/mixed (if needed)
- Review misc issues:
    - incorrect assumptions
    - missing steps
    - bad ordering or sequencing
    - ambiguity or vagueness
    - missing edge cases
    - architectural risks
    - hidden dependencies

- For each issue found, document:
    - The affected part of the plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected code patch with all issues fixed.
- Add additional tests if needed to cover new issues, or fixed issues.
- IMPORTANT: ALL code modifications (source code, tests, build files) MUST be made exclusively inside <target_source_code_dir>. Do NOT modify any files outside of this directory.
</instructions>

<output>
- Dump the corrected code patch to <output_dir>/{output_patch_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""


# ---------------------------------------------------------------------------
# LLMFrameworkUseCase -- orchestrates the LLM framework porting pipeline
# ---------------------------------------------------------------------------

class LLMFrameworkUseCase(UseCase):
    """Use case for porting optimisations between LLM serving frameworks."""

    def create_context_str(self, claude_config, config):
        return create_context_str(claude_config, config)

    def clear_target_cmd(self, config):
        return {
            "fn": lambda: clear_vllm_source_tree(config.source_code_dir),
            "fn_name": 'clear_vllm_source_tree("{}")'.format(config.source_code_dir),
        }

    async def run_runtime_iterations(
        self, context, config, claude_config, code_trace_files,
        code_port_plan_file, test_plan_file, resume=False,
    ):
        from auto_code_gen.code_gen_prompts import (
            gen_IterationHistorySummaryPrompt,
            gen_FindSmallerModelPrompt,
            RUNTIME_SMALLER_MODEL_FILE,
        )
        from auto_code_gen.run_code_gen import run_runtime_iterations as _run_runtime_iters

        output_dir = config.output_dir
        all_step_timings = []

        print("Generating iteration history summary before runtime iterations...")
        history_prompt = gen_IterationHistorySummaryPrompt(
            context=context,
            code_gen_output_dir=output_dir,
        )
        steps_history = [
            PipelineStep(
                name="iteration_history_summary",
                prompt=history_prompt.prompt(),
                output_files=[history_prompt.output_file],
            ),
        ]
        timings = await claude_run(claude_config, steps_history)
        all_step_timings.extend(timings)

        smaller_model_file = None
        if config.use_smaller_model_for_runtime:
            smaller_model_path = os.path.join(output_dir, RUNTIME_SMALLER_MODEL_FILE)
            if os.path.isfile(smaller_model_path) and os.path.getsize(smaller_model_path) > 0:
                print("Reusing existing smaller model selection ({})".format(
                    RUNTIME_SMALLER_MODEL_FILE
                ))
                smaller_model_file = RUNTIME_SMALLER_MODEL_FILE
            else:
                print("Finding smaller model for runtime iterations...")
                smaller_prompt = gen_FindSmallerModelPrompt(
                    context=context,
                    code_trace_files=code_trace_files,
                    code_port_plan_file=code_port_plan_file,
                )
                steps_smaller = [
                    PipelineStep(
                        name="find_smaller_model",
                        prompt=smaller_prompt.prompt(),
                        output_files=[smaller_prompt.output_file],
                    ),
                ]
                timings = await claude_run(claude_config, steps_smaller)
                all_step_timings.extend(timings)
                smaller_model_file = smaller_prompt.output_file

        runtime_phase_results, runtime_timings = await _run_runtime_iters(
            context, code_trace_files, config, claude_config,
            code_port_plan_file, test_plan_file,
            history_prompt.output_file, resume=resume,
            smaller_model_file=smaller_model_file,
            disable_new_feature=config.disable_new_feature_for_runtime,
        )
        all_step_timings.extend(runtime_timings)

        return runtime_phase_results, all_step_timings

    # -- Phase 1: code traces ------------------------------------------------

    def gen_code_trace_steps(self, context, config):
        source_fw = config.source_framework
        target_fw = config.target_framework

        source_prompt = gen_CodeTracePrompt(
            context=context,
            framework=source_fw,
            prompt_template=LLM_CODE_TRACE_TEMPLATE,
        )
        target_prompt = gen_CodeTracePrompt(
            context=context,
            framework=target_fw,
            prompt_template=LLM_CODE_TRACE_TEMPLATE,
        )

        code_trace_files = [
            source_prompt.output_file,
            target_prompt.output_file,
        ]

        steps = [
            PipelineStep(
                name="clear_target_dir",
                prompt=self.clear_target_cmd(config),
            ),
            PipelineStep(
                name="code_trace_{}".format(source_fw),
                prompt=source_prompt.prompt(),
                output_files=[source_prompt.output_file],
            ),
            PipelineStep(
                name="code_trace_{}".format(target_fw),
                prompt=target_prompt.prompt(),
                output_files=[target_prompt.output_file],
            ),
        ]

        return steps, code_trace_files

    # -- Phase 2: code port plan iterations -----------------------------------

    def gen_code_port_plan_iter_steps(
        self, context, config, code_trace_files,
        prev_output_file, prev_output_summary_file, iteration,
    ):
        plan_prompt = gen_CodePortPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            disallowed_modules=config.disallowed_modules,
            output_file="{}_V{}.txt".format(CODE_PORT_PLAN_FILE_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
            prompt_template=LLM_CODE_PORT_PLAN_TEMPLATE,
        )

        review_prompt = gen_ReviewCodePortPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            input_file=plan_prompt.output_file,
            input_summary_file=plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, iteration,
            ),
            iteration=iteration,
            prompt_template=LLM_REVIEW_CODE_PORT_PLAN_TEMPLATE,
        )

        steps = [
            PipelineStep(
                name="code_port_plan_V{}".format(iteration),
                prompt=plan_prompt.prompt(),
                output_files=[plan_prompt.output_file, plan_prompt.output_summary_file],
            ),
            PipelineStep(
                name="code_port_plan_review_V{}".format(iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_prompt.output_file, review_prompt.output_summary_file],
            ),
        ]

        return steps, review_prompt

    # -- Phase 3: test plan iterations ----------------------------------------

    def gen_test_plan_iter_steps(
        self, context, config, code_trace_files,
        code_port_plan_file,
        prev_output_file, prev_output_summary_file, iteration,
    ):
        test_plan_prompt = gen_TestPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            output_file="{}_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
            output_summary_file="{}_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, iteration,
            ),
            prev_output_file=prev_output_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
            prompt_template=LLM_TEST_PLAN_TEMPLATE,
        )

        review_prompt = gen_ReviewTestPlanPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            input_file=test_plan_prompt.output_file,
            input_summary_file=test_plan_prompt.output_summary_file,
            output_file="{}_review_V{}.txt".format(TEST_PLAN_PREFIX, iteration),
            output_summary_file="{}_review_summary_V{}.txt".format(
                TEST_PLAN_PREFIX, iteration,
            ),
            iteration=iteration,
            prompt_template=LLM_REVIEW_TEST_PLAN_TEMPLATE,
        )

        steps = [
            PipelineStep(
                name="test_plan_V{}".format(iteration),
                prompt=test_plan_prompt.prompt(),
                output_files=[test_plan_prompt.output_file, test_plan_prompt.output_summary_file],
            ),
            PipelineStep(
                name="test_plan_review_V{}".format(iteration),
                prompt=review_prompt.prompt(),
                output_files=[review_prompt.output_file, review_prompt.output_summary_file],
            ),
        ]

        return steps, review_prompt

    # -- Phase 4: code gen iterations -----------------------------------------

    def gen_code_gen_iter_steps(
        self, context, config, code_trace_files,
        code_port_plan_file, test_plan_file,
        prev_output_patch_file, prev_output_summary_file, iteration,
    ):
        code_gen_prompt = gen_CodeGenPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            output_patch_file="{}_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            prev_output_patch_file=prev_output_patch_file,
            prev_output_summary_file=prev_output_summary_file,
            iteration=iteration,
            prompt_template=LLM_CODE_GEN_TEMPLATE,
        )

        code_review_prompt = gen_ReviewCodeGenPrompt(
            context=context,
            code_trace_files=code_trace_files,
            code_port_plan_file=code_port_plan_file,
            test_plan_file=test_plan_file,
            input_patch_file=code_gen_prompt.output_patch_file,
            input_summary_file=code_gen_prompt.output_summary_file,
            output_patch_file="{}_review_V{}.patch".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            output_summary_file="{}_review_summary_V{}.txt".format(
                CODE_GEN_FILE_PREFIX, iteration,
            ),
            iteration=iteration,
            prompt_template=LLM_REVIEW_CODE_GEN_TEMPLATE,
        )

        steps = [
            PipelineStep(
                name="clear_target_dir_code_gen_V{}".format(iteration),
                prompt=self.clear_target_cmd(config),
            ),
            PipelineStep(
                name="code_gen_V{}".format(iteration),
                prompt=code_gen_prompt.prompt(),
                output_files=[code_gen_prompt.output_patch_file, code_gen_prompt.output_summary_file],
            ),
            PipelineStep(
                name="code_gen_review_V{}".format(iteration),
                prompt=code_review_prompt.prompt(),
                output_files=[code_review_prompt.output_patch_file, code_review_prompt.output_summary_file],
            ),
        ]

        return steps, code_review_prompt
