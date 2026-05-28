from dataclasses import dataclass
from typing import ClassVar, Optional

from common.claude_utils import ClaudeConfig
from auto_code_gen.code_gen_configs import CodeGenConfig


def _prev_iteration_section(prev_output_file: Optional[str], prev_output_summary_file: Optional[str]) -> str:
    if prev_output_file is None:
        return ""
    return """
<prev_output_file>
{prev_output_file}
</prev_output_file>
<prev_output_summary_file>
{prev_output_summary_file}
</prev_output_summary_file>

Read and analyze in-detail <prev_output_file> and <prev_output_summary_file> which contain the results from the previous iterations. Understand what was done, what issues were found and fixed, and what the iteration evolution looks like. The current attempt is done from scratch, but must incorporate all learnings from the previous iterations to avoid repeating mistakes and to build on what worked.
""".format(prev_output_file=prev_output_file, prev_output_summary_file=prev_output_summary_file)


def create_context_str(claude_config: ClaudeConfig, code_gen_config: CodeGenConfig):
    return """
<context>

<output_dir>
{output_dir}
</output_dir>

<model>
{model}
</model>
<gpu_type>
{gpu_type}
</gpu_type>
<batch_size>
{batch_size}
</batch_size>
<isl>
{isl}
</isl>
<osl>
{osl}
</osl>

<tested_execution>
Execution of model <model> on <gpu_type> GPU with ISL <isl>, OSL <osl> and batch size <batch_size>
</tested_execution>

<framework_names>
{framework_names}
</framework_names>
<source_framework>
{source_framework}
</source_framework>
<target_framework>
{target_framework}
</target_framework>
<framework_source_codes>
{framework_source_codes}
</framework_source_codes>
<framework_test_dirs>
{framework_test_dirs}
</framework_test_dirs>
<transformer_block_high_level_ops_files>
{transformer_block_high_level_ops_files}
</transformer_block_high_level_ops_files>
<median_transformer_block_files>
{median_transformer_block_files}
</median_transformer_block_files>

<plan_file>
{plan_file}
</plan_file>
<plan_step>
{plan_step}
</plan_step>

<slower_framework>
{slower_framework}
</slower_framework>

<target_source_code_dir>
{target_source_code_dir}
</target_source_code_dir>

</context>

<definitions>
<code_trace>
code-paths, code-pieces, and their associated call-chains
</code_trace>
</definitions>

<context_explanations>
- <output_dir> is the output directory where all generated artifacts (plans, summaries, patches, test plans) must be saved. ALL output files MUST be written to this directory.

- <framework_names> is the list of frameworks involved.
- <source_framework> is the "source" framework from which code pieces are ported
- <target_framework> is the "target" framework to which code pieces are ported (same as <slower_framework>)
- <framework_source_codes> is the list of framework source codes for <framework_names> respectively.
- <framework_test_dirs> is the list of test directories for <framework_names> respectively, testing the <tested_execution>. Each directory has run log files that can be inspected to detect the active code pieces during the run of <tested_execution>.
- <transformer_block_high_level_ops_files> is the list of high-level transformer block operation files for <framework_names> respectively.
- <median_transformer_block_files> is the list of median low-level => high-level transformer block operation files for <framework_names> respectively.

- <target_source_code_dir> is the directory containing the "target" framework source code. ALL code modifications MUST be made exclusively inside this directory. Do NOT modify files outside of <target_source_code_dir>.

- The file <plan_file> has a sequence of improvement steps for <slower_framework> based on the comparison between frameworks running <tested_execution>.
- The improvement plan step <plan_step> from <plan_file> is what we want to implement for <slower_framework>
</context_explanations>

""".format(
        output_dir=claude_config.cwd,
        model=code_gen_config.model,
        gpu_type=code_gen_config.gpu_type,
        batch_size=code_gen_config.batch_size,
        isl=code_gen_config.isl,
        osl=code_gen_config.osl,
        framework_names=code_gen_config.framework_names,
        source_framework=code_gen_config.source_framework,
        target_framework=code_gen_config.target_framework,
        framework_source_codes=code_gen_config.framework_source_codes,
        framework_test_dirs=code_gen_config.framework_test_dirs,
        transformer_block_high_level_ops_files=code_gen_config.transformer_block_high_level_ops_files,
        median_transformer_block_files=code_gen_config.median_transformer_block_files,
        plan_file=code_gen_config.plan_file,
        plan_step=code_gen_config.plan_step,
        slower_framework=code_gen_config.target_framework,
        target_source_code_dir=code_gen_config.source_code_dir,
    )


@dataclass
class CodeTracePrompt:
    context: str
    framework: str
    output_file: str
    prompt_template: ClassVar[str] = """

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

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework=self.framework,
            output_file=self.output_file,
        )


CODE_TRACE_FILE = "code_trace.txt"


def gen_CodeTracePrompt(
    context: str,
    framework: str,
):
    return CodeTracePrompt(
        context=context,
        framework=framework,
        output_file="{}_{}".format(framework, CODE_TRACE_FILE),
    )


@dataclass
class CodePortPlanPrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_disallowed_modules: list[str]
    output_file: str
    output_summary_file: str
    prev_output_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int
    prompt_template: ClassVar[str] = """

{context}

{prev_iteration_section}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_disallowed_modules>
{code_port_disallowed_modules}
</code_port_disallowed_modules>
</definitions>

<definition_explanations>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
</definition_explanations>

<instructions>
The goal of this task is to provide a high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside "target" framework by porting <code_trace> parts from the "source" framework to "target" framework. Think hard for this task and follow these guidelines:
- Analyze and understand in-detail the <code_trace> of the "source" framework from <framework_code_trace_files>.
- Analyze and understand in-detail the <code_trace> of the "target" framework from <framework_code_trace_files>.
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
    - DO NOT import or use any of the <code_port_disallowed_modules>
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

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(self.prev_output_file, self.prev_output_summary_file),
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_disallowed_modules=self.code_port_disallowed_modules,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


CODE_PORT_PLAN_FILE_PREFIX = "code_port_plan"


def gen_CodePortPlanPrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_disallowed_modules: list[str],
    output_file: str,
    output_summary_file: str,
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
    iteration: int,
):
    assert len(framework_code_trace_files) == 2

    return CodePortPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_disallowed_modules=code_port_disallowed_modules,
        output_file=output_file,
        output_summary_file=output_summary_file,
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
    )


@dataclass
class ReviewCodePortPlanPrompt:
    context: str
    framework_code_trace_files: list[str]
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
</definitions>

<definition_explanations>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework_code_trace_files=self.framework_code_trace_files,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


def gen_ReviewCodePortPlanPrompt(
    context: str,
    framework_code_trace_files: list[str],
    input_file: str,
    input_summary_file: str,
    output_file: str,
    output_summary_file: str,
    iteration: int,
):
    assert len(framework_code_trace_files) == 2

    return ReviewCodePortPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        input_file=input_file,
        input_summary_file=input_summary_file,
        output_file=output_file,
        output_summary_file=output_summary_file,
        iteration=iteration,
    )


@dataclass
class TestPlanPrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_plan_file: str
    output_file: str
    output_summary_file: str
    prev_output_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int
    prompt_template: ClassVar[str] = """

{context}

{prev_iteration_section}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
</definitions>

<definition_explanations>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(self.prev_output_file, self.prev_output_summary_file),
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


TEST_PLAN_PREFIX = "test_plan"


def gen_TestPlanPrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
    output_file: str,
    output_summary_file: str,
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
    iteration: int,
):
    assert len(framework_code_trace_files) == 2

    return TestPlanPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_file=output_file,
        output_summary_file=output_summary_file,
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
    )


@dataclass
class CodeGenPrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_plan_file: list[str]
    test_plan_file: list[str]
    output_patch_file: str
    output_summary_file: str
    prev_output_patch_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int
    prompt_template: ClassVar[str] = """

{context}

{prev_iteration_section}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
</definitions>

<definition_explanations>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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
</instructions>

<output>
- Dump the code patch to <output_dir>/{output_patch_file} (it must have both code and its tests inside)
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the code patch and tests generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(self.prev_output_patch_file, self.prev_output_summary_file),
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


CODE_GEN_FILE_PREFIX = "code_gen"


def gen_CodeGenPrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    output_patch_file: str,
    output_summary_file: str,
    prev_output_patch_file: Optional[str],
    prev_output_summary_file: Optional[str],
    iteration: int,
):
    assert len(framework_code_trace_files) == 2

    return CodeGenPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        output_patch_file=output_patch_file,
        output_summary_file=output_summary_file,
        prev_output_patch_file=prev_output_patch_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
    )


@dataclass
class ReviewCodeGenPrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    input_patch_file: str
    input_summary_file: str
    output_patch_file: str
    output_summary_file: str
    iteration: int
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
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
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            input_patch_file=self.input_patch_file,
            input_summary_file=self.input_summary_file,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


def gen_ReviewCodeGenPrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    input_patch_file: str,
    input_summary_file: str,
    output_patch_file: str,
    output_summary_file: str,
    iteration: int,
):
    assert len(framework_code_trace_files) == 2

    return ReviewCodeGenPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        input_patch_file=input_patch_file,
        input_summary_file=input_summary_file,
        output_patch_file=output_patch_file,
        output_summary_file=output_summary_file,
        iteration=iteration,
    )


@dataclass
class InvestigateIssuePrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    code_port_plan_review_evolution_file: str
    code_pr_info_file: str
    code_pr_file: str
    code_pr_review_evolution_file: str
    issue_desc_file: str
    issue_fix_previous_attempt_file: str
    issue_fix_previous_attempt_review_evolution_file: str
    issue_fix_file: str
    code_pr_fixed_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<code_port_plan_review_evolution_file>
{code_port_plan_review_evolution_file}
</code_port_plan_review_evolution_file>
<code_pr_info_file>
{code_pr_info_file}
</code_pr_info_file>
<code_pr_file>
{code_pr_file}
</code_pr_file>
<code_pr_review_evolution_file>
{code_pr_review_evolution_file}
</code_pr_review_evolution_file>
<issue_desc_file>
{issue_desc_file}
</issue_desc_file>
<issue_fix_previous_attempt_file>
{issue_fix_previous_attempt_file}
</issue_fix_previous_attempt_file>
<issue_fix_previous_attempt_review_evolution_file>
{issue_fix_previous_attempt_review_evolution_file}
</issue_fix_previous_attempt_review_evolution_file>
</definitions>

<definition_explanations>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <test_plan_file> is a file that describes the high-level multi-step testing plan for the implementation in <code_port_plan_file>.
- <code_port_plan_review_evolution_file> describes the review evolution process during the code port plan => review generation iterations that lead to <code_port_plan_file>.
- <code_pr_file> is the code patch for <target_framework> that implements the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file>.
- <code_pr_info_file> is a file that describes the <code_pr_file>
- <code_pr_review_evolution_file> describes the review evolution process during the code => review generation iterations that lead to <code_pr_file>
- <issue_desc_file> describes the issue that arises with the application of <code_pr_file> and needs to be investigated for a potential fix.
</definition_explanations>

<instructions>
The goal of this task is to investigate the issue described in <issue_desc_file> that arises after the <code_pr_file> is applied to <target_framework>. Do the following and think hard:
- If <issue_fix_previous_attempt_file> is provided and is an existing real file, then read and analyze in-detail this contents to get all of the information about the previous attempt to generate a fix to the issue described here. Use all of the learnings and details from the previous attempt to improve the current attempt. Note that the current attempt is still done from scratch, but it can use the learnings from the previous attempt.
- If <issue_fix_previous_attempt_review_evolution_file> is provided and is an existing real file, then read and analyze in-detail this contents to get all of the information about the previous attempt to generate a fix to the issue described here. Use all of the learnings and details from the previous attempt to improve the current attempt. Note that the current attempt is still done from scratch, but it can use the learnings from the previous attempt.
- Read, analyze and understand in-detail all previous issues that were reported and are in the <output_dir>. These are may be relevant to avoid repeating mistakes, bugs or misleading information. Take all of the learning of previous issues into account while working on this issue from scratch.
- Analyze and understand in-detail the code port plan in <code_port_plan_file>, and restate the code port plan process step-by-step.
    - Analyze and understand in-detail the code port plan => review iteration evolution that is described in <code_port_plan_review_evolution_file> that lead to the final <code_port_plan_file>.
- Analyze and understand in-detail the code patch in <code_pr_file>, and restate the coding process step-by-step.
    - Analyze and understand in-detail the code gen => review iteration evolution that is described in <code_pr_review_evolution_file> that lead to the final <code_pr_file>.
- Detect and analyze in-detail the root causes that make issue <issue_desc_file> to appear in <target_framework>.
- Detect and analyze in-detail the root causes that make issue <issue_desc_file> to NOT appear in <source_framework>.
- Dive deep into the source code of both frameworks, and their related third party libraries, to get full picture of the source code end-to-end as it related to the <code_trace> of both frameworks. 
    - For example, if an external kernel is used, then find/fetch the source code of this kernel and trace all of the wrappers till this kernel is invoked. Make sure to find the actual full source code of the kernel. This is important.
- Analyze and read any necessary extra information to get deeper understanding of the issue, including:
    - run logs
    - high level transformer blocks
    - median transformer blocks that correlate low-level kernels to high-level source codes
    - code port planning
    - code pr and code pr info files
    - third party library source codes and their wrappers, all of the way from high-level calls to lowest level function calls.
    - Any related commits, their descriptions and more
- Understand how to fix the issue in <issue_desc_file> and provide a detailed explanation of:
    - Why it happens.
    - What are the key reasons with source code references for both frameworks.    
    - Steps to fix
</instructions>

<output>
- Dump the detailed explanation of the issue, key reasons, and how to fix to <output_dir>/{issue_fix_file}
- Add new tests to verify that the issue is fully fixed.
- Dump the fixed code pr patch with old and new tests to <output_dir>/{code_pr_fixed_file}
- Apply the new code patch to the "target" source code inside <target_source_code_dir>. ALL code modifications MUST be made exclusively inside <target_source_code_dir>.
- Run the tests and ensure ALL PASS (NO SKIPS). If some test fails, then review the work, fix the issue again, and re-run again. DO NOT STOP UNTIL THE ISSUE IS FIXED.
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            code_port_plan_review_evolution_file=self.code_port_plan_review_evolution_file,
            code_pr_info_file=self.code_pr_info_file,
            code_pr_file=self.code_pr_file,
            code_pr_review_evolution_file=self.code_pr_review_evolution_file,
            issue_desc_file=self.issue_desc_file,
            issue_fix_previous_attempt_file=self.issue_fix_previous_attempt_file,
            issue_fix_previous_attempt_review_evolution_file=self.issue_fix_previous_attempt_review_evolution_file,
            issue_fix_file=self.issue_fix_file,
            code_pr_fixed_file=self.code_pr_fixed_file,
        )


def gen_InvestigateIssuePrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    code_port_plan_review_evolution_file: str,
    code_pr_info_file: str,
    code_pr_file: str,
    code_pr_review_evolution_file: str,
    issue_desc_file: str,
    issue_fix_previous_attempt_file: str,
    issue_fix_previous_attempt_review_evolution_file: str,
    issue_fix_file: str,
    code_pr_fixed_file: str,
):
    assert len(framework_code_trace_files) == 2

    return InvestigateIssuePrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        code_port_plan_review_evolution_file=code_port_plan_review_evolution_file,
        code_pr_info_file=code_pr_info_file,
        code_pr_file=code_pr_file,
        code_pr_review_evolution_file=code_pr_review_evolution_file,
        issue_desc_file=issue_desc_file,
        issue_fix_previous_attempt_file=issue_fix_previous_attempt_file,
        issue_fix_previous_attempt_review_evolution_file=issue_fix_previous_attempt_review_evolution_file,
        issue_fix_file=issue_fix_file,
        code_pr_fixed_file=code_pr_fixed_file,
    )


@dataclass
class ReviewInvestigatedIssuePrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    code_port_plan_review_evolution_file: str
    code_pr_info_file: str
    code_pr_file: str
    code_pr_review_evolution_file: str
    issue_desc_file: str
    issue_fix_file: str
    issue_fix_review_file: str
    issue_fix_fixed_file: str
    issue_fix_review_evolution_file: str
    code_pr_review_fixed_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<code_port_plan_review_evolution_file>
{code_port_plan_review_evolution_file}
</code_port_plan_review_evolution_file>
<code_pr_info_file>
{code_pr_info_file}
</code_pr_info_file>
<code_pr_file>
{code_pr_file}
</code_pr_file>
<code_pr_review_evolution_file>
{code_pr_review_evolution_file}
</code_pr_review_evolution_file>
<issue_desc_file>
{issue_desc_file}
</issue_desc_file>
<issue_fix_file>
{issue_fix_file}
</issue_fix_file>
</definitions>

<definition_explanations>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <test_plan_file> is a file that describes the high-level multi-step testing plan for the implementation in <code_port_plan_file>.
- <code_port_plan_review_evolution_file> describes the review evolution process during the code port plan => review generation iterations that lead to <code_port_plan_file>.
- <code_pr_file> is the code patch for <target_framework> that implements the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file>.
- <code_pr_info_file> is a file that describes the <code_pr_file>
- <code_pr_review_evolution_file> describes the review evolution process during the code => review generation iterations that lead to <code_pr_file>
- <issue_desc_file> describes the issue that arises with the application of <code_pr_file> and needs to be investigated for a potential fix.
- <issue_fix_file> describes the key reasons for the issue <issue_desc_file> and how to fix it.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of issue fix in <issue_fix_file>. Do the following and think hard:
- Analyze and understand in-detail the issue in <issue_desc_file> 
- Analyze and understand in-detail the fix described in <issue_fix_file> for the issue <issue_desc_file>.
- Review the issue fix for any:
    - incorrect assumptions
    - missing steps
    - bad ordering or sequencing
    - ambiguity or vagueness
    - missing edge cases
    - architectural risks
    - hidden dependencies
    
- For each problem found, document:
    - The affected part of the plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected issue fix.
</instructions>

<output>
- Dump the documentation of the fixes to the <issue_fix_file> to <output_dir>/{issue_fix_review_file}
- Dump the corrected issue fix to <output_dir>/{issue_fix_fixed_file}
- Dump the corrected code PR file to <output_dir>/{code_pr_review_fixed_file}. For this add new tests if needed, apply the new patch inside <target_source_code_dir>, and re-run the tests. ALL code modifications MUST be made exclusively inside <target_source_code_dir>.
- If multiple issue investigation => review iterations were done till now, then summarize the iteration evolution in <output_dir>/{issue_fix_review_evolution_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            code_port_plan_review_evolution_file=self.code_port_plan_review_evolution_file,
            code_pr_info_file=self.code_pr_info_file,
            code_pr_file=self.code_pr_file,
            code_pr_review_evolution_file=self.code_pr_review_evolution_file,
            issue_desc_file=self.issue_desc_file,
            issue_fix_file=self.issue_fix_file,
            issue_fix_review_file=self.issue_fix_review_file,
            issue_fix_fixed_file=self.issue_fix_fixed_file,
            issue_fix_review_evolution_file=self.issue_fix_review_evolution_file,
            code_pr_review_fixed_file=self.code_pr_review_fixed_file,
        )


def gen_ReviewInvestigatedIssuePrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    code_port_plan_review_evolution_file: str,
    code_pr_info_file: str,
    code_pr_file: str,
    code_pr_review_evolution_file: str,
    issue_desc_file: str,
    issue_fix_file: str,
    issue_fix_review_file: str,
    issue_fix_fixed_file: str,
    issue_fix_review_evolution_file: str,
    code_pr_review_fixed_file: str,
):
    assert len(framework_code_trace_files) == 2

    return ReviewInvestigatedIssuePrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        code_port_plan_review_evolution_file=code_port_plan_review_evolution_file,
        code_pr_info_file=code_pr_info_file,
        code_pr_file=code_pr_file,
        code_pr_review_evolution_file=code_pr_review_evolution_file,
        issue_desc_file=issue_desc_file,
        issue_fix_file=issue_fix_file,
        issue_fix_review_file=issue_fix_review_file,
        issue_fix_fixed_file=issue_fix_fixed_file,
        issue_fix_review_evolution_file=issue_fix_review_evolution_file,
        code_pr_review_fixed_file=code_pr_review_fixed_file,
    )


@dataclass
class WorkItemsPrompt:
    context: str
    code_gen_dir: str
    work_items_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_gen_dir>
{code_gen_dir}
</code_gen_dir>
<work_items_file>
{work_items_file}
</work_items_file>
</definitions>

<definition_explanations>
- <code_gen_dir> is a directory that holds the results of the code generation process that implemented <plan_step> (from <plan_file>) inside "target" framework. This directory includes:
    - Framework code traces
    - code port plan with review iterations
    - code generation with review iterations
    - issue fixing sequences
</definition_explanations>

<instructions>
The goal of this task is to execute the work items that are described in <work_items_file>. Do the following and think hard:
- Read all result files in <code_gen_dir> to get a detailed understanding of the implementation process that occured. Understand everything in great detail, and take all the learnings from the review evolutions and the issue fixing processes that were executed.
- Read the "context section"
- Read the "work_items section"
- Execute the work one by one. For each work item:
    - Find, analyze and understand in-depth any relevant data that will help to execute the work item. Be very thorough.
    - If the work item is complex, split to smaller steps, execute each one and verify before proceeding to next step.
    - Do final verification that the step completed successfully. Perform a critical review and fix issues.
</instructions>

<output>
- Read the "output section" from <work_items_file> and generate the outputs based on that.
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_gen_dir=self.code_gen_dir,
            work_items_file=self.work_items_file,
        )


def gen_WorkItemsPrompt(
    context: str,
    code_gen_dir: str,
    work_items_file: str,
):

    return WorkItemsPrompt(
        context=context,
        code_gen_dir=code_gen_dir,
        work_items_file=work_items_file,
    )


@dataclass
class SummarizeCodeGenProcessPrompt:
    context: str
    framework_code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    code_port_plan_review_evolution_file: str
    code_pr_info_file: str
    code_pr_file: str
    code_pr_review_evolution_file: str
    issue_desc_files: list[str]
    issue_fix_review_evolution_files: list[str]
    auto_analyze_project_brief: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<code_port_plan_review_evolution_file>
{code_port_plan_review_evolution_file}
</code_port_plan_review_evolution_file>
<code_pr_info_file>
{code_pr_info_file}
</code_pr_info_file>
<code_pr_file>
{code_pr_file}
</code_pr_file>
<code_pr_review_evolution_file>
{code_pr_review_evolution_file}
</code_pr_review_evolution_file>
<issue_desc_files>
{issue_desc_files}
</issue_desc_files>
<issue_fix_review_evolution_files>
{issue_fix_review_evolution_files}
</issue_fix_review_evolution_files>
<auto_analyze_project_brief>
{auto_analyze_project_brief}
</auto_analyze_project_brief>
</definitions>

<definition_explanations>
- <auto_analyze_project_brief> is a PDF file that summarizes the auto-analyze process that resulted in the improvement plan file <plan_file>
- <framework_code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
- <test_plan_file> is a file that describes the high-level multi-step testing plan for the implementation in <code_port_plan_file>.
- <code_port_plan_review_evolution_file> describes the review evolution process during the code port plan => review generation iterations that lead to <code_port_plan_file>.
- <code_pr_file> is the code patch for <target_framework> that implements the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file>. Note that this code patch also incorporates the fixes to the issues in <issue_desc_files>.
- <code_pr_info_file> is a file that describes the <code_pr_file> in general (before the fixed issues)
- <code_pr_review_evolution_file> describes the review evolution process during the code => review generation iterations that lead to <code_pr_file> (before the issues were fixed)
- <issue_desc_files> is a list of files that describes the issues encountered that needed to be fixed after running a full DeepSeek V3.2 on 8 Hopper GPUs. All of them needed to be fixed to arrive to full correctness.
- <issue_fix_review_evolution_files> describes the reviews that were applied to fixing the issues in <issue_desc_files>.

</definition_explanations>

<PPTX_formatting>
Create a professional PowerPoint (.pptx) presentation for a highly technical audience of low-level GPU programmers, inference engineers, kernel/performance engineers, systems engineers, and experts in vLLM internals.

Primary goal:
Produce a deck that looks like it was made by a strong senior performance engineer with excellent technical judgment and good design sense. The presentation must be suitable for an internal deep-dive review of vLLM behavior, performance bottlenecks, kernel execution, scheduling behavior, memory movement, attention execution, and optimization opportunities.

Audience assumptions:
	•	The audience already understands LLM inference concepts, CUDA/GPU execution, model serving, and vLLM at a technical level
	•	They care about correctness, performance methodology, kernel-level behavior, architecture tradeoffs, runtime overheads, and implementation details
	•	They will quickly notice weak technical claims, shallow explanations, unreadable slides, cluttered layouts, imprecise terminology, and poorly formatted code or traces

Desired tone and style:
	•	Deep technical engineering presentation
	•	Professional, clean, modern, understated
	•	Not marketing-like
	•	Visually polished, but optimized for technical clarity over decoration
	•	Strong visual hierarchy, consistent formatting, clean spacing, and good contrast
	•	The deck should look appropriate for a vLLM architecture review, performance review, or optimization deep dive

Critical formatting requirements:
	•	Generate the final output as a real .pptx file
	•	Never use tiny fonts to make content fit
	•	Minimum font sizes:
	•	Slide titles: 28 pt or larger
	•	Section headers / box headers: 20 pt or larger
	•	Body bullets: 18 pt or larger
	•	Text inside diagrams, trace views, tables, and annotations: 16 pt or larger
	•	Code snippets: 16 pt or larger whenever possible
	•	If content does not fit, do NOT shrink the font
	•	Instead:
	•	shorten visible text
	•	split material across more slides
	•	crop code snippets to the most relevant lines
	•	simplify diagrams
	•	move extra detail into speaker notes
	•	The slides must remain comfortably readable during screen sharing and on a projected display

Density rules:
	•	Prefer more light slides over fewer dense slides
	•	Each slide should communicate one main technical point
	•	Avoid overloaded diagrams, overloaded code slides, or too many bullets
	•	If a comparison becomes dense, split it into multiple progressive slides
	•	If a trace or call chain is long, show only the relevant section and continue on the next slide
	•	Prioritize readability and reasoning flow over compression

vLLM-specific content expectations:
Focus the presentation on the kinds of details that matter to vLLM experts, such as:
	•	request lifecycle and scheduler behavior
	•	prefill vs decode execution paths
	•	paged attention behavior
	•	KV cache structure, allocation, reuse, and movement
	•	block management and memory fragmentation considerations
	•	CUDA kernel launch behavior and kernel sequence
	•	attention backend differences
	•	communication overheads, tensor parallel behavior, and synchronization points
	•	CPU overhead vs GPU bottlenecks
	•	graph capture / cuda graph behavior where relevant
	•	trace analysis, operator-level bottlenecks, and timeline interpretation
	•	source-code-level explanations of important hot paths
	•	comparisons across frameworks or execution modes when relevant
	•	concrete optimization ideas and expected impact

Slide content rules:
	•	Be precise, technical, and concise
	•	Avoid generic AI language, buzzwords, and fluff
	•	Use exact engineering terminology where appropriate
	•	Prefer short bullets over paragraphs
	•	Maximum 4 bullets per slide unless the slide is primarily code, trace, or benchmark oriented
	•	Every slide should have a clear technical takeaway
	•	Do not invent metrics, code behavior, or implementation details that are not provided

Preferred slide types:
Use these kinds of slides where appropriate:
	•	problem/context
	•	request execution flow
	•	prefill vs decode comparison
	•	scheduler behavior walkthrough
	•	KV cache / block manager explanation
	•	call-chain comparison
	•	code trace comparison
	•	timeline / profiling view explanation
	•	kernel sequence and hotspot analysis
	•	focused code snippet walkthrough
	•	bottleneck summary
	•	optimization proposal
	•	before/after benchmark comparison
	•	tradeoffs / risks
	•	next steps

Requirements for code traces, call-chains, and timeline comparisons:
	•	Use side-by-side layout when comparing two traces / paths / implementations
	•	Clearly label the two sides
	•	Align equivalent stages visually
	•	Highlight only the important differences
	•	Use color sparingly and purposefully to indicate:
	•	added or removed steps
	•	different kernels
	•	bottlenecks
	•	regressions or improvements
	•	synchronization points
	•	Add a short takeaway sentence on each comparison slide explaining the key difference
	•	If the comparison is too dense, split it into multiple slides by subsystem or stage

Requirements for code snippets:
	•	Use short, focused snippets only
	•	Crop to the most relevant functions, loops, branches, or call sites
	•	Preserve indentation and readability
	•	Add brief annotations explaining why the snippet matters
	•	Highlight only the relevant lines
	•	Do not place long code snippets and long prose on the same slide
	•	If needed, use one slide for the snippet and a follow-up slide for explanation or performance implications

Requirements for benchmarks and performance slides:
	•	Present metrics clearly with simple charts or tables
	•	Show before vs after where possible
	•	Include throughput, latency, GPU utilization, CPU overhead, kernel time, memory effects, or other relevant low-level metrics when available
	•	Make benchmark conditions explicit: model, batch/concurrency, input/output lengths, GPU type, framework/mode, and any important runtime flags
	•	Do not hide methodology
	•	Show caveats and tradeoffs where relevant

Visual design rules:
	•	Use a restrained professional color palette
	•	Favor neutral backgrounds with strong text contrast
	•	Use one primary accent color and one secondary accent color, plus neutrals
	•	Use color to guide attention, not decorate
	•	Avoid loud or overly saturated colors
	•	Avoid decorative graphics that add no technical value
	•	Prefer clean architecture/flow diagrams, comparison tables, and trace visuals over stock art or generic icons
	•	Use whitespace intentionally, but do not leave large empty areas while the content itself is cramped
</PPTX_formatting>

<instructions>
The goal of this task is to generate a sequence of PPTX slides that describe the AI-based automatic code generation process that implemented improvement step <plan_step> from the <plan_file>. Do the following and think hard:
- Analyze and understand in-depth the AI-based automatic code generation process that is composed of the sequence of generated files in <output_dir>. Read all of these files in <output_dir> and analyze their contents. The general steps are as follows:
    - Generate a code trace for the "source" framework, to get the call-chain of active code pieces 
    - Generate a code trace for the "target" framework, to get the call-chain of active code pieces 
    - Generate a code port plan from "source" to "target" framework that implements the improvement plan.
        - The code port plan is done in iterations, where each iteration is "generate code port plan" => "review and fix"
        - Learnings from previous iterations are used in the current iteration to improve the quality of the result and avoid bugs
        - In general, the iteration-based generation is critical to provide a correct code port plan due to the complexity of the problem. It is highly unlikely that AI can generate a working code port plan from first shot, and it does need these iterations to fix bugs and issues before actually running the code. This iteration "evolution" is key for success.  
    - Generate a test plan, from simple unit tests to larger end-to-end tests that are focused on critical things like: decode-only, prefill-only, mixed execution modes, cuda graphs support and more.
    - Generate a the code patch based on the code port plan
        - Here, we also apply the iterations to do "code gen" => "review" to fix issues and bugs. Also in this step, recompilation occurs and real tests are ran.
    - After the code patch is done, the code was ran manually with the model <model> and a couple of issues, issue_1 and issue_2, where for each issue AI was used to fix the issue in the context of the previous generations. AI was able to fix these issues and after these fixes everything worked: has both correctness and speedups. 

- Generate a sequence of PPTX slides as follows:
    - A comparison one-to-one, on the same slide, of the code traces of both frameworks from <framework_code_trace_files>. Make sure to show:
        - The full operation call-chains with all the necessary details, with the specific per-operation time breakdowns based on the median transformer blocks. 
        - Explain the differences that can be seen in both the code traces and the transformer blocks, how the correlate and why the "target" is faster.
    - Explain what needs to be done to port the "source" code pieces to the "target" codebase. Base the explanation on the code port plan file. Make sure to explain:
        - Each critical code piece that is ported from the "source", what stays the same, what is changed, and how this code piece is integrated into the "target" codebase. Make sure to show actual code, and explain fully and clearly, so expert programmer can understand.
    - Show and explain in-detail the Claude query prompt that is used to generate the code port plan, based on the source code here /home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/code_gen_prompts.py.
    - Show and explain in-detail the Claude query prompt that is used to generate the code port plan review, based on the source code here /home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/code_gen_prompts.py.
    - Show and explain in-detail the "code port plan" => "review" iterations that are used to fix bugs and issues before running the code:
        - The key problem is that a code that is generated on first iteration is usually incorrect due to the complexity of the task, and the key idea to solve it is to use iterations to evolve the generated code to the point where it is fully correct. 
        - Show each iteration (based on the files in <output_dir>), what bugs/issues it found (in-detail), why it is important, and how it is fixed. This "gen" => "review" evolution flow is important to understand since it is the key for correctness.
    - For code patch generation, same as for previous bullet:
        - Show the generated code vs the iteration evolution process that fixes bugs and issues. Show each iterations, with what it found, why it is important and how it is fixed.
    - Present the final pipeline of the process with all steps in a diagram, where "gen" => "review" is annotated properly with back arrows.
    - Show the resulting code after code patch is applied in the "target"
        - Show the new kernel components and compilation modifications
        - Show the new/modified classes/functions and how they solve the previously detected issues that made "source" framework slower.
        - Explain what is ported AS IS (copy-pasted) and what is the integration/amalgamation code.
    - Show the correctness and performance improvement of 16-17 percent for TPOT. Correctnes was verified via lm_eval and this is the result:
        |Tasks|Version|   Filter   |n-shot| Metric  |  |Value |  |Stderr|
        |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
        |gsm8k|   3|flexible-extract|   5|exact_match|↑ |0.9545|± |0.0057|
        |   |    |strict-match  |   5|exact_match|↑ |0.9553|± |0.0057|
- Follow formatting in <PPTX_formatting>
</instructions>

<output>
- Dump the resulting PPTX slides to <output_dir>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            code_port_plan_review_evolution_file=self.code_port_plan_review_evolution_file,
            code_pr_info_file=self.code_pr_info_file,
            code_pr_file=self.code_pr_file,
            code_pr_review_evolution_file=self.code_pr_review_evolution_file,
            issue_desc_files=self.issue_desc_files,
            issue_fix_review_evolution_files=self.issue_fix_review_evolution_files,
            auto_analyze_project_brief=self.auto_analyze_project_brief,
            output_file=self.output_file,
        )


def gen_SummarizeCodeGenProcessPrompt(
    context: str,
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    code_port_plan_review_evolution_file: str,
    code_pr_info_file: str,
    code_pr_file: str,
    code_pr_review_evolution_file: str,
    issue_desc_files: list[str],
    issue_fix_review_evolution_files: list[str],
    auto_analyze_project_brief: str,
    output_file: str,
):
    assert len(framework_code_trace_files) == 2

    return SummarizeCodeGenProcessPrompt(
        context=context,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        code_port_plan_review_evolution_file=code_port_plan_review_evolution_file,
        code_pr_info_file=code_pr_info_file,
        code_pr_file=code_pr_file,
        code_pr_review_evolution_file=code_pr_review_evolution_file,
        issue_desc_files=issue_desc_files,
        issue_fix_review_evolution_files=issue_fix_review_evolution_files,
        auto_analyze_project_brief=auto_analyze_project_brief,
        output_file=output_file,
    )
