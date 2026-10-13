# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
from dataclasses import dataclass
from typing import ClassVar

from common.claude_utils import ClaudeConfig
from common.iteration_utils import prev_iteration_section as _prev_iteration_section
from llm.auto_code_gen.code_gen_configs import CodeGenConfig


def _resolve_knowledge_base_path():
    """Resolve the global knowledge base file path."""
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        KNOWLEDGE_BASE_FILE,
    )
    return kb_path if os.path.isfile(kb_path) else None


def create_context_str(claude_config: ClaudeConfig, code_gen_config: CodeGenConfig):
    # Extract branch names if the use-case config provides them
    source_branch = getattr(code_gen_config, "source_branch", None)
    target_branch = getattr(code_gen_config, "target_branch", None)

    branch_section = ""
    if source_branch and target_branch:
        branch_section = f"""
<source_branch>
{source_branch}
</source_branch>
<target_branch>
{target_branch}
</target_branch>
"""

    knowledge_base_section = ""
    use_kb = getattr(code_gen_config, "use_knowledge_base", False)
    if use_kb:
        kb_path = _resolve_knowledge_base_path()
        if kb_path:
            knowledge_base_section = f"""
<knowledge_base_file>
{kb_path}
</knowledge_base_file>
"""

    return f"""
<context>

<output_dir>
{claude_config.cwd}
</output_dir>

<model>
{code_gen_config.model}
</model>
<gpu_type>
{code_gen_config.gpu_type}
</gpu_type>
<batch_size>
{code_gen_config.batch_size}
</batch_size>
<isl>
{code_gen_config.isl}
</isl>
<osl>
{code_gen_config.osl}
</osl>

<tested_execution>
Execution of model <model> on <gpu_type> GPU with ISL <isl>, OSL <osl> and batch size <batch_size>
</tested_execution>

<framework_names>
{code_gen_config.framework_names}
</framework_names>
<source_framework>
{code_gen_config.source_framework}
</source_framework>
<target_framework>
{code_gen_config.target_framework}
</target_framework>
<framework_source_codes>
{code_gen_config.framework_source_codes}
</framework_source_codes>
<framework_test_dirs>
{code_gen_config.framework_test_dirs}
</framework_test_dirs>
<transformer_block_high_level_ops_files>
{code_gen_config.transformer_block_high_level_ops_files}
</transformer_block_high_level_ops_files>
<median_transformer_block_files>
{code_gen_config.median_transformer_block_files}
</median_transformer_block_files>

<plan_file>
{code_gen_config.plan_file}
</plan_file>
<plan_step>
{code_gen_config.plan_step}
</plan_step>

<slower_framework>
{code_gen_config.target_framework}
</slower_framework>

<target_source_code_dir>
{code_gen_config.source_code_dir}
</target_source_code_dir>

{branch_section}
{knowledge_base_section}
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

- If <source_branch> and <target_branch> are provided: ONLY analyze code from these specific branches. Do NOT look at other branches, commits, or git history outside of these branches. When using git commands or examining source code, restrict all operations to the specified branches.

- If <knowledge_base_file> is provided: Read it BEFORE starting work. It contains learnings from previous runs. USAGE RULES:
    - Use KB entries as HINTS, not as absolute truths. They may be outdated, context-specific, or wrong.
    - NEVER use a KB entry to conclude an approach is impossible or non-viable. KB entries document what happened in PREVIOUS runs — a different implementation may succeed where a previous one failed.
    - NEVER dismiss E2E benchmark results based on KB entries about "benchmark artifacts" or "measurement noise". E2E is ground truth.
    - ALWAYS verify KB claims against the CURRENT source code — the codebase may have changed since the entry was written.
    - Treat KB entries about thresholds (e.g. "must save >840us") as guidelines from past observations, not universal laws.
</context_explanations>

"""


@dataclass
class CodeTracePrompt:
    context: str
    framework: str
    output_file: str
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework=self.framework,
            output_file=self.output_file,
        )


CODE_TRACES_DIR = "code_traces"
CODE_PORT_PLAN_DIR = "code_port_plan"
TEST_PLAN_DIR = "test_plan"
CODE_GEN_DIR = "code_gen"
RUNTIME_DIR = "runtime"
# Legacy spelling kept for callers predating the runtime/ consolidation
# (llm.auto_code_gen.run_code_gen on this branch still imports it).
RUNTIME_LOGS_DIR = RUNTIME_DIR

CODE_TRACE_FILE = "code_trace.txt"


def code_trace_filename(framework: str) -> str:
    return os.path.join(CODE_TRACES_DIR, f"{framework}_{CODE_TRACE_FILE}")


def gen_CodeTracePrompt(
    context: str,
    framework: str,
    prompt_template: str,
):
    return CodeTracePrompt(
        context=context,
        framework=framework,
        output_file=code_trace_filename(framework),
        prompt_template=prompt_template,
    )


@dataclass
class CodePortPlanPrompt:
    context: str
    code_trace_files: list[str]
    disallowed_modules: list[str]
    output_file: str
    output_summary_file: str
    prev_output_file: str | None
    prev_output_summary_file: str | None
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file
            ),
            code_trace_files=self.code_trace_files,
            disallowed_modules=self.disallowed_modules,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


CODE_PORT_PLAN_FILE_PREFIX = "code_port_plan"


def gen_CodePortPlanPrompt(
    context: str,
    code_trace_files: list[str],
    disallowed_modules: list[str],
    output_file: str,
    output_summary_file: str,
    prev_output_file: str | None,
    prev_output_summary_file: str | None,
    iteration: int,
    prompt_template: str,
):
    assert len(code_trace_files) == 2

    return CodePortPlanPrompt(
        context=context,
        code_trace_files=code_trace_files,
        disallowed_modules=disallowed_modules,
        output_file=output_file,
        output_summary_file=output_summary_file,
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
        prompt_template=prompt_template,
    )


@dataclass
class ReviewCodePortPlanPrompt:
    context: str
    code_trace_files: list[str]
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


def gen_ReviewCodePortPlanPrompt(
    context: str,
    code_trace_files: list[str],
    input_file: str,
    input_summary_file: str,
    output_file: str,
    output_summary_file: str,
    iteration: int,
    prompt_template: str,
):
    assert len(code_trace_files) == 2

    return ReviewCodePortPlanPrompt(
        context=context,
        code_trace_files=code_trace_files,
        input_file=input_file,
        input_summary_file=input_summary_file,
        output_file=output_file,
        output_summary_file=output_summary_file,
        iteration=iteration,
        prompt_template=prompt_template,
    )


@dataclass
class TestPlanPrompt:
    context: str
    code_trace_files: list[str]
    code_port_plan_file: str
    output_file: str
    output_summary_file: str
    prev_output_file: str | None
    prev_output_summary_file: str | None
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file
            ),
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


TEST_PLAN_PREFIX = "test_plan"


def gen_TestPlanPrompt(
    context: str,
    code_trace_files: list[str],
    code_port_plan_file: str,
    output_file: str,
    output_summary_file: str,
    prev_output_file: str | None,
    prev_output_summary_file: str | None,
    iteration: int,
    prompt_template: str,
):
    assert len(code_trace_files) == 2

    return TestPlanPrompt(
        context=context,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_file=output_file,
        output_summary_file=output_summary_file,
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
        prompt_template=prompt_template,
    )


@dataclass
class ReviewTestPlanPrompt:
    context: str
    code_trace_files: list[str]
    code_port_plan_file: str
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


def gen_ReviewTestPlanPrompt(
    context: str,
    code_trace_files: list[str],
    code_port_plan_file: str,
    input_file: str,
    input_summary_file: str,
    output_file: str,
    output_summary_file: str,
    iteration: int,
    prompt_template: str,
):
    assert len(code_trace_files) == 2

    return ReviewTestPlanPrompt(
        context=context,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        input_file=input_file,
        input_summary_file=input_summary_file,
        output_file=output_file,
        output_summary_file=output_summary_file,
        iteration=iteration,
        prompt_template=prompt_template,
    )


@dataclass
class CodeGenPrompt:
    context: str
    code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    output_patch_file: str
    output_summary_file: str
    prev_output_patch_file: str | None
    prev_output_summary_file: str | None
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_patch_file, self.prev_output_summary_file
            ),
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


CODE_GEN_FILE_PREFIX = "code_gen"


def gen_CodeGenPrompt(
    context: str,
    code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    output_patch_file: str,
    output_summary_file: str,
    prev_output_patch_file: str | None,
    prev_output_summary_file: str | None,
    iteration: int,
    prompt_template: str,
):
    assert len(code_trace_files) == 2

    return CodeGenPrompt(
        context=context,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        output_patch_file=output_patch_file,
        output_summary_file=output_summary_file,
        prev_output_patch_file=prev_output_patch_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
        prompt_template=prompt_template,
    )


@dataclass
class ReviewCodeGenPrompt:
    context: str
    code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    input_patch_file: str
    input_summary_file: str
    output_patch_file: str
    output_summary_file: str
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
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
    code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    input_patch_file: str,
    input_summary_file: str,
    output_patch_file: str,
    output_summary_file: str,
    iteration: int,
    prompt_template: str,
):
    assert len(code_trace_files) == 2

    return ReviewCodeGenPrompt(
        context=context,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        input_patch_file=input_patch_file,
        input_summary_file=input_summary_file,
        output_patch_file=output_patch_file,
        output_summary_file=output_summary_file,
        iteration=iteration,
        prompt_template=prompt_template,
    )


RUNTIME_FILE_PREFIX = "code_gen_runtime"
RUNTIME_SUCCESS_FILE = "success_result.txt"
RUNTIME_RESULTS_MANIFEST = "results_manifest.json"
PR_COMMANDS_DIR = "pr-generate"
RUNTIME_SMALLER_MODEL_FILE = "smaller_model.txt"
RUNTIME_EXPECTED_IMPROVEMENT_FILE = "expected_improvement.json"
DIAGNOSTIC_PROFILE_DIR = "diagnostic_profile"
DIAGNOSTIC_REPORT_FILE = "diagnostic_report.txt"
KNOWLEDGE_BASE_FILE = "llm/auto_analyze/knowledge_base.txt"


@dataclass
class ExtractExpectedImprovementPrompt:
    context: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<instructions>
Read the improvement plan from <plan_file> and find the improvement step <plan_step>.

Extract the expected performance improvement percentage for this specific improvement step.
Look for the Impact line (or similar) in the improvement step section that states the expected
wall-time or per-layer improvement as a percentage (e.g., "6.1%", "1.2% of wall").

If the improvement is marked as OVL (overlapped streams only) with no critical-path percentage,
set the percentage to null — these improvements affect auxiliary streams and cannot be measured
via end-to-end throughput.

Write a JSON file to <output_dir>/{output_file} with exactly this format:
{{
    "expected_improvement_pct": <number or null>,
    "improvement_label": "<IMP-N or similar label>",
    "improvement_title": "<title of the improvement>",
    "source": "<brief quote of the Impact line or equivalent>"
}}

Only write the JSON file. Do not write any other files or output.
</instructions>

<output>
- <output_dir>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            output_file=self.output_file,
        )


def gen_ExtractExpectedImprovementPrompt(context: str):
    return ExtractExpectedImprovementPrompt(
        context=context,
        output_file=os.path.join(RUNTIME_DIR, RUNTIME_EXPECTED_IMPROVEMENT_FILE),
    )


@dataclass
class FindSmallerModelPrompt:
    context: str
    code_trace_files: list[str]
    code_port_plan_file: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively.
- <code_port_plan_file> is the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside <target_framework>.
</definition_explanations>

<instructions>
The goal of this task is to find a smaller version of the model <model> that can be used for runtime testing of the improvement plan implementation during runtime iterations. The smaller model must exercise the same code paths that the improvement plan modifies, but require fewer GPUs of the same type (<gpu_type>).

Follow these steps and think hard:

Step 1: Analyze the Current Model
- Identify <model> architecture family, total parameter count, active parameter count, precision format, and key characteristics.
- Determine the current GPU requirements: how many <gpu_type> GPUs are needed, what tensor-parallel-size is used.
- Understand the model's architecture components: attention mechanism (MHA, MQA, GQA, MLA, CSA, HCA, etc.), MoE structure (number of experts, active experts), KV cache format, and any special features.

Step 2: Analyze the Improvement Plan
- Read and understand the code port plan from <code_port_plan_file>.
- Read the framework code traces from <code_trace_files>.
- Identify which architectural components are modified by the improvement plan: attention kernels, MoE routing, KV cache management, scheduling, CUDA graph handling, specific operator implementations, etc.
- Determine what code paths MUST be exercised to validate the improvement.

Step 3: Find a Smaller Model
- Search for smaller models in the same family as <model>. Look for:
    - Smaller variants from the same organization (e.g., Flash, Mini, Lite, Small versions)
    - Models with fewer total parameters but the same architecture
    - Models that use the same attention mechanism and kernel interfaces
- The smaller model MUST:
    - Share the same architecture and code paths that the improvement plan modifies
    - Be compatible with <target_framework> (same serving flags, same attention mechanism, same kernel interfaces)
    - Require fewer <gpu_type> GPUs than <model>
    - Be publicly available with open weights (e.g., on Hugging Face)
- Prefer models that:
    - Are from the same model family and organization as <model>
    - Use the same or compatible precision format (FP4/FP8)
    - Exercise identical kernel code paths for the improvement plan
    - Have the same MoE structure (if the improvement involves MoE-related code)

Step 4: Validate the Choice
- Verify that the improvement plan's code changes would be exercised when running the smaller model.
- Check that the smaller model uses the same attention mechanism, MoE structure (if applicable), and kernel interfaces that the improvement plan modifies.
- Confirm the GPU count reduction is significant.
- Assess risks: different expert counts, different hidden dimensions, different head counts, different batch behavior, and whether these differences affect the code paths under test.

Step 5: Write the Output
- Write the results to <output_dir>/{output_file} with the following structure:
    - First line must be exactly: SMALLER_MODEL_NAME: <full_model_name_on_huggingface>
    - Second line must be exactly: SMALLER_MODEL_NUM_GPUS: <number_of_gpus_needed>
    - Third line: blank
    - Then a detailed summary explaining:
        - Why this smaller model is appropriate for testing the improvement plan
        - Which architectural components are shared with <model>
        - What code paths from the improvement plan are exercised by both models
        - The GPU reduction achieved (from N GPUs to M GPUs of <gpu_type>)
        - Any limitations or differences to be aware of during testing
        - The recommended execution command adjustments (--model, --tensor-parallel-size, CUDA_VISIBLE_DEVICES)
</instructions>

<output>
- <output_dir>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            output_file=self.output_file,
        )


def gen_FindSmallerModelPrompt(
    context: str,
    code_trace_files: list[str],
    code_port_plan_file: str,
):
    assert len(code_trace_files) == 2

    return FindSmallerModelPrompt(
        context=context,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_file=os.path.join(RUNTIME_DIR, RUNTIME_SMALLER_MODEL_FILE),
    )


@dataclass
class ApplyCodeAndCompilePrompt:
    context: str
    patch_file: str
    iteration: int
    output_status_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<patch_file>
{patch_file}
</patch_file>
</definitions>

<instructions>
The goal of this task is to apply the code patch <patch_file> to the "target" framework source code in <target_source_code_dir> and perform incremental compilation (runtime iteration {iteration}).

IMPORTANT: If <target_branch> is provided in the context, ensure you are on that branch before proceeding. Do NOT examine or use code from other branches.

Follow these steps precisely:

Step 1: Check if patch is already committed (post-rebase case)
- Run `git -C <target_source_code_dir> log --oneline -1` to see the latest commit.
- Run `git -C <target_source_code_dir> diff --stat HEAD~1..HEAD` to see what the latest commit changed.
- Check if the patch file <output_dir>/{patch_file} content matches what's already committed:
    - Compare `git -C <target_source_code_dir> diff HEAD~1..HEAD` against the patch file content
    - If they match (same files modified, same changes), the patch is already committed (e.g., after a rebase). Print: "Patch already committed in HEAD. Skipping apply, proceeding to compile."
    - Skip Step 2 entirely and proceed to Step 3.
- If the patch is NOT already committed, proceed to Step 2.

Step 2: Verify Clean Repository State and Correct Branch
- If <target_branch> is provided, verify you are on that branch: `git -C <target_source_code_dir> branch --show-current`. If not on <target_branch>, print a clear ERROR and STOP.
- Run `git -C <target_source_code_dir> status --porcelain` to check for any modifications or uncommitted changes.
- If the repository has ANY modifications (staged or unstaged tracked files):
    - Print a clear WARNING message: "WARNING: Repository at <target_source_code_dir> is not clean. Cannot apply patch."
    - List all dirty files.
    - Write "REPO_NOT_CLEAN" to <output_dir>/{output_status_file}
    - STOP immediately. Do NOT proceed to the next steps.
- If the repository is clean, apply the patch:
    - Apply the patch to <target_source_code_dir> using: `cd <target_source_code_dir> && git apply <output_dir>/{patch_file}`
    - If `git apply` fails, try with: `cd <target_source_code_dir> && git apply --3way <output_dir>/{patch_file}`
    - If that also fails, try with: `cd <target_source_code_dir> && patch -p1 < <output_dir>/{patch_file}`
    - Verify the patch was applied correctly by checking `git -C <target_source_code_dir> diff --stat`
    - IMPORTANT: ALL code modifications MUST be exclusively inside <target_source_code_dir>. Do NOT modify files outside of it.

Step 3: Incremental Compilation
- Run incremental compilation for the "target" framework in <target_source_code_dir>:
    1. `cd <target_source_code_dir> && python tools/generate_cmake_presets.py --force-overwrite`
    2. `cd <target_source_code_dir> && cmake --preset release`
    3. `cd <target_source_code_dir> && cmake --build --preset release --target install` (use all available CPUs for speed)
- If compilation fails:
    - Analyze the compilation errors carefully.
    - Fix the code issues directly in <target_source_code_dir>.
    - Re-run the compilation commands.
    - Repeat until compilation fully succeeds with zero errors.
    - Do NOT give up on compilation — iterate until it succeeds.

Step 4: Write Success Status
- Write "SUCCESS" to <output_dir>/{output_status_file}
</instructions>

<output>
- <output_dir>/{output_status_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            patch_file=self.patch_file,
            iteration=self.iteration,
            output_status_file=self.output_status_file,
        )


def gen_ApplyCodeAndCompilePrompt(
    context: str,
    patch_file: str,
    iteration: int,
):
    return ApplyCodeAndCompilePrompt(
        context=context,
        patch_file=patch_file,
        iteration=iteration,
        output_status_file=os.path.join(
            RUNTIME_DIR, f"V{iteration}", "apply_status.txt"
        ),
    )


@dataclass
class RunAndLogPrompt:
    context: str
    execution_command: str
    runtime_logs_dir: str
    iteration: int
    gpu_wait_timeout_minutes: int
    smaller_model_file: str | None
    disable_new_feature: bool
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<execution_command>
{execution_command}
</execution_command>
<runtime_logs_dir>
{runtime_logs_dir}
</runtime_logs_dir>
<gpu_wait_timeout_minutes>
{gpu_wait_timeout_minutes}
</gpu_wait_timeout_minutes>
{smaller_model_section}
</definitions>

<instructions>
The goal of this task is to run the benchmark execution command for <tested_execution> and capture all output (runtime iteration {iteration}).
{smaller_model_instructions}

Follow these steps precisely:

Step 1: Wait for GPUs to Become Available
- Determine the number of GPUs needed (K):
{gpu_count_instructions}
- Run `nvidia-smi` and analyze the output for ALL GPUs on the machine.
- Find K GPUs that are free — i.e. not running other significant workloads (training jobs, other inference servers, large GPU processes). Small monitoring processes (a few MB of VRAM) are acceptable. The free GPUs do NOT need to be consecutive — any K free GPUs will work.
- If fewer than K GPUs are free:
    - Print a status message listing the busy GPUs and their conflicting processes (PID, process name, GPU memory usage), and how many free GPUs were found vs. needed.
    - Sleep for 60 seconds, then re-check by running `nvidia-smi` again.
    - Keep polling in this loop until K GPUs are free.
    - Track elapsed wait time. If the total wait exceeds <gpu_wait_timeout_minutes> minutes:
        - Print "ERROR: Not enough free GPUs after <gpu_wait_timeout_minutes> minutes, giving up."
        - Write "GPUS_TIMEOUT" to {runtime_logs_dir}/run_status.txt
        - STOP immediately. Do NOT proceed.
    - Each poll iteration, print the current wait time and remaining timeout so progress is visible.
- Once K free GPUs are found, record their GPU IDs (e.g. 2,5,6). These will be used for CUDA_VISIBLE_DEVICES in Step 3.

Step 2: Create Output Directory
- Ensure {runtime_logs_dir} exists: `mkdir -p {runtime_logs_dir}`

Step 3: Run the Execution Command
- The original execution command is: <execution_command>
- Before running, make these adjustments:
    - Set CUDA_VISIBLE_DEVICES to the specific free GPU IDs discovered in Step 1 (e.g. CUDA_VISIBLE_DEVICES=2,5,6). Replace any existing CUDA_VISIBLE_DEVICES in the command.
{command_model_adjustments}
{feature_toggle_env_var}
    - If the command contains Docker-internal paths (paths starting with `/app/`), replace them with the corresponding local paths.
    - If the command has `--output-json <path>`, update the path to: {runtime_logs_dir}/bench_result.json
    - Ensure all other paths in the command are valid local paths.
- Write the final adjusted command (the exact command you are about to run, including all env var prefixes) to {runtime_logs_dir}/run_command.txt
- Run the adjusted command and capture ALL output:
    - Use a script or shell construct to capture stdout and stderr separately AND as a combined stream:
      ```
      <adjusted_command> > {runtime_logs_dir}/run_stdout.log 2> {runtime_logs_dir}/run_stderr.log ; echo $? > {runtime_logs_dir}/run_exit_code.txt
      ```
    - Then create the combined log:
      ```
      cat {runtime_logs_dir}/run_stdout.log {runtime_logs_dir}/run_stderr.log > {runtime_logs_dir}/run_combined.log
      ```
    - Alternatively, use `tee` and process substitution to capture everything in real time.

Step 4: Write Run Status
- Read the exit code from {runtime_logs_dir}/run_exit_code.txt
- If exit code is 0: write "COMPLETED" to {runtime_logs_dir}/run_status.txt
- If exit code is non-zero: write "FAILED" to {runtime_logs_dir}/run_status.txt
- In both cases, the detailed output is in the log files for subsequent analysis.

{feature_verification_step}
</instructions>

<output>
- {runtime_logs_dir}/run_command.txt
- {runtime_logs_dir}/run_stdout.log
- {runtime_logs_dir}/run_stderr.log
- {runtime_logs_dir}/run_combined.log
- {runtime_logs_dir}/run_exit_code.txt
- {runtime_logs_dir}/run_status.txt
{baseline_output_files}
</output>

"""

    def prompt(self):
        if self.smaller_model_file:
            smaller_section = (
                "<smaller_model_file>\n"
                f"{self.smaller_model_file}\n"
                "</smaller_model_file>"
            )
            smaller_instructions = (
                "\n\nIMPORTANT: A smaller model is being used for runtime testing instead of <model>. "
                "Read <smaller_model_file> from <output_dir> to get the smaller model name (SMALLER_MODEL_NAME) "
                "and GPU count (SMALLER_MODEL_NUM_GPUS). Use these values when adjusting the execution command "
                "in Step 3, and use SMALLER_MODEL_NUM_GPUS as K in Step 1."
            )
            gpu_count_instructions = "    - Read SMALLER_MODEL_NUM_GPUS from <smaller_model_file> in <output_dir> — that is K."
            command_model_adjustments = (
                "    - Replace the --model value with SMALLER_MODEL_NAME from <smaller_model_file>.\n"
                "    - Replace --tensor-parallel-size with K (SMALLER_MODEL_NUM_GPUS)."
            )
        else:
            smaller_section = ""
            smaller_instructions = ""
            gpu_count_instructions = (
                "    - K is the tensor-parallel-size from <execution_command> "
                "(or the count of GPUs in CUDA_VISIBLE_DEVICES)."
            )
            command_model_adjustments = ""

        if self.disable_new_feature:
            feature_toggle_env_var = (
                "    - Read the latest code gen or runtime summary file in <output_dir> to find the "
                "environment variable name that disables the new feature.\n"
                '    - Set this environment variable to "1" in the command prefix to disable the new feature.'
            )
            feature_verification_step = (
                "Step 5: Verify Feature is DISABLED\n"
                "- Search the combined log for the expected disabled message containing "
                '"is DISABLED (env override)".\n'
                "- If found, the feature was successfully disabled.\n"
                "- If NOT found, print a WARNING that the feature disable mechanism may not be working."
            )
        else:
            feature_toggle_env_var = ""
            feature_verification_step = (
                "Step 5: Verify Feature is ENABLED\n"
                "- Search the combined log for the expected enabled message containing "
                '"is ENABLED".\n'
                "- If found, the new feature is confirmed active — good.\n"
                "- If NOT found, print a WARNING that the new feature may not be running. "
                'Write "FEATURE_NOT_VERIFIED" to {runtime_logs_dir}/run_status.txt '
                "and include details about what was searched for and what was found instead.\n"
                "\n"
                "Step 6: Baseline Comparison Run (only if the patched run in Step 3 succeeded with exit code 0 AND Step 5 confirmed the feature is enabled)\n"
                "- Read the latest code gen or runtime summary file in <output_dir> to find the environment variable "
                "name that disables the new feature.\n"
                '- Re-run the SAME adjusted command from Step 3, but with the disable env var set to "1" in the '
                "command prefix. This runs the original code paths WITHOUT the new feature, to establish a baseline.\n"
                "- Write the final baseline command (including the disable env var prefix) to "
                "{runtime_logs_dir}/baseline_command.txt\n"
                "- Capture output to:\n"
                "    - {runtime_logs_dir}/baseline_stdout.log\n"
                "    - {runtime_logs_dir}/baseline_stderr.log\n"
                "    - {runtime_logs_dir}/baseline_combined.log\n"
                "    - {runtime_logs_dir}/baseline_exit_code.txt\n"
                "  If --output-json is present, set it to {runtime_logs_dir}/baseline_bench_result.json\n"
                '- Verify the baseline run log contains the "is DISABLED (env override)" message.\n'
                "- The baseline run MUST succeed (exit code 0), since it runs the original unmodified code paths. "
                "If it fails:\n"
                "    - This indicates something is wrong with the patch's disable mechanism or the patch broke original code paths.\n"
                "    - Analyze the error, document it, and note it in the status file.\n"
                "    - Fix the issue in <target_source_code_dir> and retry the baseline run until it succeeds.\n"
                "- Once the baseline run succeeds, extract performance results from BOTH runs:\n"
                "    - Patched run (Step 3): throughput, TTFT, TPOT\n"
                "    - Baseline run (Step 6): throughput, TTFT, TPOT\n"
                "- Write the comparison to {runtime_logs_dir}/perf_comparison.txt:\n"
                "    - Patched results (feature ENABLED)\n"
                "    - Baseline results (feature DISABLED)\n"
                "    - Delta and percentage change for each metric (TTFT, TPOT, throughput)\n"
                "    - Analysis: whether the patch provides the expected performance improvement\n"
                "    - Clear statement of whether the patched version is faster or slower than baseline, and by how much"
            )

        if self.disable_new_feature:
            baseline_output_files = ""
        else:
            baseline_output_files = (
                f"- {self.runtime_logs_dir}/baseline_command.txt\n"
                f"- {self.runtime_logs_dir}/baseline_stdout.log\n"
                f"- {self.runtime_logs_dir}/baseline_stderr.log\n"
                f"- {self.runtime_logs_dir}/baseline_combined.log\n"
                f"- {self.runtime_logs_dir}/baseline_exit_code.txt\n"
                f"- {self.runtime_logs_dir}/perf_comparison.txt"
            )

        return self.prompt_template.format(
            context=self.context,
            execution_command=self.execution_command,
            runtime_logs_dir=self.runtime_logs_dir,
            iteration=self.iteration,
            gpu_wait_timeout_minutes=self.gpu_wait_timeout_minutes,
            smaller_model_section=smaller_section,
            smaller_model_instructions=smaller_instructions,
            gpu_count_instructions=gpu_count_instructions,
            command_model_adjustments=command_model_adjustments,
            feature_toggle_env_var=feature_toggle_env_var,
            feature_verification_step=feature_verification_step,
            baseline_output_files=baseline_output_files,
        )


def gen_RunAndLogPrompt(
    context: str,
    execution_command: str,
    runtime_logs_dir: str,
    iteration: int,
    gpu_wait_timeout_minutes: int = 30,
    smaller_model_file: str | None = None,
    disable_new_feature: bool = False,
):
    return RunAndLogPrompt(
        context=context,
        execution_command=execution_command,
        runtime_logs_dir=runtime_logs_dir,
        iteration=iteration,
        gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
        smaller_model_file=smaller_model_file,
        disable_new_feature=disable_new_feature,
    )


RUNTIME_LM_EVAL_FILE_PREFIX = "runtime_lm_eval"


@dataclass
class RunLMEvalPrompt:
    context: str
    execution_command: str
    runtime_logs_dir: str
    iteration: int
    gpu_wait_timeout_minutes: int
    smaller_model_file: str | None
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<execution_command>
{execution_command}
</execution_command>
<runtime_logs_dir>
{runtime_logs_dir}
</runtime_logs_dir>
<gpu_wait_timeout_minutes>
{gpu_wait_timeout_minutes}
</gpu_wait_timeout_minutes>
{smaller_model_section}
</definitions>

<instructions>
The goal of this task is to run lm_eval correctness checks on the patched "target" framework with the new feature both ENABLED and DISABLED, and compare the results (runtime iteration {iteration}).

Follow these steps precisely:

Step 1: Wait for GPUs to Become Available
- Determine the number of GPUs needed (K):
{gpu_count_instructions}
- Run `nvidia-smi` and analyze the output for ALL GPUs on the machine.
- Find K GPUs that are free — i.e. not running other significant workloads. Small monitoring processes are acceptable. The free GPUs do NOT need to be consecutive.
- If fewer than K GPUs are free:
    - Print a status message listing the busy GPUs and their conflicting processes.
    - Sleep for 60 seconds, then re-check.
    - Track elapsed wait time. If the total wait exceeds <gpu_wait_timeout_minutes> minutes:
        - Print "ERROR: Not enough free GPUs after <gpu_wait_timeout_minutes> minutes, giving up."
        - Write "GPUS_TIMEOUT" to {runtime_logs_dir}/{output_file}
        - STOP immediately. Do NOT proceed.
    - Each poll iteration, print the current wait time and remaining timeout.
- Once K free GPUs are found, record their GPU IDs (e.g. 2,5,6).

Step 2: Determine Model and Parameters
- Determine the model to use:
{model_selection_instructions}
- Determine tensor_parallel_size = K (number of GPUs).
- Determine max_model_len: use the --max-model-len value from <execution_command>. If this value is very large and may cause OOM for lm_eval, reduce it to a reasonable value (e.g., 16384 or 32768) that fits in available VRAM while being sufficient for gsm8k 5-shot evaluation.

Step 3: Run lm_eval with Feature ENABLED
- Read the latest code gen or runtime summary file in <output_dir> to find the environment variable name that disables the new feature.
- Ensure this env var is NOT set (or unset it) so the new feature is active.
- Construct and run the lm_eval command:
    ```
    CUDA_VISIBLE_DEVICES=<free_gpu_ids> lm_eval --model vllm --model_args pretrained=<model>,max_model_len=<len>,tensor_parallel_size=<K> --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto
    ```
- Capture all output to {runtime_logs_dir}/lm_eval_enabled_stdout.log and stderr to {runtime_logs_dir}/lm_eval_enabled_stderr.log
- Record exit code to {runtime_logs_dir}/lm_eval_enabled_exit_code.txt
- Extract the exact_match scores (flexible-extract and strict-match) from the output.

Step 4: Run lm_eval with Feature DISABLED
- Set the disable env var to "1" in the command prefix to disable the new feature.
- Run the same lm_eval command with the disable env var:
    ```
    <DISABLE_ENV_VAR>=1 CUDA_VISIBLE_DEVICES=<free_gpu_ids> lm_eval --model vllm --model_args pretrained=<model>,max_model_len=<len>,tensor_parallel_size=<K> --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto
    ```
- Capture output to {runtime_logs_dir}/lm_eval_disabled_stdout.log and stderr to {runtime_logs_dir}/lm_eval_disabled_stderr.log
- Record exit code to {runtime_logs_dir}/lm_eval_disabled_exit_code.txt
- Extract the exact_match scores from the output.

Step 5: Compare and Report
- Compare the exact_match scores from both runs:
    - Feature ENABLED: flexible-extract exact_match, strict-match exact_match
    - Feature DISABLED (baseline): flexible-extract exact_match, strict-match exact_match
    - Delta for each metric (enabled - disabled)
- Analyze the results:
    - If the enabled scores are equal or very close to disabled scores (within ±0.005), correctness is preserved — GOOD.
    - If the enabled scores are significantly LOWER than disabled scores (drop > 0.01), this indicates a correctness regression — the patch may be introducing errors. Document this clearly.
    - If the enabled scores are HIGHER than disabled scores, this is unexpected but acceptable — document it.
    - If either run failed (non-zero exit code), document the error and which run failed.
- Write the full comparison to {runtime_logs_dir}/{output_file}:
    - Feature ENABLED results: exact_match scores, exit code
    - Feature DISABLED results: exact_match scores, exit code
    - Delta and analysis
    - Clear PASS/FAIL verdict: PASS if correctness is preserved, FAIL if significant degradation detected
    - If FAIL: detailed description of the degradation for the investigation step to fix
</instructions>

<output>
- {runtime_logs_dir}/{output_file}
- {runtime_logs_dir}/lm_eval_enabled_stdout.log
- {runtime_logs_dir}/lm_eval_enabled_stderr.log
- {runtime_logs_dir}/lm_eval_enabled_exit_code.txt
- {runtime_logs_dir}/lm_eval_disabled_stdout.log
- {runtime_logs_dir}/lm_eval_disabled_stderr.log
- {runtime_logs_dir}/lm_eval_disabled_exit_code.txt
</output>

"""

    def prompt(self):
        if self.smaller_model_file:
            smaller_section = (
                "<smaller_model_file>\n"
                f"{self.smaller_model_file}\n"
                "</smaller_model_file>"
            )
            gpu_count_instructions = "    - Read SMALLER_MODEL_NUM_GPUS from <smaller_model_file> in <output_dir> — that is K."
            model_selection_instructions = "    - Read SMALLER_MODEL_NAME from <smaller_model_file> in <output_dir> — use that as the model."
        else:
            smaller_section = ""
            gpu_count_instructions = (
                "    - K is the tensor-parallel-size from <execution_command> "
                "(or the count of GPUs in CUDA_VISIBLE_DEVICES)."
            )
            model_selection_instructions = (
                "    - Use <model> (the model from the context)."
            )

        return self.prompt_template.format(
            context=self.context,
            execution_command=self.execution_command,
            runtime_logs_dir=self.runtime_logs_dir,
            iteration=self.iteration,
            gpu_wait_timeout_minutes=self.gpu_wait_timeout_minutes,
            smaller_model_section=smaller_section,
            gpu_count_instructions=gpu_count_instructions,
            model_selection_instructions=model_selection_instructions,
            output_file=self.output_file,
        )


def gen_RunLMEvalPrompt(
    context: str,
    execution_command: str,
    runtime_logs_dir: str,
    iteration: int,
    gpu_wait_timeout_minutes: int = 30,
    smaller_model_file: str | None = None,
):
    return RunLMEvalPrompt(
        context=context,
        execution_command=execution_command,
        runtime_logs_dir=runtime_logs_dir,
        iteration=iteration,
        gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
        smaller_model_file=smaller_model_file,
        output_file="lm_eval.txt",
    )


@dataclass
class InvestigateRuntimeOutputAndFixCodePrompt:
    context: str
    code_trace_files: list[str]
    code_port_plan_file: str
    test_plan_file: str
    runtime_logs_dir: str
    iteration: int
    prev_patch_file: str
    prev_summary_file: str | None
    iteration_history_summary_file: str
    smaller_model_file: str | None
    lm_eval_result_file: str | None
    output_patch_file: str
    output_summary_file: str
    success_result_file: str
    results_output_dir: str | None
    review_feedback: str | None = None
    prompt_template: ClassVar[str] = """

{context}

{prev_iteration_section}

{review_feedback_section}

<definitions>
<code_trace_files>{code_trace_files}</code_trace_files>
<code_port_plan_file>{code_port_plan_file}</code_port_plan_file>
<test_plan_file>{test_plan_file}</test_plan_file>
<runtime_logs_dir>{runtime_logs_dir}</runtime_logs_dir>
<prev_patch_file>{prev_patch_file}</prev_patch_file>
<iteration_history_summary_file>{iteration_history_summary_file}</iteration_history_summary_file>
{smaller_model_section}
</definitions>

<instructions>
Runtime iteration {iteration}. Analyze results, identify root causes, fix the issue.

STEP 1: UNDERSTAND THE FULL PICTURE
Read in this order:
A) NSYS diagnostic: <runtime_logs_dir>/diagnostic_profile/diagnostic_report.txt
   This is GROUND TRUTH — per-kernel timing comparison between patched and baseline.
   Shows exactly which kernels improved, regressed, or changed.
B) Benchmark results: <runtime_logs_dir>/perf_comparison.txt, run_status.txt, lm_eval.txt
C) Previous iterations: runtime/summary_all.txt, then ALL runtime/V*/summary.txt
   Focus on: what was tried, what approach was used, why it failed, what was learned.
D) Current patch: <output_dir>/<prev_patch_file> — the code changes applied for this run.
E) History: <iteration_history_summary_file> for code gen phases context.
F) Source framework: from <code_trace_files>, trace how <source_framework> implements
   this optimization end-to-end. It works there — understand the full mechanism,
   including how it integrates with compilation, graph capture, and runtime dispatch.
G) Target framework patterns: search <target_framework> codebase for concrete examples
   of how similar optimizations are implemented successfully — patterns that add kernel
   replacements, caching, or dispatch without causing regressions. These are proven
   templates from the same codebase. Read the actual code and note file paths.

STEP 2: DIAGNOSE ROOT CAUSES FROM NSYS
From the diagnostic report, identify EVERY root cause:
- Which specific kernels regressed and by how much (ns)?
- Which kernels improved and by how much?
- Is the net per-layer result positive or negative?
- If kernel-level improves but E2E regresses: what explains the gap?
- Compare the kernel sequence between patched and baseline — any new/removed/renamed kernels?
- What does this tell you about how the patch interacts with the framework's compilation
  and execution pipeline?
For each root cause: state the mechanism, the evidence, and what constraint it places on the fix.

STEP 3: SCORE DIRECTIONS AND DECIDE
Build a DECISION TABLE with AT LEAST 5 distinct directions before proposing any fix.
Think broadly — consider different levels of the stack (kernel, dispatch, compilation,
runtime config, benchmark methodology) and different strategies (fix the regression,
avoid the regression, work around it, measure differently, reduce the scope):
  | # | Direction | Tried before? | Result | Probability (H/M/L) | Reasoning |
For EACH direction, score based on:
  - NSYS evidence (which kernels are the problem and why?)
  - Previous iteration outcomes (what partially worked? what completely failed?)
  - Source framework approach (how does it solve this?)
  - Target framework patterns (found a working example in the codebase?)
  - Change surface area (fewer touched code paths = less risk of side effects)
  - Risk of collateral effects (compilation, graph topology, cache invalidation)
Pick the HIGHEST probability direction.
If re-trying a previously failed approach, state SPECIFICALLY what's different this time.
E2E improvement is the goal — kernel-only improvement with E2E regression means the
implementation approach is wrong, not the benchmark.

STEP 4: DETERMINE RESULT
- SUCCEEDED: exit code 0, perf delta > +2%, lm_eval PASS (if ran).
- FAILED: any error, any regression, neutral (within +-1.5%), lm_eval FAIL.
- GPUS_TIMEOUT: not a code error — note it, do NOT attempt fixes.

STEP 5A: IF SUCCEEDED
- Write success report to {results_output_dir}/{success_result_file}:
  patch name, perf delta, lm_eval scores, execution context.
- Write summary (same format as 5B). STOP — do NOT generate a new patch.

STEP 5B: IF FAILED
1. Fix: implement the highest-scored direction from Step 3.
   Apply in <target_source_code_dir> ONLY. Trace all execution modes (decode/prefill/mixed).
2. Test: run ONLY patch-introduced tests. Repeat until all pass.
3. Feature toggle: preserve env var disable switch + startup log messages.
4. Generate patch: `git diff` from <target_source_code_dir> (include untracked files).
   Must be standalone (applicable on clean base commit) and DIFFERENT from <prev_patch_file>.
   Save to EXACTLY <output_dir>/{output_patch_file} — do NOT change the path or iteration number.
5. Write summary to EXACTLY <output_dir>/{output_summary_file} — same path, no changes:
   1. RESULT: iteration, pass/fail, perf delta, lm_eval
   2. DECISION TABLE: all 5+ directions scored (full table), chosen one and why
   3. ROOT CAUSES: from NSYS — which kernels, what mechanism
   4. FIX APPLIED: what changed, why, how it addresses each root cause
   5. FILES MODIFIED: each file, what changed, why it's necessary
   6. TESTS: added/modified
   7. FEATURE TOGGLE: env var, startup messages
   8. IF THIS FAILS: next-best direction from decision table + what to look for

The optimization works in <source_framework>. If it regresses in <target_framework>,
the implementation is wrong — not the approach. Never declare it impossible.
</instructions>

<output>
IMPORTANT: Write ALL output files to EXACTLY these paths. Do NOT create new
directories or change the iteration number. These paths are pre-determined.

Always:
- <output_dir>/{output_summary_file}

On SUCCESS:
- {results_output_dir}/{success_result_file}

On FAILURE:
- <output_dir>/{output_patch_file}
</output>

"""

    def prompt(self):
        if self.smaller_model_file:
            smaller_section = (
                "<smaller_model_file>\n"
                f"{self.smaller_model_file}\n"
                "</smaller_model_file>"
            )
        else:
            smaller_section = ""

        if self.review_feedback:
            review_section = (
                "<review_feedback>\n"
                "PREVIOUS REVIEW REJECTED YOUR DRAFT. You MUST address every issue listed below.\n"
                "Do NOT repeat the same approach — the review explains what was wrong.\n\n"
                f"{self.review_feedback}\n"
                "</review_feedback>"
            )
        else:
            review_section = ""

        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_patch_file if self.prev_summary_file else None,
                self.prev_summary_file,
            ),
            review_feedback_section=review_section,
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            runtime_logs_dir=self.runtime_logs_dir,
            prev_patch_file=self.prev_patch_file,
            iteration_history_summary_file=self.iteration_history_summary_file,
            smaller_model_section=smaller_section,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            success_result_file=self.success_result_file,
            results_output_dir=self.results_output_dir,
            iteration=self.iteration,
        )


def gen_InvestigateRuntimeOutputAndFixCodePrompt(
    context: str,
    code_trace_files: list[str],
    code_port_plan_file: str,
    test_plan_file: str,
    runtime_logs_dir: str,
    iteration: int,
    prev_patch_file: str,
    prev_summary_file: str | None,
    iteration_history_summary_file: str,
    smaller_model_file: str | None = None,
    lm_eval_result_file: str | None = None,
    results_output_dir: str | None = None,
    output_patch_file: str | None = None,
    output_summary_file: str | None = None,
    review_feedback: str | None = None,
):
    assert len(code_trace_files) == 2

    if output_patch_file is None:
        output_patch_file = os.path.join(RUNTIME_DIR, f"V{iteration}", "patch.patch")
    if output_summary_file is None:
        output_summary_file = os.path.join(RUNTIME_DIR, f"V{iteration}", "summary.txt")

    return InvestigateRuntimeOutputAndFixCodePrompt(
        context=context,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        runtime_logs_dir=runtime_logs_dir,
        iteration=iteration,
        prev_patch_file=prev_patch_file,
        prev_summary_file=prev_summary_file,
        iteration_history_summary_file=iteration_history_summary_file,
        smaller_model_file=smaller_model_file,
        lm_eval_result_file=lm_eval_result_file,
        output_patch_file=output_patch_file,
        output_summary_file=output_summary_file,
        success_result_file=RUNTIME_SUCCESS_FILE,
        results_output_dir=results_output_dir,
        review_feedback=review_feedback,
    )


@dataclass
class ReviewInvestigationPrompt:
    runtime_logs_dir: str
    iteration: int
    revision: int
    patch_draft_file: str
    summary_draft_file: str
    output_review_file: str
    prior_review_files: list[str]
    prompt_template: ClassVar[str] = """
<instructions>
Review the proposed fix for runtime iteration V{iteration} (revision R{revision}).

Read these files:
1. Proposed patch: <output_dir>/{patch_draft_file}
2. Proposed summary: <output_dir>/{summary_draft_file}
3. NSYS diagnostic: {runtime_logs_dir}/diagnostic_profile/diagnostic_report.txt (if exists)
4. Benchmark results: {runtime_logs_dir}/perf_comparison.txt, run_status.txt
5. Previous iteration summaries: runtime/summary_all.txt, runtime/V*/summary.txt
{prior_reviews_section}

Evaluate each dimension and write your verdict to <output_dir>/{output_review_file}:

1. ROOT CAUSE COVERAGE
   From the NSYS diagnostic report, list every identified root cause.
   For each: does the proposed patch address it? Mark [OK] or [MISSING].
   If any root cause is ignored without justification, flag it.

2. PRIOR ATTEMPTS
   Check ALL previous iteration summaries (V1 through V{prev_iteration}).
   Was this approach (or a close variant) tried before?
   If yes: what SPECIFICALLY differs this time? If nothing differs, mark [REPEAT].

3. CHANGE SURFACE
   List every file and function the patch modifies.
   For each: is the change necessary for the fix? Mark [NECESSARY] or [UNNECESSARY].
   Could the fix be achieved with fewer modifications?

4. EXECUTION MODES
   Will the fix work correctly in decode, prefill, and mixed modes?
   Flag any path that is untested or has conditional logic that may diverge.

5. DECISION TABLE
   Does the scoring in the summary make sense given the NSYS evidence?
   Are the probability ratings justified?

Write the final verdict on the FIRST LINE of the review file:
  VERDICT: APPROVED
  VERDICT: NEEDS_REVISION

If NEEDS_REVISION, list each issue as a numbered item with specific guidance.
</instructions>

<output>
- <output_dir>/{output_review_file}
</output>
"""

    def prompt(self):
        if self.prior_review_files:
            lines = ["6. Prior review feedback (issues that should now be fixed):"]
            for rf in self.prior_review_files:
                lines.append(f"   - <output_dir>/{rf}")
            prior_section = "\n".join(lines)
        else:
            prior_section = ""

        return self.prompt_template.format(
            runtime_logs_dir=self.runtime_logs_dir,
            iteration=self.iteration,
            revision=self.revision,
            patch_draft_file=self.patch_draft_file,
            summary_draft_file=self.summary_draft_file,
            output_review_file=self.output_review_file,
            prior_reviews_section=prior_section,
            prev_iteration=self.iteration - 1,
        )


def gen_ReviewInvestigationPrompt(
    runtime_logs_dir: str,
    iteration: int,
    revision: int,
    prior_review_files: list[str] | None = None,
):
    iter_dir = os.path.join(RUNTIME_DIR, f"V{iteration}")
    return ReviewInvestigationPrompt(
        runtime_logs_dir=runtime_logs_dir,
        iteration=iteration,
        revision=revision,
        patch_draft_file=os.path.join(iter_dir, f"patch_draft_R{revision}.patch"),
        summary_draft_file=os.path.join(iter_dir, f"summary_draft_R{revision}.txt"),
        output_review_file=os.path.join(iter_dir, f"review_R{revision}.txt"),
        prior_review_files=prior_review_files or [],
    )


@dataclass
class IterationHistorySummaryPrompt:
    context: str
    code_gen_output_dir: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_gen_output_dir>
{code_gen_output_dir}
</code_gen_output_dir>
</definitions>

<instructions>
The goal of this task is to read and analyze ALL previous code generation iteration results to build a comprehensive understanding of the work done so far. This is required before starting runtime iterations.

Follow these steps:

Step 0: Check for Existing Summaries
- Check if <output_dir>/{output_file} already exists from a previous run.
- If it exists, read it first — use it as a starting point, only read NEW artifacts.
- Check if <output_dir>/runtime/{summary_all_file} exists (cumulative iteration evolution summary).
- If it exists, read it — it contains the executive summary, results table, timeline, key findings, and next steps from all previous runtime iterations. Use it as the authoritative source for runtime iteration history instead of re-reading individual V*/summary.txt files.

Step 1: Read Generated Artifacts
- List all files in <code_gen_output_dir> and its subdirectories.
- Read and analyze each file in the following order of phases:
    1. Framework code traces (code_traces/*_code_trace.txt) — understand the active code paths in both "source" and "target" frameworks for <tested_execution>.
    2. Code port plan iterations (code_port_plan/V*.txt, code_port_plan/V*_summary.txt) — understand the planned porting steps and their evolution.
    3. Code port plan reviews (code_port_plan/V*_review.txt, code_port_plan/V*_review_summary.txt) — understand issues found and fixed during plan review.
    4. Test plan iterations (test_plan/V*.txt, test_plan/V*_summary.txt) — understand the testing strategy.
    5. Test plan reviews (test_plan/V*_review.txt, test_plan/V*_review_summary.txt) — understand test review fixes.
    6. Code gen iterations (code_gen/V*.patch, code_gen/V*_summary.txt) — understand the generated code patches.
    7. Code gen reviews (code_gen/V*_review.patch, code_gen/V*_review_summary.txt) — understand code review fixes and the final patch.
    8. Any previous runtime iteration files (runtime/V*/patch.patch, runtime/V*/summary.txt) — understand previous runtime fix attempts.
    9. Any runtime logs in runtime/V*/ subdirectories — understand previous execution results.
    10. Any runtime success result (runtime/success_result.txt) — if already succeeded.

Step 2: Build Comprehensive Understanding
- For each phase, understand:
    - What was generated in each iteration
    - What issues/bugs were found during reviews
    - How issues were fixed and what the key learnings were
    - The convergence status (did it converge, at which iteration?)
- For code patches, understand:
    - What code was changed/added/ported in <target_source_code_dir>
    - What tests were created
    - What compilation issues were encountered and how they were resolved
- For any runtime iterations already done:
    - What runtime errors occurred during execution
    - How they were diagnosed and fixed
    - What the current state of the code is
- Identify the latest/final code patch (either code_gen/V<N>_review.patch or runtime/V<N>/patch.patch, where <N> is the highest iteration number)

Step 3: Generate History Summary
- Write a comprehensive history summary to <output_dir>/{output_file} that includes:
    - Overview of all phases completed (code traces, port plan, test plan, code gen)
    - Key decisions and changes in each phase, with iteration counts
    - The iteration evolution across all phases: what was found, fixed, and learned
    - Issues found and fixed throughout the entire process
    - Current state: which patch file is the latest, what is the code status
    - Patterns of recurring issues across iterations
    - Key learnings that should inform upcoming runtime iterations
    - The final code port plan summary and test plan summary
    - A clear statement of what the code is expected to do and what risks remain

- NEVER DECLARE IMPOSSIBLE: if the optimization works in the source framework, a working implementation EXISTS. State what was tried, why it failed, what the source does differently, and what to try next. Keep the door open.

- IMPORTANT: The VERY LAST LINE of the file MUST be a marker indicating the highest runtime iteration directory (runtime/V<K>/) that existed when this summary was generated. Use this exact format:
    LAST_ITERATION_USED=V<K>
  If no runtime iterations exist yet (no runtime/V*/ directories), write:
    LAST_ITERATION_USED=V0
</instructions>

<output>
- <output_dir>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_gen_output_dir=self.code_gen_output_dir,
            output_file=self.output_file,
            summary_all_file=SUMMARY_ALL_FILE,
        )


def gen_IterationHistorySummaryPrompt(
    context: str,
    code_gen_output_dir: str,
):
    return IterationHistorySummaryPrompt(
        context=context,
        code_gen_output_dir=code_gen_output_dir,
        output_file=os.path.join(RUNTIME_DIR, "iterations_history.txt"),
    )


SUMMARY_ALL_FILE = "summary_all.txt"


@dataclass
class CumulativeSummaryPrompt:
    runtime_dir: str
    iteration: int
    output_file: str
    prompt_template: ClassVar[str] = """
<instructions>
Write a cumulative iteration evolution summary covering ALL runtime iterations V1 through V{iteration}.

Read these files:
- {runtime_dir}/summary_all.txt (previous cumulative summary — if it exists, UPDATE it, don't start from scratch)
- {runtime_dir}/V*/summary.txt (per-iteration investigation summaries)
- {runtime_dir}/V*/perf_comparison.txt (benchmark results)
- {runtime_dir}/V*/lm_eval.txt (correctness results)
- {runtime_dir}/V*/candidates/*/summary.txt (candidate investigation summaries, if parallel)
- {runtime_dir}/V*/diagnostic_profile/diagnostic_report.txt (nsys kernel analysis, if exists)

Write to <output_dir>/{output_file} with this structure:

1. EXECUTIVE SUMMARY (3-5 sentences): current best result, overall trend, biggest blocker, next action.

2. RESULTS TABLE:
   | Iter | Throughput Delta | lm_eval | Winner | Key Change |
   One row per iteration.

3. ITERATION TIMELINE: for each V{{N}}, one paragraph: what patch was applied, what happened, why it failed/succeeded, what was learned.

4. KEY FINDINGS: bullet list of confirmed facts (from NSYS, benchmarks, lm_eval).

5. OPEN QUESTIONS: what's still unknown or needs investigation.

6. NEXT STEPS: what the next iteration should try, based on the accumulated evidence.

Keep it concise — one page per 3 iterations. Update the executive summary to reflect the latest state.
</instructions>

<output>
- <output_dir>/{output_file}
</output>
"""

    def prompt(self):
        return self.prompt_template.format(
            runtime_dir=self.runtime_dir,
            iteration=self.iteration,
            output_file=self.output_file,
        )


def gen_CumulativeSummaryPrompt(runtime_dir: str, iteration: int):
    return CumulativeSummaryPrompt(
        runtime_dir=runtime_dir,
        iteration=iteration,
        output_file=os.path.join(RUNTIME_DIR, SUMMARY_ALL_FILE),
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
    code_trace_files: list[str]
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
<code_trace_files>
{code_trace_files}
</code_trace_files>
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
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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
    - A comparison one-to-one, on the same slide, of the code traces of both frameworks from <code_trace_files>. Make sure to show:
        - The full operation call-chains with all the necessary details, with the specific per-operation time breakdowns based on the median transformer blocks.
        - Explain the differences that can be seen in both the code traces and the transformer blocks, how the correlate and why the "target" is faster.
    - Explain what needs to be done to port the "source" code pieces to the "target" codebase. Base the explanation on the code port plan file. Make sure to explain:
        - Each critical code piece that is ported from the "source", what stays the same, what is changed, and how this code piece is integrated into the "target" codebase. Make sure to show actual code, and explain fully and clearly, so expert programmer can understand.
    - Show and explain in-detail the Claude query prompt that is used to generate the code port plan, based on the source code here llm/auto_code_gen/code_gen_prompts.py (in this repository).
    - Show and explain in-detail the Claude query prompt that is used to generate the code port plan review, based on the source code here llm/auto_code_gen/code_gen_prompts.py (in this repository).
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
            code_trace_files=self.code_trace_files,
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
    code_trace_files: list[str],
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
    assert len(code_trace_files) == 2

    return SummarizeCodeGenProcessPrompt(
        context=context,
        code_trace_files=code_trace_files,
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


@dataclass
class CollectBenchmarkResultsPrompt:
    context: str
    code_gen_output_dir: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_gen_output_dir>
{code_gen_output_dir}
</code_gen_output_dir>
</definitions>

<definition_explanations>
- <code_gen_output_dir> is the directory containing all pipeline output files, including runtime results and any additional benchmark subdirectories.
</definition_explanations>

<instructions>
The goal of this task is to collect all benchmark results (performance and correctness) from the pipeline output into a single structured JSON manifest. Do NOT run any benchmarks yourself. Only read existing result files and produce the manifest.

Follow these steps:

Step 1: Check for existing manifest.
- Check if <output_dir>/{output_file} already exists. If it does, read it — it is the starting point. You will update it with any new results found.

Step 2: Collect benchmark results from all model runs.
- Each model run stores its results in a separate subdirectory inside <code_gen_output_dir> matching the pattern runtime_*/ (e.g., runtime_original/, runtime_smaller/, runtime_deepseek_v4_flash_2xh200/).
- List all runtime_*/ subdirectories in <code_gen_output_dir>.
- For each subdirectory, read the following files (if they exist):
    - runtime/success_result.txt — the benchmark success result for this model
    - runtime/V*/perf_comparison.txt — performance comparison files (use the highest iteration number)
    - runtime/V*/lm_eval.txt — lm_eval correctness results (use the highest iteration number)
- Extract from each: model name, gpu_type, num_gpus, batch_size, isl, osl, throughput/TPOT/TTFT for both baseline (feature disabled) and patched (feature enabled), lm_eval scores for both, verdict.
- The subdirectory name indicates the model (e.g., runtime_original/ is the primary model, runtime_smaller/ is the auto-detected smaller model).

Step 4: Write the manifest.
- Write <output_dir>/{output_file} as a JSON array where each entry has:
    {{
        "label": "<human-readable label, e.g., DeepSeek-V3 (8xH200)>",
        "model": "<model name>",
        "gpu_type": "<GPU type>",
        "num_gpus": <number>,
        "batch_size": "<batch size>",
        "isl": "<input sequence length>",
        "osl": "<output sequence length>",
        "throughput_baseline": <number or null>,
        "throughput_patched": <number or null>,
        "tpot_baseline_ms": <number or null>,
        "tpot_patched_ms": <number or null>,
        "ttft_baseline_ms": <number or null>,
        "ttft_patched_ms": <number or null>,
        "lm_eval_baseline": {{"metric": <score>}} or null,
        "lm_eval_patched": {{"metric": <score>}} or null,
        "lm_eval_verdict": "<PASS/FAIL/null>",
        "source_dir": "<which directory the results came from>"
    }}
- The primary benchmark is the first entry. Additional benchmarks follow in order.
- If a field cannot be extracted, set it to null.
- Preserve existing entries in the manifest that are not updated.
</instructions>

<output>
- Write the JSON manifest to <output_dir>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_gen_output_dir=self.code_gen_output_dir,
            output_file=self.output_file,
        )


def gen_CollectBenchmarkResultsPrompt(
    context: str,
    code_gen_output_dir: str,
):
    return CollectBenchmarkResultsPrompt(
        context=context,
        code_gen_output_dir=code_gen_output_dir,
        output_file=os.path.join(RUNTIME_DIR, RUNTIME_RESULTS_MANIFEST),
    )


@dataclass
class RebasePRPrompt:
    context: str
    target_repo_dir: str
    branch_name: str
    pr_base_branch: str
    pr_remote: str
    code_gen_output_dir: str
    latest_main_branch: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<target_repo_dir>
{target_repo_dir}
</target_repo_dir>
<branch_name>
{branch_name}
</branch_name>
<pr_base_branch>
{pr_base_branch}
</pr_base_branch>
<pr_remote>
{pr_remote}
</pr_remote>
<code_gen_output_dir>
{code_gen_output_dir}
</code_gen_output_dir>
<latest_main_branch>
{latest_main_branch}
</latest_main_branch>
</definitions>

<definition_explanations>
- <target_repo_dir> is the path to the target framework/project repository where the patch has been applied and committed.
- <branch_name> is the local branch that has the committed patch (from a previous PR gen commit step).
- <pr_base_branch> is the upstream base branch (e.g., "main").
- <pr_remote> is the git remote name (e.g., "origin").
- <code_gen_output_dir> is the output directory where all pipeline artifacts (patches, summaries) are stored.
- <latest_main_branch> is the local branch name that holds the freshly fetched latest main (created by the previous step).
</definition_explanations>

<instructions>
The goal of this task is to rebase the patched branch onto the latest main, resolving all conflicts. Follow these steps carefully:

Step 0: Build full context from pipeline history
- Before any git operations, read and understand ALL previous pipeline iteration results from <code_gen_output_dir>. This is critical for resolving conflicts correctly.
- Read in this order:
    - Code traces (code_traces/*_code_trace.txt) — understand the code paths the feature touches in both source and target frameworks
    - Code port plan iterations (code_port_plan/V*.txt, code_port_plan/V*_review.txt) — understand the porting strategy, what was changed and why
    - Test plan iterations (test_plan/V*.txt, test_plan/V*_review.txt) — understand the test strategy
    - Code gen iterations (code_gen/V*.patch, code_gen/V*_review.patch) — understand how the code evolved across iterations, what issues were found and fixed
    - Runtime iteration patches and summaries (runtime/V*/patch.patch, runtime/V*/summary.txt) — understand runtime fixes applied
    - All runtime iteration patches and summaries (runtime/V*/patch.patch, runtime/V*/summary.txt) — understand runtime fixes and any previous rebase iterations (rebases are also stored as runtime iterations)
    - Iteration history summary (runtime/iterations_history.txt) — overall iteration evolution and key learnings
- Build a comprehensive understanding of:
    - What the feature does and every file it modifies
    - Key design decisions and constraints (e.g., CUDA graph safety, execution mode handling, scheduling)
    - Issues that were encountered and how they were fixed — avoid re-introducing bugs that were already fixed
    - Which code changes are load-bearing (must be preserved exactly) vs which are adaptable to new APIs
- This context is essential for Step 4 (conflict resolution) — without it, conflicts cannot be resolved correctly.

Step 1: Verify state
- cd into <target_repo_dir>
- Verify the current branch is <branch_name>: `git branch --show-current`
- Verify the working tree is clean: `git status --porcelain` should show no changes
- If not clean, abort with an error message

Step 2: Create backup branch
- Create a backup of the current branch before rebasing:
    `git branch <branch_name>__store_before_rebase_$(date +%Y%m%d_%H%M%S)`
- Print the backup branch name so the user knows where the pre-rebase state is stored

Step 3: Rebase onto latest main
- Run: `git rebase <latest_main_branch>`
- If the rebase completes with no conflicts, proceed to Step 5

Step 4: Resolve conflicts (if any)
- If the rebase reports conflicts:
    - Use the full context from Step 0 to guide resolution — you must understand what the feature does, why each code change exists, and what issues were previously fixed
    - For each conflicted file, read the conflict markers and understand both sides
    - The "ours" side is the patch we want to keep (the feature implementation)
    - The "theirs" side is the latest main changes
    - Resolve each conflict by:
        - Keeping the feature implementation logic from "ours"
        - Adapting it to work with any API changes, renames, or refactors from "theirs"
        - Cross-reference with the code port plan and runtime fix summaries to ensure you do NOT re-introduce bugs that were already fixed in previous iterations
        - If a function signature changed in main, update the feature code to use the new signature while preserving the feature's behavior
        - If a file was deleted in main but modified in the patch, check if the functionality moved elsewhere and adapt
        - If main added new code paths (e.g., new execution modes, new model architectures), ensure the feature handles them correctly based on the code trace analysis from Step 0
    - After resolving each file: `git add <file>`
    - Continue the rebase: `git rebase --continue`
    - Repeat until all conflicts are resolved
- If a conflict is too complex to resolve automatically, document it clearly and abort the rebase with `git rebase --abort`, then restore from the backup branch

Step 5: Verify the rebase
- Confirm the rebase completed: `git log --oneline -5` should show the patch commit on top of the latest main commits
- Run a quick compilation check using the build process to ensure the rebased code compiles
- If compilation fails, investigate and fix the issues (they may be caused by API changes in the latest main)

Step 6: Save the rebased patch
- Generate a new patch file from the rebased commit:
    `git diff <latest_main_branch>..HEAD > <code_gen_output_dir>/{rebase_patch_file}`
- Generate a summary file at <code_gen_output_dir>/{rebase_summary_file} that includes:
    - The base commit (HEAD of <latest_main_branch>) that this patch applies on top of
    - The base commit hash: `git rev-parse <latest_main_branch>`
    - What conflicts were encountered and how they were resolved (if any)
    - Whether compilation succeeded after rebase
    - The backup branch name for reference
    - A note that this is a rebase iteration — the patch is the same feature rebased onto a newer base commit, superseding all previous patches

Step 7: Verify the patch can be re-applied
- To confirm the patch is self-contained, verify it could be applied to a clean checkout of <latest_main_branch>:
    - `git stash` (if any unstaged changes)
    - `git checkout <latest_main_branch>`
    - `git apply --check <code_gen_output_dir>/{rebase_patch_file}`
    - `git checkout <branch_name>` (go back)
    - `git stash pop` (if stashed)
- If the check fails, the patch file is incomplete — regenerate it
</instructions>

<output>
- The rebased branch <branch_name> with the patch on top of latest main
- <code_gen_output_dir>/{rebase_patch_file} — the rebased patch
- <code_gen_output_dir>/{rebase_summary_file} — summary with base commit, conflicts, and verification
</output>

"""

    def prompt(self):
        import os
        import re

        max_ver = 0
        runtime_dir = os.path.join(self.code_gen_output_dir, RUNTIME_DIR)
        if os.path.isdir(runtime_dir):
            pat = re.compile(r"^V(\d+)$")
            for d in os.listdir(runtime_dir):
                m = pat.match(d)
                if m:
                    ver = int(m.group(1))
                    if ver > max_ver:
                        max_ver = ver
        next_ver = max_ver + 1
        rebase_patch_file = os.path.join(RUNTIME_DIR, f"V{next_ver}", "patch.patch")
        rebase_summary_file = os.path.join(RUNTIME_DIR, f"V{next_ver}", "summary.txt")

        return self.prompt_template.format(
            context=self.context,
            target_repo_dir=self.target_repo_dir,
            branch_name=self.branch_name,
            pr_base_branch=self.pr_base_branch,
            pr_remote=self.pr_remote,
            code_gen_output_dir=self.code_gen_output_dir,
            latest_main_branch=self.latest_main_branch,
            rebase_patch_file=rebase_patch_file,
            rebase_summary_file=rebase_summary_file,
        )


def gen_RebasePRPrompt(
    context: str,
    target_repo_dir: str,
    branch_name: str,
    pr_base_branch: str,
    pr_remote: str,
    code_gen_output_dir: str,
    latest_main_branch: str,
):
    return RebasePRPrompt(
        context=context,
        target_repo_dir=target_repo_dir,
        branch_name=branch_name,
        pr_base_branch=pr_base_branch,
        pr_remote=pr_remote,
        code_gen_output_dir=code_gen_output_dir,
        latest_main_branch=latest_main_branch,
    )


PR_COMMIT_DESC_FILE = "commit_desc.txt"
PR_PR_DESC_FILE = "pr_desc.txt"
PR_METADATA_FILE = "pr_gen_metadata.txt"


@dataclass
class UpdateCommitAndPRDescsPrompt:
    context: str
    code_gen_output_dir: str
    output_dir_pr: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_gen_output_dir>
{code_gen_output_dir}
</code_gen_output_dir>
<output_dir_pr>
{output_dir_pr}
</output_dir_pr>
</definitions>

<definition_explanations>
- <code_gen_output_dir> is the directory containing all pipeline output files (code traces, plans, patches, runtime results).
- <output_dir_pr> is the directory where the PR description files will be written.
</definition_explanations>

<instructions>
The goal of this task is to read all pipeline results and generate up-to-date commit and PR descriptions. Do NOT run any git or gh commands yourself. Only generate the two description files.

Follow these steps and think hard:

Step 0: Check for existing descriptions.
- Check if <output_dir_pr>/commit_desc.txt and <output_dir_pr>/pr_desc.txt already exist from a previous run.
- If they exist, read them first — they are the starting point. Your job is to UPDATE them with any new information from the latest pipeline results (new runtime iterations, fixes, updated performance numbers, etc.). Preserve the existing structure and wording where it is still accurate, and only modify sections that are outdated or incomplete.
- If they do not exist, generate them from scratch.

Step 1: Read and analyze all result files in <code_gen_output_dir>:
- Shared pipeline artifacts (in <code_gen_output_dir> root):
    - code_port_plan/V*_review.txt or code_port_plan/V*.txt — the final code port plan (use the highest iteration number)
    - runtime/iterations_history.txt — iteration history summary
    - code_gen/V*_review_summary.txt or code_gen/V*_summary.txt — code gen summary
    - Any runtime fix summaries (runtime/V*/summary.txt) — to capture the latest fixes
- Per-model runtime results (each model run has its own subdirectory inside <code_gen_output_dir> matching runtime_*/, e.g., runtime_original/, runtime_smaller/):
    - runtime_*/runtime/success_result.txt — performance results and success confirmation for that model
    - runtime_*/runtime/V*/perf_comparison.txt — performance comparison (enabled vs disabled baseline)
    - runtime_*/runtime/V*/lm_eval.txt — correctness verification results
- runtime/results_manifest.json — if this file exists in <code_gen_output_dir>, it contains structured benchmark results from all model configurations. Use this as the authoritative source for performance and correctness numbers.
- PR-flow runtime results (directly under <code_gen_output_dir>/runtime/, no runtime_* prefix):
    - runtime/V*/verify_verdict.json — gate verdict (perf, lm_eval, overall) for each PR-flow iteration (rebase or review-comment rounds)
    - runtime/V*/perf_comparison.txt — patched-vs-baseline benchmark numbers for that iteration
    - runtime/V*/lm_eval_run/lm_eval_output/ and runtime/V*/lm_eval_run/lm_eval_output_disabled/ (or lm_eval_off_run/lm_eval_output_disabled/) — lm_eval scores for the ON and OFF arms
    - runtime/V*/followup_commit_msg.txt — commit message for a fix made during that iteration (mention such fixes in the description when they change behavior)
- IMPORTANT: the highest-numbered runtime/V* directory whose verify_verdict.json has overall == "PASS" is the authoritative result. Prefer it over older entries in results_manifest.json and over earlier V directories, and quote its numbers in the Performance Results and Correctness tables.

Step 2: Extract key information:
- What the patch does (from the code port plan) — summarize in 2-3 sentences
- Performance results: if runtime/results_manifest.json exists, extract results for ALL model configurations listed in it. Otherwise, read the per-model runtime_*/runtime/success_result.txt files directly.
- Correctness results: lm_eval scores for all model configurations (if available)
- Feature toggle: the environment variable name and startup log messages
- Test coverage: summary of what tests are included in the patch

Step 3: Create the directory <output_dir_pr> (if it doesn't exist) and generate:

File: <output_dir_pr>/commit_desc.txt
- First line: concise commit title (type: short description, e.g., "perf: replace attention kernel with fused MLA")
- Empty line
- Body: explaining what was changed and why, mentioning the performance improvement
- This file is the complete commit message, ready to be used with `git commit -F`
- IMPORTANT: Do NOT include any Co-Authored-By, Signed-off-by, or other trailers. Do NOT mention AI, Claude, auto_code_gen, or any tool attribution. The commit message must read as if written by a human engineer.

File: <output_dir_pr>/pr_desc.txt
- A well-structured PR description in markdown that includes:
    * "## Summary" — what was changed and why (2-3 paragraphs)
    * "## Performance Results" — a markdown table with throughput/TPOT/TTFT for baseline vs patched with delta %. If multiple model configurations are available (from runtime/results_manifest.json), include one row per model configuration with a label column.
    * "## Correctness (lm_eval)" — a markdown table with scores if available, or "Not run" if not. If multiple model configurations, include one row per model.
    * "## Feature Toggle" — the env var name and startup log messages
    * "## Test Coverage" — summary of tests
- This file is the complete PR body, ready to be used with `gh pr create --body-file`
- IMPORTANT: Do NOT mention AI, Claude, auto_code_gen, or any tool attribution anywhere in the PR description.
</instructions>

<output>
- Write <output_dir_pr>/commit_desc.txt and <output_dir_pr>/pr_desc.txt
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_gen_output_dir=self.code_gen_output_dir,
            output_dir_pr=self.output_dir_pr,
        )


def gen_UpdateCommitAndPRDescsPrompt(
    context: str,
    code_gen_output_dir: str,
):
    import os

    output_dir_pr = os.path.join(code_gen_output_dir, PR_COMMANDS_DIR)
    return UpdateCommitAndPRDescsPrompt(
        context=context,
        code_gen_output_dir=code_gen_output_dir,
        output_dir_pr=output_dir_pr,
    )


@dataclass
class GeneratePRCommandsPrompt:
    context: str
    code_gen_output_dir: str
    target_repo_dir: str
    branch_name: str
    pr_base_branch: str
    pr_remote: str
    output_dir_pr: str
    config_path: str
    agent_config_path: str = ""
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<code_gen_output_dir>
{code_gen_output_dir}
</code_gen_output_dir>
<target_repo_dir>
{target_repo_dir}
</target_repo_dir>
<branch_name>
{branch_name}
</branch_name>
<pr_base_branch>
{pr_base_branch}
</pr_base_branch>
<pr_remote>
{pr_remote}
</pr_remote>
<output_dir_pr>
{output_dir_pr}
</output_dir_pr>
<config_path>
{config_path}
</config_path>
</definitions>

<definition_explanations>
- <code_gen_output_dir> is the directory containing all pipeline output files (code traces, plans, patches, runtime results).
- <target_repo_dir> is the path to the target framework/project repository where the patch has been applied.
- <branch_name> is the git branch name in <target_repo_dir> that contains the patch.
- <pr_base_branch> is the upstream branch to create the PR against (e.g., "main").
- <pr_remote> is the git remote name to push to (e.g., "origin").
- <output_dir_pr> is the directory where all PR step scripts and description files live. It already contains commit_desc.txt and pr_desc.txt.
- <config_path> is the absolute path to the JSON config file used for this pipeline run.
</definition_explanations>

<instructions>
The goal of this task is to generate individual shell scripts for each step of the PR creation and update process. Do NOT run any git or gh commands yourself. Only generate the script files.

The description files <output_dir_pr>/commit_desc.txt and <output_dir_pr>/pr_desc.txt already exist (generated by a previous step). The scripts below must reference them.

Follow these steps and think hard:

Step 1: Read <output_dir_pr>/commit_desc.txt to get the commit title (first line) — this is also used as the PR title.

Step 2: Choose a readable remote branch name.
- Based on the commit title, choose a short, human-readable branch name for the remote.
- Format: `perf/<short-kebab-description>` (e.g., `perf/fused-mla-attention`, `perf/kv-cache-paged-copy`).
- Use lowercase, hyphens between words, max 4-5 words after the prefix.
- This readable name is used ONLY for the remote branch and the PR. The local branch stays as <branch_name>.

Step 3: Identify the latest code patch file.
- Look for the latest patch in <code_gen_output_dir>: check for runtime/V*/patch.patch files first (highest N), then code_gen/V*_review.patch, then code_gen/V*.patch.
- This is the patch that produced the successful runtime result.

Step 4: Generate the following script files inside <output_dir_pr>. Each file must be a valid, self-contained bash script (with #!/bin/bash header) that can be run independently. Each file must have a header comment block documenting what it does, what the expected outcome is, and any prerequisites.

File: <output_dir_pr>/pr_gen_step_0_generate_descs.sh
- Runs the UpdateCommitAndPRDescs prompt to regenerate commit_desc.txt and pr_desc.txt with the latest results.
- The script must first cd into the repo root directory that contains llm/ (derive it from <config_path>: the config lives at llm/auto_code_gen/configs/, so the repo root is three levels up from <config_path>).
- Then execute: `python -m llm.auto_code_gen.run_generate_pr_commands --config {config_command_arg}{agent_command_arg} --descs-only`
- Use the exact path from <config_path>.

File: <output_dir_pr>/pr_gen_step_1_apply_patch.sh
- cd into <target_repo_dir>
- First, check if the latest patch is already applied by running `git diff --stat` and comparing against the patch content. If the working tree already has the changes (non-empty diff matching the patch), skip application and print a message that the patch is already applied.
- If not applied:
    - Reset the repo to a clean state: `git checkout -- .` and `git clean -fd`
    - Apply the latest patch file identified in Step 3: `git apply <code_gen_output_dir>/<latest_patch_file>`
    - If `git apply` fails, try `git apply --3way <code_gen_output_dir>/<latest_patch_file>`
- After applying, print the `git diff --stat` output to show what changed
- The script must contain the absolute path to the patch file

File: <output_dir_pr>/pr_gen_step_2_commit.sh
- cd into <target_repo_dir>
- `git add -A`
- `git commit -F <output_dir_pr>/commit_desc.txt`
- Use the absolute path to commit_desc.txt

File: <output_dir_pr>/pr_gen_step_3_signoff.sh
- cd into <target_repo_dir>
- Run `git rebase HEAD~1 --signoff` to add the Signed-off-by trailer to the last commit

File: <output_dir_pr>/pr_gen_step_4_push.sh
- cd into <target_repo_dir>
- Push the local branch to the remote under the readable branch name chosen in Step 2, with tracking:
    `git push -u <pr_remote> <branch_name>:<readable_remote_branch_name>`
- After the push, echo the remote branch name so the user can see it
- Write a metadata file <output_dir_pr>/pr_gen_metadata.txt with:
    local_branch: <branch_name>
    remote_branch: <readable_remote_branch_name>
    remote: <pr_remote>
    pr_base: <pr_base_branch>

File: <output_dir_pr>/pr_gen_step_5_create_pr.sh
- Read the PR title from the first line of <output_dir_pr>/commit_desc.txt
- `gh pr create` command with:
    - --repo flag pointing to the <pr_remote> repo (use `git -C <target_repo_dir> remote get-url <pr_remote>` to resolve it)
    - --base <pr_base_branch>
    - --head <readable_remote_branch_name> (the readable name from Step 2, NOT <branch_name>)
    - --title: read from first line of commit_desc.txt (use `head -1 <output_dir_pr>/commit_desc.txt`)
    - --body-file <output_dir_pr>/pr_desc.txt

File: <output_dir_pr>/pr_gen_step_6_update_pr.sh
- This script is for updating an existing PR after additional fixes/iterations.
- cd into <target_repo_dir>
- Amend the last commit with updated changes: `git add -A && git commit --amend -F <output_dir_pr>/commit_desc.txt`
- Add signoff: `git rebase HEAD~1 --signoff`
- Force-push to update the remote branch: `git push --force-with-lease <pr_remote> <branch_name>:<readable_remote_branch_name>`
- Update the PR description: read the readable_remote_branch_name from <output_dir_pr>/pr_gen_metadata.txt, then run:
    `gh pr edit <readable_remote_branch_name> --repo <repo_url> --title "$(head -1 <output_dir_pr>/commit_desc.txt)" --body-file <output_dir_pr>/pr_desc.txt`
- Print a message confirming the PR was updated

File: <output_dir_pr>/pr_gen_step_7_get_latest_main.sh
- cd into <target_repo_dir>
- Fetch the latest from <pr_remote>: `git fetch <pr_remote> <pr_base_branch>`
- Create a local branch from the fetched main with a timestamped name:
    `git branch latest_main_$(date +%Y%m%d_%H%M%S) <pr_remote>/<pr_base_branch>`
- Print the created branch name
- Save the branch name to <output_dir_pr>/latest_main_branch.txt so the next step can read it

File: <output_dir_pr>/pr_gen_step_8_rebase.sh
- This script runs the RebasePR prompt to rebase the patched branch onto the latest main.
- The script must first cd into the repo root directory that contains llm/ (derive from <config_path>: three levels up from the llm/auto_code_gen/configs/ path).
- Read the latest main branch name from <output_dir_pr>/latest_main_branch.txt
- Run: `python -m llm.auto_code_gen.run_rebase_pr --config {config_command_arg}{agent_command_arg} --latest-main-branch "$(cat <output_dir_pr>/latest_main_branch.txt)"`
- Print the result: whether rebase succeeded and where the rebased patch is saved
</instructions>

<output>
- Write the nine script files inside <output_dir_pr>.
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            code_gen_output_dir=self.code_gen_output_dir,
            target_repo_dir=self.target_repo_dir,
            branch_name=self.branch_name,
            pr_base_branch=self.pr_base_branch,
            pr_remote=self.pr_remote,
            output_dir_pr=self.output_dir_pr,
            config_path=self.config_path,
            config_command_arg=shlex.quote(self.config_path),
            agent_command_arg=(
                (" --agent-config " + shlex.quote(self.agent_config_path))
                if self.agent_config_path
                else ""
            ),
        )


def gen_GeneratePRCommandsPrompt(
    context: str,
    code_gen_output_dir: str,
    target_repo_dir: str,
    branch_name: str,
    pr_base_branch: str,
    config_path: str,
    pr_remote: str = "origin",
    agent_config_path: str = "",
):
    import os

    output_dir_pr = os.path.join(code_gen_output_dir, PR_COMMANDS_DIR)
    return GeneratePRCommandsPrompt(
        agent_config_path=agent_config_path,
        context=context,
        code_gen_output_dir=code_gen_output_dir,
        target_repo_dir=target_repo_dir,
        branch_name=branch_name,
        pr_base_branch=pr_base_branch,
        pr_remote=pr_remote,
        output_dir_pr=output_dir_pr,
        config_path=os.path.abspath(config_path),
    )


@dataclass
class DiagnosticProfileAndAnalyzePrompt:
    context: str
    execution_command: str
    delay_iterations: int
    feature_disable_env_var: str
    original_analysis_dir: str
    diagnostic_dir: str
    iteration: int
    trace_gpu_focus: int
    gpu_wait_timeout_minutes: int
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<execution_command>
{execution_command}
</execution_command>
<delay_iterations>
{delay_iterations}
</delay_iterations>
<feature_disable_env_var>
{feature_disable_env_var}
</feature_disable_env_var>
<original_analysis_dir>
{original_analysis_dir}
</original_analysis_dir>
<diagnostic_dir>
{diagnostic_dir}
</diagnostic_dir>
<trace_gpu_focus>
{trace_gpu_focus}
</trace_gpu_focus>
</definitions>

<instructions>
The goal of this task is to run kernel-level NSYS diagnostic profiling for runtime iteration {iteration}, extract transformer block timings, and identify the KEY REASONS for the performance result (improvement, regression, or neutral). Think hard and be extremely thorough.

All output files MUST be written to <diagnostic_dir>. Create it first: `mkdir -p <diagnostic_dir>`.
Each step below writes an ongoing output file that can be monitored with `tail -f`.

Step 0: Wait for GPUs (output: <diagnostic_dir>/step0_gpu_wait.txt)
- Run `nvidia-smi` to check if GPUs are available (not running other significant workloads).
- If GPUs are busy:
    - Log the current state (busy GPUs, PIDs, process names, GPU memory usage) to the output file with a timestamp.
    - Sleep 30 seconds, then re-check.
    - Do NOT give up — keep waiting until GPUs become available.
    - Log each poll iteration with timestamp and status.
- Once GPUs are free, log "GPUs available — proceeding" with timestamp.

Step 1: Run NSYS Profiles (output: <diagnostic_dir>/step1_nsys_profiles.txt)
- Adapt <execution_command> for local execution (replace any Docker-internal paths like /app/ with local equivalents).
- Log "Running PATCHED NSYS profile..." to the output file.
- Run the patched benchmark with NSYS profiling:
    nsys profile -t cuda,nvtx -c cudaProfilerApi --cuda-graph-trace=node --trace-fork-before-exec=true \\
        -o <diagnostic_dir>/trace_patched \\
        <adapted_execution_command> --profile \\
        --profiler-config '{{"profiler":"cuda","delay_iterations":{delay_iterations},"max_iterations":50}}'
- Log "Running BASELINE NSYS profile..." to the output file.
- Run the baseline benchmark (same command but with <feature_disable_env_var>=1 prepended):
    env <feature_disable_env_var>=1 nsys profile -t cuda,nvtx -c cudaProfilerApi --cuda-graph-trace=node --trace-fork-before-exec=true \\
        -o <diagnostic_dir>/trace_baseline \\
        <adapted_execution_command> --profile \\
        --profiler-config '{{"profiler":"cuda","delay_iterations":{delay_iterations},"max_iterations":50}}'
- Export both traces to SQLite:
    nsys export --type=sqlite --output=<diagnostic_dir>/trace_patched.sqlite <diagnostic_dir>/trace_patched.nsys-rep
    nsys export --type=sqlite --output=<diagnostic_dir>/trace_baseline.sqlite <diagnostic_dir>/trace_baseline.nsys-rep
- Log completion status, file sizes, any errors.
- If either profile fails, diagnose the error, fix, and retry.

Step 2: Extract GPU Ops (output: <diagnostic_dir>/step2_extract_gpu_ops.txt)
- Look for the original analysis's GPU ops extraction script in <original_analysis_dir> (e.g., extract_gpu_ops.py or similar). Read it to understand the extraction approach.
- Run it (or adapt it) against both new SQLite traces to produce:
    <diagnostic_dir>/gpu_ops_patched.txt
    <diagnostic_dir>/gpu_ops_baseline.txt
- Focus on GPU <trace_gpu_focus> only.
- The output format must match the original gpu_ops.txt format (pipe-separated table with columns: Idx, Category, Short Name, Start, End, Duration, Stream, Grid, Block, Regs, ShMem, CommAlgo, CommSize, XferBytes, XferKind, Full Name).
- Log the number of ops extracted from each trace, any errors or adaptations needed.
- If the extraction script fails (e.g., trace format changed, missing tables), diagnose the error, fix the script, and retry.

Step 3: Correlate to Transformer Blocks (output: <diagnostic_dir>/step3_correlation.txt)
- Read the original analysis results from <original_analysis_dir> as REFERENCE CONTEXT (not as ground truth — the patch may have changed kernels, added/removed ops, or altered execution patterns):
    - transformer_block_high_level_ops_V*.txt (latest version) — the expected block structure and operation sequence
    - gpu_ops_to_blocks_V*.txt (latest version) — the original kernel-to-operation mapping
    - median_block_V*.txt (latest version) — the original per-op timings and high_level_op names
- Using these as reference, correlate the new GPU ops (from Step 2) to transformer blocks:
    - Detect block boundaries using the same kernel signatures as the original analysis
    - Classify block types (the model may have multiple block types, e.g., with/without indexer)
    - Map each kernel to its high_level_op name using (kernel_short_name, stream) identity from the original correlation
    - If the patch added new kernels or removed existing ones, adapt the correlation and document the differences
    - If a kernel's name changed or a new kernel replaced an existing one, map it to the closest logical operation and document it
- Select median-representative blocks (by total wall time) for both patched and baseline
- Write:
    <diagnostic_dir>/median_block_patched.txt
    <diagnostic_dir>/median_block_baseline.txt
- Log: block count, block types found, any differences from the original analysis

Step 4: Compare and Identify Key Reasons (output: <diagnostic_dir>/step4_comparison.txt)
- Produce a per-operation comparison table between baseline and patched median blocks:
    | # | Operation | Baseline (ns) | Patched (ns) | Delta (ns) | Delta (%) |
    Each row is one high_level_op. Include a TOTAL row for the full block wall time.
- Identify the KEY REASONS for the performance result. Determine:
    - If the target kernel(s) time is UNCHANGED: the optimization is NOT taking effect. Investigate why — wrong config file being loaded? Autotuner overriding at runtime? Config key mismatch? The config change not reaching the kernel?
    - If the target kernel(s) is SLOWER: the improvement plan's assumption is wrong for this workload. Investigate why — is the kernel memory-bound (larger tiles increase padding waste)? Are there occupancy effects? Is the comparison between frameworks invalid due to architectural differences?
    - If the target kernel(s) is FASTER but the total block time didn't improve: identify what is offsetting the gain — did another kernel regress? Is there new scheduling overhead? Did the optimization shift the bottleneck?
    - Are there new or removed kernels compared to the original analysis? What do they indicate about how the patch changed execution?
    - Does the improvement plan's predicted mechanism match observed reality?
- Write all findings to the output file.

Step 5: Write Final Diagnostic Report (output: <diagnostic_dir>/diagnostic_report.txt)
- This is the single file that the runtime investigation prompt will read. It MUST contain ALL of the following in a clear, structured format:
    1. SUMMARY: One-paragraph assessment of the performance result and its root cause
    2. PER-OPERATION COMPARISON TABLE: The full table from Step 4
    3. KEY REASONS: The specific reasons identified in Step 4, each with supporting kernel-level evidence
    4. BLOCK STRUCTURE CHANGES: Any differences in block structure, new/removed kernels, or correlation adaptations from Step 3
    5. RAW TIMING DATA: Patched block total wall time, baseline block total wall time, absolute and percentage delta
    6. IMPROVEMENT PLAN VALIDATION: Whether the improvement plan's predicted mechanism matches observed reality, and if not, what the actual mechanism is
    7. RECOMMENDED DIRECTION: What the next fix attempt should target, justified by the key reasons above
</instructions>

<output>
All files in <diagnostic_dir>/:
- step0_gpu_wait.txt (GPU availability log)
- step1_nsys_profiles.txt (NSYS profiling log)
- step2_extract_gpu_ops.txt (extraction log)
- step3_correlation.txt (correlation log)
- step4_comparison.txt (comparison and key reasons)
- diagnostic_report.txt (FINAL REPORT — the investigate prompt reads this)
- gpu_ops_patched.txt, gpu_ops_baseline.txt (raw GPU ops)
- median_block_patched.txt, median_block_baseline.txt (median block timings)
- trace_patched.sqlite, trace_baseline.sqlite (NSYS traces)
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            execution_command=self.execution_command,
            delay_iterations=self.delay_iterations,
            feature_disable_env_var=self.feature_disable_env_var,
            original_analysis_dir=self.original_analysis_dir,
            diagnostic_dir=self.diagnostic_dir,
            trace_gpu_focus=self.trace_gpu_focus,
            iteration=self.iteration,
        )


def gen_DiagnosticProfileAndAnalyzePrompt(
    context: str,
    execution_command: str,
    delay_iterations: int,
    feature_disable_env_var: str,
    original_analysis_dir: str,
    runtime_logs_dir: str,
    iteration: int,
    trace_gpu_focus: int = 0,
    gpu_wait_timeout_minutes: int = 30,
):
    import os

    diagnostic_dir = os.path.join(
        runtime_logs_dir,
        DIAGNOSTIC_PROFILE_DIR,
    )
    return DiagnosticProfileAndAnalyzePrompt(
        context=context,
        execution_command=execution_command,
        delay_iterations=delay_iterations,
        feature_disable_env_var=feature_disable_env_var,
        original_analysis_dir=os.path.abspath(original_analysis_dir),
        diagnostic_dir=diagnostic_dir,
        iteration=iteration,
        trace_gpu_focus=trace_gpu_focus,
        gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
    )


@dataclass
class LocalDiagnosticAnalysisPrompt:
    """Steps 2-5 of diagnostic profiling — local analysis only.

    Used after the cluster subprocess has completed nsys profiling (Step 1)
    and copied the SQLite traces back to diagnostic_dir.  This prompt runs
    in a fresh agent session with no cluster output in its context.
    """

    context: str
    original_analysis_dir: str
    diagnostic_dir: str
    iteration: int
    trace_gpu_focus: int
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<original_analysis_dir>
{original_analysis_dir}
</original_analysis_dir>
<diagnostic_dir>
{diagnostic_dir}
</diagnostic_dir>
<trace_gpu_focus>
{trace_gpu_focus}
</trace_gpu_focus>
</definitions>

<instructions>
Analyze kernel-level NSYS diagnostic traces for runtime iteration {iteration}. \
The NSYS profiling has already been completed — the SQLite traces are in \
<diagnostic_dir>. Your job is to extract transformer block timings and identify \
the KEY REASONS for the performance result. Think hard and be extremely thorough.

Before starting, verify that these files exist in <diagnostic_dir>:
- trace_patched.sqlite
- trace_baseline.sqlite
If either is missing, write "DIAGNOSTIC_ANALYSIS_FAILED: missing trace files" \
to <diagnostic_dir>/diagnostic_report.txt and STOP.

Step 2: Extract GPU Ops (output: <diagnostic_dir>/step2_extract_gpu_ops.txt)
- Look for the original analysis's GPU ops extraction script in <original_analysis_dir> \
(e.g., extract_gpu_ops.py or similar). Read it to understand the extraction approach.
- Run it (or adapt it) against both new SQLite traces to produce:
    <diagnostic_dir>/gpu_ops_patched.txt
    <diagnostic_dir>/gpu_ops_baseline.txt
- Focus on GPU <trace_gpu_focus> only.
- The output format must match the original gpu_ops.txt format (pipe-separated table \
with columns: Idx, Category, Short Name, Start, End, Duration, Stream, Grid, Block, \
Regs, ShMem, CommAlgo, CommSize, XferBytes, XferKind, Full Name).
- Log the number of ops extracted from each trace, any errors or adaptations needed.
- If the extraction script fails (e.g., trace format changed, missing tables), \
diagnose the error, fix the script, and retry.

Step 3: Correlate to Transformer Blocks (output: <diagnostic_dir>/step3_correlation.txt)
- Read the original analysis results from <original_analysis_dir> as REFERENCE CONTEXT \
(not as ground truth — the patch may have changed kernels, added/removed ops, or \
altered execution patterns):
    - transformer_block_high_level_ops_V*.txt (latest version)
    - gpu_ops_to_blocks_V*.txt (latest version)
    - median_block_V*.txt (latest version)
- Using these as reference, correlate the new GPU ops (from Step 2) to transformer blocks:
    - Detect block boundaries using the same kernel signatures as the original analysis
    - Classify block types (the model may have multiple block types, e.g., with/without indexer)
    - Map each kernel to its high_level_op name using (kernel_short_name, stream) identity \
from the original correlation
    - If the patch added new kernels or removed existing ones, adapt the correlation and \
document the differences
    - If a kernel's name changed or a new kernel replaced an existing one, map it to the \
closest logical operation and document it
- Select median-representative blocks (by total wall time) for both patched and baseline
- Write:
    <diagnostic_dir>/median_block_patched.txt
    <diagnostic_dir>/median_block_baseline.txt
- Log: block count, block types found, any differences from the original analysis

Step 4: Compare and Identify Key Reasons (output: <diagnostic_dir>/step4_comparison.txt)
- Produce a per-operation comparison table between baseline and patched median blocks:
    | # | Operation | Baseline (ns) | Patched (ns) | Delta (ns) | Delta (%) |
    Each row is one high_level_op. Include a TOTAL row for the full block wall time.
- Identify the KEY REASONS for the performance result. Determine:
    - If the target kernel(s) time is UNCHANGED: the optimization is NOT taking effect. \
Investigate why — wrong config file being loaded? Autotuner overriding at runtime? \
Config key mismatch? The config change not reaching the kernel?
    - If the target kernel(s) is SLOWER: the improvement plan's assumption is wrong for \
this workload. Investigate why — is the kernel memory-bound (larger tiles increase \
padding waste)? Are there occupancy effects?
    - If the target kernel(s) is FASTER but the total block time didn't improve: identify \
what is offsetting the gain — did another kernel regress? Is there new scheduling \
overhead? Did the optimization shift the bottleneck?
    - Are there new or removed kernels compared to the original analysis?
    - Does the improvement plan's predicted mechanism match observed reality?
- Write all findings to the output file.

Step 5: Write Final Diagnostic Report (output: <diagnostic_dir>/diagnostic_report.txt)
- This is the single file that the runtime investigation prompt will read. It MUST contain \
ALL of the following in a clear, structured format:
    1. SUMMARY: One-paragraph assessment of the performance result and its root cause
    2. PER-OPERATION COMPARISON TABLE: The full table from Step 4
    3. KEY REASONS: The specific reasons identified in Step 4, each with supporting \
kernel-level evidence
    4. BLOCK STRUCTURE CHANGES: Any differences in block structure, new/removed kernels, \
or correlation adaptations from Step 3
    5. RAW TIMING DATA: Patched block total wall time, baseline block total wall time, \
absolute and percentage delta
    6. IMPROVEMENT PLAN VALIDATION: Whether the improvement plan's predicted mechanism \
matches observed reality, and if not, what the actual mechanism is
    7. RECOMMENDED DIRECTION: What the next fix attempt should target, justified by \
the key reasons above
</instructions>

<output>
All files in <diagnostic_dir>/:
- step2_extract_gpu_ops.txt (extraction log)
- step3_correlation.txt (correlation log)
- step4_comparison.txt (comparison and key reasons)
- diagnostic_report.txt (FINAL REPORT — the investigate prompt reads this)
- gpu_ops_patched.txt, gpu_ops_baseline.txt (raw GPU ops)
- median_block_patched.txt, median_block_baseline.txt (median block timings)
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            original_analysis_dir=self.original_analysis_dir,
            diagnostic_dir=self.diagnostic_dir,
            trace_gpu_focus=self.trace_gpu_focus,
            iteration=self.iteration,
        )


def gen_LocalDiagnosticAnalysisPrompt(
    context: str,
    original_analysis_dir: str,
    runtime_logs_dir: str,
    iteration: int,
    trace_gpu_focus: int = 0,
):
    diagnostic_dir = os.path.join(
        runtime_logs_dir,
        DIAGNOSTIC_PROFILE_DIR,
    )
    return LocalDiagnosticAnalysisPrompt(
        context=context,
        original_analysis_dir=os.path.abspath(original_analysis_dir),
        diagnostic_dir=diagnostic_dir,
        iteration=iteration,
        trace_gpu_focus=trace_gpu_focus,
    )


@dataclass
class RecordLearningsPrompt:
    code_gen_output_dir: str
    knowledge_base_file: str
    prompt_template: ClassVar[str] = """

<definitions>
<code_gen_output_dir>
{code_gen_output_dir}
</code_gen_output_dir>
<knowledge_base_file>
{knowledge_base_file}
</knowledge_base_file>
</definitions>

<instructions>
Extract learnings from a completed code generation run and update the knowledge base.

Step 1: Read code generation results from <code_gen_output_dir>:
- runtime/iterations_history.txt, runtime/V*/summary.txt, diagnostic reports, results_manifest.json

Step 2: Read the current knowledge base from <knowledge_base_file>.

Step 3: Extract NEW learnings — things not already in the KB. Focus on actionable rules, not narrative.

Step 4: Update the knowledge base. MATCH THE EXISTING FORMAT EXACTLY:
- The KB is organized by topic sections (== SECTION HEADER ==)
- Each rule is ONE bullet point, ONE line — two lines max if unavoidable
- No per-entry headers, no Source/Date/Category/Evidence/Implication fields
- No narrative or history ("V7 added X, then V8 fixed Y") — only the final rule
- Include thresholds/numbers only when the agent needs to check against them
- Place new rules under the appropriate existing section, or create a new section only if no existing one fits

FORMAT EXAMPLE (match this style):
```
== SECTION NAME ==

- Rule statement — concise, actionable, one line
- Another rule with a threshold: must save >840us/step to break even
- NEVER do X — consequence. Fix: do Y instead
```

Rules for updating:
- Do NOT delete existing rules unless factually wrong
- Do NOT add duplicates — if an existing rule covers it, leave it alone
- Do NOT re-add rules that were already compressed into the current KB from earlier verbose entries
- Add new rules at the end of the appropriate section
- Update the "Last updated" date in the header
</instructions>

<output>
- <knowledge_base_file> (created or updated)
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            code_gen_output_dir=self.code_gen_output_dir,
            knowledge_base_file=self.knowledge_base_file,
        )


def gen_RecordLearningsPrompt(
    code_gen_output_dir: str, knowledge_base_file: str | None = None
):
    import os

    if knowledge_base_file is None:
        knowledge_base_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            KNOWLEDGE_BASE_FILE,
        )
    return RecordLearningsPrompt(
        code_gen_output_dir=os.path.abspath(code_gen_output_dir),
        knowledge_base_file=os.path.abspath(knowledge_base_file),
    )


@dataclass
class InvestigateIssuePrompt:
    context: str
    code_trace_files: list[str]
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
<code_trace_files>
{code_trace_files}
</code_trace_files>
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
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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

IMPORTANT: If <source_branch> and <target_branch> are provided, ONLY examine code from these specific branches. Do NOT look at other branches or git history outside of these branches.

- If <issue_fix_previous_attempt_file> is provided and is an existing real file, then read and analyze in-detail this contents to get all of the information about the previous attempt to generate a fix to the issue described here. Use all of the learnings and details from the previous attempt to improve the current attempt. Note that the current attempt is still done from scratch, but it can use the learnings from the previous attempt.
- If <issue_fix_previous_attempt_review_evolution_file> is provided and is an existing real file, then read and analyze in-detail this contents to get all of the information about the previous attempt to generate a fix to the issue described here. Use all of the learnings and details from the previous attempt to improve the current attempt. Note that the current attempt is still done from scratch, but it can use the learnings from the previous attempt.
- Read, analyze and understand in-detail all previous issues that were reported and are in the <output_dir>. These are may be relevant to avoid repeating mistakes, bugs or misleading information. Take all of the learning of previous issues into account while working on this issue from scratch.
- Analyze and understand in-detail the code port plan in <code_port_plan_file>, and restate the code port plan process step-by-step.
    - Analyze and understand in-detail the code port plan => review iteration evolution that is described in <code_port_plan_review_evolution_file> that lead to the final <code_port_plan_file>.
- Analyze and understand in-detail the code patch in <code_pr_file>, and restate the coding process step-by-step.
    - Analyze and understand in-detail the code gen => review iteration evolution that is described in <code_pr_review_evolution_file> that lead to the final <code_pr_file>.
- Detect and analyze in-detail the root causes that make issue <issue_desc_file> to appear in <target_framework>.
- Detect and analyze in-detail the root causes that make issue <issue_desc_file> to NOT appear in <source_framework>.
- Dive deep into the source code of both frameworks (ON THE SPECIFIED BRANCHES ONLY), and their related third party libraries, to get full picture of the source code end-to-end as it related to the <code_trace> of both frameworks.
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
            code_trace_files=self.code_trace_files,
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
    code_trace_files: list[str],
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
    assert len(code_trace_files) == 2

    return InvestigateIssuePrompt(
        context=context,
        code_trace_files=code_trace_files,
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
    code_trace_files: list[str]
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
<code_trace_files>
{code_trace_files}
</code_trace_files>
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
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
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
            code_trace_files=self.code_trace_files,
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
    code_trace_files: list[str],
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
    assert len(code_trace_files) == 2

    return ReviewInvestigatedIssuePrompt(
        context=context,
        code_trace_files=code_trace_files,
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
class FixIssuePrompt:
    claude_config: ClaudeConfig
    code_gen_config: CodeGenConfig
    high_level_code_plan_file: str
    prs_dir: str
    issue_to_fix_file: str
    issue_cwd: str
    prompt_template: ClassVar[str] = """
<cwd> = {cwd}
<issue_cwd> = {issue_cwd}

<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<batch_size> = {batch_size}
<isl> = {isl}
<osl> = {osl}
<framework_names> = {framework_names}
<framework_source_codes> = {framework_source_codes}
<plan_file> = {plan_file}
<plan_step> = {plan_step}
<faster_framework> = {faster_framework}
<slower_framework> = {slower_framework}
<faster_framework_test_dir> = {faster_framework_test_dir}
<slower_framework_test_dir> = {slower_framework_test_dir}
<faster_transformer_block_high_level_ops_file> = {faster_transformer_block_high_level_ops_file}
<slower_transformer_block_high_level_ops_file> = {slower_transformer_block_high_level_ops_file}
<faster_median_transformer_block_file> = {faster_median_transformer_block_file}
<slower_median_transformer_block_file> = {slower_median_transformer_block_file}
<high_level_code_plan_file> = {high_level_code_plan_file}
<prs_dir> = {prs_dir}
<issue_to_fix_file> = {issue_to_fix_file}

- <faster_transformer_block_high_level_ops_file> is a list of high-level transformer block operations of the faster framework.
- <slower_transformer_block_high_level_ops_file> is a list of high-level transformer block operations of the slower framework.
- <faster_median_transformer_block_file> is a median transformer block of the faster framework, with low-level kernel to high-level source code per-operation correlations.
- <slower_median_transformer_block_file> is a median transformer block of the slower framework, with low-level kernel to high-level source code per-operation correlations.

<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>

The <faster_framework_test_dir> and <slower_framework_test_dir> are the test/profile/run-logs directories for the faster and slower framework respectively. Each directory has the following format (that encodes test parameters): ../<test_id_with_batch>/[framework], and it includes the following files:
    - bench-<full_test_id>.json file that has the benchmark results of running the test on the framework
    - run-log-<full_test_id>.txt file that has the run log of executing the framework
    - run-log-profile-<full_test_id>.txt file that has the profile run log of executing the framework
    - trace-<full_test_id>.nsys-rep file that has the NSYS profile results
    - trace-<full_test_id>.sqlite file that is the SQLite version of the nsys-rep file

The traces provided are only for pure decode operations (no prefill).

- <framework_names> is a list of framework names involved
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- The file <plan_file> has a sequence of improvement steps for <slower_framework> based on the comparison between frameworks running the model <model> in precision <precision> on <gpu_type> GPU, with ISL <isl>, OSL <osl> and batch size <batch_size>.
- The file <high_level_code_plan_file> has the multi-step high-level coding plan to apply the improvement <plan_step> to <slower_framework>.
- Inside <prs_dir> directory are the code patches that implement the multi-step high-level coding plan from <high_level_code_plan_file> that applies the improvement <plan_step> to <slower_framework>.

The goal of this task is to fix the issue described in <issue_to_fix_file> that happens after the code patches from <prs_dir> are applied to <slower_framework> source code (that is inside the related directory from <framework_source_codes>), and the new code is executed. Do the following and think hard:
- Analyze and understand in-detail the issue <issue_to_fix>.
- Detect and analyze in-detail the root causes that make issue <issue_to_fix> to appear in <slower_framework>. Ensure to correlate the findings here with the multi-step high-level coding plan from <high_level_code_plan_file>.
- Detect and analyze in-detail the root causes that make issue <issue_to_fix> to NOT appear in <faster_framework>. Ensure to correlate the findings here with the multi-step high-level coding plan from <high_level_code_plan_file>.
- Fix <slower_framework> code patches to remove the issue <issue_to_fix> with minimal changes to the code patches.
- Strongly prefer to port fixes/ideas/code-pieces from <faster_framework> to fix issue <issue_to_fix> in <slower_framework>.
- For the sequence of changes made to fix the issue, write a sequence of tests that verify each small step taken for correctness, and eventually verifying the full fix end-to-end as follows:
    - The sequence of tests must go from smaller low-level tests, to larger higher-level tests to gradually test the correctness.
    - Make sure everything passes, for prefill, decode and mix of prefill and decode executions.
    - Make sure to test cuda graph modes enabled with the new code-paths for full correctness.
    - Make sure to have a large end-to-end tests that compare vs baseline and test all possible modes of execution (prefill/decode, cuda graph and more).
    - Ensure the issue <issue_to_fix> is fixed in <slower_framework> with the new fix.
- Ensure all existing tests are still working. Run them and verify.
- Review the new fixed code patches again for issues and fix anything necessary. Repeat the review multiple times till high confidence is reached.

Finally:
- Apply the new fixed code patches, run all of the tests (do NOT SKIP any tests), and verify all works and the issue is fixed.
- Ensure there is more than 1 test inside each test file, and ensure that all of them pass when run together in one file. If there is an error due to CUDA illegal memory, then it means that the fix is incorrect. Do not try to run the tests separately. Make sure they pass all together while fixing the relevant issues.
- Try to verify your changes by running a sequence of small tests that verify the sequence of small changes, that together compromise the full change. Be detailed, precise and ensure everything works.

Output:
- Dump a brief explanation of the fix applied to <issue_cwd>/issue_info.txt so a professional programmer can understand the fix applied here step-by-step.
- Dump the sequence of new fixed small code patches to <issue_cwd>/<output_file_prefix>_[seq_id].patch
- Dump the new fixed final full code patch to <issue_cwd>/<output_file_prefix>_full.patch
- Dump all new/fixed and old tests to <issue_cwd>/

Do not modify any files inside <prs_dir>.

"""

    def prompt(self):
        return self.prompt_template.format(
            cwd=self.claude_config.cwd,
            model=self.code_gen_config.model,
            precision=self.code_gen_config.precision,
            gpu_type=self.code_gen_config.gpu_type,
            batch_size=self.code_gen_config.batch_size,
            isl=self.code_gen_config.isl,
            osl=self.code_gen_config.osl,
            framework_names=self.code_gen_config.framework_names,
            framework_source_codes=self.code_gen_config.framework_source_codes,
            plan_file=self.code_gen_config.plan_file,
            plan_step=self.code_gen_config.plan_step,
            faster_framework=self.code_gen_config.faster_framework,
            slower_framework=self.code_gen_config.slower_framework,
            faster_framework_test_dir=self.code_gen_config.faster_framework_test_dir,
            slower_framework_test_dir=self.code_gen_config.slower_framework_test_dir,
            faster_transformer_block_high_level_ops_file=self.code_gen_config.faster_transformer_block_high_level_ops_file,
            slower_transformer_block_high_level_ops_file=self.code_gen_config.slower_transformer_block_high_level_ops_file,
            faster_median_transformer_block_file=self.code_gen_config.faster_median_transformer_block_file,
            slower_median_transformer_block_file=self.code_gen_config.slower_median_transformer_block_file,
            high_level_code_plan_file=self.high_level_code_plan_file,
            prs_dir=self.prs_dir,
            issue_to_fix_file=self.issue_to_fix_file,
            issue_cwd=self.issue_cwd,
        )


def gen_FixIssuePrompt(
    claude_config: ClaudeConfig,
    code_gen_config: CodeGenConfig,
    high_level_code_plan_file: str,
    prs_dir: str,
    issue_to_fix_file: str,
    issue_cwd: str,
):
    return FixIssuePrompt(
        claude_config=claude_config,
        code_gen_config=code_gen_config,
        high_level_code_plan_file=high_level_code_plan_file,
        prs_dir=prs_dir,
        issue_to_fix_file=issue_to_fix_file,
        issue_cwd=issue_cwd,
    )
