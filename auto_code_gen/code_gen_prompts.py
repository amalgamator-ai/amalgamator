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
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework=self.framework,
            output_file=self.output_file,
        )


CODE_TRACE_FILE = "code_trace.txt"


def code_trace_filename(framework: str) -> str:
    return "{}_{}".format(framework, CODE_TRACE_FILE)


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
    prev_output_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(self.prev_output_file, self.prev_output_summary_file),
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
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
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
    prev_output_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(self.prev_output_file, self.prev_output_summary_file),
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
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
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
    code_port_plan_file: list[str]
    test_plan_file: list[str]
    output_patch_file: str
    output_summary_file: str
    prev_output_patch_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int
    prompt_template: str

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(self.prev_output_patch_file, self.prev_output_summary_file),
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
    prev_output_patch_file: Optional[str],
    prev_output_summary_file: Optional[str],
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
RUNTIME_SUCCESS_FILE = "runtime_success_result.txt"
RUNTIME_LOGS_DIR = "runtime_logs"
RUNTIME_SMALLER_MODEL_FILE = "runtime_smaller_model.txt"


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
        output_file=RUNTIME_SMALLER_MODEL_FILE,
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

Follow these steps precisely:

Step 1: Verify Clean Repository State
- Run `git -C <target_source_code_dir> status --porcelain` to check for any modifications or uncommitted changes.
- If the repository has ANY modifications (staged or unstaged tracked files):
    - Print a clear WARNING message: "WARNING: Repository at <target_source_code_dir> is not clean. Cannot apply patch."
    - List all dirty files.
    - Write "REPO_NOT_CLEAN" to <output_dir>/{output_status_file}
    - STOP immediately. Do NOT proceed to the next steps.
- If the repository is clean, proceed to Step 2.

Step 2: Apply the Code Patch
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
        output_status_file="runtime_apply_status_V{}.txt".format(iteration),
    )


@dataclass
class RunAndLogPrompt:
    context: str
    execution_command: str
    runtime_logs_dir: str
    iteration: int
    gpu_wait_timeout_minutes: int
    smaller_model_file: Optional[str]
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
        - Write "GPUS_TIMEOUT" to {runtime_logs_dir}/runtime_run_status_V{iteration}.txt
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
    - If the command has `--output-json <path>`, update the path to: {runtime_logs_dir}/runtime_bench_result_V{iteration}.json
    - Ensure all other paths in the command are valid local paths.
- Run the adjusted command and capture ALL output:
    - Use a script or shell construct to capture stdout and stderr separately AND as a combined stream:
      ```
      <adjusted_command> > {runtime_logs_dir}/runtime_run_V{iteration}_stdout.log 2> {runtime_logs_dir}/runtime_run_V{iteration}_stderr.log ; echo $? > {runtime_logs_dir}/runtime_run_V{iteration}_exit_code.txt
      ```
    - Then create the combined log:
      ```
      cat {runtime_logs_dir}/runtime_run_V{iteration}_stdout.log {runtime_logs_dir}/runtime_run_V{iteration}_stderr.log > {runtime_logs_dir}/runtime_run_V{iteration}_combined.log
      ```
    - Alternatively, use `tee` and process substitution to capture everything in real time.

Step 4: Write Run Status
- Read the exit code from {runtime_logs_dir}/runtime_run_V{iteration}_exit_code.txt
- If exit code is 0: write "COMPLETED" to {runtime_logs_dir}/runtime_run_status_V{iteration}.txt
- If exit code is non-zero: write "FAILED" to {runtime_logs_dir}/runtime_run_status_V{iteration}.txt
- In both cases, the detailed output is in the log files for subsequent analysis.

{feature_verification_step}
</instructions>

<output>
- {runtime_logs_dir}/runtime_run_V{iteration}_stdout.log
- {runtime_logs_dir}/runtime_run_V{iteration}_stderr.log
- {runtime_logs_dir}/runtime_run_V{iteration}_combined.log
- {runtime_logs_dir}/runtime_run_V{iteration}_exit_code.txt
- {runtime_logs_dir}/runtime_run_status_V{iteration}.txt
{baseline_output_files}
</output>

"""

    def prompt(self):
        if self.smaller_model_file:
            smaller_section = (
                "<smaller_model_file>\n"
                "{}\n"
                "</smaller_model_file>".format(self.smaller_model_file)
            )
            smaller_instructions = (
                "\n\nIMPORTANT: A smaller model is being used for runtime testing instead of <model>. "
                "Read <smaller_model_file> from <output_dir> to get the smaller model name (SMALLER_MODEL_NAME) "
                "and GPU count (SMALLER_MODEL_NUM_GPUS). Use these values when adjusting the execution command "
                "in Step 3, and use SMALLER_MODEL_NUM_GPUS as K in Step 1."
            )
            gpu_count_instructions = (
                "    - Read SMALLER_MODEL_NUM_GPUS from <smaller_model_file> in <output_dir> — that is K."
            )
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
                "environment variable name that disables the new feature (e.g., VLLM_DISABLE_AUTO_PERF_PLAN_<plan_step>).\n"
                "    - Set this environment variable to \"1\" in the command prefix to disable the new feature."
            )
            feature_verification_step = (
                "Step 5: Verify Feature is DISABLED\n"
                "- Search the combined log for the expected disabled message: "
                "\"[AutoPerf] Plan step <plan_step>:\" followed by \"is DISABLED (env override)\".\n"
                "- If found, the feature was successfully disabled.\n"
                "- If NOT found, print a WARNING that the feature disable mechanism may not be working."
            )
        else:
            feature_toggle_env_var = ""
            feature_verification_step = (
                "Step 5: Verify Feature is ENABLED\n"
                "- Search the combined log for the expected enabled message: "
                "\"[AutoPerf] Plan step <plan_step>:\" followed by \"is ENABLED\".\n"
                "- If found, the new feature is confirmed active — good.\n"
                "- If NOT found, print a WARNING that the new feature may not be running. "
                "Write \"FEATURE_NOT_VERIFIED\" to {runtime_logs_dir}/runtime_run_status_V{iteration}.txt "
                "and include details about what was searched for and what was found instead.\n"
                "\n"
                "Step 6: Baseline Comparison Run (only if the patched run in Step 3 succeeded with exit code 0 AND Step 5 confirmed the feature is enabled)\n"
                "- Read the latest code gen or runtime summary file in <output_dir> to find the environment variable "
                "name that disables the new feature (e.g., VLLM_DISABLE_AUTO_PERF_PLAN_<plan_step>).\n"
                "- Re-run the SAME adjusted command from Step 3, but with the disable env var set to \"1\" in the "
                "command prefix. This runs the original code paths WITHOUT the new feature, to establish a baseline.\n"
                "- Capture output to:\n"
                "    - {runtime_logs_dir}/runtime_baseline_V{iteration}_stdout.log\n"
                "    - {runtime_logs_dir}/runtime_baseline_V{iteration}_stderr.log\n"
                "    - {runtime_logs_dir}/runtime_baseline_V{iteration}_combined.log\n"
                "    - {runtime_logs_dir}/runtime_baseline_V{iteration}_exit_code.txt\n"
                "  If --output-json is present, set it to {runtime_logs_dir}/runtime_baseline_bench_result_V{iteration}.json\n"
                "- Verify the baseline run log contains the \"[AutoPerf]...is DISABLED (env override)\" message.\n"
                "- The baseline run MUST succeed (exit code 0), since it runs the original unmodified code paths. "
                "If it fails:\n"
                "    - This indicates something is wrong with the patch's disable mechanism or the patch broke original code paths.\n"
                "    - Analyze the error, document it, and note it in the status file.\n"
                "    - Fix the issue in <target_source_code_dir> and retry the baseline run until it succeeds.\n"
                "- Once the baseline run succeeds, extract performance results from BOTH runs:\n"
                "    - Patched run (Step 3): throughput, TTFT, TPOT\n"
                "    - Baseline run (Step 6): throughput, TTFT, TPOT\n"
                "- Write the comparison to {runtime_logs_dir}/runtime_perf_comparison_V{iteration}.txt:\n"
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
                "- {runtime_logs_dir}/runtime_baseline_V{iteration}_stdout.log\n"
                "- {runtime_logs_dir}/runtime_baseline_V{iteration}_stderr.log\n"
                "- {runtime_logs_dir}/runtime_baseline_V{iteration}_combined.log\n"
                "- {runtime_logs_dir}/runtime_baseline_V{iteration}_exit_code.txt\n"
                "- {runtime_logs_dir}/runtime_perf_comparison_V{iteration}.txt"
            ).format(runtime_logs_dir=self.runtime_logs_dir, iteration=self.iteration)

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
    smaller_model_file: Optional[str] = None,
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
    smaller_model_file: Optional[str]
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
- Read the latest code gen or runtime summary file in <output_dir> to find the environment variable name that disables the new feature (e.g., VLLM_DISABLE_AUTO_PERF_PLAN_<plan_step>).
- Ensure this env var is NOT set (or unset it) so the new feature is active.
- Construct and run the lm_eval command:
    ```
    CUDA_VISIBLE_DEVICES=<free_gpu_ids> lm_eval --model vllm --model_args pretrained=<model>,max_model_len=<len>,tensor_parallel_size=<K> --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto
    ```
- Capture all output to {runtime_logs_dir}/runtime_lm_eval_V{iteration}_enabled_stdout.log and stderr to {runtime_logs_dir}/runtime_lm_eval_V{iteration}_enabled_stderr.log
- Record exit code to {runtime_logs_dir}/runtime_lm_eval_V{iteration}_enabled_exit_code.txt
- Extract the exact_match scores (flexible-extract and strict-match) from the output.

Step 4: Run lm_eval with Feature DISABLED
- Set the disable env var to "1" in the command prefix to disable the new feature.
- Run the same lm_eval command with the disable env var:
    ```
    <DISABLE_ENV_VAR>=1 CUDA_VISIBLE_DEVICES=<free_gpu_ids> lm_eval --model vllm --model_args pretrained=<model>,max_model_len=<len>,tensor_parallel_size=<K> --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto
    ```
- Capture output to {runtime_logs_dir}/runtime_lm_eval_V{iteration}_disabled_stdout.log and stderr to {runtime_logs_dir}/runtime_lm_eval_V{iteration}_disabled_stderr.log
- Record exit code to {runtime_logs_dir}/runtime_lm_eval_V{iteration}_disabled_exit_code.txt
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
- {runtime_logs_dir}/runtime_lm_eval_V{iteration}_enabled_stdout.log
- {runtime_logs_dir}/runtime_lm_eval_V{iteration}_enabled_stderr.log
- {runtime_logs_dir}/runtime_lm_eval_V{iteration}_enabled_exit_code.txt
- {runtime_logs_dir}/runtime_lm_eval_V{iteration}_disabled_stdout.log
- {runtime_logs_dir}/runtime_lm_eval_V{iteration}_disabled_stderr.log
- {runtime_logs_dir}/runtime_lm_eval_V{iteration}_disabled_exit_code.txt
</output>

"""

    def prompt(self):
        if self.smaller_model_file:
            smaller_section = (
                "<smaller_model_file>\n"
                "{}\n"
                "</smaller_model_file>".format(self.smaller_model_file)
            )
            gpu_count_instructions = (
                "    - Read SMALLER_MODEL_NUM_GPUS from <smaller_model_file> in <output_dir> — that is K."
            )
            model_selection_instructions = (
                "    - Read SMALLER_MODEL_NAME from <smaller_model_file> in <output_dir> — use that as the model."
            )
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
    smaller_model_file: Optional[str] = None,
):
    return RunLMEvalPrompt(
        context=context,
        execution_command=execution_command,
        runtime_logs_dir=runtime_logs_dir,
        iteration=iteration,
        gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
        smaller_model_file=smaller_model_file,
        output_file="{}_V{}.txt".format(RUNTIME_LM_EVAL_FILE_PREFIX, iteration),
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
    prev_summary_file: Optional[str]
    iteration_history_summary_file: str
    smaller_model_file: Optional[str]
    lm_eval_result_file: Optional[str]
    output_patch_file: str
    output_summary_file: str
    success_result_file: str
    prompt_template: ClassVar[str] = """

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
<runtime_logs_dir>
{runtime_logs_dir}
</runtime_logs_dir>
<prev_patch_file>
{prev_patch_file}
</prev_patch_file>
<iteration_history_summary_file>
{iteration_history_summary_file}
</iteration_history_summary_file>
{smaller_model_section}
{lm_eval_section}
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for <source_framework> and <target_framework> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is the high-level multi-step coding plan that implements the improvement plan step.
- <test_plan_file> is the testing plan for the implementation.
- <runtime_logs_dir> contains the captured output from the benchmark execution for this iteration.
- <prev_patch_file> is the code patch that was applied for this runtime execution.
- <iteration_history_summary_file> is a comprehensive summary of ALL previous code generation phases (code traces, port plan, test plan, code gen iterations) and any previous runtime iterations. Read it first to get full context before analyzing the current runtime output.
{smaller_model_explanation}
{lm_eval_explanation}
</definition_explanations>

<instructions>
The goal of this task is to analyze the runtime benchmark output from iteration {iteration} of <tested_execution> and determine whether the execution succeeded or failed. Think hard and be extremely thorough.

Step 1: Analyze Previous History
- Read <iteration_history_summary_file> from <output_dir> to get full context of all previous code generation phases and runtime iterations.
- Read the previous/latest code patch from <output_dir>/<prev_patch_file> to understand the current state of the code changes.
- Read all previous runtime iteration summaries (code_gen_runtime_summary_V*.txt) in <output_dir> to understand what was tried before, what errors occurred, what was fixed, and what patterns emerged across iterations.
- Read all other relevant files in <output_dir> (code port plan, test plan, code gen summaries, etc.) to have full context of the implementation.

Step 2: Read and Analyze Runtime Output
- Read the runtime logs from <runtime_logs_dir>:
    - runtime_run_V{iteration}_stdout.log (standard output)
    - runtime_run_V{iteration}_stderr.log (error output)
    - runtime_run_V{iteration}_combined.log (combined view)
    - runtime_run_V{iteration}_exit_code.txt (process exit code)
    - runtime_run_status_V{iteration}.txt (run status: COMPLETED, FAILED, GPUS_TIMEOUT, or FEATURE_NOT_VERIFIED)
- Determine whether the execution SUCCEEDED or FAILED:
    - SUCCEEDED means ALL of the following:
        - The benchmark run (RunAndLog) succeeded:
            - Process exited with code 0
            - Benchmark completed fully without errors
            - Performance results (throughput, latency, tokens/s) are present in the output
            - No error tracebacks, crashes, segmentation faults, or assertion failures in stderr
        - The lm_eval correctness check (if <lm_eval_result_file> is provided) passed:
            - Read <lm_eval_result_file> from <runtime_logs_dir>
            - The verdict must be PASS (no significant correctness degradation)
            - If the lm_eval verdict is FAIL, the overall result is FAILED even if the benchmark run succeeded
    - FAILED means ANY of the following:
        - Non-zero exit code from the benchmark run
        - Error tracebacks or Python exceptions
        - Segmentation faults, CUDA errors, or OOM errors
        - Process was killed (SIGKILL, SIGTERM) or timed out
        - Missing or incomplete performance results
        - Any runtime error that prevented normal completion
        - runtime_run_status file contains GPUS_TIMEOUT (GPUs were not available — this is NOT a code error, note it in the summary and do NOT attempt code fixes)
        - runtime_run_status file contains FEATURE_NOT_VERIFIED (the new feature startup log was not detected — investigate why the feature is not activating)
        - lm_eval verdict is FAIL (correctness regression detected — the patch must be fixed to preserve correctness)

Step 3A: If SUCCEEDED (both benchmark AND lm_eval passed)
- Extract the performance results from the benchmark output, including:
    - Throughput (requests/s, tokens/s)
    - TTFT (Time To First Token) — extract directly if reported, or calculate from the output data (e.g., time from request submission to first generated token)
    - TPOT (Time Per Output Token) — extract directly if reported, or calculate as: (total_generation_time - TTFT) / (output_tokens - 1)
    - Latency percentiles (p50, p90, p99) if available
    - Total generation time
    - Any other relevant metrics
- Also read the baseline performance comparison file from <runtime_logs_dir>/runtime_perf_comparison_V{iteration}.txt (if it exists). This contains side-by-side patched vs baseline results from the RunAndLog step.
- Also read the lm_eval results from <lm_eval_result_file> in <runtime_logs_dir> (if it exists). This contains correctness scores for both feature enabled and disabled.
- Write a comprehensive summary to <output_dir>/{success_result_file}:
    - Performance results:
        - Patched run performance: throughput, TTFT, TPOT with units
        - Baseline run performance (from comparison file): throughput, TTFT, TPOT with units
        - Performance delta: absolute and percentage change for each metric
        - Clear statement of whether the patch delivers the expected improvement
    - Correctness results (from lm_eval):
        - Feature ENABLED: gsm8k exact_match scores (flexible-extract, strict-match)
        - Feature DISABLED: gsm8k exact_match scores (flexible-extract, strict-match)
        - Delta and analysis of correctness impact
    - Execution context: model=<model>, gpu=<gpu_type>, ISL=<isl>, OSL=<osl>, batch_size=<batch_size>
    - Number of runtime iterations required to reach success
    - Summary of all issues fixed across runtime iterations (if any)
- STOP here. Do NOT generate a new patch.

Step 3B: If FAILED
- Detect ALL errors and issues in the runtime output. Be extremely thorough and precise:
    - Analyze every error traceback line by line
    - Identify the root cause by tracing through the source code in both <target_source_code_dir> and the framework code traces
    - Look for: CUDA errors, shape mismatches, type errors, missing attributes, import errors, memory errors, assertion failures, timeout issues, incorrect kernel invocations, wrong tensor dtypes/devices, and any other runtime errors
    - Cross-reference errors with the applied code patch to understand what in the patch caused each error
    - If an error is recurring from a previous iteration (visible in previous runtime summaries in <output_dir>), understand why the previous fix was insufficient and fix the root cause properly this time
- For EACH error/issue found, document:
    - The exact error message and full stack trace location
    - The root cause in the source code (file path, function name, line)
    - Why this error occurs (what assumption was wrong, what was missed, what code path triggers it)
    - The precise fix, with explanation of why this fix is correct
- Apply ALL fixes to the code in <target_source_code_dir>:
    - Fix each error precisely at its root cause
    - Trace through the execution flow for all modes (decode-only, prefill-only, mixed) to ensure fixes are correct
    - Do NOT introduce new bugs while fixing existing ones
    - IMPORTANT: ALL code modifications MUST be exclusively inside <target_source_code_dir>. Do NOT modify files outside of it.
- Add NEW tests that specifically verify these runtime fixes:
    - For each fixed error, add a targeted test that would catch the same error if it regressed
    - Include edge cases related to the errors
    - Follow the existing test patterns in <target_source_code_dir>
- Run ONLY the tests that this patch introduces (from all iterations including the current one, and the NEW TESTS from above):
    - Identify every test file and test function that was added or modified by the patch (use `git diff` to find them).
    - Do NOT run general framework tests — only the patch's own tests.
    - Run every one of those patch-introduced tests. Do NOT skip any.
    - If any test fails: analyze the failure, fix the issue, and re-run all patch-introduced tests again.
    - Repeat until every patch-introduced test PASSES with zero failures and zero skips.
    - Do NOT stop until full success on all patch-introduced tests + NEW tests from above.
- Feature Toggle and Startup Verification:
    - Ensure the fixed patch preserves the feature toggle mechanism from the original code gen patch:
        1. STARTUP LOG: A log message during framework startup that prints `[AutoPerf] Plan step <plan_step>: <description> is ENABLED` when the new feature is active.
        2. ENVIRONMENT VARIABLE DISABLE SWITCH: The env var `VLLM_DISABLE_AUTO_PERF_PLAN_<plan_step>` that, when set to "1", disables the new feature and falls back to original code paths, printing `[AutoPerf] Plan step <plan_step>: <description> is DISABLED (env override)`.
    - If these mechanisms are missing or broken in the current patch, add or fix them.
    - Do NOT remove or weaken these mechanisms during fixes.
- Generate the fixed code patch:
    - Generate from <target_source_code_dir> using `git diff` (include both staged and unstaged changes)
    - The patch MUST include the code fixes AND all patch-introduced tests (previous iterations + new)
    - Save to <output_dir>/{output_patch_file}
- Write a detailed summary to <output_dir>/{output_summary_file}:
    - All errors/issues found in the runtime output with exact error messages
    - Root cause analysis for each error with source code references
    - How each error was fixed with the specific code changes made
    - New tests added and what they verify
    - The exact environment variable name used to disable the new feature
    - The exact startup log message that confirms the feature is enabled
    - The exact startup log message that confirms the feature is disabled
    - Iteration evolution summary from iteration 1 through iteration {iteration}:
        - What happened in each runtime iteration
        - How the errors evolved (which were fixed, which are new, which recurred and why)
        - Key learnings accumulated across iterations
        - Current state and any remaining concerns
</instructions>

<output>
On SUCCESS:
- <output_dir>/{success_result_file}

On FAILURE:
- <output_dir>/{output_patch_file}
- <output_dir>/{output_summary_file}
</output>

"""

    def prompt(self):
        if self.smaller_model_file:
            smaller_section = (
                "<smaller_model_file>\n"
                "{}\n"
                "</smaller_model_file>".format(self.smaller_model_file)
            )
            smaller_explanation = (
                "- <smaller_model_file> describes the smaller model being used for runtime "
                "testing instead of <model>. The runtime output comes from this smaller model. "
                "Take this into account when interpreting performance numbers and errors."
            )
        else:
            smaller_section = ""
            smaller_explanation = ""

        if self.lm_eval_result_file:
            lm_eval_section = (
                "<lm_eval_result_file>\n"
                "{}\n"
                "</lm_eval_result_file>".format(self.lm_eval_result_file)
            )
            lm_eval_explanation = (
                "- <lm_eval_result_file> contains lm_eval (gsm8k) correctness results comparing "
                "the new feature ENABLED vs DISABLED. Read it from <runtime_logs_dir> to check if "
                "there is a correctness regression. If the verdict is FAIL, the patch has a correctness "
                "issue that must be investigated and fixed."
            )
        else:
            lm_eval_section = ""
            lm_eval_explanation = ""

        return self.prompt_template.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_patch_file if self.prev_summary_file else None,
                self.prev_summary_file,
            ),
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            runtime_logs_dir=self.runtime_logs_dir,
            prev_patch_file=self.prev_patch_file,
            iteration_history_summary_file=self.iteration_history_summary_file,
            smaller_model_section=smaller_section,
            smaller_model_explanation=smaller_explanation,
            lm_eval_section=lm_eval_section,
            lm_eval_explanation=lm_eval_explanation,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            success_result_file=self.success_result_file,
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
    prev_summary_file: Optional[str],
    iteration_history_summary_file: str,
    smaller_model_file: Optional[str] = None,
    lm_eval_result_file: Optional[str] = None,
):
    assert len(code_trace_files) == 2

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
        output_patch_file="{}_V{}.patch".format(RUNTIME_FILE_PREFIX, iteration),
        output_summary_file="{}_summary_V{}.txt".format(RUNTIME_FILE_PREFIX, iteration),
        success_result_file=RUNTIME_SUCCESS_FILE,
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

Step 0: Check for Existing Summary
- Check if <output_dir>/{output_file} already exists from a previous run.
- If it exists, read it first — it contains a summary of artifacts that were already analyzed in a previous run. Use it as a starting point: you do NOT need to re-read artifacts that are already covered by this summary. Focus on reading only NEW or UPDATED artifacts that were created after the previous summary was written.
- If it does not exist, proceed to read all artifacts from scratch.

Step 1: Read Generated Artifacts
- List all files in <code_gen_output_dir> and its subdirectories.
- Read and analyze each file in the following order of phases:
    1. Framework code traces (*_code_trace.txt) — understand the active code paths in both "source" and "target" frameworks for <tested_execution>.
    2. Code port plan iterations (code_port_plan_V*.txt, code_port_plan_summary_V*.txt) — understand the planned porting steps and their evolution.
    3. Code port plan reviews (code_port_plan_review_V*.txt, code_port_plan_review_summary_V*.txt) — understand issues found and fixed during plan review.
    4. Test plan iterations (test_plan_V*.txt, test_plan_summary_V*.txt) — understand the testing strategy.
    5. Test plan reviews (test_plan_review_V*.txt, test_plan_review_summary_V*.txt) — understand test review fixes.
    6. Code gen iterations (code_gen_V*.patch, code_gen_summary_V*.txt) — understand the generated code patches.
    7. Code gen reviews (code_gen_review_V*.patch, code_gen_review_summary_V*.txt) — understand code review fixes and the final patch.
    8. Any previous runtime iteration files (code_gen_runtime_V*.patch, code_gen_runtime_summary_V*.txt) — understand previous runtime fix attempts.
    9. Any runtime logs in runtime_logs/ subdirectory — understand previous execution results.
    10. Any runtime success result (runtime_success_result.txt) — if already succeeded.

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
- Identify the latest/final code patch (either code_gen_review_V<N>.patch or code_gen_runtime_V<N>.patch, where <N> is the highest iteration number)

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
        )


def gen_IterationHistorySummaryPrompt(
    context: str,
    code_gen_output_dir: str,
):
    return IterationHistorySummaryPrompt(
        context=context,
        code_gen_output_dir=code_gen_output_dir,
        output_file="runtime_iterations_history.txt",
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
