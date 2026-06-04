import pytest
from auto_code_gen.use_cases.bug_fix import (
    BugFixUseCase,
    RunAndFixPrompt,
    RUN_AND_FIX_FAILURE_FILE,
    _create_bug_fix_context_str,
)
from common.claude_utils import PipelineStep


@pytest.fixture
def use_case():
    return BugFixUseCase()


@pytest.fixture
def context(claude_cfg, bug_fix_cfg):
    return _create_bug_fix_context_str(claude_cfg, bug_fix_cfg)


# ---------------------------------------------------------------------------
# create_context_str
# ---------------------------------------------------------------------------

class TestCreateContextStr:
    def test_contains_branch_info(self, context):
        assert "gcc-14-branch" in context
        assert "gcc-13-branch" in context
        assert "abc1234" in context

    def test_contains_bug_info(self, context):
        assert "CVE-2024-1234" in context

    def test_contains_build_info(self, context):
        assert "make -j$(nproc)" in context
        assert "make check -j$(nproc)" in context
        assert "/path/to/gcc/build" in context

    def test_contains_output_dir(self, context):
        assert "/tmp/test_cwd" in context

    def test_no_gpu_language(self, context):
        assert "gpu_type" not in context
        assert "batch_size" not in context
        assert "precision" not in context


# ---------------------------------------------------------------------------
# gen_code_trace_steps
# ---------------------------------------------------------------------------

class TestGenCodeTraceSteps:
    def test_returns_steps_and_code_trace_files(self, use_case, context, bug_fix_cfg):
        steps, code_trace_files = use_case.gen_code_trace_steps(context, bug_fix_cfg)
        assert isinstance(steps, list)
        assert len(steps) >= 2
        assert isinstance(code_trace_files, list)
        assert len(code_trace_files) == 1

    def test_code_trace_file_references_source_branch(self, use_case, context, bug_fix_cfg):
        _, code_trace_files = use_case.gen_code_trace_steps(context, bug_fix_cfg)
        assert "gcc-14-branch" in code_trace_files[0]

    def test_first_step_is_clear_target(self, use_case, context, bug_fix_cfg):
        steps, _ = use_case.gen_code_trace_steps(context, bug_fix_cfg)
        assert steps[0].name == "clear_target_repo"
        assert isinstance(steps[0].prompt, dict)

    def test_second_step_is_code_trace(self, use_case, context, bug_fix_cfg):
        steps, _ = use_case.gen_code_trace_steps(context, bug_fix_cfg)
        assert "code_trace" in steps[1].name
        assert isinstance(steps[1].prompt, str)
        assert "gcc-14-branch" in steps[1].prompt

    def test_code_trace_prompt_no_cuda_language(self, use_case, context, bug_fix_cfg):
        steps, _ = use_case.gen_code_trace_steps(context, bug_fix_cfg)
        text = steps[1].prompt
        assert "cuda" not in text.lower()
        assert "prefill" not in text.lower()
        assert "decode" not in text.lower()


# ---------------------------------------------------------------------------
# gen_code_port_plan_iter_steps
# ---------------------------------------------------------------------------

class TestGenCodePortPlanIterSteps:
    def test_returns_steps_and_review_prompt(self, use_case, context, bug_fix_cfg):
        steps, review_prompt = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"],
            None, None, 1,
        )
        assert isinstance(steps, list)
        assert len(steps) == 2
        assert hasattr(review_prompt, "output_file")
        assert hasattr(review_prompt, "output_summary_file")

    def test_plan_step_contains_branches(self, use_case, context, bug_fix_cfg):
        steps, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"],
            None, None, 1,
        )
        plan_prompt_text = steps[0].prompt
        assert "gcc-14-branch" in plan_prompt_text
        assert "gcc-13-branch" in plan_prompt_text

    def test_plan_step_contains_disallowed_modules(self, use_case, context, bug_fix_cfg):
        steps, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"],
            None, None, 1,
        )
        assert "gcc/config/arm/" in steps[0].prompt

    def test_iteration_tracking(self, use_case, context, bug_fix_cfg):
        steps_v1, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"],
            None, None, 1,
        )
        steps_v2, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"],
            "prev.txt", "prev_summary.txt", 2,
        )
        assert "V1" in steps_v1[0].output_files[0]
        assert "V2" in steps_v2[0].output_files[0]

    def test_no_cuda_language_in_plan(self, use_case, context, bug_fix_cfg):
        steps, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"],
            None, None, 1,
        )
        text = steps[0].prompt
        assert "cuda" not in text.lower()
        assert "kernel vendor" not in text.lower()


# ---------------------------------------------------------------------------
# Combined code+test port plan mode
# ---------------------------------------------------------------------------

class TestCombinedCodeAndTestPortPlan:
    def test_skip_test_plan_when_combined(self, use_case, bug_fix_cfg):
        bug_fix_cfg.use_combined_code_and_test_port_plan = True
        assert use_case.skip_test_plan_phase(bug_fix_cfg) is True

    def test_no_skip_test_plan_when_separate(self, use_case, bug_fix_cfg):
        bug_fix_cfg.use_combined_code_and_test_port_plan = False
        assert use_case.skip_test_plan_phase(bug_fix_cfg) is False

    def test_combined_plan_contains_test_planning(self, use_case, context, bug_fix_cfg):
        bug_fix_cfg.use_combined_code_and_test_port_plan = True
        steps, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"], None, None, 1,
        )
        text = steps[0].prompt
        assert "Test Plan" in text or "test plan" in text.lower()
        assert "git show" in text
        assert "supplemental" in text.lower()

    def test_separate_plan_does_not_contain_test_planning(self, use_case, context, bug_fix_cfg):
        bug_fix_cfg.use_combined_code_and_test_port_plan = False
        steps, _ = use_case.gen_code_port_plan_iter_steps(
            context, bug_fix_cfg, ["trace.txt"], None, None, 1,
        )
        text = steps[0].prompt
        assert "git show" not in text

    def test_combined_default_is_true(self, bug_fix_cfg):
        assert bug_fix_cfg.use_combined_code_and_test_port_plan is True


# ---------------------------------------------------------------------------
# run_runtime_iterations (RunAndFix)
# ---------------------------------------------------------------------------

class TestRunRuntimeIterations:
    def test_run_and_fix_template_contains_build_command(self):
        from auto_code_gen.use_cases.bug_fix import RunAndFixPrompt, gen_RunAndFixPrompt
        prompt = gen_RunAndFixPrompt(
            context="<context>test</context>",
            build_command="make -j$(nproc)",
            test_command="make check -j$(nproc)",
            build_dir="/path/to/build",
            max_build_test_retries=3,
        )
        text = prompt.prompt()
        assert "make -j$(nproc)" in text

    def test_run_and_fix_template_contains_test_command(self):
        from auto_code_gen.use_cases.bug_fix import gen_RunAndFixPrompt
        prompt = gen_RunAndFixPrompt(
            context="<context>test</context>",
            build_command="make -j$(nproc)",
            test_command="make check -j$(nproc)",
            build_dir="/path/to/build",
            max_build_test_retries=3,
        )
        text = prompt.prompt()
        assert "make check -j$(nproc)" in text

    def test_run_and_fix_template_references_failure_file(self):
        from auto_code_gen.use_cases.bug_fix import gen_RunAndFixPrompt
        prompt = gen_RunAndFixPrompt(
            context="<context>test</context>",
            build_command="make",
            test_command="make check",
            build_dir="/path/to/build",
            max_build_test_retries=3,
        )
        text = prompt.prompt()
        assert RUN_AND_FIX_FAILURE_FILE in text

    def test_run_and_fix_template_contains_retry_limit(self):
        from auto_code_gen.use_cases.bug_fix import gen_RunAndFixPrompt
        prompt = gen_RunAndFixPrompt(
            context="<context>test</context>",
            build_command="make",
            test_command="make check",
            build_dir="/path/to/build",
            max_build_test_retries=3,
        )
        text = prompt.prompt()
        assert "3" in text

    def test_run_and_fix_template_mentions_incremental_build(self):
        from auto_code_gen.use_cases.bug_fix import gen_RunAndFixPrompt
        prompt = gen_RunAndFixPrompt(
            context="<context>test</context>",
            build_command="make",
            test_command="make check",
            build_dir="/path/to/build",
            max_build_test_retries=3,
        )
        text = prompt.prompt()
        assert "incremental" in text.lower() or "stale artifact" in text.lower()

    def test_method_exists_and_is_async(self, use_case):
        import inspect
        assert hasattr(use_case, "run_runtime_iterations")
        assert inspect.iscoroutinefunction(use_case.run_runtime_iterations)


# ---------------------------------------------------------------------------
# Bug fix prompts: no GPU/CUDA language
# ---------------------------------------------------------------------------

class TestBugFixPromptsNoGpuLanguage:
    def test_code_trace_template_no_gpu(self):
        from auto_code_gen.use_cases.bug_fix import BUGFIX_CODE_TRACE_TEMPLATE
        text = BUGFIX_CODE_TRACE_TEMPLATE.lower()
        assert "cuda" not in text
        assert "gpu" not in text
        assert "prefill" not in text
        assert "decode" not in text

    def test_code_port_plan_template_no_gpu(self):
        from auto_code_gen.use_cases.bug_fix import BUGFIX_CODE_PORT_PLAN_TEMPLATE
        text = BUGFIX_CODE_PORT_PLAN_TEMPLATE.lower()
        assert "cuda" not in text
        assert "kernel vendor" not in text

    def test_run_and_fix_template_no_gpu(self):
        text = RunAndFixPrompt.prompt_template.lower()
        assert "cuda" not in text
        assert "gpu" not in text
