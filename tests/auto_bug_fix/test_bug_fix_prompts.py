import pytest
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_TestPortPrompt,
    TEST_PORT_MANIFEST_FILE,
)
from auto_bug_fix.bug_fix_prompts import gen_CodePortPlanPrompt, gen_ReviewCodePortPlanPrompt
from auto_bug_fix.bug_fix_prompts import (
    gen_TestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
)


class TestCreateContextStr:
    def test_contains_branch_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "gcc-14-branch" in ctx
        assert "gcc-13-branch" in ctx
        assert "abc1234" in ctx

    def test_contains_bug_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "CVE-2024-1234" in ctx

    def test_contains_build_info(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "make -j$(nproc)" in ctx
        assert "make check -j$(nproc)" in ctx
        assert "/path/to/gcc/build" in ctx

    def test_no_gpu_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "gpu_type" not in ctx
        assert "batch_size" not in ctx
        assert "precision" not in ctx

    def test_contains_cwd(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        assert "/tmp/test_cwd" in ctx


class TestCodeTracePrompt:
    def test_contains_source_branch(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "gcc-14-branch" in p.prompt()

    def test_references_fix_commit(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "source_fix_commit" in p.prompt()

    def test_no_cuda_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        text = p.prompt()
        assert "cuda" not in text.lower()
        assert "prefill" not in text.lower()
        assert "decode" not in text.lower()

    def test_output_file_named_after_branch(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_CodeTracePrompt(ctx, bug_cfg.source_branch)
        assert "gcc-14-branch" in p.output_file


class TestTestPortPrompt:
    def test_references_git_show(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert "git show" in p.prompt()

    def test_contains_commit_sha(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert "abc1234" in p.prompt()

    def test_references_manifest_file(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert TEST_PORT_MANIFEST_FILE in p.prompt()

    def test_output_manifest_is_constant(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPortPrompt(ctx, bug_cfg.source_fix_commit, bug_cfg.target_branch)
        assert p.output_manifest_file == TEST_PORT_MANIFEST_FILE


class TestCodePortPlanPrompt:
    def _make(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        return gen_CodePortPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["source_trace.txt"],
            disallowed_modules=bug_cfg.disallowed_modules,
            previous_attempt_file="",
            output_file="plan_V1.txt",
        )

    def test_contains_branches(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "gcc-14-branch" in p.prompt()
        assert "gcc-13-branch" in p.prompt()

    def test_contains_disallowed_modules(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "gcc/config/arm/" in p.prompt()

    def test_no_cuda_language(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "cuda" not in text.lower()
        assert "kernel vendor" not in text.lower()

    def test_asserts_two_branches(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        with pytest.raises(AssertionError):
            gen_CodePortPlanPrompt(
                context=ctx,
                branches=["only-one"],
                branch_code_trace_files=["trace.txt"],
                disallowed_modules=[],
                previous_attempt_file="",
                output_file="plan.txt",
            )


class TestReviewCodePortPlanPrompt:
    def test_references_plan_and_output_files(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_ReviewCodePortPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            output_review_file="review.txt",
            output_fixed_file="fixed.txt",
            output_total_review_summary_file="evolution.txt",
        )
        text = p.prompt()
        assert "plan.txt" in text
        assert "review.txt" in text
        assert "fixed.txt" in text


class TestTestPlanPrompt:
    def test_references_manifest(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
        )
        assert TEST_PORT_MANIFEST_FILE in p.prompt()

    def test_no_performance_language(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_TestPlanPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
        )
        text = p.prompt()
        assert "speedup" not in text.lower()
        assert "throughput" not in text.lower()


class TestCodeGenPrompt:
    def _make(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        return gen_CodeGenPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_plan_file="test_plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            previous_attempt_file="",
            output_info_file="info.txt",
            output_pr_file="patch.patch",
        )

    def test_references_build_command(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert "build_command" in p.prompt()

    def test_no_vllm_cmake_steps(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        text = p.prompt()
        assert "generate_cmake_presets" not in text
        assert "cmake --preset" not in text

    def test_references_manifest(self, claude_cfg, bug_cfg):
        p = self._make(claude_cfg, bug_cfg)
        assert TEST_PORT_MANIFEST_FILE in p.prompt()


class TestReviewCodeGenPrompt:
    def test_references_pr_and_output_files(self, claude_cfg, bug_cfg):
        ctx = create_context_str(claude_cfg, bug_cfg)
        p = gen_ReviewCodeGenPrompt(
            context=ctx,
            branches=[bug_cfg.source_branch, bug_cfg.target_branch],
            branch_code_trace_files=["trace.txt"],
            code_port_plan_file="plan.txt",
            test_plan_file="test_plan.txt",
            test_port_manifest_file=TEST_PORT_MANIFEST_FILE,
            code_pr_info_file="info.txt",
            code_pr_file="patch.patch",
            output_review_file="review.txt",
            output_fixed_file="fixed.txt",
            output_total_review_summary_file="evolution.txt",
        )
        text = p.prompt()
        assert "patch.patch" in text
        assert "review.txt" in text
