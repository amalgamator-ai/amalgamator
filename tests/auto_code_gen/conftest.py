import pytest
from common.claude_utils import ClaudeConfig
from auto_code_gen.code_gen_configs import BugFixConfig


@pytest.fixture
def claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
        thinking={"type": "adaptive"},
        effort="max",
        max_thinking_tokens=1048576,
    )


@pytest.fixture
def bug_fix_cfg():
    return BugFixConfig(
        output_dir="/tmp/test_output",
        source_code_dir="/path/to/gcc",
        num_code_port_plan_iterations=3,
        num_test_plan_iterations=3,
        num_code_gen_iterations=3,
        disallowed_modules=["gcc/config/arm/"],
        thinking_mode="deep",
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: buffer overflow in fold_convert()",
        issue_id="CVE-2024-1234",
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
