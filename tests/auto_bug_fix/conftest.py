import pytest
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig


@pytest.fixture
def claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
    )


@pytest.fixture
def bug_cfg():
    return BugFixConfig(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: buffer overflow in fold_convert()",
        issue_id="CVE-2024-1234",
        disallowed_modules=["gcc/config/arm/"],
        port_tests=True,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
