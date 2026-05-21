"""Configuration dataclass and default instance for the auto_bug_fix pipeline.

Edit the ``bug_fix_config`` instance at the bottom of this file before running
``python -m auto_bug_fix.run_bug_fix``.
"""

from dataclasses import dataclass
from common.claude_utils import ClaudeConfig


@dataclass
class BugFixConfig:
    # Repository
    repo_path: str
    build_dir: str

    # Branch identity
    source_branch: str
    target_branch: str
    source_fix_commit: str

    # Bug context
    bug_description: str
    issue_id: str

    # Porting constraints
    disallowed_modules: list[str]
    port_tests: bool

    # Build & test
    build_command: str
    test_command: str
    max_build_test_retries: int


claude_config = ClaudeConfig(
    model="claude-opus-4-6",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",
    cwd="/path/to/output/dir",
)

bug_fix_config = BugFixConfig(
    repo_path="/path/to/gcc",
    build_dir="/path/to/gcc/build",
    source_branch="gcc-14-branch",
    target_branch="gcc-13-branch",
    source_fix_commit="abc1234",
    bug_description="CVE-2024-XXXX: buffer overflow in fold_convert()",
    issue_id="CVE-2024-XXXX",
    disallowed_modules=["gcc/config/arm/"],
    port_tests=True,
    build_command="make -j$(nproc)",
    test_command="make check -j$(nproc)",
    max_build_test_retries=3,
)
