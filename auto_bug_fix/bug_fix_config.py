"""Configuration dataclass and default instance for the auto_bug_fix pipeline.

Edit the ``bug_fix_config`` instance at the bottom of this file before running
``python -m auto_bug_fix.run_bug_fix``.
"""

from dataclasses import dataclass, field
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
    allowed_modules: list[str] | None = None
    disallowed_modules: list[str] = field(default_factory=list)
    port_tests: bool = True

    # Build & test
    build_command: str = "make -j$(nproc)"
    test_command: str = "make check -j$(nproc)"
    max_build_test_retries: int = 3
    max_resolution_retries: int = 3  # Max attempts for LLM conflict resolution

    # Quorum
    require_unanimity: bool = True
    allow_abstention: bool = True

    # Bisect
    bisect_max_commits: int = 200

    # Patch-ID
    forward_patch_id_lookback: str = "12 months"

    # Test fixture
    manifest_patch_command: str | None = None

    # Worktree
    worktree_root: str | None = None
    use_per_branch_worktrees: bool = False
    use_per_cve_worktrees: bool = False
    rerere_enabled: bool = False

    # Allowlist
    allowlist_rename_expansion_cap: int = 3

    def __post_init__(self):
        if self.rerere_enabled and not self.use_per_cve_worktrees:
            raise ValueError(
                "rerere_enabled=True requires use_per_cve_worktrees=True "
                "(per-branch worktrees share rr-cache across CVEs)"
            )


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
