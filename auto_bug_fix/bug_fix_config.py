"""Configuration dataclass and default instance for the auto_bug_fix pipeline."""

from dataclasses import dataclass, field


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

    # Build & test
    build_command: str = "make -j$(nproc)"
    test_command: str = "make check -j$(nproc)"
    max_build_test_retries: int = 3
    max_resolution_retries: int = 3

    # Patch-ID
    forward_patch_id_lookback: str = "12 months"

    # Allowlist
    allowlist_rename_expansion_cap: int = 3
