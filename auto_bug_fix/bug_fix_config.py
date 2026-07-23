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

    # Build & test (empty = skip, not specified in YAML = autodetect)
    build_command: str = ""
    test_command: str = ""
    lint_command: str = ""
    auto_discover_tests: bool = False
    max_build_test_retries: int = 3
    max_resolution_retries: int = 3

    # Per-command timeout in seconds (None = no limit). Set via YAML
    # advanced.timeout_minutes. Builds/tests can legitimately run for hours, so
    # the default is unbounded; a fired timeout is treated as a hard failure.
    command_timeout: int | None = None

    # Patch-ID
    forward_patch_id_lookback: str = "12 months"

    # Allowlist
    allowlist_rename_expansion_cap: int = 3

    # Deterministic adaptation machinery (Phase 3a/3a.1). Individually
    # toggleable so eval runs can ablate each mechanism.
    mechanical_resolution: bool = True   # git-apply rungs before the LLM
    snapback: bool = True                # snap drifted lines back to source bytes
    evidence_checks: bool = True         # token audit + substitution mining + follow-up scan
    post_resolution_check: bool = True   # Phase 3a.1 verification agent
    blind_review: bool = True            # Phase 3a.2 triage-blind review agent
    # Optional per-file check run over touched files after resolution,
    # e.g. "gcc -fsyntax-only {file}". {file} is replaced with the absolute path.
    file_check_command: str = ""
