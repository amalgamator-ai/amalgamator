"""Configuration dataclass for the auto_bug_fix pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.model_registry import ModelProfile


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

    # Maximum context window in tokens for the pipeline model. Used by the
    # LiteLLM agent loop to trigger context compaction before a single call
    # exceeds the model's limit. Set via YAML advanced.context_length.
    # None = auto-detect from litellm model info, else default 128K.
    context_length: int | None = None

    # Patch-ID
    forward_patch_id_lookback: str = "12 months"

    # Allowlist
    allowlist_rename_expansion_cap: int = 3

    # Deterministic adaptation machinery (Phase 3a/3a.1). Individually
    # toggleable so eval runs can ablate each mechanism.
    mechanical_resolution: bool = True  # git-apply rungs before the LLM
    snapback: bool = True  # snap drifted lines back to source bytes
    evidence_checks: bool = True  # token audit + substitution mining + follow-up scan
    post_resolution_check: bool = True  # Phase 3a.1 verification agent
    blind_review: bool = True  # Phase 3a.2 triage-blind review agent
    # Non-bugfix ports only: fraction of the fix's files that may be dropped
    # (skipped and never regenerated) before the outcome is downgraded from
    # success to partial_success.
    partial_success_skip_ratio: float = 0.5
    # Model for Phase 4b recovery. Defaults to the pipeline model.
    # Set a different profile id in models.yaml to use a stronger model.
    phase_4b_model: str = ""
    phase_4b_profile: ModelProfile | None = None
    # Optional per-file check run over touched files after resolution,
    # e.g. "gcc -fsyntax-only {file}". {file} is replaced with the absolute path.
    file_check_command: str = ""

    # Patch-file ingest (Phase -1). Mutually exclusive with source_fix_commit
    # at config time; source_fix_commit is filled in after ingest.
    fix_patch: str = ""
    fix_upstream_ref: str = ""
