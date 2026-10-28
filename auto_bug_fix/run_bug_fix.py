# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 0-5 orchestrator for the auto_bug_fix pipeline.

Usage:
    Edit ``auto_bug_fix/bug_fix_config.py`` to configure your project, then run:

        python -m auto_bug_fix.run_bug_fix
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TextIO, TypedDict, cast

from auto_bug_fix.claude_utils import (
    BugFixClaudeConfig,
    ClaudeConfig,
    PipelineStep,
)
from auto_bug_fix.claude_utils import (
    agent_run as _agent_run,
)
from common.utils import Tee

if TYPE_CHECKING:
    from common.model_registry import ModelProfile


def _phase_config(
    claude_config: ClaudeConfig,
    config: BugFixConfig,
    *,
    model: str | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
    profile: ModelProfile | None = None,
) -> BugFixClaudeConfig:
    """Create a per-phase ClaudeConfig that inherits transport settings.

    When *profile* is provided, overrides the model and transport config.
    Explicit model changes must carry the corresponding resolved profile.
    Copies ``command_timeout`` and ``context_length`` from the parent
    config so that context window management stays active across all
    pipeline phases.
    """
    effective_profile = profile or claude_config.profile
    effective_model = model or (profile.model if profile else claude_config.model)

    if model and effective_profile and effective_profile.model != effective_model:
        raise ConfigError(
            "Phase model does not match its profile; supply a resolved profile"
        )

    return BugFixClaudeConfig(
        model=effective_model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
        command_timeout=claude_config.command_timeout,
        context_length=claude_config.context_length,
        thinking=thinking,
        effort=effort or claude_config.effort,
        profile=effective_profile,
    )


async def _run_llm(claude_config, prompts, tracker=None):
    """Compatibility wrapper: call agent_run and feed timings into tracker.

    Uses the model-agnostic ``agent_run`` dispatcher which routes to the
    Claude Agent SDK for Claude models and to a LiteLLM tool-use loop
    for everything else.
    """
    batch_start = time.time()
    step_timings = await _agent_run(claude_config, prompts)

    if tracker and step_timings:
        cursor = batch_start
        for st in step_timings:
            # A recorded duration of 0 is a valid measurement; only a missing
            # entry falls back to the default.
            raw_dur = st.get("duration")
            dur = float(raw_dur) if raw_dur is not None else 0
            sdk_cost = st.get("cost_usd")
            record = tracker.record_query(
                prompt_name=st.get("name", "query"),
                start_time=cursor,
                end_time=cursor + dur,
                input_tokens=st.get("input_tokens", 0),
                output_tokens=st.get("output_tokens", 0),
                sdk_cost_usd=sdk_cost,
                model=claude_config.model,
                profile=claude_config.profile,
            )
            log.info(
                "LLM query %s: duration=%.1fs, tokens=%d/%d, cost=$%.4f (sdk=$%s)",
                st.get("name", "query"),
                dur,
                st.get("input_tokens", 0),
                st.get("output_tokens", 0),
                record.cost_usd,
                sdk_cost,
            )
            cursor += dur

    return step_timings


claude_run = _run_llm

from auto_bug_fix.allowlist import (
    derive_seed,
    enforce_allowlist,
    enforce_test_file_veto,
    resolve_renames,
)
from auto_bug_fix.baseline import (
    TIMEOUT_EXIT_CODE,
    check_regression,
    parse_test_failures,
    run_test_suite,
)
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.bug_fix_prompts import (
    RED_TRIAGE_FILE,
    SEMANTIC_TRIAGE_FILE,
    TEST_PORT_MANIFEST_FILE,
    BuildErrorRecoveryPrompt,
    NarrowResolutionAgentPrompt,
    PostResolutionCheckPrompt,
    RedTriagePrompt,
    SemanticTriagePrompt,
    TestPortAgentPrompt,
    TriageBlindReviewPrompt,
    create_context_str,
)
from auto_bug_fix.config_loader import (
    ConfigError,
    TargetSpec,
    _sanitize_ref,
    load_multi_target_config,
)
from auto_bug_fix.dossier import Dossier, build_trailers, format_dossier
from auto_bug_fix.evidence import (
    added_lines_by_file,
    annotate_dead_config,
    apply_substitutions,
    audit_added_tokens,
    audit_line_provenance,
    find_pure_rename_restorations,
    mine_substitutions,
    port_line_changes,
    render_evidence,
    run_file_checks,
    scan_followup_fixes,
    target_config_state,
    token_exists_on_ref,
    validate_triage_prerequisites,
)
from auto_bug_fix.git_tools import (
    COAUTHOR_TRAILERS as _COAUTHOR_TRAILERS,
)
from auto_bug_fix.git_tools import (
    _run_git,
    git_cat_file_exists,
    git_cherry_pick,
    git_cherry_pick_abort,
    git_diff_name_only,
    git_diff_name_only_range,
    git_format_patch_bytes,
    git_is_ancestor,
    git_merge_base,
    git_reset_hard,
    git_rev_parse,
    git_show_files,
    git_status_porcelain,
    is_merge_commit,
    resolve_merge_commit,
)
from auto_bug_fix.git_tools import (
    git_commit as _git_commit_raw,
)
from auto_bug_fix.ingest import PATCH_TRAILER_KEY, IngestError, ingest_patch
from auto_bug_fix.mechanical import resolve_mechanically
from auto_bug_fix.patch_id import backward_patch_id_check, forward_patch_id_check
from auto_bug_fix.phase_4b import (
    RUN_AND_FIX_FAILURE_FILE,
    RUN_AND_FIX_SUCCESS_FILE,
    gen_RunAndFixPrompt,
)
from auto_bug_fix.range_diff import parse_equivalence, run_range_diff
from auto_bug_fix.signature import capture_output
from auto_bug_fix.snapback import (
    _git_blob,
    snap_back_files,
)
from auto_bug_fix.tracker import Tracker
from auto_bug_fix.worktree import cleanup_worktree, create_fresh_worktree

LOG_FILE = "__run_log_bug_fix.txt"
DOSSIER_FILE = "dossier.md"
PATCH_FILE = "candidate.patch"
_PIPELINE_ARTIFACTS = (
    SEMANTIC_TRIAGE_FILE,
    TEST_PORT_MANIFEST_FILE,
    RED_TRIAGE_FILE,
    RUN_AND_FIX_FAILURE_FILE,
    RUN_AND_FIX_SUCCESS_FILE,
    os.path.join("runs", SEMANTIC_TRIAGE_FILE),
    os.path.join("runs", RED_TRIAGE_FILE),
)

log = logging.getLogger(__name__)


def _sanitize_branch_slug(raw: str) -> str:
    """Make *raw* safe for use inside a git branch name (``git check-ref-format``)."""
    s = re.sub(r"[~^: \t\\]+", "-", raw)
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"@\{", "at-", s)
    s = s.strip("-./")
    return s or "unknown"


def _unstage_pipeline_artifacts(repo_path: str) -> None:
    """Remove pipeline artifact files from the git index so they don't end up in commits."""
    for artifact in _PIPELINE_ARTIFACTS:
        subprocess.run(
            ["git", "rm", "--cached", "--ignore-unmatch", "-f", artifact],
            cwd=repo_path,
            capture_output=True,
        )


def git_commit(
    repo_path: str, message: str, trailers: dict[str, str] | None = None
) -> None:
    """Wrapper around git_commit that unstages pipeline artifacts first."""
    _unstage_pipeline_artifacts(repo_path)
    _git_commit_raw(repo_path, message, trailers=trailers)


def git_amend(repo_path: str) -> None:
    """Amend HEAD with the currently staged changes, keeping the message."""
    _unstage_pipeline_artifacts(repo_path)
    result = subprocess.run(
        ["git", "commit", "--amend", "--no-edit"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        log.warning(
            "git commit --amend failed: %s", (result.stderr or result.stdout)[:300]
        )


class PipelineEscalation(Exception):
    pass


class PipelineStop(Exception):
    pass


_CONFLICT_PREFIXES = ("UU ", "AA ", "DU ", "UD ", "DD ", "AU ", "UA ")

_MARKER_START_RE = re.compile(r"^<{7}( |\r?$)", re.MULTILINE)
_MARKER_END_RE = re.compile(r"^>{7}( |\r?$)", re.MULTILINE)

_DANGEROUS_SHELL_PATTERNS = (";", "&&", "||", "|", "`", "$(", "${", ">", "<", "\n")

_BUGFIX_TYPES = ("bugfix", "bug-fix", "bug_fix", "fix", "security")


def _validate_shell_command(cmd: str) -> bool:
    """Reject commands with shell metacharacters that could indicate injection."""
    return not any(p in cmd for p in _DANGEROUS_SHELL_PATTERNS)


def _conflicted_files(status_lines: list[str]) -> list[str]:
    """Extract conflicted file paths from porcelain status output."""
    return [
        line[3:]
        for line in status_lines
        if any(line.startswith(p) for p in _CONFLICT_PREFIXES)
    ]


def _has_conflicts(status_lines: list[str]) -> list[str]:
    """Return status lines that represent merge conflicts."""
    return [
        line
        for line in status_lines
        if any(line.startswith(p) for p in _CONFLICT_PREFIXES)
    ]


@dataclass
class PipelineState:
    seed: list[str] = field(default_factory=list)
    allowed_modules: list[str] = field(default_factory=list)
    baseline: set[str] = field(default_factory=set)
    ported_test_files: list[str] = field(default_factory=list)
    cherry_pick_path: str = ""
    dossier: Dossier = field(default_factory=Dossier)
    triage_assessment: dict = field(default_factory=dict)
    priority_files: list[str] = field(default_factory=list)
    baseline_build_exit: int = 0
    baseline_test_exit: int = 0
    baseline_build_output: str = ""
    baseline_verification: list[dict] = field(default_factory=list)
    red_passed_output: str = ""
    followup_fixes: list[str] = field(default_factory=list)
    followup_fixes_commit: str = ""
    failed_prerequisites: list[str] = field(default_factory=list)
    skipped_prerequisites: list[str] = field(default_factory=list)
    squashed_backport_range: str = ""


def phase_0_triage(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
) -> dict[str, str]:
    """Run deterministic triage gates (patch-id, ancestry, seed files) and derive the allowlist."""
    repo = config.repo_path
    fix = resolve_merge_commit(repo, config.source_fix_commit)
    target = config.target_branch

    state.seed = derive_seed(repo, config.source_fix_commit)
    if not state.seed:
        seed_from_diff = (
            subprocess.run(
                [
                    "git",
                    "diff",
                    "--name-only",
                    f"{config.source_fix_commit}^1",
                    config.source_fix_commit,
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            .stdout.strip()
            .splitlines()
        )
        state.seed = [f for f in seed_from_diff if f.strip()]
        if state.seed:
            log.info("Seed derived from diff range (%d files)", len(state.seed))

    fork_point = git_merge_base(repo, config.source_branch, target)
    resolved, _ = resolve_renames(
        repo,
        state.seed,
        target,
        fork_point,
        cap=config.allowlist_rename_expansion_cap,
    )
    state.allowed_modules = resolved

    fwd = forward_patch_id_check(repo, fix, target, config.forward_patch_id_lookback)
    if fwd:
        raise PipelineStop(f"Fix already present on target as {fwd}")

    bwd = backward_patch_id_check(repo, fix, target, config.forward_patch_id_lookback)
    is_ancestor = git_is_ancestor(repo, fix, target)

    ancestry_status = "affected"
    if not is_ancestor and not bwd:
        seed_exists_on_target = any(
            git_cat_file_exists(repo, target, f) for f in state.seed
        )
        if not seed_exists_on_target:
            ancestry_status = "no_seed_files"
            state.allowed_modules = list(state.seed)
            log.warning(
                "No ancestry, no patch-id match, and seed files not found on target. "
                "The upstream fix touches: %s — the equivalent code may live in a "
                "different file on the target branch. Proceeding anyway.",
                ", ".join(state.seed),
            )
        else:
            ancestry_status = "seed_files_only"
            log.info(
                "Ancestry/patch-id checks failed but seed files exist on target — proceeding with caution"
            )

    # The operator-supplied allowlist is authoritative. Apply it after the
    # ancestry block so the no_seed_files fallback cannot clobber it.
    if config.allowed_modules is not None:
        state.allowed_modules = config.allowed_modules

    return {"forward_patch_id": "not_found", "ancestry": ancestry_status}


def _diff_char_caps(config: BugFixConfig) -> tuple[int, int]:
    """Return (max_diff_chars, max_chunk_chars) scaled to the context budget.

    Floors at the legacy defaults (200K / 50K characters). With an estimate
    of 4 characters/token, full-diff growth starts above ~400K configured
    tokens and chunk growth above ~416.7K. Above those floors, the caps use
    ~12.5% / ~3% of the configured window. A 262,144-token window therefore
    retains both legacy caps. Unset context_length uses 128K here; the agent
    loop's model-window auto-detection does not change these diff limits.
    """
    ctx = config.context_length or 128_000
    chars_per_token = 4
    max_diff = max(200_000, int(ctx * 0.125 * chars_per_token))
    max_chunk = max(50_000, int(ctx * 0.03 * chars_per_token))
    return max_diff, max_chunk


def _compute_fix_diff(
    config: BugFixConfig, commit: str | None = None, for_files: list[str] | None = None
) -> str:
    """Compute the fix diff with truncation for large commits.

    When for_files is provided, only include diffs for those specific files
    (used during chunked resolution to keep prompt size manageable).

    Caps scale with ``config.context_length`` so large-context models get
    the full diff instead of falling back to --stat.
    """
    sha = commit or config.source_fix_commit
    max_diff_chars, max_chunk_chars = _diff_char_caps(config)

    if for_files:
        if is_merge_commit(config.repo_path, sha):
            stat = subprocess.run(
                ["git", "diff", "--stat", f"{sha}^1..{sha}"],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
            partial_diff = subprocess.run(
                ["git", "diff", f"{sha}^1..{sha}", "--"] + for_files,
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
        else:
            stat = subprocess.run(
                ["git", "show", "--stat", sha],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
            partial_diff = subprocess.run(
                ["git", "show", sha, "--"] + for_files,
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
        fix_diff = stat + "\n\n--- Diff for chunk files ---\n\n" + partial_diff
        if len(fix_diff) > max_chunk_chars:
            fix_diff = (
                stat
                + "\n\n[Chunk diffs too large — stat only. Agent must read files directly.]"
            )
        return fix_diff

    if is_merge_commit(config.repo_path, sha):
        fix_diff = subprocess.run(
            ["git", "diff", f"{sha}^1..{sha}"],
            cwd=config.repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    else:
        fix_diff = subprocess.run(
            ["git", "show", sha],
            cwd=config.repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout

    if len(fix_diff) > max_diff_chars:
        log.warning(
            "fix_diff too large (%d chars, cap %d), using --stat only",
            len(fix_diff),
            max_diff_chars,
        )
        if is_merge_commit(config.repo_path, sha):
            fix_diff = subprocess.run(
                ["git", "diff", "--stat", f"{sha}^1..{sha}"],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
        else:
            fix_diff = subprocess.run(
                ["git", "show", "--stat", sha],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout

    return fix_diff


def phase_3b_regeneration(
    config: BugFixConfig,
    state: PipelineState,
    regen_cmds: list[str],
) -> None:
    """Run project-specific regeneration commands to rebuild generated files.

    Captures HEAD SHA before running any command. On failure, resets to that
    exact commit so codegen scripts that delete files before crashing don't
    destroy the Phase 3a result.
    """
    safe_point = git_rev_parse(config.repo_path, "HEAD")
    log.info("Phase 3b: safe point is %s", safe_point[:12])
    any_succeeded = False

    for cmd in regen_cmds:
        log.info("Phase 3b: running regeneration command: %s", cmd)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.command_timeout,
            )
        except subprocess.TimeoutExpired:
            log.warning("Phase 3b: regeneration timed out (30min): %s", cmd)
            git_reset_hard(config.repo_path, safe_point)
            state.dossier.add(
                f"Regeneration timed out: {cmd}",
                "Timed out after 30 minutes",
                "Phase 3b",
            )
            continue

        if result.returncode == 0:
            log.info("Phase 3b: regeneration succeeded: %s", cmd)
            any_succeeded = True
        else:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            log.warning("Phase 3b: regeneration exited %d: %s", result.returncode, cmd)
            try:
                build_check = subprocess.run(
                    config.build_command,
                    shell=True,
                    cwd=config.build_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=config.command_timeout,
                )
            except subprocess.TimeoutExpired:
                log.warning("Phase 3b: build check timed out, rolling back")
                git_reset_hard(config.repo_path, safe_point)
                continue
            if build_check.returncode == 0:
                log.info(
                    "Phase 3b: codegen exited non-zero but build still passes — keeping partial regeneration"
                )
                any_succeeded = True
                state.dossier.add(
                    f"Regeneration partial (kept): {cmd}",
                    f"Codegen exited {result.returncode} but build passes.\n{stderr_tail}",
                    "Phase 3b",
                )
            else:
                log.warning(
                    "Phase 3b: build broken after codegen, rolling back to %s",
                    safe_point[:12],
                )
                git_reset_hard(config.repo_path, safe_point)
                state.dossier.add(
                    f"Regeneration failed (rolled back): {cmd}",
                    stderr_tail,
                    "Phase 3b",
                )

    changed = git_diff_name_only(config.repo_path)
    if changed and any_succeeded:
        log.info("Phase 3b: %d files regenerated, committing", len(changed))
        subprocess.run(
            ["git", "add", "-A"],
            cwd=config.repo_path,
            capture_output=True,
        )
        git_commit(
            config.repo_path,
            f"Regenerate files for {config.issue_id}\n\nCommands: {', '.join(regen_cmds)}",
        )
        state.dossier.add(
            "Regenerated files",
            "\n".join(changed[:50])
            + (f"\n... and {len(changed)-50} more" if len(changed) > 50 else ""),
            "Phase 3b",
        )
    else:
        log.info(
            "Phase 3b: no successful regeneration — keeping auto-resolved versions"
        )


def phase_0_5_semantic_triage(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> dict:
    """LLM-powered analysis of the commit before cherry-pick."""

    sha = config.source_fix_commit
    if is_merge_commit(config.repo_path, sha):
        fix_diff = subprocess.run(
            ["git", "diff", "--stat", f"{sha}^1..{sha}"],
            cwd=config.repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    else:
        fix_diff = subprocess.run(
            ["git", "show", "--stat", sha],
            cwd=config.repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout

    context = create_context_str(claude_config, config)
    prompt = SemanticTriagePrompt(
        context=context,
        fix_diff=fix_diff,
        source_branch=config.source_branch,
        target_branch=config.target_branch,
        seed_files=state.seed,
    )

    claude_config_triage = _phase_config(claude_config, config)

    asyncio.run(claude_run(claude_config_triage, [prompt.prompt()], tracker=tracker))

    triage_candidates = [
        os.path.join(config.repo_path, SEMANTIC_TRIAGE_FILE),
        os.path.join(config.repo_path, "runs", SEMANTIC_TRIAGE_FILE),
    ]
    triage_path = next((p for p in triage_candidates if os.path.exists(p)), None)
    assessment = {}
    if triage_path:
        try:
            with open(triage_path) as f:
                assessment = json.load(f)
            log.info(
                "Semantic triage: difficulty=%s, recommendation=%s",
                assessment.get("estimated_difficulty", "?"),
                assessment.get("recommendation", "?"),
            )
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to parse semantic triage output: %s", e)

    state.triage_assessment = assessment
    state.priority_files = assessment.get("priority_files", state.allowed_modules)

    return assessment


_TEST_PATTERNS = (
    "_test.",
    "test_",
    "tests/",
    "/test/",
    "testing/",
    "spec/",
    "_spec.",
    "testdata/",
)


def identify_test_files(seed_files: list[str]) -> list[str]:
    """Return seed files that look like test files based on naming patterns."""
    test_files = []
    for f in seed_files:
        lower = f.lower()
        basename = lower.rsplit("/", 1)[-1]
        if any(p in lower for p in _TEST_PATTERNS) or basename.startswith("test_"):
            test_files.append(f)
    return test_files


def phase_0_5c_test_port(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> list[str]:
    """Identify test files in the fix commit and port them to the target branch."""
    files_to_skip = set(state.triage_assessment.get("files_to_skip", []))
    test_files = [f for f in identify_test_files(state.seed) if f not in files_to_skip]
    if not test_files:
        log.info("Phase 0.5c: no test files found in fix commit — skipping test port")
        return []

    log.info(
        "Phase 0.5c: found %d test files in fix commit: %s",
        len(test_files),
        ", ".join(test_files),
    )

    manifest_abs = os.path.join(config.repo_path, TEST_PORT_MANIFEST_FILE)
    context = create_context_str(claude_config, config)
    sha = config.source_fix_commit
    if is_merge_commit(config.repo_path, sha):
        diff_command = f"git diff {sha}^1..{sha} --"
    else:
        diff_command = f"git show {sha} --"
    prompt = TestPortAgentPrompt(
        context=context,
        source_fix_commit=config.source_fix_commit,
        target_branch=config.target_branch,
        test_files=test_files,
        output_manifest_file=manifest_abs,
        diff_command=diff_command,
    )

    claude_config_test = _phase_config(claude_config, config)

    asyncio.run(claude_run(claude_config_test, [prompt.prompt()], tracker=tracker))

    ported = []
    if os.path.exists(manifest_abs):
        with open(manifest_abs) as f:
            ported = [line.strip() for line in f if line.strip()]
        log.info("Phase 0.5c: ported %d test files: %s", len(ported), ", ".join(ported))
    else:
        log.warning("Phase 0.5c: no manifest file produced — test port may have failed")

    state.ported_test_files = ported
    return ported


def phase_1_baseline_and_red(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
) -> None:
    """Capture comprehensive baseline on the clean target branch before any changes.

    Runs build, test, and all lint/verification commands discovered by triage.
    Stores results so Phase 4+ can distinguish regressions from pre-existing failures.
    """
    # Step 1: Build baseline
    if config.build_command:
        build_exit, build_output = capture_output(
            [config.build_command],
            config.build_dir,
            timeout=config.command_timeout,
        )
        state.baseline_build_exit = build_exit
        state.baseline_build_output = build_output
        if build_exit == 0:
            log.info("Baseline build: PASSED")
        else:
            log.warning(
                "Baseline build: FAILED (exit %d) — target branch has pre-existing build issues",
                build_exit,
            )
    else:
        log.info("Baseline build: skipped (no build command)")

    # Step 2: Test baseline (existing logic)
    if config.test_command:
        red_exit, red_output = capture_output(
            [config.test_command],
            config.build_dir,
            timeout=config.command_timeout,
        )

        state.baseline_test_exit = red_exit
        state.baseline = parse_test_failures(red_output, "")
        is_bugfix = (
            state.triage_assessment.get("commit_type", "").lower() in _BUGFIX_TYPES
        )
        if red_exit == 0:
            if state.ported_test_files and is_bugfix:
                log.warning("RED check passed — deferring to Phase 1.5 triage")
                state.red_passed_output = red_output
            else:
                log.info(
                    "Baseline tests pass — %s",
                    (
                        "feature/refactor commit, RED check not applicable"
                        if not is_bugfix
                        else "no ported vulnerability test available — proceeding"
                    ),
                )
    else:
        log.info("Baseline test: skipped (no test command)")

    # Step 3: Extended verification baseline (lint + optionally triage-discovered tests)
    lint_cmds = [
        c
        for c in state.triage_assessment.get("lint_commands", [])
        if _validate_shell_command(c)
    ]
    if config.lint_command and config.lint_command not in lint_cmds:
        lint_cmds.insert(0, config.lint_command)
    verification_cmds = list(lint_cmds)
    if config.auto_discover_tests:
        for c in state.triage_assessment.get("verification_commands", []):
            if _validate_shell_command(c):
                verification_cmds.append(c)
    state.baseline_verification = []
    for vcmd in verification_cmds:
        try:
            vresult = subprocess.run(
                vcmd,
                shell=True,
                cwd=config.build_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.command_timeout,
            )
            entry = {
                "cmd": vcmd,
                "exit_code": vresult.returncode,
                "output": (vresult.stdout + "\n" + vresult.stderr)[-3000:],
            }
        except subprocess.TimeoutExpired:
            entry = {
                "cmd": vcmd,
                "exit_code": TIMEOUT_EXIT_CODE,
                "output": f"Timed out after {config.command_timeout}s",
            }
        state.baseline_verification.append(entry)
        if entry["exit_code"] == 0:
            log.info("Baseline verification passed: %s", vcmd)
        else:
            log.info(
                "Baseline verification failed (exit %d, pre-existing): %s",
                entry["exit_code"],
                vcmd,
            )


def phase_1_5_red_triage(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> dict:
    """LLM triage when RED check passes — determine if vulnerability exists via a different vector."""
    context = create_context_str(claude_config, config)
    triage_json = (
        json.dumps(state.triage_assessment, indent=2)
        if state.triage_assessment
        else "{}"
    )

    prompt = RedTriagePrompt(
        context=context,
        bug_description=config.bug_description,
        test_output=state.red_passed_output,
        ported_test_files=state.ported_test_files,
        triage_assessment=triage_json,
    )

    claude_config_triage = _phase_config(claude_config, config)

    asyncio.run(claude_run(claude_config_triage, [prompt.prompt()], tracker=tracker))

    triage_candidates = [
        os.path.join(config.repo_path, RED_TRIAGE_FILE),
        os.path.join(config.repo_path, "runs", RED_TRIAGE_FILE),
    ]
    triage_path = next((p for p in triage_candidates if os.path.exists(p)), None)
    result = {}
    if triage_path:
        try:
            with open(triage_path) as f:
                result = json.load(f)
            log.info("RED triage verdict=%s", result.get("verdict", "?"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to parse RED triage output: %s", e)

    return result


MAX_ERROR_CHARS = 50_000
MAX_DIFF_PROMPT_CHARS = 200_000


def _safe_git_diff(repo_path: str, *refs: str) -> str:
    """Run ``git diff`` with size cap and decode safety.

    Returns at most MAX_DIFF_PROMPT_CHARS of output; appends a stat
    summary when truncated.  Binary/encoding noise is replaced rather
    than raising.
    """
    result = subprocess.run(
        ["git", "diff", *refs],
        cwd=repo_path,
        capture_output=True,
    )
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        text = result.stdout.decode("utf-8", errors="replace")

    if len(text) <= MAX_DIFF_PROMPT_CHARS:
        return text

    log.warning("git diff output too large (%d chars), truncating", len(text))
    stat = subprocess.run(
        ["git", "diff", "--stat", *refs],
        cwd=repo_path,
        capture_output=True,
    )
    try:
        stat_text = stat.stdout.decode("utf-8")
    except UnicodeDecodeError:
        stat_text = stat.stdout.decode("utf-8", errors="replace")
    return (
        text[:MAX_DIFF_PROMPT_CHARS] + f"\n\n[TRUNCATED — full stat below]\n{stat_text}"
    )


def _resolve_lint_command(config: BugFixConfig, state: PipelineState) -> str:
    """Return the lint command: YAML field takes priority, then triage-discovered.

    Returns empty string if the lint command is identical to the test command
    (already runs as part of the test step).
    """
    if config.lint_command:
        cmd = config.lint_command
    else:
        triage_lints = [
            c
            for c in state.triage_assessment.get("lint_commands", [])
            if _validate_shell_command(c)
        ]
        cmd = triage_lints[0] if triage_lints else ""
    if cmd and cmd == config.test_command:
        return ""
    return cmd


def _run_build_and_test(
    config: BugFixConfig, state: PipelineState
) -> tuple[bool, str, str, str, str]:
    """Run build, test, and lint commands. Return (passed, build_out, build_err, test_out, test_err).

    Empty build/test commands are skipped (for interpreted languages).
    """
    build_stdout = build_stderr = ""
    if config.build_command:
        build_exit, build_stdout, build_stderr = run_test_suite(
            config.build_command, config.build_dir, timeout=config.command_timeout
        )
        if build_exit != 0:
            return False, build_stdout, build_stderr, "", ""

    test_stdout = test_stderr = ""
    if config.test_command:
        test_exit, test_stdout, test_stderr = run_test_suite(
            config.test_command, config.build_dir, timeout=config.command_timeout
        )
        if test_exit == TIMEOUT_EXIT_CODE:
            log.warning(
                "Test command timed out — treating as verification failure: %s",
                config.test_command,
            )
            return False, build_stdout, build_stderr, test_stdout, test_stderr
        failures = parse_test_failures(test_stdout, test_stderr)
        if failures:
            # Named failures — the baseline says which of them are pre-existing.
            passed, _ = check_regression(failures, state.baseline)
        elif test_exit != 0:
            # The command failed but produced nothing the parser recognized, so
            # there is nothing to compare against the baseline. Treating that as
            # a pass would let any failing test command through. Excuse it only
            # if the baseline failed the same opaque way — the same rule the
            # lint check below already applies.
            passed = state.baseline_test_exit != 0
            if passed:
                log.info(
                    "Test command failed (exit %d) with no parseable failures, "
                    "but the baseline failed the same way — treating as "
                    "pre-existing: %s",
                    test_exit,
                    config.test_command,
                )
            else:
                log.warning(
                    "Test command failed (exit %d) with no parseable "
                    "failures and a clean baseline — treating as a "
                    "verification failure: %s",
                    test_exit,
                    config.test_command,
                )
        else:
            passed = True
    else:
        passed = True

    if not passed:
        return False, build_stdout, build_stderr, test_stdout, test_stderr

    lint_cmd = _resolve_lint_command(config, state)
    if lint_cmd:
        lint_exit, lint_stdout, lint_stderr = run_test_suite(
            lint_cmd, config.build_dir, timeout=config.command_timeout
        )
        if lint_exit == TIMEOUT_EXIT_CODE:
            log.warning("Lint command timed out — treating as failure: %s", lint_cmd)
            return False, build_stdout, build_stderr, lint_stdout, lint_stderr
        if lint_exit != 0:
            baseline_lint_failed = any(
                e["cmd"] == lint_cmd and e["exit_code"] != 0
                for e in state.baseline_verification
            )
            if baseline_lint_failed:
                log.info("Lint failed but pre-existing on baseline: %s", lint_cmd)
            else:
                log.warning("Lint failed (NEW): %s", lint_cmd)
                return False, build_stdout, build_stderr, lint_stdout, lint_stderr

    repo = config.repo_path
    ok, extra_files = enforce_allowlist(
        repo, state.allowed_modules, config.disallowed_modules
    )
    if not ok:
        log.info("Files modified outside priority list (advisory): %s", extra_files)
    test_ok, vetoed = enforce_test_file_veto(repo, state.seed)
    if not test_ok:
        log.info("Test files modified outside seed (advisory): %s", vetoed)
    return True, build_stdout, build_stderr, test_stdout, test_stderr


def phase_4_verify(
    config: BugFixConfig,
    state: PipelineState,
) -> bool:
    """Build, test, and check allowlist conformance (single attempt)."""
    passed, _, _, _, _ = _run_build_and_test(config, state)
    if passed:
        return True
    raise PipelineEscalation("Phase 4 verify failed")


def _non_artifact_dirt(repo_path: str) -> list[str]:
    """Working-tree changes that are not pipeline artifacts.

    ``git_commit`` unstages everything in ``_PIPELINE_ARTIFACTS`` before
    committing, so those files legitimately stay dirty afterwards. Anything
    else left over is real work that did not make it into a commit.
    """
    dirt = []
    for line in git_status_porcelain(repo_path):
        path = line[3:].strip() if len(line) > 3 else line.strip()
        # Renames are reported as "old -> new"; the new path is what matters.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if path and path not in _PIPELINE_ARTIFACTS:
            dirt.append(line)
    return dirt


def _extra_verification_section(commands: list[str]) -> str:
    """Render the commands that must pass on top of build and test."""
    if not commands:
        return ""
    listed = "\n".join(commands)
    return f"""
<extra_verification_commands>
{listed}
</extra_verification_commands>

IMPORTANT: the commands in <extra_verification_commands> are what caused this
escalation. They run from <build_dir>. They are NOT optional: do not write the
success file until every one of them passes in addition to the build and the
tests. They are re-run automatically afterwards, so reporting success while one
still fails will simply fail the run.
"""


def _rerun_verification(config: BugFixConfig, commands: list[str]) -> list[str]:
    """Re-run verification commands, returning those that still fail."""
    still_failing = []
    for vcmd in commands:
        try:
            result = subprocess.run(
                vcmd,
                shell=True,
                cwd=config.build_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.command_timeout,
            )
        except subprocess.TimeoutExpired:
            still_failing.append(vcmd)
            continue
        if result.returncode != 0:
            still_failing.append(vcmd)
    return still_failing


def _commit_recovery_edits(config: BugFixConfig, phase_label: str) -> bool:
    """Commit agent recovery edits so they reach the deliverables.

    The candidate patch (git format-patch), the preserved branch, and the
    Phase 4.5 range-diff are all commit-based — recovery edits left in the
    working tree by the 4a/4b agents would silently vanish from every one
    of them. Returns True if a commit was created, False if there was
    nothing to commit.

    Raises PipelineEscalation if edits exist but could not be committed. A
    verified working tree whose deliverables omit those edits is precisely
    the failure this helper exists to prevent, so it must never be reported
    as a success.
    """
    for marker in (RUN_AND_FIX_FAILURE_FILE, RUN_AND_FIX_SUCCESS_FILE):
        path = os.path.join(config.repo_path, marker)
        if os.path.exists(path):
            os.remove(path)
    if not git_status_porcelain(config.repo_path):
        return False
    head_before = git_rev_parse(config.repo_path, "HEAD")
    subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
    git_commit(
        config.repo_path,
        f"fixup: build/test recovery for {config.issue_id} ({phase_label})",
        trailers=_COAUTHOR_TRAILERS,
    )
    if git_rev_parse(config.repo_path, "HEAD") != head_before:
        log.info("%s: recovery edits committed", phase_label)
        return True

    # Nothing landed on HEAD. That is expected when the only dirt was pipeline
    # artifacts, which git_commit unstages by design. Anything else still dirty
    # is real work the deliverables would silently drop, so it must not be
    # allowed to pass as a success.
    remaining = _non_artifact_dirt(config.repo_path)
    if remaining:
        raise PipelineEscalation(
            f"{phase_label}: recovery edits could not be committed — the "
            f"candidate patch, preserved branch and range-diff would omit "
            f"them. Still uncommitted:\n" + "\n".join(remaining)
        )
    log.info("%s: only pipeline artifacts were dirty, no commit needed", phase_label)
    return False


def phase_4a_build_recovery(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
    fix_diff: str,
) -> None:
    """Agent-assisted build/test error recovery. Adapts code to the target branch API."""
    context = create_context_str(claude_config, config)

    claude_config_4a = _phase_config(claude_config, config)

    for attempt in range(1, config.max_build_test_retries + 1):
        build_exit, build_stdout, build_stderr = run_test_suite(
            config.build_command, config.build_dir, timeout=config.command_timeout
        )
        if build_exit == 0:
            test_exit, test_stdout, test_stderr = run_test_suite(
                config.test_command, config.build_dir, timeout=config.command_timeout
            )

            failures = parse_test_failures(test_stdout, test_stderr)
            no_regression, _ = check_regression(failures, state.baseline)
            if test_exit != TIMEOUT_EXIT_CODE and no_regression:
                log.info("Phase 4a: build+test passed on attempt %d", attempt)
                _commit_recovery_edits(config, "Phase 4a")
                return
        else:
            test_stdout, test_stderr = "", ""

        build_errors = (build_stdout + "\n" + build_stderr).strip()
        test_errors = (test_stdout + "\n" + test_stderr).strip()

        if len(build_errors) > MAX_ERROR_CHARS:
            build_errors = build_errors[:MAX_ERROR_CHARS] + "\n\n[TRUNCATED]"
        if len(test_errors) > MAX_ERROR_CHARS:
            test_errors = test_errors[:MAX_ERROR_CHARS] + "\n\n[TRUNCATED]"

        current_diff = _safe_git_diff(config.repo_path, "HEAD")

        pre_fix_files = set(git_diff_name_only(config.repo_path))

        prompt = BuildErrorRecoveryPrompt(
            context=context,
            fix_diff=fix_diff,
            current_diff=current_diff,
            build_errors=build_errors,
            test_errors=test_errors,
            allowed_modules=state.allowed_modules,
            allowed_seed=state.seed,
            target_branch=config.target_branch,
            build_command=config.build_command,
            test_command=config.test_command,
            iteration=attempt,
            max_retries=config.max_build_test_retries,
        )

        log.info(
            "Phase 4a attempt %d/%d — agent diagnosing build/test errors",
            attempt,
            config.max_build_test_retries,
        )
        asyncio.run(claude_run(claude_config_4a, [prompt.prompt()], tracker=tracker))

        post_fix_files = set(git_diff_name_only(config.repo_path))
        if post_fix_files == pre_fix_files:
            raise PipelineEscalation(
                "Phase 4a: agent made no changes — likely CANNOT_FIX"
            )

        # Strict pre-verification boundary: the agent's edits are committed
        # now, before the next build/test attempt runs. If this phase later
        # escalates, candidate.patch still carries the work; committing only
        # on the success path (as this code used to) loses it on every
        # escalation.
        _commit_recovery_edits(config, f"Phase 4a attempt {attempt}")

    build_exit, _, _ = run_test_suite(
        config.build_command, config.build_dir, timeout=config.command_timeout
    )
    if build_exit == 0:
        test_exit, test_stdout, test_stderr = run_test_suite(
            config.test_command, config.build_dir, timeout=config.command_timeout
        )

        failures = parse_test_failures(test_stdout, test_stderr)
        no_regression, _ = check_regression(failures, state.baseline)
        if test_exit != TIMEOUT_EXIT_CODE and no_regression:
            log.info("Phase 4a: build+test passed after final attempt")
            _commit_recovery_edits(config, "Phase 4a")
            return

    raise PipelineEscalation(
        f"Phase 4a: build/test still failing after {config.max_build_test_retries} recovery attempts"
    )


def phase_4b_run_and_fix(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
    extra_verification: list[str] | None = None,
) -> None:
    """Escalate to the autonomous build-test-fix loop (last resort).

    Assembles a rich context with ALL pipeline data (dossier, triage, config)
    and runs the RunAndFixPrompt as a single long agent session that iterates
    build → test → targeted fix until clean or retries are exhausted.

    A success marker is the agent's own report, so it never clears the gate by
    itself: build and test are re-run here, and so is anything in
    extra_verification.

    extra_verification: commands beyond build/test whose failure caused this
    escalation (lint, extended verification). The loop's own success criteria
    only cover build and test, so an agent can honestly report success while
    the command that got it here still fails. They are named in the context so
    the agent fixes them, and re-run afterwards alongside the build/test gate.
    """
    extra_verification = list(extra_verification or [])
    dossier_text = format_dossier(state.dossier) if state.dossier else ""
    triage_json = (
        json.dumps(state.triage_assessment, indent=2)
        if state.triage_assessment
        else "{}"
    )

    enriched_context = f"""
<context>
<output_dir>{config.repo_path}</output_dir>
<repo_path>{config.repo_path}</repo_path>
<build_dir>{config.build_dir}</build_dir>
<source_branch>{config.source_branch}</source_branch>
<target_branch>{config.target_branch}</target_branch>
<source_fix_commit>{config.source_fix_commit}</source_fix_commit>
<bug_description>{config.bug_description}</bug_description>
<issue_id>{config.issue_id}</issue_id>
<build_command>{config.build_command}</build_command>
<test_command>{config.test_command}</test_command>
<max_build_test_retries>{config.max_build_test_retries}</max_build_test_retries>
</context>
{_extra_verification_section(extra_verification)}

<pipeline_dossier>
{dossier_text}
</pipeline_dossier>

<semantic_triage>
{triage_json}
</semantic_triage>
"""

    phase_4b_profile = config.phase_4b_profile
    phase_4b_model = config.phase_4b_model or claude_config.model
    if phase_4b_profile is None:
        if phase_4b_model == claude_config.model:
            phase_4b_profile = claude_config.profile
        elif claude_config.profile is not None:
            # A bare override must not inherit an unrelated profile's credentials.
            from common.model_registry import build_legacy_profile

            phase_4b_profile = build_legacy_profile(phase_4b_model)
        # Truly unprofiled programmatic callers retain legacy transport lookup.
    if phase_4b_profile is not None:
        phase_4b_model = phase_4b_profile.model
    ext_claude_config = _phase_config(
        claude_config,
        config,
        model=phase_4b_model,
        thinking={"type": "adaptive"},
        effort="max",
        profile=phase_4b_profile,
    )
    ext_claude_config.max_thinking_tokens = 1048576
    log.info(
        "Phase 4b: recovery with %s (pipeline model is %s)",
        phase_4b_model,
        claude_config.model,
    )

    prompt = gen_RunAndFixPrompt(
        context=enriched_context,
        build_command=config.build_command,
        test_command=config.test_command,
        build_dir=config.build_dir,
        max_build_test_retries=config.max_build_test_retries,
    )

    steps = [
        PipelineStep(
            name="phase_4b_external_fix",
            prompt=prompt.prompt(),
        ),
    ]

    # Bound unconditionally: the finalizers below close over both, and 4b is
    # called without a tracker in tests and from the standalone paths.
    phase_4b_json_path = None
    phase_4b_data: dict = {}
    if tracker:
        phase_4b_data.update(
            {
                "output_dir": tracker._output_dir,
                "repo_path": config.repo_path,
                "build_dir": config.build_dir,
                "source_branch": config.source_branch,
                "target_branch": config.target_branch,
                "source_fix_commit": config.source_fix_commit,
                "issue_id": config.issue_id,
                "build_command": config.build_command,
                "test_command": config.test_command,
                "max_build_test_retries": config.max_build_test_retries,
                "model": ext_claude_config.model,
                "dossier": dossier_text,
                "semantic_triage": (
                    state.triage_assessment if state.triage_assessment else {}
                ),
            }
        )
        phase_4b_json_path = os.path.join(tracker._output_dir, "phase_4b_input.json")
        with open(phase_4b_json_path, "w") as f:
            json.dump(phase_4b_data, f, indent=2)
        log.info("Phase 4b input written to %s", phase_4b_json_path)

    def _write_phase_4b_json() -> None:
        if phase_4b_json_path:
            with open(phase_4b_json_path, "w") as f:
                json.dump(phase_4b_data, f, indent=2)

    def _finalize(outcome: str) -> None:
        """Record the verified outcome before leaving the phase."""
        phase_4b_data["outcome"] = outcome
        _write_phase_4b_json()

    failure_file = os.path.join(config.repo_path, RUN_AND_FIX_FAILURE_FILE)
    success_file = os.path.join(config.repo_path, RUN_AND_FIX_SUCCESS_FILE)
    # 4b can run more than once per target (cherry-pick failure path, then
    # verify failure). Remove markers from any earlier invocation so whatever
    # exists after the session is provably this session's verdict.
    for stale in (failure_file, success_file):
        if os.path.exists(stale):
            os.remove(stale)

    log.info("Phase 4b: starting run-and-fix loop (build-test-fix)")

    _4b_start = time.time()

    phase_4b_log_path = (
        os.path.join(tracker._output_dir, "phase_4b_log.txt") if tracker else None
    )
    _4b_log_file: TextIO | None = None
    _prev_stdout = sys.stdout
    if phase_4b_log_path:
        _4b_log_file = open(  # noqa: SIM115 (tee target, closed in finally)
            phase_4b_log_path, "w", encoding="utf-8"
        )
        sys.stdout = Tee(_prev_stdout, _4b_log_file)

    try:
        timings = asyncio.run(_agent_run(ext_claude_config, steps))
    finally:
        if _4b_log_file is not None:
            sys.stdout = _prev_stdout
            _4b_log_file.close()
            log.info("Phase 4b log written to %s", phase_4b_log_path)

    _4b_end = time.time()

    query_records = []
    if tracker and timings:
        for st in timings:
            # A recorded duration of 0 is a valid measurement; only a missing
            # entry falls back to the whole-phase span.
            raw_dur = st.get("duration")
            dur = float(raw_dur) if raw_dur is not None else (_4b_end - _4b_start)
            sdk_cost = st.get("cost_usd")
            query_records.append(
                tracker.record_query(
                    prompt_name=st.get("name", "phase_4b"),
                    start_time=_4b_start,
                    end_time=_4b_start + dur,
                    input_tokens=st.get("input_tokens", 0),
                    output_tokens=st.get("output_tokens", 0),
                    sdk_cost_usd=sdk_cost,
                    model=ext_claude_config.model,
                    profile=ext_claude_config.profile,
                )
            )

    if tracker and phase_4b_json_path:
        phase_4b_data["duration_seconds"] = _4b_end - _4b_start
        if timings:
            phase_4b_data["cost_usd"] = sum(record.cost_usd for record in query_records)
            phase_4b_data["input_tokens"] = sum(
                int(cast(Any, st.get("input_tokens", 0))) for st in timings
            )
            phase_4b_data["output_tokens"] = sum(
                int(cast(Any, st.get("output_tokens", 0))) for st in timings
            )

        # Failure first — must match the decision precedence below, so a
        # session that somehow writes both files is recorded as a failure.
        for result_file in (failure_file, success_file):
            if os.path.exists(result_file):
                dest = os.path.join(tracker._output_dir, os.path.basename(result_file))
                shutil.copy2(result_file, dest)
                log.info("Phase 4b result copied to %s", dest)
                with open(result_file) as f:
                    phase_4b_data["agent_reported"] = (
                        "success" if result_file == success_file else "failure"
                    )
                    phase_4b_data["result_summary"] = f.read()
                break
        # The agent's marker is a claim, not the verdict — build/test and the
        # escalating commands are re-run below. Record the claim now (so the
        # artifact survives a crash) but leave the outcome pending until the
        # deterministic checks have actually run.
        phase_4b_data["outcome"] = "pending_verification"
        _write_phase_4b_json()

    if os.path.exists(failure_file):
        with open(failure_file) as f:
            failure_text = f.read()
        state.dossier.add("Run-and-fix failure", failure_text, "Phase 4b")
        # Commit the agent-session work now, at the abort boundary: the
        # deliverable must contain what the agent produced, and nothing that
        # ran after this point (there is nothing left to run) can contaminate it.
        _commit_worktree_edits(config, "Phase 4b (agent-reported failure)")
        _finalize("failure")
        raise PipelineEscalation("Phase 4b: run-and-fix loop exhausted retries")
    elif os.path.exists(success_file):
        state.dossier.add(
            "Run-and-fix loop", "Build-test-fix loop succeeded", "Phase 4b"
        )
        log.info("Phase 4b: run-and-fix loop reported success — verifying")

        # Commit the agent's edits *before* verification runs, strictly: the
        # checks below re-run the build/test commands, which dirty the tree
        # themselves (centpkg prep re-unpacks, Maven writes target/). Only
        # this commit carries the agent's work into candidate.patch, and the
        # re-verification's side effects stay out of it. If the work cannot
        # be committed, escalating (with the failure-path patch export) is
        # safer than reporting success without it.
        if _commit_recovery_edits(config, "Phase 4b (pre-verification)"):
            state.dossier.add(
                "Run-and-fix loop",
                "Recovery edits committed (fixup, pre-verification)",
                "Phase 4b",
            )

        # The marker says the agent believes it is done. Prove it: on the
        # cherry-pick-failure and verify-failure paths extra_verification is
        # empty, so without this the basic build/test gate would rest entirely
        # on the agent's self-report.
        try:
            phase_4_verify(config, state)
        except PipelineEscalation as e:
            state.dossier.add(
                "Run-and-fix loop",
                f"Reported success, but build/test still fails: {e}",
                "Phase 4b",
            )
            _finalize("verification_failed")
            raise PipelineEscalation(
                f"Phase 4b: reported success while build/test still fails ({e})"
            ) from None

        still_failing = _rerun_verification(config, extra_verification)
        if still_failing:
            state.dossier.add(
                "Run-and-fix loop",
                "Reported success, but the commands that triggered the "
                "escalation still fail:\n" + "\n".join(still_failing),
                "Phase 4b",
            )
            _finalize("verification_failed")
            raise PipelineEscalation(
                "Phase 4b: reported success while extended verification still "
                f"fails: {', '.join(still_failing)}"
            )
        # No post-verification commit: the agent's work is already in the
        # pre-verification commit above, and everything the checks left in
        # the tree is side-effect dirt that must not reach the deliverable.
        _finalize("success")
    else:
        # The agent session ended without writing either marker (crash, early
        # exit, ignored instructions) — success may not be assumed.
        state.dossier.add(
            "Run-and-fix loop",
            "Agent ended without writing a success or failure file",
            "Phase 4b",
        )
        _finalize("no_result_file")
        _commit_worktree_edits(config, "Phase 4b (no result file)")
        raise PipelineEscalation(
            "Phase 4b: run-and-fix loop ended without writing a result file"
        )


MAX_PREREQUISITE_DEPTH = 5


def cherry_pick_and_resolve(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
    commit: str,
    label: str,
    files_to_skip: list[str] | None = None,
) -> str:
    """Cherry-pick a single commit trying multiple strategies. Returns 'clean', 'resolved', or raises."""

    class _CherryPickStrategy(TypedDict):
        strategy: str | None
        strategy_option: str | None
        name: str

    strategies: list[_CherryPickStrategy] = [
        {"strategy": None, "strategy_option": None, "name": "default"},
        {"strategy": None, "strategy_option": "patience", "name": "patience"},
        {"strategy": "ort", "strategy_option": None, "name": "ort"},
    ]

    best_strategy: tuple[_CherryPickStrategy, int] | None = None
    for strat in strategies:
        result = git_cherry_pick(
            config.repo_path,
            commit,
            strategy=strat["strategy"],
            strategy_option=strat["strategy_option"],
        )
        status = git_status_porcelain(config.repo_path)
        uu_files = _conflicted_files(status)

        log.info(
            "%s: strategy %s: exit=%d, conflicts=%d, success=%s",
            label,
            strat["name"],
            result.exit_code,
            len(uu_files),
            result.success,
        )

        if result.success and not uu_files:
            log.info("%s: clean cherry-pick (strategy: %s)", label, strat["name"])
            if state.dossier:
                state.dossier.add_strategy(strat["name"], "clean")
            return "clean"

        if uu_files:
            if best_strategy is None or len(uu_files) < best_strategy[1]:
                best_strategy = (strat, len(uu_files))
            if state.dossier:
                state.dossier.add_strategy(
                    strat["name"], f"conflict ({len(uu_files)} files)"
                )
        else:
            log.warning(
                "%s: strategy %s: no conflicts but exit=%d, stderr=%s",
                label,
                strat["name"],
                result.exit_code,
                result.stderr[:500],
            )

        git_cherry_pick_abort(config.repo_path)

    if best_strategy is None:
        raise PipelineEscalation(
            f"{label}: unmappable cherry-pick (all strategies failed)"
        )

    log.info(
        "%s: using strategy %s (%d conflicts, fewest)",
        label,
        best_strategy[0]["name"],
        best_strategy[1],
    )
    strat = best_strategy[0]
    git_cherry_pick(
        config.repo_path,
        commit,
        strategy=strat["strategy"],
        strategy_option=strat["strategy_option"],
    )
    status = git_status_porcelain(config.repo_path)
    uu_files = _conflicted_files(status)

    if not uu_files:
        log.info("%s: clean cherry-pick on retry (strategy: %s)", label, strat["name"])
        return "clean"

    conflict_summary_parts = []
    skip_set = set(files_to_skip or [])
    if skip_set:
        # Reset ALL skip files to target version — both conflicting and cleanly-applied
        all_changed = (
            subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            .stdout.strip()
            .splitlines()
        )
        skip_applied_clean = [
            f for f in all_changed if f in skip_set and f not in uu_files
        ]
        if skip_applied_clean:
            log.info(
                "%s: reverting %d skip files that applied cleanly",
                label,
                len(skip_applied_clean),
            )
            for f in skip_applied_clean:
                subprocess.run(
                    ["git", "checkout", "HEAD", "--", f],
                    cwd=config.repo_path,
                    capture_output=True,
                )

        auto_resolved = [f for f in uu_files if f in skip_set]
        if auto_resolved:
            log.info(
                "%s: reverting %d conflicting skip files to target version",
                label,
                len(auto_resolved),
            )
            for f in auto_resolved:
                subprocess.run(
                    ["git", "checkout", "HEAD", "--", f],
                    cwd=config.repo_path,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "add", "--", f], cwd=config.repo_path, capture_output=True
                )

        total_skipped = len(skip_applied_clean) + len(auto_resolved)
        if total_skipped:
            conflict_summary_parts.append(
                f"Reverted {total_skipped} files to target version (files_to_skip)"
            )
            uu_files = [f for f in uu_files if f not in skip_set]

    if not uu_files:
        msg = (
            f"Port {label}\n\n[ Conflict summary: {'; '.join(conflict_summary_parts)} ]"
        )
        git_commit(config.repo_path, msg)
        return "resolved"

    original_uu_files = list(uu_files)
    base_head = git_rev_parse(config.repo_path, "HEAD")
    fix_diff = _compute_fix_diff(config, commit)
    context = create_context_str(claude_config, config)
    priority = state.priority_files if state.priority_files else state.allowed_modules

    claude_config_resolve = _phase_config(claude_config, config)

    if config.mechanical_resolution:
        try:
            mech = resolve_mechanically(config.repo_path, commit, uu_files)
        except Exception as e:
            log.warning(
                "%s: mechanical resolution failed (continuing to LLM): %s", label, e
            )
            mech = {}
        if mech:
            log.info(
                "%s: mechanically re-applied %d/%d conflicted files",
                label,
                len(mech),
                len(uu_files),
            )
            if state.dossier:
                for f, rung in sorted(mech.items()):
                    state.dossier.add_strategy(f"mechanical:{rung}", f"resolved {f}")
            conflict_summary_parts.append(
                f"Mechanically re-applied {len(mech)} files (git apply, relaxed context)"
            )
            uu_files = [f for f in uu_files if f not in mech]
            if tracker:
                tracker.record_gate(
                    "mechanical_resolution", f"{len(mech)}/{len(original_uu_files)}"
                )

    if not uu_files:
        return _finalize_resolution(
            config,
            claude_config_resolve,
            state,
            tracker,
            commit,
            label,
            conflict_summary_parts,
            base_head,
            fix_diff,
            context,
        )

    followup_note = ""
    if state.followup_fixes and _is_followup_target(config.repo_path, commit, state):
        followup_note = "\n".join(f"  - {f}" for f in state.followup_fixes)

    _max_diff, max_chunk = _diff_char_caps(config)
    CHUNK_SIZE = 30 if max_chunk <= 50_000 else 45

    for attempt in range(1, config.max_resolution_retries + 1):
        if len(uu_files) > CHUNK_SIZE:
            if len(uu_files) > 100:
                chunk_size = 10 if max_chunk <= 50_000 else 15
            elif len(fix_diff) > max_chunk:
                chunk_size = 15 if max_chunk <= 50_000 else 20
            else:
                chunk_size = CHUNK_SIZE
            log.info(
                "%s attempt %d/%d — resolving %d conflicts in chunks of %d",
                label,
                attempt,
                config.max_resolution_retries,
                len(uu_files),
                chunk_size,
            )
            for chunk_start in range(0, len(uu_files), chunk_size):
                chunk = uu_files[chunk_start : chunk_start + chunk_size]
                log.info(
                    "%s: chunk %d-%d of %d",
                    label,
                    chunk_start + 1,
                    chunk_start + len(chunk),
                    len(uu_files),
                )
                chunk_diff = _compute_fix_diff(config, commit, for_files=chunk)
                prompt = NarrowResolutionAgentPrompt(
                    fix_diff=chunk_diff,
                    conflicted_files=chunk,
                    context=context,
                    allowed_modules=priority,
                    triage_assessment=(
                        json.dumps(state.triage_assessment, indent=2)
                        if state.triage_assessment
                        else ""
                    ),
                    target_branch=config.target_branch,
                    baseline_ref=base_head,
                    followup_fixes=followup_note,
                )
                asyncio.run(
                    claude_run(
                        claude_config_resolve, [prompt.prompt()], tracker=tracker
                    )
                )
        else:
            prompt = NarrowResolutionAgentPrompt(
                fix_diff=fix_diff,
                conflicted_files=uu_files,
                context=context,
                allowed_modules=priority,
                triage_assessment=(
                    json.dumps(state.triage_assessment, indent=2)
                    if state.triage_assessment
                    else ""
                ),
                target_branch=config.target_branch,
                baseline_ref=base_head,
                followup_fixes=followup_note,
            )
            log.info(
                "%s attempt %d/%d — resolving %d conflicts",
                label,
                attempt,
                config.max_resolution_retries,
                len(uu_files),
            )
            asyncio.run(
                claude_run(claude_config_resolve, [prompt.prompt()], tracker=tracker)
            )

        remaining = git_status_porcelain(config.repo_path)
        still_conflicted = _has_conflicts(remaining)
        if not still_conflicted:
            conflict_summary_parts.append(
                f"LLM resolved {len(original_uu_files)} source conflicts ({attempt} attempt(s))"
            )
            return _finalize_resolution(
                config,
                claude_config_resolve,
                state,
                tracker,
                commit,
                label,
                conflict_summary_parts,
                base_head,
                fix_diff,
                context,
            )
        log.warning(
            "%s: attempt %d: %d conflicts remain", label, attempt, len(still_conflicted)
        )
        uu_files = _conflicted_files(still_conflicted)

    git_cherry_pick_abort(config.repo_path)
    raise PipelineEscalation(
        f"{label}: failed after {config.max_resolution_retries} attempts"
    )


def _is_followup_target(repo_path: str, commit: str, state: PipelineState) -> bool:
    """True if the follow-up fixes scanned in Phase 0 refer to this commit.

    Prerequisite ports go through the same resolution path but must not be
    told that the main commit's follow-ups correct THEM.
    """
    if not state.followup_fixes_commit:
        return False
    try:
        return git_rev_parse(repo_path, commit) == state.followup_fixes_commit
    except RuntimeError:
        return False


def _finalize_resolution(
    config: BugFixConfig,
    claude_config_resolve: ClaudeConfig,
    state: PipelineState,
    tracker,
    commit: str,
    label: str,
    conflict_summary_parts: list[str],
    base_head: str,
    fix_diff: str,
    context: str,
) -> str:
    """Commit the resolution, run deterministic post-passes, then Phase 3a.1.

    Post-passes read only the upstream history and the target branch:
    - substitution mining + auto-apply of unambiguous downstream renames
      whose upstream form does not exist on the target at all
    - whitespace snap-back of LLM-drifted lines to source bytes
    - token audit + optional per-file checks, injected as evidence into the
      Phase 3a.1 verification agent
    """
    repo = config.repo_path

    # The resolution agent is instructed to commit; conclude the cherry-pick
    # ourselves if it didn't.
    if git_rev_parse(repo, "HEAD") == base_head:
        summary = (
            "; ".join(conflict_summary_parts) if conflict_summary_parts else "resolved"
        )
        git_commit(
            repo,
            f"Port {label}\n\n[ Conflict summary: {summary} ]",
            trailers=_COAUTHOR_TRAILERS,
        )
        if git_rev_parse(repo, "HEAD") == base_head:
            # Empty resolution or a rejecting commit hook — never amend or
            # post-process base_head itself.
            log.warning(
                "%s: resolution commit did not move HEAD — skipping post-passes", label
            )
            return "resolved"

    changed_files = git_diff_name_only_range(repo, base_head, "HEAD")

    findings = []
    line_findings = []
    applied_subs: dict[str, str] = {}
    advisory_subs: dict[str, str] = {}
    auto_restored: dict[str, int] = {}
    file_check_failures: list[str] = []
    marker_files: list[str] = []

    # --- Deterministic rewrites (run before any auditing) -----------------
    if config.evidence_checks:
        try:
            port_diff = subprocess.run(
                ["git", "diff", base_head, "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
            added_by_file = added_lines_by_file(port_diff)
            # Mining references base_head — the tree the port actually lands
            # on (target tip PLUS ported prerequisites/tests). Using the bare
            # target branch would mine a ported prerequisite's own rename
            # backwards and mechanically revert it.
            mined = mine_substitutions(
                repo, f"{commit}^1", base_head, sorted(added_by_file)
            )
            guarded = {
                old: new
                for old, new in mined.items()
                if not token_exists_on_ref(repo, base_head, old)
            }
            advisory_subs = {o: n for o, n in mined.items() if o not in guarded}
            for f in changed_files:
                abs_path = os.path.join(repo, f)
                if not guarded or not os.path.isfile(abs_path):
                    continue
                try:
                    with open(abs_path, encoding="utf-8", newline="") as fh:
                        current = fh.read()
                except (UnicodeDecodeError, OSError):
                    continue
                preimage = _git_blob(repo, base_head, f) or ""
                new_text, hits = apply_substitutions(current, preimage, guarded)
                if hits:
                    with open(abs_path, "w", encoding="utf-8", newline="") as fh:
                        fh.write(new_text)
                    for token in hits:
                        applied_subs[token] = guarded[token]
            if applied_subs:
                log.info(
                    "Auto-applied downstream substitutions: %s",
                    ", ".join(f"{o}->{n}" for o, n in sorted(applied_subs.items())),
                )
                if state.dossier:
                    state.dossier.add(
                        "Auto-applied downstream substitutions",
                        "\n".join(
                            f"{o} -> {n}" for o, n in sorted(applied_subs.items())
                        ),
                        "Phase 3a post-pass (mined from upstream-base vs target delta)",
                    )

            # Deterministically restore fix bytes for pure token-renames: the
            # clearest three-way violations must not depend on the verifier's
            # obedience. Guards inside the finder ensure every restored token
            # exists on the target.
            restorations = find_pure_rename_restorations(
                repo, base_head, commit, port_diff
            )
            for f, items in restorations.items():
                abs_path = os.path.join(repo, f)
                if not os.path.isfile(abs_path):
                    continue
                try:
                    with open(abs_path, encoding="utf-8", newline="") as fh:
                        lines = fh.read().split("\n")
                except (UnicodeDecodeError, OSError):
                    continue
                changed = False
                for lineno, old_line, fix_line in items:
                    idx = lineno - 1
                    if not (0 <= idx < len(lines)):
                        continue
                    # Diff-derived lines lose their \r; the worktree keeps it.
                    eol = "\r" if lines[idx].endswith("\r") else ""
                    if lines[idx].rstrip("\r") == old_line:
                        lines[idx] = fix_line + eol
                        auto_restored[f] = auto_restored.get(f, 0) + 1
                        changed = True
                if changed:
                    with open(abs_path, "w", encoding="utf-8", newline="") as fh:
                        fh.write("\n".join(lines))
            if auto_restored:
                log.info(
                    "Auto-restored fix bytes for %d pure-renamed lines in %d files",
                    sum(auto_restored.values()),
                    len(auto_restored),
                )
                if state.dossier:
                    state.dossier.add(
                        "Fix lines auto-restored (pure renames)",
                        "\n".join(
                            f"{f}: {n} lines" for f, n in sorted(auto_restored.items())
                        ),
                        "Phase 3a post-pass",
                    )
        except Exception as e:
            log.warning("Evidence checks failed (continuing): %s", e)

    if config.snapback:
        try:
            snap_counts = snap_back_files(repo, changed_files, base_head, commit)
            if snap_counts:
                log.info(
                    "Snap-back: restored source bytes in %d files (%d lines)",
                    len(snap_counts),
                    sum(snap_counts.values()),
                )
                if state.dossier:
                    state.dossier.add(
                        "Snap-back (drifted formatting restored to source bytes)",
                        "\n".join(
                            f"{f}: {n} lines" for f, n in sorted(snap_counts.items())
                        ),
                        "Phase 3a post-pass",
                    )
        except Exception as e:
            log.warning("Snap-back failed (continuing): %s", e)

    if git_diff_name_only(repo):
        subprocess.run(["git", "add", "-u"], cwd=repo, capture_output=True)
        git_amend(repo)
        changed_files = git_diff_name_only_range(repo, base_head, "HEAD")

    # --- Deterministic audits (on the final rewritten state) --------------
    current_texts: dict[str, str] = {}
    for f in changed_files:
        abs_path = os.path.join(repo, f)
        if not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, encoding="utf-8", newline="") as fh:
                current_texts[f] = fh.read()
        except (UnicodeDecodeError, OSError):
            continue

    try:
        # Require BOTH an opening and closing marker: a lone seven-equals
        # line is legitimate content (doc tables, setext underlines).
        marker_files = [
            f
            for f, text in current_texts.items()
            if _MARKER_START_RE.search(text) and _MARKER_END_RE.search(text)
        ]
        if marker_files:
            log.warning(
                "Conflict markers committed in %d files: %s",
                len(marker_files),
                ", ".join(marker_files),
            )
            if tracker:
                tracker.record_gate("conflict_markers", f"{len(marker_files)} files")
            if state.dossier:
                state.dossier.add(
                    "Conflict markers committed",
                    "\n".join(marker_files),
                    "Phase 3a post-pass (deterministic scan)",
                )
    except Exception as e:
        log.warning("Conflict-marker scan failed (continuing): %s", e)

    if config.evidence_checks:
        try:
            port_diff = subprocess.run(
                ["git", "diff", base_head, "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
            added_by_file = added_lines_by_file(port_diff)
            findings = audit_added_tokens(repo, base_head, commit, added_by_file)
            port_changes = port_line_changes(port_diff)
            annotate_dead_config(
                findings,
                port_changes,
                current_texts,
                lambda s: target_config_state(repo, base_head, s),
            )
            if findings:
                log.info(
                    "Token audit: %d identifiers flagged (%s)",
                    len(findings),
                    ", ".join(f.token for f in findings[:8]),
                )
            if tracker:
                tracker.record_gate("token_audit", f"{len(findings)} flagged")

            line_findings = audit_line_provenance(
                repo,
                base_head,
                commit,
                port_diff,
                current_texts,
                substitutions=applied_subs,
            )
            if line_findings:
                log.info(
                    "Three-way discipline: %d line findings (%s)",
                    len(line_findings),
                    ", ".join(sorted({lf.kind for lf in line_findings})),
                )
            if tracker:
                tracker.record_gate("line_provenance", f"{len(line_findings)} findings")
        except Exception as e:
            log.warning("Evidence audits failed (continuing): %s", e)

    if config.file_check_command:
        try:
            file_check_failures = run_file_checks(
                repo, changed_files, config.file_check_command, config.command_timeout
            )
            if file_check_failures:
                log.warning(
                    "Per-file checks failed for %d files", len(file_check_failures)
                )
                if state.dossier:
                    state.dossier.add(
                        "Per-file check failures",
                        "\n".join(file_check_failures),
                        "Phase 3a post-pass",
                    )
        except Exception as e:
            log.warning("Per-file checks failed to run (continuing): %s", e)

    followups = state.followup_fixes if _is_followup_target(repo, commit, state) else []
    evidence_report = render_evidence(
        findings,
        applied_subs,
        advisory_subs,
        followups,
        file_check_failures,
        line_findings=line_findings,
        conflict_marker_files=marker_files,
        auto_restored=auto_restored,
    )

    current_diff = _safe_git_diff(repo, base_head)
    if config.post_resolution_check and current_diff.strip():
        log.info("Phase 3a.1 — post-resolution adaptation check")
        check_prompt = PostResolutionCheckPrompt(
            context=context,
            fix_diff=fix_diff,
            current_diff=current_diff,
            target_branch=config.target_branch,
            triage_assessment=(
                json.dumps(state.triage_assessment, indent=2)
                if state.triage_assessment
                else ""
            ),
            evidence_report=evidence_report,
            baseline_ref=base_head,
        )
        try:
            asyncio.run(
                claude_run(
                    claude_config_resolve, [check_prompt.prompt()], tracker=tracker
                )
            )
        except Exception:
            log.warning(
                "Phase 3a.1 — post-resolution check failed, continuing with existing resolution"
            )
        else:
            # No snap-back after the verifier: its deliberate corrections must
            # never be rewritten, and every pre-verifier line was already
            # snapped with the same sources — a second pass could only undo
            # the verifier's work (observed on real runs) or do nothing.
            post_status = git_status_porcelain(repo)
            if post_status:
                subprocess.run(["git", "add", "-u"], cwd=repo, capture_output=True)
                git_commit(
                    repo,
                    "fixup: adapt backport to target branch API",
                    trailers=_COAUTHOR_TRAILERS,
                )
                log.info("Phase 3a.1 — post-resolution check committed adaptations")
            if config.evidence_checks and findings:
                try:
                    port_diff = subprocess.run(
                        ["git", "diff", base_head, "HEAD"],
                        cwd=repo,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    ).stdout
                    remaining = audit_added_tokens(
                        repo, base_head, commit, added_lines_by_file(port_diff)
                    )
                    if tracker:
                        tracker.record_gate(
                            "token_audit_post", f"{len(remaining)} remaining"
                        )
                    if remaining:
                        log.warning(
                            "Token audit: %d identifiers still flagged after Phase 3a.1",
                            len(remaining),
                        )
                        if state.dossier:
                            state.dossier.add(
                                "Token audit — unresolved after Phase 3a.1",
                                "\n".join(
                                    f"{f.token} [{f.status}] in {', '.join(sorted(set(f.files))[:3])}"
                                    for f in remaining
                                ),
                                "Phase 3a.1 advisory re-audit",
                            )
                except Exception as e:
                    log.warning("Post-3a.1 token re-audit failed: %s", e)

    if config.blind_review:
        blind_diff = _safe_git_diff(repo, base_head)
        if blind_diff.strip():
            log.info("Phase 3a.2 — triage-blind independent review")
            blind_prompt = TriageBlindReviewPrompt(
                context=context,
                fix_diff=fix_diff,
                current_diff=blind_diff,
                target_branch=config.target_branch,
                baseline_ref=base_head,
            )
            try:
                asyncio.run(
                    claude_run(
                        claude_config_resolve, [blind_prompt.prompt()], tracker=tracker
                    )
                )
            except Exception:
                log.warning("Phase 3a.2 — blind review failed, continuing")
            else:
                blind_status = git_status_porcelain(repo)
                if blind_status:
                    subprocess.run(["git", "add", "-u"], cwd=repo, capture_output=True)
                    git_commit(
                        repo,
                        "fixup: independent review corrections (Phase 3a.2)",
                        trailers=_COAUTHOR_TRAILERS,
                    )
                    log.info("Phase 3a.2 — blind review committed corrections")
                    if state.dossier:
                        state.dossier.add(
                            "Independent review corrections",
                            "Triage-blind reviewer found and fixed symbol "
                            "adaptations that were incorrect.",
                            "Phase 3a.2",
                        )

    return "resolved"


def port_prerequisite_chain(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> list[dict]:
    """Port prerequisite commits identified by Phase 0.5 before the main commit."""
    prerequisites = state.triage_assessment.get("prerequisites", [])
    portable = [
        p for p in prerequisites if p.get("portable", False) and p.get("introduced_by")
    ]

    if not portable:
        return []

    log.info(
        "Phase 0.5 identified %d portable prerequisites to port first", len(portable)
    )

    overflow = portable[MAX_PREREQUISITE_DEPTH:]
    if overflow:
        skipped = [p.get("symbol", "unknown") for p in overflow]
        state.skipped_prerequisites.extend(skipped)
        log.warning(
            "Phase 0.5: %d portable prerequisites exceed the depth cap (%d); "
            "skipping: %s",
            len(overflow),
            MAX_PREREQUISITE_DEPTH,
            ", ".join(skipped),
        )
        state.dossier.add(
            "Prerequisites skipped (depth cap)",
            "\n".join(
                f"{p.get('symbol', 'unknown')}: {p.get('introduced_by', '')}"
                for p in overflow
            ),
            f"Phase 0.5 prerequisite chain (cap {MAX_PREREQUISITE_DEPTH})",
        )

    ported = []

    for i, prereq in enumerate(portable[:MAX_PREREQUISITE_DEPTH]):
        sha = prereq["introduced_by"]
        symbol = prereq.get("symbol", "unknown")
        subject = prereq.get("subject", "")
        label = f"prerequisite {i+1}/{len(portable)}: {symbol} ({sha[:10]})"

        log.info("Porting %s — %s", label, subject)
        state.dossier.add(
            f"Prerequisite {i+1}: {symbol}",
            f"Commit: {sha}\nSubject: {subject}\nReason: {prereq.get('reason', '')}",
            "Phase 0.5 prerequisite chain",
        )

        old_commit = config.source_fix_commit
        try:
            config.source_fix_commit = sha
            result = cherry_pick_and_resolve(
                config,
                claude_config,
                state,
                tracker,
                commit=sha,
                label=label,
                files_to_skip=state.triage_assessment.get("files_to_skip", []),
            )
            ported.append({"sha": sha, "symbol": symbol, "result": result})
            log.info("Prerequisite %s ported: %s", symbol, result)
        except PipelineEscalation as e:
            log.warning("Prerequisite %s failed: %s — continuing without it", symbol, e)
            state.failed_prerequisites.append(symbol)
            state.dossier.add(
                f"Prerequisite failed: {symbol}",
                str(e),
                "Phase 0.5 prerequisite chain",
            )
        finally:
            config.source_fix_commit = old_commit

    return ported


def _squashed_backport_range(config: BugFixConfig, state: PipelineState) -> str | None:
    """Synthetic single-commit range spanning the whole backport.

    Pipeline commits (test port, resolution, review fixups, 4a/4b recovery)
    make target..HEAD multi-commit, and range-diff pairs commits one-to-one,
    so every extra commit shows up as an unmatched row regardless of content.
    Squashing the backport into one synthetic commit (tree = HEAD's tree,
    parent = the target base) makes the comparison about content, not
    commit count.

    Cached on ``state``: commit objects embed the committer timestamp, so
    calling this twice yields two different SHAs (and two dangling objects)
    for the same tree. The verdict and the dossier's range-diff must quote
    the same object.
    """
    if state.squashed_backport_range:
        return state.squashed_backport_range
    repo = config.repo_path
    try:
        # Range from the frozen baseline, not the branch name — the agent
        # may have moved the branch (see run_pipeline); merge-base against
        # the moved tip can collapse to HEAD and return None here.
        target_base = git_merge_base(
            repo, config.frozen_target_sha or config.target_branch, "HEAD"
        )
        head = git_rev_parse(repo, "HEAD")
        if not target_base or not head or target_base == head:
            return None
        result = subprocess.run(
            [
                "git",
                "commit-tree",
                f"{head}^{{tree}}",
                "-p",
                target_base,
                "-m",
                f"squashed backport for {config.issue_id}",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return None
        squash = result.stdout.strip()
        if not squash:
            return None
        state.squashed_backport_range = f"{squash}^..{squash}"
        return state.squashed_backport_range
    except Exception:
        return None


def _backport_completeness(
    config: BugFixConfig, state: PipelineState
) -> tuple[str, str]:
    """Classify the finished backport as "success" or "partial_success".

    Fires ONLY for non-bugfix commits (feature/api-change ports). A bug fix
    that reached this point ported its whole file set by construction, while
    a feature port can "succeed" by dropping most of its files —
    files_to_skip reverts them to the target's version, so a passing build
    mostly proves the target still compiles. Returns (outcome, detail).
    """
    raw_type = state.triage_assessment.get("commit_type")
    commit_type = raw_type if isinstance(raw_type, str) else ""
    # Normalize LLM variants ("Bug Fix", " bug_fix ") before matching, so a
    # sloppily-labeled bug fix can never be downgraded.
    commit_type = commit_type.strip().lower().replace("_", "-").replace(" ", "-")
    if not commit_type or commit_type in _BUGFIX_TYPES:
        return "success", ""

    reasons = []
    try:
        if state.failed_prerequisites:
            reasons.append(
                "prerequisites failed to port: "
                + ", ".join(str(s) for s in state.failed_prerequisites)
            )

        if state.skipped_prerequisites:
            reasons.append(
                "prerequisites skipped by the depth cap: "
                + ", ".join(str(s) for s in state.skipped_prerequisites)
            )

        # state.seed is the fix's file set with merge-commit (-m1) handling;
        # git show --name-only is the fallback and is empty for merges.
        fix_files = set(state.seed) or set(
            git_show_files(config.repo_path, config.source_fix_commit)
        )
        skipped = set(state.triage_assessment.get("files_to_skip", [])) & fix_files
        if fix_files and skipped:
            target_base = git_merge_base(
                config.repo_path,
                config.frozen_target_sha or config.target_branch,
                "HEAD",
            )
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{target_base}..HEAD"],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            backported = (
                set(result.stdout.splitlines()) if result.returncode == 0 else set()
            )
            # A skipped file that reappears in the final diff was regenerated
            # (Phase 3b) or otherwise recovered — only count real drops.
            dropped = sorted(skipped - backported)
            ratio = len(dropped) / len(fix_files)
            if dropped and ratio >= config.partial_success_skip_ratio:
                reasons.append(
                    f"{len(dropped)}/{len(fix_files)} of the fix's files were "
                    f"dropped (reverted to target, never regenerated): "
                    + ", ".join(dropped[:10])
                    + (" …" if len(dropped) > 10 else "")
                )
    except Exception as e:
        log.warning("Completeness check failed: %s", e)

    if reasons:
        return "partial_success", "; ".join(reasons)
    return "success", ""


# Relatedness thresholds for the Phase 4.5 fallback (see
# _shares_content_with_fix). Derived by measurement, not taste — re-derive them
# before changing: compare commits with disjoint file sets and check that the
# resulting overlap ratios stay well under the threshold.
_RELATEDNESS_MIN_LINE_LEN = 6
_RELATEDNESS_MIN_OVERLAP = 0.30


def _changed_payload_lines(repo: str, diff_args: list[str]) -> set[tuple[str, str]]:
    """The +/- lines of a diff, as (polarity, normalized content) pairs.

    Polarity is kept: a line the fix *adds* matching a line the backport
    *removes* is not evidence that the two are the same work.

    Structural noise is dropped — hunk headers, file headers, and payloads too
    short to carry meaning (``}``, ``else:``, blanks), which match between any
    two diffs of the same language.
    """
    result = subprocess.run(
        ["git", "diff", "--unified=0", *diff_args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return set()
    lines = set()
    for raw in result.stdout.splitlines():
        if raw[:3] in ("+++", "---") or not raw or raw[0] not in "+-":
            continue
        payload = " ".join(raw[1:].split())  # collapse whitespace
        if len(payload) >= _RELATEDNESS_MIN_LINE_LEN:
            lines.add((raw[0], payload))
    return lines


def _shares_content_with_fix(config: BugFixConfig, state: PipelineState) -> bool:
    """Does the backport change any of the same lines the upstream fix changed?

    Used only to recover the "unmatched" verdict that forcing range-diff's
    pairing destroys. File-set overlap is not enough: a backport that touches
    the right file but changes something else entirely in it — a reworded
    comment beside the untouched bug — shares a filename while sharing no
    content, and would pass a filename test.

    Comparison is at line level, and requires a *proportion* of the fix's
    changed lines to reappear — not merely one. Any two diffs in the same
    language share boilerplate: measured over 918 pairs of commits in this
    repository with disjoint file sets, requiring a single shared line called
    204 of them related, on lines like ``return []`` and ``try:``. Their
    overlap ratio peaked at 0.23 with a 99th percentile of 0.03, so the
    threshold sits an order of magnitude above the noise floor and far below
    what a faithful port yields — this pipeline's discipline is that lines the
    fix changes carry the fix's exact bytes, so a real port overlaps heavily.

    Errs towards True: when either side cannot be determined it declines to
    override, so it can only ever add an unmatched verdict on strong evidence
    (nothing in common at all), never remove one. A port landing entirely in
    renamed files or rewritten line-by-line reads as unrelated here — the safe
    direction, since it asks a human to look.
    """
    try:
        fix_commit = config.source_fix_commit
        if is_merge_commit(config.repo_path, fix_commit):
            fix_args = [f"{fix_commit}^1..{fix_commit}"]
        else:
            fix_args = [f"{fix_commit}^..{fix_commit}"]
        fix_lines = _changed_payload_lines(config.repo_path, fix_args)
        if not fix_lines:
            return True

        target_base = git_merge_base(
            config.repo_path, config.frozen_target_sha or config.target_branch, "HEAD"
        )
        if not target_base:
            return True
        backport_lines = _changed_payload_lines(
            config.repo_path, [f"{target_base}..HEAD"]
        )
        if not backport_lines:
            return False

        shared = fix_lines & backport_lines
        overlap = len(shared) / len(fix_lines)
        if overlap < _RELATEDNESS_MIN_OVERLAP:
            log.info(
                "Relatedness: only %d/%d of the fix's changed lines appear "
                "in the backport (%.0f%%, below %.0f%%)",
                len(shared),
                len(fix_lines),
                overlap * 100,
                _RELATEDNESS_MIN_OVERLAP * 100,
            )
            return False
        return True
    except Exception as e:
        log.warning("Relatedness check failed: %s", e)
        return True


def phase_4_5_semantic_equivalence(
    config: BugFixConfig,
    state: PipelineState,
) -> str:
    """Run range-diff to classify the port as identical, modified, or unmatched.

    Two things have to be true at once, and range-diff's pairing threshold
    cannot deliver both on its own:

    - A legitimate backport carrying pipeline commits (test port, resolution,
      4a/4b recovery) must not be called "unmatched" just for containing more
      than the fix did. The squash handles the commit count; the threshold
      still rejects the pairing once the extra content is large enough.
    - A backport with nothing to do with the fix must not be called "modified".

    Measured on both cases, every creation-factor flips them together — there
    is no value that separates them. So the pairing is forced, which settles
    the first, and the "unmatched" signal is recovered independently from
    whether the backport touches any of the files the fix touched.
    """
    repo = config.repo_path
    try:
        backport_range = _squashed_backport_range(config, state)
        # See above: forcing is only sound because _shares_content_with_fix
        # supplies the unmatched signal that forcing destroys.
        creation_factor = 999 if backport_range else None
        if backport_range is None:
            # Range from the frozen baseline, not the branch name — the
            # agent may have moved the branch (see run_pipeline).
            target_base = git_merge_base(
                repo, config.frozen_target_sha or config.target_branch, "HEAD"
            )
            backport_range = f"{target_base}..HEAD"
        rd_output = run_range_diff(
            repo,
            f"{config.source_fix_commit}^..{config.source_fix_commit}",
            backport_range,
            creation_factor=creation_factor,
        )
    except Exception:
        return "unmatched"

    equivalence = parse_equivalence(rd_output)
    if (
        equivalence != "unmatched"
        and creation_factor is not None
        and not _shares_content_with_fix(config, state)
    ):
        log.warning(
            "Phase 4.5: range-diff paired the commits, but the backport "
            "changes none of the lines the fix changed — reporting unmatched"
        )
        return "unmatched"
    return equivalence


def _write_dossier_on_failure(state: PipelineState, tracker: Tracker) -> None:
    """Persist dossier.md even when the pipeline aborts.

    ``run_targets`` drops the dossier object on PipelineStop/Escalation and
    unexpected errors; without this write those runs leave no dossier.md
    even though Phases 0-4 recorded rich evidence. Best-effort: never
    raises.
    """
    try:
        if state.dossier is None:
            return
        os.makedirs(tracker._output_dir, exist_ok=True)
        path = os.path.join(tracker._output_dir, DOSSIER_FILE)
        with open(path, "w", encoding="utf-8") as f:
            f.write(format_dossier(state.dossier))
        log.info("Failure path: dossier written to %s", path)
    except Exception as e:
        log.warning("Failure-path dossier write failed: %s", e)


def _export_candidate_patch_bytes(config: BugFixConfig) -> bytes:
    """format-patch bytes for the Phase 5 deliverable; escalates if empty.

    Success paths must never ship a 0-byte patch: an empty range means no
    deliverable exists (observed 2026-10-15: agent moved the target branch,
    "target..HEAD" collapsed, a perfect backport exported as nothing). The
    net-diff check catches the degenerate sibling: commits exist in the
    range but change no file (message-only fixups), which format-patch
    renders as headers without a diff — still an empty deliverable.
    """
    patch_base = config.frozen_target_sha or config.target_branch
    patch_bytes = git_format_patch_bytes(config.repo_path, base_ref=patch_base)
    changed = _changed_files_since(config.repo_path, patch_base)
    if changed is None:
        raise PipelineEscalation(
            "Phase 5: could not determine which files changed between the "
            f"frozen baseline ({patch_base}) and HEAD (git diff failed) — "
            "refusing to report success without a verified deliverable"
        )
    if not patch_bytes.strip() or not changed:
        raise PipelineEscalation(
            "Phase 5: candidate patch is empty — no commits between the "
            f"frozen baseline ({patch_base}) and HEAD; refusing to report "
            "success without a deliverable"
        )
    return patch_bytes


def _is_test_path(path: str) -> bool:
    """Conservative heuristic: does a changed file look like a test file?

    Used only on the failure path to keep a ``PipelineStop`` (e.g. "fix
    already present", RED-triage "not vulnerable") from shipping a patch
    whose only content is the test files Phase 0.5c ported before the stop.
    Errs towards "not a test": a mixed patch always gets emitted.

    Path-segment matches (src/test/, spec/, __tests__/, testdata/) cover the
    Maven/Gradle/JVM layout where a *directory* marks the tests; basename
    patterns cover Python, Go, C, and JavaScript conventions (test_*.py,
    *_test.go, *_test.c, *.test.js). The JVM class convention is matched
    case-sensitively, at the CamelCase boundary (``[a-z0-9]Tests?\\.java``): a
    lowercased endswith would misclassify ordinary sources like Contest.java
    or Latest.java.
    """
    lowered = path.lower()
    parts = lowered.split("/")
    if any(
        seg in ("test", "tests", "spec", "__tests__", "testdata") for seg in parts[:-1]
    ):
        return True
    base = parts[-1]
    return (
        base.startswith("test_")
        or base.startswith("test.")
        or ".test." in base
        or ".spec." in base
        or base.endswith("_test.py")
        or base.endswith("_test.go")
        or base.endswith("_test.c")
        or re.search(r"[a-z0-9]Tests?\.java$", path) is not None
    )


def _changed_files_since(repo_path: str, base_ref: str) -> list[str] | None:
    """Files changed between base_ref and HEAD, or None when git fails.

    ``None`` (rather than ``[]``) on failure is deliberate: an unreadable
    diff is not an empty one. Phase 5 escalates on it with the real cause
    instead of claiming the patch range is empty, and the failure-path
    suppression stays silent rather than pretending the patch is test-only.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}..HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        log.warning(
            "git diff --name-only %s..HEAD failed (exit %d): %s",
            base_ref,
            result.returncode,
            result.stderr.strip(),
        )
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _commit_worktree_edits(config: BugFixConfig, phase_label: str) -> bool:
    """Best-effort commit of uncommitted agent work on an *aborting* path.

    Non-raising sibling of ``_commit_recovery_edits``: when the pipeline is
    already failing (agent-reported failure, missing result marker) it is
    still better to commit the session's work than to let it vanish from
    candidate.patch. On paths that can still succeed, the strict
    ``_commit_recovery_edits`` is used instead, so a failed commit is never
    silently ignored. Returns True if a commit was created; a failure is
    logged, never raised.

    Call this at the *abort boundary*, immediately after the agent session
    and before anything else runs. Committing later — at ``run_pipeline``'s
    exception handler, as an earlier revision of this code did — sweeps in
    whatever the orchestrator's own re-verification left in the tree
    (``centpkg prep`` re-unpacking a source tarball is the motivating case),
    which lands build corruption in the deliverable.

    Known limitation: the agent runs build and test commands during its own
    session, so its build output is still committed alongside its edits.
    Isolating that is tracked separately; it cannot be solved here without
    risking the fix itself.
    """
    for marker in (RUN_AND_FIX_FAILURE_FILE, RUN_AND_FIX_SUCCESS_FILE):
        path = os.path.join(config.repo_path, marker)
        if os.path.exists(path):
            os.remove(path)
    try:
        if not git_status_porcelain(config.repo_path):
            return False
        head_before = git_rev_parse(config.repo_path, "HEAD")
        subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
        git_commit(
            config.repo_path,
            f"fixup: uncommitted work for {config.issue_id} ({phase_label})",
            trailers=_COAUTHOR_TRAILERS,
        )
        committed = git_rev_parse(config.repo_path, "HEAD") != head_before
        if not committed:
            left = _non_artifact_dirt(config.repo_path)
            if left:
                log.warning(
                    "Best-effort work commit (%s) left work uncommitted:\n%s",
                    phase_label,
                    "\n".join(left),
                )
        return committed
    except Exception as e:
        log.warning("Best-effort work commit (%s) failed: %s", phase_label, e)
        return False


def _emit_candidate_patch_on_failure(
    config: BugFixConfig, tracker: Tracker, allow_test_only: bool = True
) -> None:
    """Best-effort candidate.patch export when the pipeline aborts.

    The benchmark verifier judges the exported patch on its own build+test
    gates, not the pipeline's internal verdict — and on escalation the agent
    may have produced a good backport that only Phase 4's re-verify rejected
    (observed on expat c8s, four identical runs: green tree, escalation, no
    patch, reward 0). Emit whatever exists between the frozen baseline and
    HEAD so the verifier can score the work on its merits.

    This function never commits anything: the phases commit agent work at
    their own pre-verification and abort boundaries (see
    ``_commit_worktree_edits``), so whatever is still uncommitted at abort
    time is verification-command side effects — an unpacked source tree, a
    re-patched spec — which must not be swept into the deliverable.

    ``allow_test_only=False`` suppresses the export when every changed file
    is a test file: a ``PipelineStop`` after Phase 0.5c's test port must not
    submit a test-only patch as if it were the fix. Suppression is reserved
    for ``PipelineStop`` because a stop asserts "there is nothing to port"
    — any patch from it can only be the ported tests. An escalation makes
    no such assertion; the escalated fix may be real and merely rejected by
    an over-strict re-verify, so escalations always emit and let the
    independent verifier judge. Never raises.
    """
    try:
        if not os.path.isdir(config.repo_path):
            return
        patch_base = config.frozen_target_sha or config.target_branch
        if not allow_test_only:
            changed = _changed_files_since(config.repo_path, patch_base)
            if changed is None:
                # Cannot prove the patch is test-only, and suppressing a real
                # fix is the worse error: emit and let the verifier judge.
                log.warning(
                    "Failure path: could not determine the changed files "
                    "(git diff failed) — emitting without test-only suppression"
                )
            elif changed and all(_is_test_path(f) for f in changed):
                log.info(
                    "Failure path: only test files changed since the frozen "
                    "baseline — not emitting a candidate patch"
                )
                return
        patch_bytes = git_format_patch_bytes(config.repo_path, base_ref=patch_base)
        if not patch_bytes.strip():
            return
        os.makedirs(tracker._output_dir, exist_ok=True)
        patch_path = os.path.join(tracker._output_dir, PATCH_FILE)
        with open(patch_path, "wb") as f:
            f.write(patch_bytes)
        log.info(
            "Failure path: candidate patch written to %s (%d bytes)",
            patch_path,
            len(patch_bytes),
        )
    except Exception as e:
        log.warning("Failure-path patch export failed: %s", e)


def _branch_checkout_holders(
    source_path: str, sym_ref: str, exclude: str
) -> list[str] | None:
    """Worktrees (other than ``exclude``) with ``sym_ref`` checked out.

    Returns None when the worktree list cannot be read — callers must treat
    that as "unknown" and fail closed, never as "no holders".
    """
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=source_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None
    exclude_real = os.path.realpath(exclude)
    paths: list[tuple[str | None, str | None]] = []
    current_path: str | None = None
    current_branch: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            paths.append((current_path, current_branch))
            current_path = line[len("worktree ") :]
            current_branch = None
        elif line.startswith("branch "):
            current_branch = line[len("branch ") :]
    paths.append((current_path, current_branch))
    return [
        path
        for path, branch in paths
        if path is not None
        and branch == sym_ref
        and os.path.realpath(path) != exclude_real
    ]


def _restore_target_baseline(
    source_path: str, worktree_path: str, branch: str, frozen_sha: str
) -> str | None:
    """Point the target branch back at the frozen baseline if the agent moved it.

    The task verifier builds its own worktree from the branch NAME; if the
    agent left its commits on the target branch, a pristine-baseline apply
    of candidate.patch would fail. The agent's work stays preserved in
    candidate.patch, on the ``amalgamator/`` branch for successful runs, and
    — when no deliverable exists — on the ``amalgamator-failed/`` branch
    created here, so restoring never discards it.

    Returns ``None`` when the branch is at the frozen baseline (or was
    restored), and a human-readable reason when it was left moved. Every
    skip case has the same consequence — the verifier may apply
    candidate.patch to a non-pristine tree and score 0 — so the caller
    records the reason in the run summary rather than leaving it in a log
    nobody reads on a benchmark run.

    This is a ``run_targets`` responsibility: it owns the worktree and the
    source checkout, and only it knows the frozen baseline the branch must
    return to. Callers of ``run_pipeline`` that manage their own checkouts
    must arrange their own cleanup. Never raises.
    """
    try:
        if not frozen_sha:
            return (
                "no frozen baseline was recorded — cannot tell whether the branch moved"
            )
        current = git_rev_parse(source_path, branch)
        if current == frozen_sha:
            return None
        # Only local branches (refs/heads/*) can be repointed. "branch" may
        # name a remote-tracking ref or a tag (origin/main, v1.2); moving a
        # remote-tracking ref corrupts the view of the remote, and
        # "git branch -f <tag> <sha>" creates a bogus same-named local
        # branch instead of touching the tag. Non-branch refs are left
        # alone: the verifier building from them is unaffected by the
        # agent's local commits either way.
        sym = subprocess.run(
            ["git", "rev-parse", "--symbolic-full-name", branch],
            cwd=source_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        if not sym.startswith("refs/heads/"):
            reason = (
                f"target ref resolves to {sym or 'nothing'}, not a local "
                "branch — only refs/heads/* are restored, so it is left at "
                f"{current[:12]}"
            )
            log.warning(
                "Target ref %s resolves to %s, not a local branch — leaving "
                "it at %s; only refs/heads/* are restored",
                branch,
                sym or "nothing",
                current[:12],
            )
            return reason
        # Never move a branch that another worktree holds checked out: the
        # ref is shared state, and repointing it under that checkout corrupts
        # the other worktree's view (its index and HEAD still reference the
        # old tip while the branch name resolves elsewhere). Our own
        # pipeline worktree is excluded — it is detached below. The check
        # fails closed: if the worktree list cannot be read, restoration is
        # skipped rather than guessing.
        holders = _branch_checkout_holders(source_path, sym, exclude=worktree_path)
        if holders is None:
            reason = (
                "cannot determine which worktrees hold the target branch "
                f"(git worktree list failed) — it is left at {current[:12]}"
            )
            log.warning(
                "Cannot determine which worktrees hold %s — leaving it at "
                "%s; restoration requires a readable worktree list",
                branch,
                current[:12],
            )
            return reason
        if holders:
            reason = (
                "target branch is checked out in another worktree "
                f"({', '.join(holders)}) — the shared ref is left at "
                f"{current[:12]}"
            )
            log.warning(
                "Target branch %s is checked out in %s — leaving it at %s; "
                "the ref is shared state and will not be moved under another "
                "worktree",
                branch,
                ", ".join(holders),
                current[:12],
            )
            return reason
        log.warning(
            "Target branch %s moved during the run (%s -> %s); restoring "
            "the frozen baseline so the verifier applies candidate.patch "
            "to a pristine tree",
            branch,
            frozen_sha[:12],
            current[:12],
        )
        # Preserve the moved tip first: if no candidate.patch was emitted,
        # this ref is the only surviving copy of the agent's commits.
        preserve_name = "amalgamator-failed/" + _sanitize_branch_slug(branch)
        preserved = subprocess.run(
            ["git", "branch", "-f", preserve_name, current],
            cwd=source_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if preserved.returncode == 0:
            log.info("Moved tip preserved on branch %s", preserve_name)
        else:
            log.warning(
                "Could not preserve the moved tip on %s: %s",
                preserve_name,
                (preserved.stderr or "").strip(),
            )
        # Detach the worktree HEAD first so the checkout is consistent with
        # the moved ref; the commits stay reachable from the worktree until
        # cleanup, and live on in candidate.patch regardless. If the detach
        # fails the branch is still checked out here, so moving it would
        # corrupt this worktree — skip the update entirely.
        detach = subprocess.run(
            ["git", "checkout", "--detach"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if detach.returncode != 0:
            reason = (
                "git checkout --detach failed in the pipeline worktree — the "
                f"branch is still checked out there and is left at {current[:12]}"
            )
            log.warning(
                "git checkout --detach failed (%s) — leaving %s at %s; the "
                "branch is still checked out in the pipeline worktree",
                (detach.stderr or "").strip(),
                branch,
                current[:12],
            )
            return reason
        # Compare-and-swap: if the branch moved again between the read above
        # and this update, git update-ref fails and the concurrent writer's
        # value wins — we never clobber commits we did not observe.
        result = subprocess.run(
            ["git", "update-ref", sym, frozen_sha, current],
            cwd=source_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            reason = (
                f"git update-ref (CAS) failed — {branch} appears to have "
                "moved concurrently and was left where it was"
            )
            log.warning(
                "git update-ref (CAS) failed — %s moved concurrently; "
                "leaving it alone: %s",
                sym,
                (result.stderr or "").strip(),
            )
            return reason
        return None
    except Exception as e:
        log.warning("Target baseline restore failed: %s", e)
        return f"restore failed: {e}"


def run_pipeline(
    config: BugFixConfig | None = None,
    claude_config: ClaudeConfig | None = None,
    output_dir: str = "runs",
) -> tuple[Dossier, Tracker]:
    """Orchestrate the full pipeline: triage, baseline, cherry-pick, resolve, verify, equivalence."""
    if config is None:
        raise ConfigError("run_pipeline() requires a config")
    if claude_config is None:
        raise ConfigError("run_pipeline() requires a claude_config")

    if not config.source_fix_commit:
        raise ConfigError(
            "source_fix_commit is empty. In patch mode, run ingest_patch() "
            "before calling run_pipeline."
        )

    # Freeze the target baseline before any agent phase runs. Phase 0.5
    # drives an LLM with shell access inside this worktree, and it may
    # attach HEAD to the shared target branch ref and commit there
    # (observed 2026-10-15: the django run whose format-patch
    # "target..HEAD" then exported an empty range). Everything that later
    # computes "what did the agent produce" must range from this SHA, not
    # from the movable branch name.
    # Always re-freeze, by design: this is runtime state of *this* run, and
    # a config object reused across runs must not keep ranging its patches
    # from the previous run's (possibly moved) baseline. A caller-provided
    # frozen_target_sha is therefore intentionally overwritten — passing a
    # baseline in buys nothing. run_targets constructs a fresh config per
    # target, so only direct run_pipeline callers reusing a config are
    # affected, and for them the overwrite is the safe behavior.
    try:
        config.frozen_target_sha = git_rev_parse(config.repo_path, config.target_branch)
    except Exception:
        try:
            config.frozen_target_sha = git_rev_parse(config.repo_path, "HEAD")
        except Exception:
            config.frozen_target_sha = ""
            log.warning(
                "Could not freeze the target baseline in %s; patch "
                "export falls back to the branch name",
                config.repo_path,
            )

    from common.model_registry import public_config_snapshot

    tracker = Tracker(
        issue_id=config.issue_id,
        config_dict=public_config_snapshot(config),
        model=claude_config.model,
        profile=claude_config.profile,
        output_dir=output_dir,
    )

    state = PipelineState()
    state.dossier = Dossier(
        issue_id=config.issue_id,
        source_branch=config.source_branch,
        target_branch=config.target_branch,
        fix_commit=config.source_fix_commit,
        bug_description=f"Porting fix for {config.issue_id}",
    )

    try:
        return _run_pipeline_phases(config, claude_config, state, tracker)
    except (PipelineStop, PipelineEscalation) as e:
        tracker.set_outcome(type(e).__name__.lower())
        _write_dossier_on_failure(state, tracker)
        # A PipelineStop after Phase 0.5c's test port must not submit a
        # test-only patch as if it were the fix; every other failure path
        # emits whatever work exists.
        _emit_candidate_patch_on_failure(
            config, tracker, allow_test_only=not isinstance(e, PipelineStop)
        )
        raise
    except Exception:
        tracker.set_outcome("error")
        _write_dossier_on_failure(state, tracker)
        _emit_candidate_patch_on_failure(config, tracker)
        raise
    finally:
        run_path = tracker.save()
        log.info("Run saved to %s", run_path)
        print(tracker.summary())


def _run_pipeline_phases(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker: Tracker,
) -> tuple[Dossier, Tracker]:
    """Execute all pipeline phases. Separated from run_pipeline for try/finally tracker saving."""
    if config.fix_patch:
        msg = _run_git(
            config.repo_path, ["log", "-1", "--format=%B", config.source_fix_commit]
        )
        trailer_match = [
            line
            for line in msg.stdout.splitlines()
            if line.startswith(f"{PATCH_TRAILER_KEY}:")
        ]
        ingest_lines = [
            f"Patch file: {config.fix_patch}",
            f"Source fix commit (synthetic): {config.source_fix_commit}",
        ]
        if trailer_match:
            ingest_lines.append(f"Trailer: {trailer_match[0]}")
        state.dossier.add("Patch ingest", "\n".join(ingest_lines), "Phase -1 — Ingest")

    with tracker.phase("Phase 0 — Triage"):
        triage_gates = phase_0_triage(config, claude_config, state)
        state.dossier.add("Triage result", str(triage_gates), "Phase 0")
        tracker.record_gate("forward_patch_id", triage_gates["forward_patch_id"])
        tracker.record_gate("ancestry", triage_gates["ancestry"])

        if config.evidence_checks:
            try:
                # With source == target (fix commit passed separately, e.g.
                # unrelated-history setups) the scan would walk the target's
                # entire history for upstream follow-ups — wrong corpus.
                followup_ref = config.fix_upstream_ref or config.source_fix_commit
                if git_rev_parse(
                    config.repo_path, config.source_branch
                ) != git_rev_parse(config.repo_path, config.target_branch):
                    state.followup_fixes = scan_followup_fixes(
                        config.repo_path, followup_ref, config.source_branch
                    )
                if state.followup_fixes:
                    state.followup_fixes_commit = git_rev_parse(
                        config.repo_path, config.source_fix_commit
                    )
            except Exception as e:
                log.warning("Follow-up fix scan failed: %s", e)
            if state.followup_fixes:
                log.info(
                    "Upstream has %d follow-up fixes to the ported commit",
                    len(state.followup_fixes),
                )
                state.dossier.add(
                    "Upstream follow-up fixes to the ported commit",
                    "\n".join(state.followup_fixes),
                    "Phase 0 — git log --grep 'Fixes:'",
                )
                tracker.record_gate("followup_fixes", str(len(state.followup_fixes)))

    with tracker.phase("Phase 0.5 — Semantic triage"):

        assessment = phase_0_5_semantic_triage(config, claude_config, state, tracker)
        state.dossier.add(
            "Semantic triage",
            json.dumps(assessment, indent=2) if assessment else "no assessment",
            "Phase 0.5",
        )
        tracker.record_gate(
            "difficulty", assessment.get("estimated_difficulty", "unknown")
        )
        tracker.record_gate(
            "recommendation", assessment.get("recommendation", "unknown")
        )
        triage_dest: str | None = None
        if assessment:
            triage_dest = os.path.join(tracker._output_dir, "semantic_triage.json")
            with open(triage_dest, "w") as f:
                json.dump(assessment, f, indent=2)

        if config.evidence_checks:
            prereqs = state.triage_assessment.get("prerequisites", [])
            if prereqs:
                try:
                    corrections = validate_triage_prerequisites(
                        config.repo_path, config.target_branch, prereqs
                    )
                except Exception as e:
                    log.warning("Triage prerequisite validation failed: %s", e)
                    corrections = []
                for corr in corrections:
                    log.warning(
                        "Triage correction: %s — %s", corr.symbol, corr.correction
                    )
                    for p in prereqs:
                        if p.get("symbol") == corr.symbol:
                            p["corrected"] = True
                            p["original_reason"] = corr.original_reason
                            p["reason"] = corr.correction
                    state.dossier.add(
                        f"Triage correction: {corr.symbol}",
                        f"Original: {corr.original_reason}\n"
                        f"Correction: {corr.correction}",
                        "Phase 0.5.1 — deterministic prerequisite validation",
                    )
                if corrections:
                    tracker.record_gate("triage_corrections", str(len(corrections)))
                    if assessment and triage_dest is not None:
                        with open(triage_dest, "w") as f:
                            json.dump(assessment, f, indent=2)

    cherry_pick_succeeded = False
    try:
        prereqs = state.triage_assessment.get("prerequisites", [])
        portable_prereqs = [
            p for p in prereqs if p.get("portable", False) and p.get("introduced_by")
        ]
        if portable_prereqs:
            with tracker.phase("Phase 0.5b — Prerequisite chain"):
                ported = port_prerequisite_chain(config, claude_config, state, tracker)
                state.dossier.add(
                    "Prerequisites ported",
                    json.dumps(ported, indent=2) if ported else "none",
                    "Phase 0.5b",
                )
                tracker.record_gate("prerequisites_ported", str(len(ported)))

        test_files_in_commit = identify_test_files(state.seed)
        if test_files_in_commit:
            with tracker.phase("Phase 0.5c — Test port"):
                ported_tests = phase_0_5c_test_port(
                    config, claude_config, state, tracker
                )
                state.dossier.add(
                    "Test port",
                    "\n".join(ported_tests) if ported_tests else "no tests ported",
                    "Phase 0.5c",
                )
                tracker.record_gate("test_files_ported", str(len(ported_tests)))
                if ported_tests:
                    subprocess.run(
                        ["git", "add", "-A"], cwd=config.repo_path, capture_output=True
                    )
                    git_commit(
                        config.repo_path,
                        f"Port test files for {config.issue_id} (Phase 0.5c)\n\nPorted before fix to enable RED/GREEN checking",
                    )

        with tracker.phase("Phase 1 — Baseline + RED"):
            phase_1_baseline_and_red(config, claude_config, state)

            baseline_summary = f"Build: {'PASSED' if state.baseline_build_exit == 0 else 'FAILED (exit ' + str(state.baseline_build_exit) + ')'}"
            baseline_summary += f"\nTest ({config.test_command}): {'PASSED' if not state.baseline else str(len(state.baseline)) + ' pre-existing failures'}"
            for entry in state.baseline_verification:
                status = (
                    "PASSED"
                    if entry["exit_code"] == 0
                    else f"FAILED (exit {entry['exit_code']}, pre-existing)"
                )
                baseline_summary += f"\n{entry['cmd']}: {status}"
            state.dossier.add(
                "Baseline (clean target branch)", baseline_summary, "Phase 1"
            )

        if state.red_passed_output:
            with tracker.phase("Phase 1.5 — RED triage"):
                triage_result = phase_1_5_red_triage(
                    config, claude_config, state, tracker
                )
                verdict = triage_result.get("verdict", "inconclusive").strip().lower()
                tracker.record_gate("red_triage_verdict", verdict)
                state.dossier.add(
                    "RED triage",
                    f"Verdict: {verdict}\n{triage_result.get('reasoning', 'no reasoning')}",
                    "Phase 1.5",
                )

                if verdict == "not_vulnerable":
                    raise PipelineStop(
                        f"RED triage: target branch is not vulnerable — {triage_result.get('reasoning', '')[:200]}"
                    )

                if verdict == "mitigated":
                    log.info(
                        "RED triage: root cause present but mitigated — porting fix for defense-in-depth"
                    )
                    state.red_passed_output = ""

                elif verdict == "vulnerable":
                    replacement = triage_result.get("replacement_test_file")
                    if replacement and os.path.exists(replacement):
                        log.info(
                            "RED triage produced replacement test — rerunning RED check"
                        )
                        subprocess.run(
                            ["git", "add", "-A"],
                            cwd=config.repo_path,
                            capture_output=True,
                        )
                        git_commit(
                            config.repo_path,
                            f"RED triage replacement test for {config.issue_id} (Phase 1.5)",
                        )
                        rerun_exit, rerun_output = capture_output(
                            [config.test_command],
                            config.build_dir,
                            timeout=config.command_timeout,
                        )
                        if rerun_exit != 0:
                            log.info(
                                "RED triage replacement test confirms vulnerability (exit %d) — continuing",
                                rerun_exit,
                            )
                            state.baseline = parse_test_failures(rerun_output, "")
                            state.red_passed_output = ""
                        else:
                            log.warning(
                                "RED triage replacement test also passes — escalating to Phase 4b"
                            )
                            raise PipelineEscalation(
                                "RED triage: replacement test also passes on unmodified code"
                            )
                    else:
                        log.warning(
                            "RED triage says vulnerable but no valid replacement test — escalating"
                        )
                        raise PipelineEscalation(
                            "RED triage: vulnerable verdict but no replacement test produced"
                        )
                else:
                    raise PipelineEscalation(
                        "RED triage inconclusive — escalating to Phase 4b"
                    )

        subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
        status = git_status_porcelain(config.repo_path)
        if status:
            git_commit(
                config.repo_path, f"Pre-cherry-pick cleanup for {config.issue_id}"
            )
        git_reset_hard(config.repo_path, "HEAD")

        with tracker.phase("Phase 2+3a — Cherry-pick and resolve"):
            files_to_skip = state.triage_assessment.get("files_to_skip", [])
            cp_result = cherry_pick_and_resolve(
                config,
                claude_config,
                state,
                tracker,
                commit=config.source_fix_commit,
                label=f"main commit ({config.issue_id})",
                files_to_skip=files_to_skip,
            )
            state.cherry_pick_path = cp_result
            state.dossier.cherry_pick_path = cp_result
            tracker.record_gate("cherry_pick", cp_result)

            regen_cmds = state.triage_assessment.get("regeneration_commands", [])
            if regen_cmds and files_to_skip:
                state.dossier.add(
                    "Regeneration commands needed",
                    "\n".join(regen_cmds),
                    "Phase 0.5 semantic triage",
                )

        regen_cmds = [
            c
            for c in state.triage_assessment.get("regeneration_commands", [])
            if _validate_shell_command(c)
        ]
        files_skipped = state.triage_assessment.get("files_to_skip", [])
        if regen_cmds and files_skipped:
            with tracker.phase("Phase 3b — Regeneration"):
                phase_3b_regeneration(config, state, regen_cmds)

        cherry_pick_succeeded = True

    except (PipelineEscalation, RuntimeError, OSError, subprocess.SubprocessError) as e:
        log.warning(
            "Cherry-pick resolution failed (%s: %s) — force-resolving and escalating to Phase 4b",
            type(e).__name__,
            str(e)[:200],
        )
        state.dossier.add(
            "Cherry-pick resolution failed", f"{type(e).__name__}: {e}", "Phase 2+3a"
        )

        status = git_status_porcelain(config.repo_path)
        remaining_conflicts = _conflicted_files(status)
        if remaining_conflicts:
            log.info(
                "Force-resolving %d remaining conflicts with --theirs before Phase 4b",
                len(remaining_conflicts),
            )
            for f in remaining_conflicts:
                subprocess.run(
                    ["git", "checkout", "--theirs", "--", f],
                    cwd=config.repo_path,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "add", "--", f], cwd=config.repo_path, capture_output=True
                )
            subprocess.run(
                ["git", "add", "-A"], cwd=config.repo_path, capture_output=True
            )
            git_commit(
                config.repo_path,
                f"Port {config.issue_id} (partial — force-resolved for Phase 4b)\n\n"
                f"[ {len(remaining_conflicts)} conflicts force-resolved with --theirs ]",
                trailers=_COAUTHOR_TRAILERS,
            )
        else:
            cherry_pick_in_progress = (
                subprocess.run(
                    ["git", "rev-parse", "--verify", "CHERRY_PICK_HEAD"],
                    cwd=config.repo_path,
                    capture_output=True,
                ).returncode
                == 0
            )
            if cherry_pick_in_progress:
                subprocess.run(
                    ["git", "cherry-pick", "--abort"],
                    cwd=config.repo_path,
                    capture_output=True,
                )
            # Reset to the frozen baseline, not the (movable) branch name.
            git_reset_hard(
                config.repo_path, config.frozen_target_sha or config.target_branch
            )
            git_cherry_pick(config.repo_path, config.source_fix_commit)
            status = git_status_porcelain(config.repo_path)
            force_conflicts = _conflicted_files(status)
            if force_conflicts:
                log.info(
                    "Re-applied cherry-pick, force-resolving %d conflicts with --theirs",
                    len(force_conflicts),
                )
                for f in force_conflicts:
                    subprocess.run(
                        ["git", "checkout", "--theirs", "--", f],
                        cwd=config.repo_path,
                        capture_output=True,
                    )
                    subprocess.run(
                        ["git", "add", "--", f],
                        cwd=config.repo_path,
                        capture_output=True,
                    )
            subprocess.run(
                ["git", "add", "-A"], cwd=config.repo_path, capture_output=True
            )
            git_commit(
                config.repo_path,
                f"Port {config.issue_id} (force-resolved for Phase 4b)\n\n"
                f"[ All conflicts force-resolved with --theirs ]",
                trailers=_COAUTHOR_TRAILERS,
            )

        with tracker.phase("Phase 4b — Run-and-fix loop (build-test-fix)"):
            phase_4b_run_and_fix(config, claude_config, state, tracker)
        cherry_pick_succeeded = True

    if not cherry_pick_succeeded:
        return state.dossier, tracker

    fix_diff = _compute_fix_diff(config)

    try:
        with tracker.phase("Phase 4 — Verify"):
            phase_4_verify(config, state)
            tracker.record_gate("verify", "passed")
    except PipelineEscalation:
        try:
            with tracker.phase("Phase 4a — Build error recovery"):
                phase_4a_build_recovery(config, claude_config, state, tracker, fix_diff)
                state.dossier.add(
                    "Build recovery", "Agent-assisted fix applied", "Phase 4a"
                )

            with tracker.phase("Phase 4 — Verify (post-recovery)"):
                phase_4_verify(config, state)
                tracker.record_gate("verify", "passed")
        except PipelineEscalation:
            with tracker.phase("Phase 4b — Run-and-fix loop (build-test-fix)"):
                phase_4b_run_and_fix(config, claude_config, state, tracker)

    phase4_lint = _resolve_lint_command(config, state)
    lint_cmds = [
        c
        for c in state.triage_assessment.get("lint_commands", [])
        if _validate_shell_command(c) and c != phase4_lint
    ]
    verification_cmds = list(lint_cmds)
    if config.auto_discover_tests:
        for c in state.triage_assessment.get("verification_commands", []):
            if _validate_shell_command(c):
                verification_cmds.append(c)

    baseline_failures = {
        e["cmd"]: e["exit_code"]
        for e in state.baseline_verification
        if e["exit_code"] != 0
    }

    if verification_cmds:
        ext_failures = []
        with tracker.phase("Phase 4.1 — Extended verification"):
            for vcmd in verification_cmds:
                log.info("Phase 4.1: running verification command: %s", vcmd)
                try:
                    vresult = subprocess.run(
                        vcmd,
                        shell=True,
                        cwd=config.build_dir,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=config.command_timeout,
                    )
                except subprocess.TimeoutExpired:
                    # A timeout is never treated as a pre-existing failure — we
                    # cannot conclude anything from a command that didn't finish.
                    log.warning(
                        "Phase 4.1: verification timed out — treating as failure: %s",
                        vcmd,
                    )
                    ext_failures.append(
                        {
                            "cmd": vcmd,
                            "output": f"Timed out after {config.command_timeout}s",
                        }
                    )
                    state.dossier.add(
                        f"Extended verification timed out: {vcmd}",
                        f"Timed out after {config.command_timeout}s",
                        "Phase 4.1",
                    )
                    continue
                if vresult.returncode == 0:
                    log.info("Phase 4.1: verification passed: %s", vcmd)
                elif vcmd in baseline_failures:
                    log.info(
                        "Phase 4.1: verification failed (pre-existing, exit %d): %s",
                        vresult.returncode,
                        vcmd,
                    )
                else:
                    output = (vresult.stdout + "\n" + vresult.stderr)[-3000:]
                    log.warning(
                        "Phase 4.1: verification failed (NEW, exit %d): %s",
                        vresult.returncode,
                        vcmd,
                    )
                    ext_failures.append({"cmd": vcmd, "output": output})
                    state.dossier.add(
                        f"Extended verification failed: {vcmd}",
                        output,
                        "Phase 4.1",
                    )

        if ext_failures:
            log.info(
                "Phase 4.1: %d verification commands failed — triggering Phase 4a recovery",
                len(ext_failures),
            )
            # Bound before the try: Phase 4a can raise before the post-recovery
            # sweep runs, and the handler below reads this.
            still_failing: list[str] = []
            try:
                with tracker.phase("Phase 4a — Build error recovery (from 4.1)"):
                    phase_4a_build_recovery(
                        config, claude_config, state, tracker, fix_diff
                    )
                    state.dossier.add(
                        "Build recovery (from extended verification)",
                        "Agent-assisted fix applied",
                        "Phase 4a",
                    )

                with tracker.phase("Phase 4.1 — Extended verification (post-recovery)"):
                    still_failing = []
                    for vcmd in verification_cmds:
                        try:
                            vresult = subprocess.run(
                                vcmd,
                                shell=True,
                                cwd=config.build_dir,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=config.command_timeout,
                            )
                        except subprocess.TimeoutExpired:
                            still_failing.append(vcmd)
                            continue
                        if vresult.returncode != 0 and vcmd not in baseline_failures:
                            still_failing.append(vcmd)
                    if still_failing:
                        log.warning(
                            "Phase 4.1: %d commands still failing after recovery: %s",
                            len(still_failing),
                            ", ".join(still_failing),
                        )
                        state.dossier.add(
                            "Extended verification still failing after Phase 4a",
                            "\n".join(still_failing),
                            "Phase 4.1 (post-recovery)",
                        )
                        raise PipelineEscalation(
                            f"Extended verification still failing: {', '.join(still_failing)}"
                        )
                    else:
                        log.info(
                            "Phase 4.1: all verification commands pass after recovery"
                        )
            except PipelineEscalation as e:
                log.warning(
                    "Phase 4a recovery failed — escalating to run-and-fix loop: %s", e
                )

                # Hand 4b the commands that got us here. Its own success
                # criteria cover only build and test, so without these it can
                # report success while the failing command is untouched.
                # If 4a raised before the post-recovery sweep, fall back to the
                # first-sweep failures — not every non-baseline command, which
                # would also name checks that passed.
                escalated_cmds = still_failing or [f["cmd"] for f in ext_failures]
                with tracker.phase("Phase 4b — Run-and-fix loop (build-test-fix)"):
                    phase_4b_run_and_fix(
                        config,
                        claude_config,
                        state,
                        tracker,
                        extra_verification=escalated_cmds,
                    )

    with tracker.phase("Phase 4.5 — Semantic equivalence"):
        eq_result = phase_4_5_semantic_equivalence(config, state)
        state.dossier.add("Semantic equivalence", eq_result, "Phase 4.5 range-diff")
        tracker.record_gate("semantic_equivalence", eq_result)

        try:
            squashed = _squashed_backport_range(config, state)
            # Display only, and deliberately not how eq_result was computed:
            # forcing the pairing renders a side-by-side even when the two
            # commits are too dissimilar to pair, which is exactly the case a
            # reviewer most needs to see. The verdict above uses git's default
            # threshold so that "unmatched" stays reachable.
            rd_output = run_range_diff(
                config.repo_path,
                f"{config.source_fix_commit}^..{config.source_fix_commit}",
                squashed or f"{config.frozen_target_sha or config.target_branch}..HEAD",
                creation_factor=999 if squashed else None,
            )
            if rd_output.strip():
                state.dossier.add(
                    "Range-diff (upstream vs backport)",
                    rd_output[:10000]
                    + ("\n[TRUNCATED]" if len(rd_output) > 10000 else ""),
                    f"Phase 4.5 — for reviewer: shows how the backport differs from the "
                    f"original fix. Commit pairing is forced for display, so this renders "
                    f"a diff even when the verdict is 'unmatched' (verdict: {eq_result}).",
                )
        except Exception:
            pass

    state.dossier.trailers = build_trailers(
        phase=state.cherry_pick_path,
        model=claude_config.model,
    )

    log.info("Phase 5 — Dossier ready for human review")
    if config.frozen_target_sha:
        state.dossier.add(
            "Frozen baseline",
            f"{config.target_branch} @ {config.frozen_target_sha}",
            "recorded at pipeline start, before Phase 0.5 agent shell access",
        )
    # Export against the frozen baseline, not the branch name: if the agent
    # moved the target branch mid-run, "target..HEAD" is an empty range and
    # a perfect backport would ship as a 0-byte patch (observed 2026-10-15).
    patch_bytes = _export_candidate_patch_bytes(config)
    # Written as bytes: the patch has to apply, and a repo carrying non-UTF-8
    # source would otherwise be handed a patch full of replacement characters.
    # The dossier copy is for humans, so a lossy decode is fine there.
    state.dossier.add(
        "Candidate patch", patch_bytes.decode("utf-8", "replace"), "git format-patch"
    )
    os.makedirs(tracker._output_dir, exist_ok=True)
    patch_path = os.path.join(tracker._output_dir, PATCH_FILE)
    with open(patch_path, "wb") as f:
        f.write(patch_bytes)
    log.info("Patch written to %s", patch_path)
    state.dossier.add(
        "Allowlist conformance",
        "\n".join(state.allowed_modules),
        "git diff --name-only",
    )

    with tracker.phase("Phase 5 — Completeness check"):
        outcome, completeness_detail = _backport_completeness(config, state)
        tracker.record_gate(
            "completeness", "partial" if outcome != "success" else "complete"
        )
        if outcome != "success":
            log.warning("Backport is partial: %s", completeness_detail)
            state.dossier.add(
                "Partial backport", completeness_detail, "completeness check"
            )
    tracker.set_outcome(outcome)
    return state.dossier, tracker


@dataclass
class TargetResult:
    target_branch: str
    issue_id: str
    outcome: str
    duration_s: float
    cost_usd: float
    dossier_path: str
    tracker_path: str
    error: str = ""


def _read_tracker_cost(target_run_dir: str, issue_id: str) -> tuple[float, str]:
    """Read cost and path from the tracker JSON written by run_pipeline's finally block."""
    import glob

    pattern = os.path.join(target_run_dir, f"{issue_id}_*.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return 0.0, ""
    tracker_path = matches[0]
    try:
        with open(tracker_path) as f:
            data = json.load(f)
        return data.get("total_cost_usd", 0.0), tracker_path
    except (json.JSONDecodeError, OSError):
        return 0.0, tracker_path


def _write_summary(
    results: list[TargetResult],
    parent_run_dir: str,
    warnings: list[str] | None = None,
) -> None:
    """Write summary.json aggregating all target outcomes."""
    succeeded = sum(1 for r in results if r.outcome == "success")
    summary: dict = {
        "total_targets": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "total_duration_s": round(sum(r.duration_s for r in results), 1),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "targets": [asdict(r) for r in results],
    }
    if warnings:
        summary["warnings"] = warnings
    path = os.path.join(parent_run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", path)


def run_targets(
    targets: list[TargetSpec],
    base_config: BugFixConfig,
    claude_config: ClaudeConfig,
    source_path: str,
    workdir: str,
    parent_run_dir: str,
    build_subdir: str,
    run_warnings: list[str] | None = None,
) -> list[TargetResult]:
    """Execute the pipeline serially for each target branch.

    A failure on one target does not prevent subsequent targets from running.

    Callers must ensure target branch uniqueness after sanitization — enforced
    by ``parse_targets`` upstream, not defensively here.
    """
    os.makedirs(workdir, exist_ok=True)
    run_warnings = [] if run_warnings is None else run_warnings
    results: list[TargetResult] = []

    for i, spec in enumerate(targets, 1):
        # --- Per-target config ---
        target_config = replace(
            base_config,
            target_branch=spec.branch,
            # or: empty issue_id/bug_description is meaningless, fall through to base
            issue_id=spec.issue_id or base_config.issue_id,
            bug_description=spec.bug_description or base_config.bug_description,
            # is not None: build_command="" means "skip build", not "inherit"
            build_command=(
                spec.build_command
                if spec.build_command is not None
                else base_config.build_command
            ),
            test_command=(
                spec.test_command
                if spec.test_command is not None
                else base_config.test_command
            ),
            lint_command=(
                spec.lint_command
                if spec.lint_command is not None
                else base_config.lint_command
            ),
        )
        sanitized = _sanitize_ref(spec.branch)
        target_run_dir = os.path.join(parent_run_dir, sanitized)
        os.makedirs(target_run_dir, exist_ok=True)

        log.info(
            "=== Target %d/%d: %s (issue_id=%s) ===",
            i,
            len(targets),
            spec.branch,
            target_config.issue_id,
        )

        target_start = time.time()
        worktree_created = False
        worktree_path = os.path.join(workdir, f"{target_config.issue_id}_{sanitized}")

        try:
            target_log_file = open(  # noqa: SIM115 (tee target, closed in finally)
                os.path.join(target_run_dir, "run_log.txt"), "w", encoding="utf-8"
            )
        except OSError as e:
            log.warning("Target %s: cannot open run log: %s", spec.branch, e)
            results.append(
                TargetResult(
                    target_branch=spec.branch,
                    issue_id=target_config.issue_id,
                    outcome="error",
                    duration_s=round(time.time() - target_start, 1),
                    cost_usd=0.0,
                    dossier_path="",
                    tracker_path="",
                    error=f"cannot open run log: {e}",
                )
            )
            continue

        saved_stdout = sys.stdout
        sys.stdout = Tee(saved_stdout, target_log_file)

        outcome = "error"
        error_msg = ""
        dossier_path = ""

        try:
            # --- Worktree setup ---
            print(f"Creating worktree at {worktree_path} on {spec.branch}...")
            try:
                create_fresh_worktree(source_path, worktree_path, spec.branch)
            except RuntimeError as e:
                print(f"ERROR: {e}")
                log.warning("Target %s: worktree creation failed: %s", spec.branch, e)
                results.append(
                    TargetResult(
                        target_branch=spec.branch,
                        issue_id=target_config.issue_id,
                        outcome="worktree_error",
                        duration_s=round(time.time() - target_start, 1),
                        cost_usd=0.0,
                        dossier_path="",
                        tracker_path="",
                        error=str(e),
                    )
                )
                continue
            worktree_created = True
            target_config.repo_path = worktree_path
            target_config.build_dir = os.path.join(worktree_path, build_subdir)
            print("Worktree created successfully")

            # --- Pipeline ---
            try:
                dossier, run_tracker = run_pipeline(
                    target_config, claude_config, output_dir=target_run_dir
                )
                # The tracker may have downgraded a clean finish (e.g. to
                # partial_success for an incomplete non-bugfix port).
                outcome = (
                    run_tracker.outcome
                    if run_tracker.outcome != "in_progress"
                    else "success"
                )
            except PipelineStop as e:
                dossier = None
                outcome = "pipeline_stop"
                error_msg = str(e)
                print(f"PIPELINE STOP: {e}")
            except PipelineEscalation as e:
                dossier = None
                outcome = "pipeline_escalation"
                error_msg = str(e)
                print(f"PIPELINE ESCALATION: {e}")
            except Exception as e:
                dossier = None
                outcome = "error"
                error_msg = f"{type(e).__name__}: {e}"
                print(f"PIPELINE ERROR: {error_msg}")

            # --- Dossier write (separate from outcome) ---
            if dossier is not None:
                try:
                    dossier_text = format_dossier(dossier)
                    dp = os.path.join(target_run_dir, DOSSIER_FILE)
                    with open(dp, "w", encoding="utf-8") as f:
                        f.write(dossier_text)
                    dossier_path = os.path.join(sanitized, DOSSIER_FILE)
                    print(f"Dossier written to {dp}")
                except Exception as e:
                    log.warning("Failed to write dossier: %s", e)

            # --- Cost retrieval (unified, both paths) ---
            cost_usd, tracker_json = _read_tracker_cost(
                target_run_dir, target_config.issue_id
            )
            tracker_rel = (
                os.path.join(sanitized, os.path.basename(tracker_json))
                if tracker_json
                else ""
            )

            if not dossier_path:
                dp = os.path.join(target_run_dir, DOSSIER_FILE)
                if os.path.exists(dp):
                    dossier_path = os.path.join(sanitized, DOSSIER_FILE)

            duration = round(time.time() - target_start, 1)
            results.append(
                TargetResult(
                    target_branch=spec.branch,
                    issue_id=target_config.issue_id,
                    outcome=outcome,
                    duration_s=duration,
                    cost_usd=cost_usd,
                    dossier_path=dossier_path,
                    tracker_path=tracker_rel,
                    error=error_msg,
                )
            )
            print(
                f"FINISHED target {spec.branch}: outcome={outcome}, duration={duration}s, cost=${cost_usd:.4f}"
            )

            # Preserve partial backports too — an incomplete port is exactly
            # what a human needs to inspect and finish.
            if outcome in ("success", "partial_success") and worktree_created:
                try:
                    head_sha = git_rev_parse(worktree_path, "HEAD")
                    slug = _sanitize_branch_slug(
                        target_config.issue_id
                        or f"{target_config.source_fix_commit[:12]}-to-{sanitized}"
                    )
                    branch_name = f"amalgamator/{slug}"
                    if len(targets) > 1:
                        branch_name += f"/{sanitized}"
                    result = subprocess.run(
                        ["git", "branch", "-f", branch_name, head_sha],
                        cwd=source_path,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if result.returncode == 0:
                        log.info("Commits preserved on branch %s", branch_name)
                    else:
                        log.warning("git branch failed: %s", result.stderr.strip())
                except Exception as e:
                    log.warning("Failed to create branch: %s", e)

        finally:
            sys.stdout = saved_stdout
            target_log_file.close()
            if worktree_created:
                restored = _restore_target_baseline(
                    source_path,
                    worktree_path,
                    spec.branch,
                    target_config.frozen_target_sha,
                )
                if restored:
                    run_warnings.append(
                        f"{spec.branch}: target branch not confirmed at the "
                        f"frozen baseline — {restored}"
                    )
                cleanup_worktree(source_path, worktree_path)

    _write_summary(results, parent_run_dir, warnings=run_warnings)

    log.info("=== Multi-target summary ===")
    for r in results:
        status = "OK" if r.outcome == "success" else f"FAILED ({r.outcome})"
        log.info(
            "  %s [%s]: %s  (%.1fs, $%.4f)",
            r.issue_id,
            r.target_branch,
            status,
            r.duration_s,
            r.cost_usd,
        )

    return results


def main(argv=None) -> int:
    import argparse

    from common.model_registry import RegistryError

    parser = argparse.ArgumentParser(
        description="Port a bug fix using a pipeline configuration."
    )
    parser.add_argument("config", help="Pipeline YAML/JSON configuration")
    parser.add_argument(
        "--models", metavar="PATH", help="Explicit model registry (overrides discovery)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        targets, base_config, claude_config, workdir, source_path = (
            load_multi_target_config(
                args.config,
                models_path=args.models,
            )
        )
    except (ConfigError, RegistryError, OSError) as e:
        parser.error(str(e))

    yaml_basename = os.path.splitext(os.path.basename(args.config))[0]
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    parent_run_dir = os.path.join("runs", f"{yaml_basename}_{run_timestamp}")
    os.makedirs(parent_run_dir, exist_ok=True)

    log_path = os.path.join(parent_run_dir, "run_log.txt")
    log_file = open(  # noqa: SIM115 (tee target, closed at end of main)
        log_path, "w", encoding="utf-8"
    )
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print(f"Run directory: {parent_run_dir}")
        print("Fetching latest refs from origin...")
        fetch_result = subprocess.run(
            ["git", "fetch", "--all", "--quiet"],
            cwd=source_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        fetch_warning = ""
        if fetch_result.returncode != 0:
            fetch_warning = (
                f"git fetch --all failed (exit {fetch_result.returncode}): "
                f"{(fetch_result.stderr or '').strip()[:500]}"
            )
            log.warning(fetch_warning)
            print(f"WARNING: {fetch_warning} — continuing with local refs")
        build_subdir = os.path.relpath(base_config.build_dir, base_config.repo_path)
        run_warnings = [fetch_warning] if fetch_warning else []

        if base_config.fix_patch:
            print(f"Phase -1: ingesting patch {base_config.fix_patch}...")
            ingest_result = ingest_patch(
                source_path,
                base_config.fix_patch,
                base_config.source_branch,
                base_config.issue_id,
                base_config.bug_description,
                workdir,
            )
            base_config.source_fix_commit = ingest_result.synthetic_commit
            print(
                f"Patch ingested as {ingest_result.synthetic_commit[:12]} "
                f"(method={ingest_result.method}, base={ingest_result.base_sha[:12]})"
            )
            ingest_path = os.path.join(parent_run_dir, "ingest.json")
            with open(ingest_path, "w") as f:
                json.dump(
                    {
                        "synthetic_commit": ingest_result.synthetic_commit,
                        "base_ref": ingest_result.base_ref,
                        "base_sha": ingest_result.base_sha,
                        "method": ingest_result.method,
                        "patch_sha256": ingest_result.patch_sha256,
                        "patch_path": base_config.fix_patch,
                        "warnings": ingest_result.warnings,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    f,
                    indent=2,
                )
            run_warnings.extend(ingest_result.warnings)

        results = run_targets(
            targets,
            base_config,
            claude_config,
            source_path,
            workdir,
            parent_run_dir,
            build_subdir,
            run_warnings=run_warnings,
        )
    except IngestError as e:
        print(f"INGEST ERROR: {e}")
        return 1
    finally:
        sys.stdout = original_stdout
        log_file.close()

    return 0 if all(r.outcome == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
