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
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
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
        tool_read_cap=config.tool_read_cap,
        tool_bash_cap=config.tool_bash_cap,
        thinking=thinking,
        effort=effort or claude_config.effort,
        profile=effective_profile,
        xml_tool_fallback=claude_config.xml_tool_fallback,
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
    FusedPreRedPrompt,
    FusedResolutionPrompt,
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
FUSED_RESOLUTION_REPORT_FILE = "fused_resolution_report.md"
FUSED_PRE_RED_REPORT_FILE = "fused_pre_red_report.md"
FUSED_PRE_RED_RESULT_FILE = "pre_red_result.json"
_PIPELINE_ARTIFACTS = (
    SEMANTIC_TRIAGE_FILE,
    FUSED_PRE_RED_RESULT_FILE,
    FUSED_PRE_RED_REPORT_FILE,
    FUSED_RESOLUTION_REPORT_FILE,
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
    result = _run_git(repo_path, ["commit", "--amend", "--no-edit"])
    if not result.success:
        log.warning(
            "git commit --amend failed: %s", (result.stderr or result.stdout)[:300]
        )


def _git_lines(repo_path: str, *args: str) -> list[str]:
    """Run a git command and return its non-empty stdout lines."""
    return [
        line for line in _run_git(repo_path, list(args)).stdout.splitlines() if line
    ]


class PipelineEscalation(Exception):
    pass


class PipelineStop(Exception):
    pass


_CONFLICT_PREFIXES = ("UU ", "AA ", "DU ", "UD ", "DD ", "AU ", "UA ")

_MARKER_START_RE = re.compile(r"^<{7}( |\r?$)", re.MULTILINE)
_MARKER_END_RE = re.compile(r"^>{7}( |\r?$)", re.MULTILINE)

_DANGEROUS_SHELL_PATTERNS = (";", "&&", "||", "|", "`", "$(", "${", ">", "<", "\n")
# An ``export NAME=...`` segment may carry shell expansions in its value (paths
# like ``$(dirname "$PWD")/.gocache`` are common in project verification
# commands). Only these are rejected inside such a segment; the command that
# actually runs is still held to the full list above.
_EXPORT_SEGMENT_RE = re.compile(r"^\s*export\s+[A-Za-z_][A-Za-z0-9_]*=")
_EXPORT_FORBIDDEN = ("`", "\n", ">", "<", "|", "&&", "||", "&")
# Command substitution in an export value is executed by the pipeline (Phase
# 4.1), outside the session, so only path-helper shapes are accepted.
_EXPORT_SUBST_RE = re.compile(r"\$\(([^()]*)\)")
_EXPORT_SUBST_ALLOWED_RE = re.compile(
    r"^(?:pwd|mktemp\s+-d(?:\s+\S+)*|dirname\s+\S.*)$"
)


def _export_substitution_ok(segment: str) -> bool:
    """True when every ``$(...)`` in an export segment is an allowed shape."""
    if "`" in segment:
        return False
    remainder = _EXPORT_SUBST_RE.sub("", segment)
    if "$(" in remainder:
        return False
    for inner in _EXPORT_SUBST_RE.findall(segment):
        # A `;` or `&` inside any substitution would run a second command, even
        # when the outer segment's quoted text hides it from the splitter.
        if any(ch in inner for ch in ";&"):
            return False
        if not _EXPORT_SUBST_ALLOWED_RE.match(inner.strip()):
            return False
    return True


_BUGFIX_TYPES = ("bugfix", "bug-fix", "bug_fix", "fix", "security")


def _split_shell_segments(cmd: str) -> list[str]:
    """Split on top-level semicolons, ignoring quoted and escaped text."""
    segments: list[str] = []
    current: list[str] = []
    single = double = False
    escaped = False
    for ch in cmd:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\" and not single:
            current.append(ch)
            escaped = True
            continue
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
        elif ch == ";" and not single and not double:
            segments.append("".join(current))
            current = []
            continue
        current.append(ch)
    segments.append("".join(current))
    return segments


def _outside_quotes(text: str) -> str:
    """Drop quoted spans so literal `|`/`>` in arguments are not read as syntax."""
    out: list[str] = []
    single = double = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and not single:
            escaped = True
            continue
        if ch == "'" and not double:
            single = not single
            continue
        if ch == '"' and not single:
            double = not double
            continue
        if not single and not double:
            out.append(ch)
    return "".join(out)


def _strip_shell_comment(cmd: str) -> str:
    """Drop a trailing unquoted shell comment (``# annotation``)."""
    single = double = False
    escaped = False
    previous = " "
    for index, ch in enumerate(cmd):
        if escaped:
            escaped = False
            previous = ch
            continue
        if ch == "\\" and not single:
            escaped = True
            previous = ch
            continue
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
        elif ch == "#" and not single and not double and previous.isspace():
            return cmd[:index].rstrip()
        previous = ch
    return cmd


def _validate_shell_command(cmd: str) -> bool:
    """Accept one shell command; a leading ``export NAME=...`` prefix is allowed.

    Project commands carry their own environment (PATH, caches), so export
    values may use ``$VAR``/``${VAR}`` and path-helper substitutions
    (``$(pwd)``, ``$(dirname ...)``, ``$(mktemp -d)``); the final command must
    be free of every dangerous pattern. Other substitutions in an export run
    arbitrary code in the pipeline's name, so they are rejected.
    """
    if not isinstance(cmd, str):
        return False
    # A comment strip must not hide a second line: the caller runs the original
    # string. A single trailing newline is harmless; an inner one is a second
    # command.
    outside = _outside_quotes(cmd).rstrip("\r\n")
    if "\n" in outside or "\r" in outside:
        return False
    cmd = _strip_shell_comment(cmd).rstrip("\r\n")
    if not cmd.strip():
        return False
    segments = _split_shell_segments(cmd)
    *exports, final = segments
    for segment in exports:
        if not _EXPORT_SEGMENT_RE.match(segment):
            return False
        if any(p in segment for p in _EXPORT_FORBIDDEN):
            return False
        if not _export_substitution_ok(segment):
            return False
    if not final.strip():
        return False
    # Quoted spans are argument text, not shell syntax — but command
    # substitution inside them still executes.
    if any(p in final for p in ("`", "$(", "${")):
        return False
    return not any(p in _outside_quotes(final) for p in _DANGEROUS_SHELL_PATTERNS)


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
    # Set when resolution failed and the tree was force-resolved with
    # --theirs. Such a tree may only be reported as success when build/test
    # commands independently verified it; otherwise the outcome is
    # "unverified".
    force_resolved: bool = False
    force_resolved_reason: str = ""
    # True only when `git checkout --theirs` actually ran. The no-conflict
    # re-apply path can leave a clean tree without using --theirs, and the
    # diagnostic must not claim otherwise.
    force_resolved_with_theirs: bool = False
    # Full refname -> SHA for branches an agent session must never move.
    protected_refs: dict[str, str] = field(default_factory=dict)
    # Set when the single-session resolver completed the resolution, internal
    # review, and regeneration; Phase 3b is then skipped.
    fused_resolution_ran: bool = False


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


def _compact_fix_overview(config: BugFixConfig, commit: str) -> str:
    """Stat + file list + fetch instructions instead of the full fix diff.

    Fused sessions used to resend the complete diff (hundreds of KB) on every
    tool iteration. The agent can fetch exact per-file bytes with `git show`,
    so the prompt carries only the overview.
    """
    rev = f"{commit}^1..{commit}"
    stat = "\n".join(_git_lines(config.repo_path, "diff", "--stat", rev)).strip()
    file_list = "\n".join(
        f"  - {path}"
        for path in _git_lines(config.repo_path, "diff", "--name-only", rev)
    )
    return (
        f"{stat}\n\n"
        "The full diff is omitted to save context. For exact bytes, run:\n"
        f"  git show {commit} -- <file>\n"
        "or, for the whole merge change:\n"
        f"  git diff {commit}^1 {commit} -- <file>\n\n"
        f"Changed files:\n{file_list}"
    )


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


def phase_0_5c_test_port(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> list[str]:
    """Port the test files the triage and seed discovery identified."""
    test_files = _planned_test_files(state)
    if not test_files:
        log.info("Phase 0.5c: no test files in the fix — skipping test port")
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


def _fix_diff_for_prompt(config: BugFixConfig, commit: str) -> str:
    """The fix diff a fused prompt carries (an overview when compact)."""
    if config.fused_compact_prompts:
        return _compact_fix_overview(config, commit)
    return _compute_fix_diff(config, commit)


def _run_fused_session(
    config: BugFixConfig, claude_config: ClaudeConfig, prompt: str, tracker
) -> None:
    """Run one fused agent session with the phase's model configuration."""
    asyncio.run(
        claude_run(_phase_config(claude_config, config), [prompt], tracker=tracker)
    )


def _fused_artifact_dir(tracker) -> str:
    """Directory for fused-session reports and result files (outside the repo)."""
    output_dir = getattr(tracker, "_output_dir", "") if tracker else ""
    if not output_dir:
        return tempfile.mkdtemp(prefix="fused-")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _read_fused_result(path: str) -> dict:
    """Read a fused session's JSON result; an unusable file yields ``{}``."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Fused session: cannot read %s: %s", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def _strings(value: object) -> list[str]:
    """The string entries of a JSON value (a non-list yields ``[]``)."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _dicts(value: object) -> list[dict]:
    """The object entries of a JSON value (a non-list yields ``[]``)."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# Deterministic test-file floor. Triage may add paths these patterns miss
# (golden files, fixtures); it may never remove this guarantee.
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


def _planned_test_files(state: PipelineState) -> list[str]:
    """Test files to port: triage's judgement unioned with the seed floor.

    Both inputs are filtered to files the fix actually touches — triage
    naming a path outside the fix must not widen the port — and triage's
    ``files_to_skip`` still wins.
    """
    seed = list(state.seed)
    seed_set = set(seed)
    files_to_skip = set(_strings(state.triage_assessment.get("files_to_skip")))
    triage_tests = _strings(state.triage_assessment.get("test_files"))
    unexpected = [f for f in triage_tests if f not in seed_set]
    if unexpected:
        log.warning(
            "Triage listed test files outside the fix: %s — ignoring",
            ", ".join(unexpected[:5]),
        )
    merged = dict.fromkeys(triage_tests + identify_test_files(seed))
    return [f for f in merged if f in seed_set and f not in files_to_skip]


def _read_report(path: str, limit: int = 20_000) -> str:
    """A session report, truncated for the dossier; empty when unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        log.warning("Cannot read session report %s: %s", path, e)
        return ""
    return text[:limit] + ("\n[TRUNCATED]" if len(text) > limit else "")


def _discard_fused_session(repo: str, base_head: str) -> None:
    """Drop a session's commits and edits so the caller can fall back."""
    git_reset_hard(repo, base_head)
    _run_git(repo, ["clean", "-fd"])


def _protected_refs(repo: str, refs: tuple[str, ...]) -> dict[str, str]:
    """Snapshot the full refname and SHA of branches sessions must not move."""
    protected: dict[str, str] = {}
    for ref in refs:
        full = _run_git(repo, ["rev-parse", "--symbolic-full-name", ref])
        sha = _run_git(repo, ["rev-parse", ref])
        if full.success and sha.success and full.stdout.strip():
            protected[full.stdout.strip()] = sha.stdout.strip()
    return protected


def _restore_protected_refs(config: BugFixConfig, state: PipelineState) -> None:
    """Undo a session that checked out and committed on a protected branch.

    Sessions work on the worktree's detached HEAD. Committing while attached
    to the source or target branch moves that branch, which makes the
    candidate patch (format-patch against the target) empty.
    """
    repo = config.repo_path
    moved = []
    for ref, sha in state.protected_refs.items():
        current = _run_git(repo, ["rev-parse", ref])
        if current.success and current.stdout.strip() != sha:
            moved.append((ref, sha))
    if not moved:
        return
    # Detach HEAD in place: keeps the worktree and the session's commit.
    head = _run_git(repo, ["rev-parse", "HEAD"])
    if head.success and head.stdout.strip():
        _run_git(repo, ["update-ref", "--no-deref", "HEAD", head.stdout.strip()])
    for ref, sha in moved:
        log.warning("Session moved %s — restoring %s", ref, sha[:12])
        _run_git(repo, ["update-ref", ref, sha])


_PRE_RED_COMMIT_PREFIXES = ("prereq:", "test-port:", "fused-pre-red:")


def _session_commit_files(repo: str, base_head: str, prefix: str) -> list[str]:
    """Files changed by the session's commits whose subject starts with *prefix*."""
    files: list[str] = []
    for line in _git_lines(repo, "log", "--format=%H%x09%s", f"{base_head}..HEAD"):
        sha, _, subject = line.partition("\t")
        if not subject.startswith(prefix):
            continue
        for path in git_diff_name_only_range(repo, f"{sha}^", sha):
            if path not in files:
                files.append(path)
    return files


def _fix_content_applied(repo: str, commit: str) -> bool:
    """True when HEAD already carries the fix's production bytes.

    Ancestry alone misses a cherry-picked fix (a copy, not an ancestor) that
    the session re-committed under an allowed prefix; that state makes RED
    unable to reproduce the bug. Test-only paths are excluded because the test
    port legitimately matches the fix there.
    """
    changed = [fix_path for _status, _base, fix_path in _changed_paths(repo, commit)]
    production = [p for p in changed if p not in set(identify_test_files(changed))]
    if not production:
        return False
    head_oids = _blob_oids(repo, [f"HEAD:{p}" for p in production])
    fix_oids = _blob_oids(repo, [f"{commit}:{p}" for p in production])
    return head_oids == fix_oids


def _validate_fused_pre_red(config: BugFixConfig, base_head: str) -> list[str]:
    """Objections that make a pre-RED session unusable (empty means accept).

    The prompt states these rules; checking them keeps a session that ignored
    them from poisoning the RED check or the later resolution.
    """
    repo = config.repo_path
    commit = config.source_fix_commit
    problems: list[str] = []
    if git_is_ancestor(repo, commit, "HEAD"):
        problems.append("the main fix is already applied")
    elif _fix_content_applied(repo, commit):
        problems.append("the fix's production changes are already applied")
    subjects = _git_lines(repo, "log", "--format=%s", f"{base_head}..HEAD")
    stray = [s for s in subjects if not s.startswith(_PRE_RED_COMMIT_PREFIXES)]
    if stray:
        problems.append("commits outside prereq:/test-port/: " + "; ".join(stray[:5]))
    # The committing SHA is the ground truth, not the result file's claim.
    # Only a commit reachable from the fix but NOT from its first parent is
    # carried by `git diff <fix>^1 <fix>`; commits already on the first
    # parent are mainline and can be genuine external prerequisites.
    for subject in subjects:
        parts = subject.split()
        if (
            len(parts) >= 2
            and parts[0] == "prereq:"
            and len(parts[1]) >= 7
            and all(c in "0123456789abcdef" for c in parts[1])
            and git_is_ancestor(repo, parts[1], commit)
            and not git_is_ancestor(repo, parts[1], f"{commit}^1")
        ):
            problems.append(f"ported fix ancestor {parts[1][:12]} ({subject[:60]})")
    return problems


@dataclass(frozen=True)
class _FixPath:
    """One path the fix touches, with the target branch's state for it."""

    status: str  # git name-status letter: A, M, D, R, C, ...
    state: str  # "absent", "same" or "diverged" on the target
    path: str  # the fix's path (for a rename, the new path)
    moves: tuple[str, ...] = ()  # same-basename files on the target


def _blob_oids(repo: str, names: list[str]) -> list[str | None]:
    """Object ids for ``<ref>:<path>`` names, ``None`` where the path is missing.

    One ``git cat-file --batch-check`` process answers every name; a missing
    object prints ``<name> missing`` instead of an object id.
    """
    if not names:
        return []
    out = _run_git(
        repo, ["cat-file", "--batch-check"], input="\n".join(names) + "\n"
    ).stdout
    oids: list[str | None] = []
    for line in out.splitlines():
        words = line.split()
        oids.append(None if not words or words[-1] == "missing" else words[0])
    return oids


def _changed_paths(repo: str, commit: str) -> list[tuple[str, str, str]]:
    """``(status, base path, fix path)`` for every path ``commit`` changes.

    ``-z`` separates fields with NUL, so spaces and quotes in paths are safe;
    a rename or copy reports the old and new path, every other status repeats
    the same path. ``"M\0a.go\0R100\0old.go\0new.go\0"`` becomes
    ``[("M", "a.go", "a.go"), ("R", "old.go", "new.go")]``.
    """
    out = _run_git(
        repo, ["diff", "--name-status", "-z", "-M", f"{commit}^1", commit]
    ).stdout
    fields = iter(out.split("\0"))
    paths: list[tuple[str, str, str]] = []
    while (status := next(fields, None)) is not None:
        if not status:
            continue
        base = next(fields, None)
        if base is None:
            break
        fix_path = next(fields, None) if status[0] in "RC" else base
        if fix_path is None:
            break
        paths.append((status[0], base, fix_path))
    return paths


def _path_states(repo: str, target_ref: str, commit: str) -> list[_FixPath]:
    """Classify every path the fix touches against the target branch.

    Compares the target's blob with the fix's base blob (``commit^1``, or the
    added blob for a path the fix creates) and records same-basename files on
    the target as move hints for paths the target lacks.
    """
    changed = _changed_paths(repo, commit)
    target_oids = _blob_oids(repo, [f"{target_ref}:{path}" for _, _, path in changed])
    base_oids = _blob_oids(
        repo,
        [
            f"{commit}:{path}" if status == "A" else f"{commit}^1:{base}"
            for status, base, path in changed
        ],
    )
    by_basename: dict[str, list[str]] = {}
    for path in _git_lines(repo, "ls-tree", "-r", "--name-only", target_ref):
        by_basename.setdefault(os.path.basename(path), []).append(path)

    states: list[_FixPath] = []
    for (status, _base, path), target_oid, base_oid in zip(
        changed, target_oids, base_oids, strict=False
    ):
        if target_oid is None:
            state = "absent"
        elif target_oid == base_oid:
            state = "same"
        else:
            state = "diverged"
        moves: tuple[str, ...] = ()
        if state == "absent" and status != "A":
            others = [
                candidate
                for candidate in by_basename.get(os.path.basename(path), [])
                if candidate != path
            ]
            moves = tuple(others[:3])
        states.append(_FixPath(status, state, path, moves))
    return states


def _dependency_root(path: str) -> str:
    """Prefix under which dependency-tree paths are collapsed in the facts.

    ``vendor/github.com/x/y/z.go`` -> ``vendor/github.com/x/``;
    ``internal/third_party/selinux/a.go`` -> ``internal/third_party/``;
    an ordinary source path -> ``""`` (listed individually).
    """
    if path.startswith("vendor/"):
        depth = 3
    elif path.startswith(("staging/", "third_party/", "internal/third_party/")):
        depth = 2
    else:
        return ""
    return "/".join(path.split("/")[:depth]) + "/"


def _render_facts_line(item: _FixPath) -> str:
    """One ``status state path`` facts line, with the move hint when there is one."""
    hint = ""
    if item.moves:
        hint = f"  [same name on target: {', '.join(item.moves)}]"
    return f"    {item.status} {item.state:8} {item.path}{hint}"


def _render_facts_section(lines: list[str], title: str, items: list[_FixPath]) -> None:
    """Append one facts section, collapsing dependency trees into a count."""
    lines.append(f"  {title}")
    if not items:
        lines.append("    (none)")
        return
    grouped: dict[tuple[str, str, str], list[_FixPath]] = {}
    for item in items:
        root = _dependency_root(item.path)
        if root:
            grouped.setdefault((item.status, item.state, root), []).append(item)
        else:
            lines.append(_render_facts_line(item))
    # Sorted so the block is stable regardless of the diff's path order.
    for (_status, _state, root), group in sorted(grouped.items()):
        if len(group) == 1:
            lines.append(_render_facts_line(group[0]))
        else:
            first = group[0]
            lines.append(
                f"    {first.status} {first.state:8} {root} ({len(group)} paths)"
            )


def _pre_red_facts(repo: str, target_ref: str, commit: str) -> tuple[str, bool]:
    """Render the deterministic facts block for the fused pre-RED prompt.

    The pre-RED agent must decide what to port, what to adapt and what the
    target already provides. This function answers the mechanical part from
    git alone: the commits the fix carries (already inside its diff, never
    port them), the patch size, and for every path the fix touches what the
    target has there — ``absent`` (no such path), ``same`` (bytes equal the
    fix's base, so its hunks apply as written) or ``diverged``.

    An absent path that existed in the base is either a missing package (port
    it as a prerequisite) or a file the target keeps under another name. When
    the target has a file with the same basename, it is printed as a
    ``[same name on target: ...]`` hint; the agent compares the API before
    treating it as a move.

    Example::

        `1a2b3c4d5e6f` is a merge; 2 commit(s) are carried inside the patch (reachable from the fix, not from `1a2b3c4d5e6f^1`). Never port them:
        Patch: 3 files changed, 42 insertions(+), 7 deletions(-).

        Target state per path (base = `1a2b3c4d5e6f^1`):
          supplied by the patch (the target lacks them; do not port):
            A absent   pkg/guard.go
          present in the base but missing on the target (same-name hints list possible moves; compare the API):
            M absent   lib/old/guard.go  [same name on target: lib/guard.go]
          identical on target and base (the fix's hunks apply cleanly; no inspection):
            M same     README.md
          diverged from the base (adaptation likely):
            M diverged pkg/server.go

    Returns the block and whether a pre-RED build is worth running: only a
    genuine missing package (absent, not added by the fix, no hint) can make
    the target stop compiling.
    """
    states = _path_states(repo, target_ref, commit)
    carried = _git_lines(repo, "rev-list", "--oneline", commit, f"^{commit}^1")
    shortstat = _git_lines(repo, "diff", "--shortstat", f"{commit}^1", commit)
    parents = _git_lines(repo, "rev-list", "--parents", "-n", "1", commit)
    kind = "merge" if parents and len(parents[0].split()) > 2 else "single commit"

    lines = [
        f"`{commit[:12]}` is a {kind}; {len(carried)} commit(s) are carried inside "
        f"the patch (reachable from the fix, not from `{commit}^1`). Never port them:",
    ]
    lines += [f"  - {entry}" for entry in carried[:40]]
    if len(carried) > 40:
        lines.append(f"  ... and {len(carried) - 40} more")
    lines.append(f"Patch: {shortstat[0].strip() if shortstat else 'no diff'}.")
    lines.append("")
    lines.append(f"Target state per path (base = `{commit}^1`):")
    _render_facts_section(
        lines,
        "supplied by the patch (the target lacks them; do not port):",
        [item for item in states if item.state == "absent" and item.status == "A"],
    )
    _render_facts_section(
        lines,
        "present in the base but missing on the target (same-name hints list "
        "possible moves; compare the API):",
        [item for item in states if item.state == "absent" and item.status != "A"],
    )
    _render_facts_section(
        lines,
        "identical on target and base (the fix's hunks apply cleanly; no inspection):",
        [item for item in states if item.state == "same"],
    )
    _render_facts_section(
        lines,
        "diverged from the base (adaptation likely):",
        [item for item in states if item.state == "diverged"],
    )
    needs_build = any(
        item.state == "absent" and item.status != "A" and not item.moves
        for item in states
    )
    return "\n".join(lines), needs_build


def _run_fused_pre_red(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> dict | None:
    """One session for triage, prerequisite discovery and the test port.

    Returns the triage assessment on success, or None so the caller runs the
    separate Phase 0.5/0.5b/0.5c functions.
    """
    repo = config.repo_path
    commit = config.source_fix_commit
    base_head = git_rev_parse(repo, "HEAD")
    output_dir = _fused_artifact_dir(tracker)
    result_path = os.path.join(output_dir, FUSED_PRE_RED_RESULT_FILE)
    report_path = os.path.join(output_dir, FUSED_PRE_RED_REPORT_FILE)

    facts, needs_build = _pre_red_facts(repo, config.target_branch, commit)
    prompt = FusedPreRedPrompt(
        context=create_context_str(claude_config, config),
        fix_diff=_fix_diff_for_prompt(config, commit),
        source_branch=config.source_branch,
        target_branch=config.target_branch,
        fix_commit=commit,
        base_head=base_head,
        build_command=config.build_command,
        test_command=config.test_command,
        seed_files=state.seed,
        result_path=result_path,
        report_path=report_path,
        facts=facts,
        skip_build=not needs_build,
    )
    if tracker:
        tracker.record_gate("pre_red_mode", "fused")
    log.info(
        "Phase 0.5+0.5b+0.5c — fused pre-RED session (%d seed files)", len(state.seed)
    )
    try:
        _run_fused_session(config, claude_config, prompt.prompt(), tracker)
    finally:
        _restore_protected_refs(config, state)

    result = _read_fused_result(result_path)
    triage = result.get("triage")
    if not isinstance(triage, dict) or not triage:
        # A session may write the result into the worktree despite the path.
        result = _read_fused_result(os.path.join(repo, FUSED_PRE_RED_RESULT_FILE))
        triage = result.get("triage")
    if not isinstance(triage, dict) or not triage:
        log.warning(
            "Fused pre-RED: no usable triage result at %s — falling back",
            result_path,
        )
        _discard_fused_session(repo, base_head)
        return None

    # Session artifacts must not be committed; leftover edits are the session's
    # own changes and must reach the RED check as a commit.
    for name in (
        FUSED_PRE_RED_RESULT_FILE,
        FUSED_PRE_RED_REPORT_FILE,
        SEMANTIC_TRIAGE_FILE,
    ):
        Path(repo, name).unlink(missing_ok=True)
    if git_status_porcelain(repo):
        _run_git(repo, ["add", "-A"])
        git_commit(
            repo,
            f"fused-pre-red: commit remaining pre-RED changes for {config.issue_id}",
        )
    # Validate after the leftovers commit: uncommitted production edits must
    # not slip past the "fix already applied" checks.
    problems = _validate_fused_pre_red(config, base_head)
    if problems:
        log.warning(
            "Fused pre-RED session rejected (%s) — falling back", "; ".join(problems)
        )
        _discard_fused_session(repo, base_head)
        return None

    # Attribute tests before the strip: squashing the session's commits would
    # otherwise lose the test-port subjects and silently fall back to the
    # session's self-reported list.
    tests_ported = _session_commit_files(repo, base_head, "test-port:") or _strings(
        result.get("tests_ported")
    )
    # The leftovers commit is `git add -A`, so anything force-added here would
    # reach the candidate patch; apply the same strip the resolver path uses.
    stripped = _strip_session_scratch(repo, base_head, keep=set(state.seed))
    if stripped is None:
        log.warning("Fused pre-RED: could not strip session scratch — falling back")
        _discard_fused_session(repo, base_head)
        return None
    if stripped:
        dropped = set(stripped)
        tests_ported = [f for f in tests_ported if f not in dropped]
        if state.dossier:
            state.dossier.add(
                "Build scratch stripped from fused pre-RED commit",
                "\n".join(stripped[:50]),
                "Phase 0.5+0.5b+0.5c fused session",
            )

    triage = dict(triage)
    triage["test_files"] = _strings(result.get("test_files"))
    triage["prerequisites"] = _dicts(result.get("prerequisites"))
    state.triage_assessment = triage
    missing_tests = [
        f for f in _planned_test_files(state) if f not in set(tests_ported)
    ]
    if missing_tests:
        log.warning(
            "Fused pre-RED: expected test files were not ported (%s) — falling back",
            ", ".join(missing_tests[:5]),
        )
        _discard_fused_session(repo, base_head)
        return None
    failed = [
        str(p.get("symbol", "?"))
        for p in triage["prerequisites"]
        if p.get("decision") == "failed"
    ]
    unexpected = [t for t in tests_ported if t not in state.seed]
    if unexpected:
        log.warning(
            "Fused pre-RED: ported tests outside the fix's file set: %s",
            ", ".join(unexpected[:5]),
        )

    state.ported_test_files = tests_ported
    state.triage_assessment = triage
    priority = triage.get("priority_files")
    state.priority_files = (
        priority if isinstance(priority, list) else state.allowed_modules
    )
    state.failed_prerequisites.extend(failed)

    if state.dossier:
        state.dossier.add(
            "Semantic triage (fused pre-RED)",
            json.dumps(triage, indent=2),
            "Phase 0.5+0.5b+0.5c fused session",
        )
        state.dossier.add(
            "Prerequisites (fused pre-RED)",
            json.dumps(triage["prerequisites"], indent=2),
            "Phase 0.5+0.5b+0.5c fused session",
        )
        state.dossier.add(
            "Test port (fused pre-RED)",
            "\n".join(tests_ported) if tests_ported else "no tests ported",
            "Phase 0.5+0.5b+0.5c fused session",
        )
        report = _read_report(report_path)
        state.dossier.add(
            "Fused pre-RED session report",
            report if report.strip() else f"No report produced at {report_path}",
            "Phase 0.5+0.5b+0.5c fused session",
        )

    if tracker:
        ported_count = sum(
            1 for p in triage["prerequisites"] if p.get("decision") == "ported"
        )
        tracker.record_gate("prerequisites_ported", str(ported_count))
        tracker.record_gate("test_files_ported", str(len(tests_ported)))
        if failed:
            tracker.record_gate("prerequisites_failed", str(len(failed)))
        build = result.get("build")
        if isinstance(build, dict):
            tracker.record_gate("pre_red_build_exit", str(build.get("exit_code", "?")))
    return triage


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


def _fused_check_commands(state: PipelineState) -> list[str]:
    """Lint and verification commands the fused session must run itself."""
    assessment = state.triage_assessment or {}
    commands: list[str] = []
    for key in ("lint_commands", "verification_commands"):
        value = assessment.get(key)
        # A triage that emits a bare string must yield one command, not its
        # characters.
        entries = [value] if isinstance(value, str) else _strings(value)
        for cmd in entries:
            if cmd.strip() and cmd not in commands:
                commands.append(cmd)
    return commands[:10]


# Formatting checks that fail only on whitespace, with their in-place fixer.
# Data only: add one line per ecosystem.
_FORMATTER_FIXES = (
    (r"\bgofmt\s+-l\b", "gofmt -w"),
    (r"\bgofumpt\s+-l\b", "gofumpt -w"),
    (r"\b(shfmt)\s+-d\b", r"\1 -w"),
    (r"\bblack\s+(--check|--diff)\b", "black"),
    (r"\bruff\s+format\s+--check\b", "ruff format"),
    (r"\b(isort)\s+--check-only\b", r"\1"),
    (r"\b(rustfmt)\s+--check\b", r"\1"),
    (r"\b(prettier)\s+(--check|-c)\b", r"\1 --write"),
    (r"\b(clang-format)\s+--dry-run\b", r"\1 -i"),
    (r"\bterraform\s+fmt\s+-check\b", "terraform fmt"),
)
# `hack/verify-<name>.sh` -> `hack/update-<name>.sh` is only safe when <name>
# is a formatter; verify-vendor.sh and friends update far more than whitespace.
_VERIFY_SCRIPT_RE = re.compile(r"((?:\./)?hack/)verify-([\w.-]+)\.sh\b")
_FORMATTER_SCRIPT_NAMES = frozenset(
    {
        "gofmt",
        "goimports",
        "gofumpt",
        "clang-format",
        "black",
        "isort",
        "rustfmt",
        "shfmt",
        "prettier",
        "terraform-fmt",
    }
)


def _formatter_repair_command(cmd: str) -> str | None:
    """In-place fixer for a known formatting check, if the command matches one."""
    match = _VERIFY_SCRIPT_RE.search(cmd)
    if match and match.group(2) in _FORMATTER_SCRIPT_NAMES:
        return (
            cmd[: match.start()]
            + f"{match.group(1)}update-{match.group(2)}.sh"
            + cmd[match.end() :]
        )
    for pattern, replacement in _FORMATTER_FIXES:
        repaired = re.sub(pattern, replacement, cmd)
        if repaired != cmd:
            return repaired
    return None


def _run_shell_ok(cmd: str, config: BugFixConfig) -> bool:
    try:
        exit_code, _, _ = run_test_suite(
            cmd, config.build_dir, timeout=config.command_timeout
        )
    except OSError as e:
        log.warning("Phase 4.1: formatter command failed to run: %s", e)
        return False
    return exit_code == 0


def _revert_formatter_collateral(
    config: BugFixConfig, dirty_before: set[str]
) -> list[str]:
    """Undo formatter edits to files this backport does not touch.

    Repo-wide fixers reformat pre-existing unformatted files; those lines must
    keep the target branch's bytes. Files the fix touches are kept.
    """
    repo = config.repo_path
    try:
        backport_files = set(
            _git_lines(repo, "diff", "--name-only", f"{config.target_branch}..HEAD")
        )
        dirty_now = set(git_diff_name_only(repo))
    except Exception as e:
        log.warning("Formatter repair scope check failed: %s", e)
        return []
    collateral = sorted(dirty_now - dirty_before - backport_files)
    for path in collateral:
        _run_git(repo, ["checkout", "--", path])
    return collateral


def _repair_known_formatter_failures(
    config: BugFixConfig,
    failures: list[dict],
    state: PipelineState,
) -> list[dict]:
    """Repair formatting-only verification failures without an agent session.

    A formatter check that fails because of whitespace should not cost a
    Phase 4b agent call: run the in-place fixer and re-check. Returns the
    failures that remain.
    """
    remaining: list[dict] = []
    repaired_any = False
    for failure in failures:
        cmd = str(failure.get("cmd", ""))
        repair = _formatter_repair_command(cmd)
        dirty_before: set[str] = set()
        if repair is not None:
            try:
                dirty_before = set(git_diff_name_only(config.repo_path))
            except Exception:
                dirty_before = set()
        if repair is None or not _run_shell_ok(repair, config):
            if repair is not None:
                log.warning("Phase 4.1: formatter repair failed: %s", repair)
            remaining.append(failure)
            continue
        collateral = _revert_formatter_collateral(config, dirty_before)
        if collateral:
            log.info(
                "Phase 4.1: reverted formatter edits to %d unrelated files: %s",
                len(collateral),
                ", ".join(collateral[:5]),
            )
            state.dossier.add(
                "Formatter repair scope",
                "Reverted formatter edits outside the backport: "
                + ", ".join(collateral[:20]),
                "Phase 4.1",
            )
        if not _run_shell_ok(cmd, config):
            log.warning("Phase 4.1: formatter repair did not satisfy: %s", cmd)
            remaining.append(failure)
            continue
        log.info("Phase 4.1: deterministic formatter repair fixed: %s", cmd)
        repaired_any = True
        state.dossier.add(
            "Deterministic formatter repair",
            f"Ran `{repair}`; `{cmd}` now passes.",
            "Phase 4.1",
        )
    if repaired_any:
        # Recovery edits must be commit-based or the candidate patch, preserved
        # branch, and range-diff would omit them.
        try:
            _commit_recovery_edits(config, "Phase 4.1 formatter repair")
        except PipelineEscalation as e:
            log.warning("Phase 4.1: cannot commit formatter repair: %s", e)
            return failures
    return remaining


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
        _finalize("failure")
        raise PipelineEscalation("Phase 4b: run-and-fix loop exhausted retries")
    elif os.path.exists(success_file):
        state.dossier.add(
            "Run-and-fix loop", "Build-test-fix loop succeeded", "Phase 4b"
        )
        log.info("Phase 4b: run-and-fix loop reported success — verifying")

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
        if _commit_recovery_edits(config, "Phase 4b"):
            state.dossier.add(
                "Run-and-fix loop",
                "Recovery edits committed (fixup)",
                "Phase 4b",
            )
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
        raise PipelineEscalation(
            "Phase 4b: run-and-fix loop ended without writing a result file"
        )


MAX_PREREQUISITE_DEPTH = 5


_SCRATCH_PREFIXES = (
    ".gocache/",
    ".gotmp/",
    ".tmp/",
    ".agent-tmp/",
    ".agent-home/",
    ".pytest_cache/",
    "__pycache__/",
)

# Files test runs commonly leave behind. They belong in a backport only when the
# fix itself adds them (the caller passes those paths as `keep`).
_TEST_OUTPUT_PATTERNS = (
    "*.diffs",
    "regression.out",
    "*.tap",
    "junit.xml",
    "coverage.out",
    ".coverage",
)
_TEST_OUTPUT_DIRS = ("htmlcov/", ".nyc_output/", "test-results/")


def _is_test_output(repo: str, ref: str, path: str) -> bool:
    """True for test-run output absent from *ref* (never part of a backport)."""
    name = os.path.basename(path)
    if not path.startswith(_TEST_OUTPUT_DIRS) and not any(
        fnmatch(name, pattern) for pattern in _TEST_OUTPUT_PATTERNS
    ):
        return False
    return not git_cat_file_exists(repo, ref, path)


def _strip_session_scratch(
    repo: str, base_head: str, keep: set[str] | None = None
) -> list[str] | None:
    """Drop force-added build scratch and test-run output from the commit.

    The worktree excludes prevent `git add -A` from staging the scratch paths,
    but an agent can still force-add them or commit test output. Squash the
    session's commits onto ``base_head`` and re-commit without those files, and
    delete them from the working tree so a later recovery ``git add -A`` cannot
    bring them back.

    Returns the dropped paths, or ``None`` when the strip could not complete —
    a half-stripped branch must not be reported as a success.
    """
    keep = keep or set()
    changed = git_diff_name_only_range(repo, base_head, "HEAD")

    def is_droppable(path: str) -> bool:
        if path.startswith(_SCRATCH_PREFIXES) or path in _PIPELINE_ARTIFACTS:
            return True
        if path in keep:
            return False
        return _is_test_output(repo, base_head, path)

    dropped = [f for f in changed if is_droppable(f)]
    if not dropped:
        return []

    log.warning(
        "Fused session committed %d scratch/test-output files — stripping them",
        len(dropped),
    )
    messages = _run_git(
        repo, ["log", "--reverse", "--format=%B", f"{base_head}..HEAD"]
    ).stdout.strip()
    reset = _run_git(repo, ["reset", "--soft", base_head])
    if not reset.success:
        log.warning("Could not squash fused commit: %s", reset.stderr[:300])
        return None
    removed = _run_git(
        repo, ["rm", "-r", "--cached", "--ignore-unmatch", "-q", *dropped]
    )
    if not removed.success:
        log.warning("Could not unstage stripped files: %s", removed.stderr[:300])
        return None
    commit = _run_git(
        repo,
        ["commit", "--no-verify", "-m", messages or "Fused backport"],
    )
    if not commit.success:
        log.warning("Could not re-commit without scratch: %s", commit.stderr[:300])
        return None
    for path in dropped:
        try:
            os.remove(os.path.join(repo, path))
        except OSError:
            continue
        parent = os.path.dirname(os.path.join(repo, path))
        while parent.startswith(repo + os.sep):
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)
    return dropped


def _add_fused_report_to_dossier(state: PipelineState, report_path: str) -> None:
    """Surface the fused session's own report in the pipeline dossier."""
    report = _read_report(report_path)
    if not report.strip():
        state.dossier.add(
            "Fused resolution session",
            f"No session report produced at {report_path}; Phase 4 remains the "
            "authority on build/test.",
            "Phase 2+3a fused session",
        )
        return
    state.dossier.add(
        "Fused resolution session report", report, "Phase 2+3a fused session"
    )


class _CherryPickStrategy(TypedDict):
    strategy: str | None
    strategy_option: str | None
    name: str


def _apply_cherry_pick(
    config: BugFixConfig,
    state: PipelineState,
    commit: str,
    label: str,
    files_to_skip: set[str],
) -> tuple[str, list[str], list[str]]:
    """Apply a cherry-pick with strategy fallback and revert skip-only files.

    Returns ``(outcome, conflicted_files, summary_parts)``. The outcome is
    ``"clean"`` (applied with no conflicts), ``"resolved"`` (conflicts were
    only in skip files, which were reverted and committed), or ``"conflicts"``
    (the tree is left mid-cherry-pick for resolution).
    """
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
            return "clean", [], []

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
    uu_files = _conflicted_files(git_status_porcelain(config.repo_path))
    if not uu_files:
        log.info("%s: clean cherry-pick on retry (strategy: %s)", label, strat["name"])
        return "clean", [], []

    conflict_summary_parts = []
    if files_to_skip:
        # Reset ALL skip files to target version — both conflicting and cleanly-applied
        all_changed = _git_lines(config.repo_path, "diff", "--name-only", "HEAD")
        skip_applied_clean = [
            f for f in all_changed if f in files_to_skip and f not in uu_files
        ]
        if skip_applied_clean:
            log.info(
                "%s: reverting %d skip files that applied cleanly",
                label,
                len(skip_applied_clean),
            )
            for f in skip_applied_clean:
                _run_git(config.repo_path, ["checkout", "HEAD", "--", f])

        auto_resolved = [f for f in uu_files if f in files_to_skip]
        if auto_resolved:
            log.info(
                "%s: reverting %d conflicting skip files to target version",
                label,
                len(auto_resolved),
            )
            for f in auto_resolved:
                _run_git(config.repo_path, ["checkout", "HEAD", "--", f])
                _run_git(config.repo_path, ["add", "--", f])

        total_skipped = len(skip_applied_clean) + len(auto_resolved)
        if total_skipped:
            conflict_summary_parts.append(
                f"Reverted {total_skipped} files to target version (files_to_skip)"
            )
            uu_files = [f for f in uu_files if f not in files_to_skip]

    if not uu_files:
        msg = (
            f"Port {label}\n\n[ Conflict summary: {'; '.join(conflict_summary_parts)} ]"
        )
        git_commit(config.repo_path, msg)
        return "resolved", [], conflict_summary_parts
    return "conflicts", uu_files, conflict_summary_parts


def _resolve_conflicts_mechanically(
    config: BugFixConfig,
    state: PipelineState,
    tracker,
    commit: str,
    label: str,
    uu_files: list[str],
    conflict_summary_parts: list[str],
) -> list[str]:
    """Re-apply source bytes deterministically before any LLM sees the conflicts.

    Runs before the fused session so a conflict the mechanical rungs can settle
    never reaches the model, and again after a fused fallback's fresh
    cherry-pick.
    """
    if not config.mechanical_resolution or not uu_files:
        return uu_files
    original = list(uu_files)
    try:
        mech = resolve_mechanically(config.repo_path, commit, uu_files)
    except Exception as e:
        log.warning(
            "%s: mechanical resolution failed (continuing to LLM): %s", label, e
        )
        mech = {}
    if not mech:
        return uu_files
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
    remaining = [f for f in uu_files if f not in mech]
    if tracker:
        tracker.record_gate("mechanical_resolution", f"{len(mech)}/{len(original)}")
    return remaining


def cherry_pick_and_resolve(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
    commit: str,
    label: str,
    files_to_skip: list[str] | None = None,
) -> str:
    """Cherry-pick a single commit trying multiple strategies.

    Returns 'clean' or 'resolved', or raises when resolution is impossible.
    """
    skip_set = set(files_to_skip or [])
    outcome, uu_files, conflict_summary_parts = _apply_cherry_pick(
        config, state, commit, label, skip_set
    )
    if outcome != "conflicts":
        return outcome
    base_head = git_rev_parse(config.repo_path, "HEAD")
    uu_files = _resolve_conflicts_mechanically(
        config, state, tracker, commit, label, uu_files, conflict_summary_parts
    )

    if uu_files and config.fused_resolution:
        fused_result = _run_fused_resolution(
            config,
            claude_config,
            state,
            tracker,
            commit=commit,
            label=label,
            base_head=base_head,
            conflicted_files=uu_files,
            files_to_skip=sorted(skip_set),
            conflict_summary_parts=conflict_summary_parts,
        )
        if fused_result:
            return fused_result
        log.warning(
            "%s: fused resolution did not finish — retrying with the per-chunk "
            "resolver",
            label,
        )
        if tracker:
            tracker.record_gate("resolution_mode", "fused_fallback")
        # Reset and clean: a failed session may have written untracked files
        # that would block the fresh cherry-pick.
        _discard_fused_session(config.repo_path, base_head)
        outcome, uu_files, conflict_summary_parts = _apply_cherry_pick(
            config, state, commit, label, skip_set
        )
        if outcome != "conflicts":
            return outcome
        uu_files = _resolve_conflicts_mechanically(
            config, state, tracker, commit, label, uu_files, conflict_summary_parts
        )

    original_uu_files = list(uu_files)
    fix_diff = _compute_fix_diff(config, commit)
    context = create_context_str(claude_config, config)
    priority = state.priority_files if state.priority_files else state.allowed_modules

    claude_config_resolve = _phase_config(claude_config, config)

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


def _committed_conflict_markers(repo: str, base_head: str) -> list[str]:
    """Files changed since *base_head* that carry both conflict markers."""
    marked: list[str] = []
    for path in git_diff_name_only_range(repo, base_head, "HEAD"):
        abs_path = os.path.join(repo, path)
        if not os.path.isfile(abs_path) or os.path.getsize(abs_path) > 5_000_000:
            continue
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if _MARKER_START_RE.search(text) and _MARKER_END_RE.search(text):
            marked.append(path)
    return marked


_FUSED_RESOLUTION_SCOPE = (
    "One resolution session produced this backport; it self-reviewed. The "
    "per-chunk resolver, substitution/rename/snap-back post-passes, "
    "token/line-provenance/dead-config audits, the evidence-fed verification "
    "agent and Phase 3b's rollback point did not run. Phase 4 build/test, 4.1 "
    "checks and 4.5 range-diff still apply. The triage-blind review (3a.2) is "
    "run separately when `blind_review` is enabled."
)


def _fused_resolution_unmatched(
    config: BugFixConfig, state: PipelineState, commit: str, base_head: str
) -> bool:
    """True when the fused result cannot be tied to the fix's changes.

    Range-diff pairing is forced (the pipeline's commits would otherwise read
    as unmatched), so the relatedness check supplies the verdict: a session
    that resolved nothing relevant must go through the per-chunk resolver.
    """
    repo = config.repo_path
    fix_range = (
        f"{commit}^1..{commit}"
        if is_merge_commit(repo, commit)
        else f"{commit}^..{commit}"
    )
    # Squash for the comparison: a session may leave several commits, and
    # range-diff would report each extra one as unmatched.
    cached_squash = state.squashed_backport_range
    try:
        backport_range = _squashed_backport_range(config, state)
        if backport_range is None:
            backport_range = f"{base_head}..HEAD"
        rd_output = run_range_diff(repo, fix_range, backport_range, creation_factor=999)
    except Exception as e:
        log.warning("Fused semantic check failed, keeping the result: %s", e)
        return False
    finally:
        # Phase 4.5 must squash the final tree, not this intermediate one.
        state.squashed_backport_range = cached_squash
    if parse_equivalence(rd_output) == "unmatched":
        return True
    return not _shares_content_with_fix(config, state)


def _run_fused_resolution(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
    *,
    commit: str,
    label: str,
    base_head: str,
    conflicted_files: list[str],
    files_to_skip: list[str],
    conflict_summary_parts: list[str],
) -> str | None:
    """One session to resolve, complete, build/test, review and commit.

    Returns "resolved", or None when the caller must redo the cherry-pick and
    use the per-chunk resolver.
    """
    repo = config.repo_path
    report_path = os.path.join(
        _fused_artifact_dir(tracker), FUSED_RESOLUTION_REPORT_FILE
    )
    followup_fixes = (
        "\n".join(f"  - {f}" for f in state.followup_fixes)
        if state.followup_fixes and _is_followup_target(repo, commit, state)
        else ""
    )
    triage_regeneration = state.triage_assessment.get("regeneration_commands")
    prompt = FusedResolutionPrompt(
        context=create_context_str(claude_config, config),
        fix_diff=_fix_diff_for_prompt(config, commit),
        fix_commit=commit,
        conflicted_files=conflicted_files,
        triage_assessment=(
            json.dumps(state.triage_assessment, indent=2)
            if state.triage_assessment
            else ""
        ),
        build_command=config.build_command,
        test_command=config.test_command,
        check_commands=_fused_check_commands(state),
        regeneration_commands=[
            cmd for cmd in _strings(triage_regeneration) if cmd.strip()
        ],
        files_to_skip=files_to_skip,
        baseline_ref=base_head,
        target_branch=config.target_branch,
        report_path=report_path,
        followup_fixes=followup_fixes,
    )
    log.info(
        "%s: fused session — resolving %d conflicts in one prompt",
        label,
        len(conflicted_files),
    )
    if tracker:
        tracker.record_gate("resolution_mode", "fused")
    try:
        _run_fused_session(config, claude_config, prompt.prompt(), tracker)
    finally:
        _restore_protected_refs(config, state)
    _add_fused_report_to_dossier(state, report_path)

    remaining = _conflicted_files(git_status_porcelain(repo))
    if remaining:
        log.warning("%s: fused session left %d conflicts", label, len(remaining))
        return None
    if git_rev_parse(repo, "HEAD") == base_head:
        if not _non_artifact_dirt(repo):
            log.warning("%s: fused session changed nothing", label)
            return None
        git_commit(
            repo,
            f"Port {label}\n\n[ Fused resolution session ]",
            trailers=_COAUTHOR_TRAILERS,
        )
        if git_rev_parse(repo, "HEAD") == base_head:
            log.warning("%s: fused changes could not be committed", label)
            return None
    if _non_artifact_dirt(repo):
        # Leftover edits would affect the verified build but miss the
        # candidate patch, so fold them into the session's commit.
        _run_git(repo, ["add", "-A"])
        git_amend(repo)
        log.info("%s: committed leftover session edits", label)
    markers = _committed_conflict_markers(repo, base_head)
    if markers:
        log.warning(
            "%s: fused session committed conflict markers in %d files",
            label,
            len(markers),
        )
        return None
    stripped = _strip_session_scratch(repo, base_head, keep=set(state.seed))
    if stripped is None:
        log.warning("%s: could not strip session scratch cleanly — falling back", label)
        return None
    if stripped and state.dossier:
        state.dossier.add(
            "Build scratch stripped from fused commit",
            "\n".join(stripped[:50])
            + (f"\n... and {len(stripped) - 50} more" if len(stripped) > 50 else ""),
            "Phase 2+3a fused session",
        )
    if _fused_resolution_unmatched(config, state, commit, base_head):
        log.warning(
            "%s: fused result shares too little with the fix — falling back", label
        )
        if tracker:
            tracker.record_gate("semantic_equivalence", "fused_unmatched")
        if state.dossier:
            state.dossier.add(
                "Fused semantic check",
                "The fused result has too little in common with the fix; the "
                "per-chunk resolver is used instead.",
                "Phase 2+3a fused session",
            )
        return None
    if config.blind_review:
        _run_triage_blind_review(
            config, _phase_config(claude_config, config), state, tracker, base_head
        )
    if state.dossier:
        state.dossier.add(
            "Fused resolution",
            f"{len(conflicted_files)} source conflicts resolved in one session; "
            "completion, build/test, checks and review ran in-session.",
            "Phase 2+3a fused session",
        )
        state.dossier.add(
            "Evidence scope (fused resolution)",
            _FUSED_RESOLUTION_SCOPE,
            "Phase 2+3a fused session",
        )
    conflict_summary_parts.append(
        f"Fused session resolved {len(conflicted_files)} source conflicts "
        "in one prompt"
    )
    state.fused_resolution_ran = True
    return "resolved"


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
        _run_triage_blind_review(
            config,
            claude_config_resolve,
            state,
            tracker,
            base_head,
        )

    return "resolved"


def _run_triage_blind_review(
    config: BugFixConfig,
    claude_config_resolve: ClaudeConfig,
    state: PipelineState,
    tracker,
    base_head: str,
) -> None:
    """One triage-blind review of the current resolution (Phase 3a.2).

    Runs in the ladder and after a fused resolution session: the resolver has
    seen the fix and its own adaptations, so an independent read of the diff is
    the check that catches anchoring errors.
    """
    repo = config.repo_path
    blind_diff = _safe_git_diff(repo, base_head)
    if not blind_diff.strip():
        return
    log.info("Phase 3a.2 — triage-blind independent review")
    prompt = TriageBlindReviewPrompt(
        context=create_context_str(claude_config_resolve, config),
        fix_diff=_compute_fix_diff(config, config.source_fix_commit),
        current_diff=blind_diff,
        target_branch=config.target_branch,
        baseline_ref=base_head,
    )
    try:
        asyncio.run(
            claude_run(
                claude_config_resolve,
                [prompt.prompt()],
                tracker=tracker,
            )
        )
    except Exception:
        log.warning("Phase 3a.2 — blind review failed, continuing")
        return
    finally:
        _restore_protected_refs(config, state)
    if not git_status_porcelain(repo):
        return
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
        old_fused = config.fused_resolution
        try:
            config.source_fix_commit = sha
            # The fused session is tailored to the main fix's triage and
            # regeneration commands; prerequisites must use the legacy path.
            config.fused_resolution = False
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
            config.fused_resolution = old_fused

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
        target_base = git_merge_base(repo, config.target_branch, "HEAD")
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
            target_base = git_merge_base(config.repo_path, config.target_branch, "HEAD")
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


def _has_build_or_test_commands(config: BugFixConfig) -> bool:
    """True when build/test can independently verify a resolved tree."""
    return bool(
        (config.build_command or "").strip() or (config.test_command or "").strip()
    )


def _record_verify_gate(config: BugFixConfig, state: PipelineState, tracker) -> None:
    """Record Phase 4 evidence without claiming a vacuous pass.

    A force-resolved tree with no build/test commands has nothing to verify;
    the same rule applies to the post-recovery sweep, so both call sites use
    this helper instead of hard-coding ``passed``.
    """
    if state.force_resolved and not _has_build_or_test_commands(config):
        tracker.record_gate("verify", "unverified")
        state.dossier.add(
            "Verification unavailable",
            "Build and test commands are both empty; the "
            "force-resolved tree cannot be verified automatically",
            "Phase 4",
        )
    else:
        tracker.record_gate("verify", "passed")


def _force_resolve_outcome(
    config: BugFixConfig,
    state: PipelineState,
    outcome: str,
    detail: str,
) -> tuple[str, str]:
    """Downgrade a force-resolved port that was never independently verified.

    A failed resolution that is papered over with ``git checkout --theirs``
    (or a clean re-apply) may only be reported as success when build/test
    commands ran and passed. With both commands empty, Phase 4's
    "verification" is vacuous: the conflicts were discarded, not resolved.
    Keep the explicit ``unverified`` outcome instead of promoting it to
    success, and preserve any completeness evidence that came before it.
    """
    if state.force_resolved and not _has_build_or_test_commands(config):
        reason = state.force_resolved_reason or "resolution failed"
        how = (
            "conflicts were force-resolved with --theirs"
            if state.force_resolved_with_theirs
            else "the cherry-pick was re-applied after resolution failed"
        )
        parts = [
            reason,
            f"{how}; no build/test commands were available to verify the result",
        ]
        if detail:
            parts.append(detail)
        return "unverified", "; ".join(parts)
    return outcome, detail


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

        target_base = git_merge_base(config.repo_path, config.target_branch, "HEAD")
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
            target_base = git_merge_base(repo, config.target_branch, "HEAD")
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
        raise
    except Exception:
        tracker.set_outcome("error")
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
    state.protected_refs = _protected_refs(
        config.repo_path, (config.target_branch, config.source_branch)
    )
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
        assessment: dict = {}
        fused_pre_red_ok = False
        if config.fused_pre_red:
            assessment = _run_fused_pre_red(config, claude_config, state, tracker) or {}
            fused_pre_red_ok = bool(assessment)
            if not fused_pre_red_ok:
                log.warning(
                    "Fused pre-RED session failed — falling back to separate "
                    "Phase 0.5/0.5b/0.5c"
                )
                tracker.record_gate("pre_red_mode", "fused_fallback")
        if not assessment:
            assessment = phase_0_5_semantic_triage(
                config, claude_config, state, tracker
            )
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
        if portable_prereqs and not fused_pre_red_ok:
            with tracker.phase("Phase 0.5b — Prerequisite chain"):
                ported = port_prerequisite_chain(config, claude_config, state, tracker)
                state.dossier.add(
                    "Prerequisites ported",
                    json.dumps(ported, indent=2) if ported else "none",
                    "Phase 0.5b",
                )
                tracker.record_gate("prerequisites_ported", str(len(ported)))

        test_files_in_commit = _planned_test_files(state)
        if test_files_in_commit and not fused_pre_red_ok:
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

        # Test-run output left by the RED check must not enter the candidate.
        untracked = [
            line[3:].strip()
            for line in git_status_porcelain(config.repo_path)
            if line.startswith("?? ")
        ]
        artifacts = [
            path
            for path in untracked
            if _is_test_output(config.repo_path, config.target_branch, path)
        ]
        if artifacts:
            log.warning(
                "Removing %d untracked test-output files before the cherry-pick",
                len(artifacts),
            )
            _run_git(config.repo_path, ["clean", "-f", "-d", "--", *artifacts])
        _run_git(config.repo_path, ["add", "-A"])
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
            _restore_protected_refs(config, state)
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
        if regen_cmds and files_skipped and not state.fused_resolution_ran:
            with tracker.phase("Phase 3b — Regeneration"):
                phase_3b_regeneration(config, state, regen_cmds)

        cherry_pick_succeeded = True

    except (PipelineEscalation, RuntimeError, OSError, subprocess.SubprocessError) as e:
        # A session may have moved a branch before failing; the recovery path
        # resets to the target branch, so restore the refs first.
        _restore_protected_refs(config, state)
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
            state.force_resolved_with_theirs = True
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
            git_reset_hard(config.repo_path, config.target_branch)
            git_cherry_pick(config.repo_path, config.source_fix_commit)
            status = git_status_porcelain(config.repo_path)
            force_conflicts = _conflicted_files(status)
            state.force_resolved_with_theirs = bool(force_conflicts)
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
            if force_conflicts:
                conflict_note = (
                    f"[ {len(force_conflicts)} conflicts force-resolved "
                    "with --theirs ]"
                )
            else:
                conflict_note = (
                    "[ cherry-pick re-applied without --theirs after "
                    "resolution failed ]"
                )
            git_commit(
                config.repo_path,
                f"Port {config.issue_id} (force-resolved for Phase 4b)\n\n"
                f"{conflict_note}",
                trailers=_COAUTHOR_TRAILERS,
            )

        # Resolution failed; the tree below is force-resolved, not resolved.
        # It may only become a success outcome if build/test independently
        # verify it (Phase 4b). Otherwise the final outcome is "unverified".
        state.force_resolved = True
        state.force_resolved_reason = (
            f"resolution failed ({type(e).__name__}: {str(e)[:200]})"
        )

        # Enter 4b only when it has something to drive. With both build and
        # test commands empty the run-and-fix loop has no success criterion:
        # it cannot distinguish a good port from a --theirs-poisoned one, and
        # a marker-less session would fail the whole target after the
        # force-resolved tree is already committed. Let the port stand, but
        # keep it explicitly unverified.
        if _has_build_or_test_commands(config):
            with tracker.phase("Phase 4b — Run-and-fix loop (build-test-fix)"):
                phase_4b_run_and_fix(config, claude_config, state, tracker)
        else:
            log.warning(
                "Phase 4b skipped after force-resolve: build and test "
                "commands are both empty — the force-resolved tree is "
                "UNVERIFIED and will not be reported as success"
            )
            state.dossier.add(
                "Run-and-fix loop skipped — result unverified",
                "Build and test commands are both empty — Phase 4b not "
                "entered after force-resolution; the force-resolved tree "
                "is unverified and must be reviewed by a human",
                "Phase 4b",
            )
        cherry_pick_succeeded = True

    if not cherry_pick_succeeded:
        return state.dossier, tracker

    fix_diff = _compute_fix_diff(config)

    try:
        with tracker.phase("Phase 4 — Verify"):
            phase_4_verify(config, state)
            _record_verify_gate(config, state, tracker)
    except PipelineEscalation:
        try:
            with tracker.phase("Phase 4a — Build error recovery"):
                phase_4a_build_recovery(config, claude_config, state, tracker, fix_diff)
                state.dossier.add(
                    "Build recovery", "Agent-assisted fix applied", "Phase 4a"
                )

            with tracker.phase("Phase 4 — Verify (post-recovery)"):
                phase_4_verify(config, state)
                _record_verify_gate(config, state, tracker)
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
            ext_failures = _repair_known_formatter_failures(config, ext_failures, state)

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
                squashed or "HEAD^..HEAD",
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
    _restore_protected_refs(config, state)
    patch_bytes = git_format_patch_bytes(
        config.repo_path, base_ref=config.target_branch
    )
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
        outcome, completeness_detail = _force_resolve_outcome(
            config, state, outcome, completeness_detail
        )
        tracker.record_gate(
            "completeness",
            (
                "complete"
                if outcome == "success"
                else "unverified" if outcome == "unverified" else "partial"
            ),
        )
        if outcome != "success":
            log.warning("Backport outcome is %s: %s", outcome, completeness_detail)
            label = (
                "Unverified force-resolved port"
                if outcome == "unverified"
                else "Partial backport"
            )
            state.dossier.add(label, completeness_detail, "completeness check")
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
    pre_red_mode: str = ""
    resolution_mode: str = ""


def _read_tracker_modes(target_run_dir: str, issue_id: str) -> dict[str, str]:
    """The pre-RED and resolution modes recorded as tracker gates."""
    import glob

    pattern = os.path.join(target_run_dir, f"{issue_id}_*.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    modes = {"pre_red_mode": "", "resolution_mode": ""}
    if not matches:
        return modes
    try:
        with open(matches[0]) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return modes
    for phase in data.get("phases", []):
        gates = phase.get("gate_decisions") or {}
        for key in modes:
            if gates.get(key):
                modes[key] = str(gates[key])
    return modes


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
            modes = _read_tracker_modes(target_run_dir, target_config.issue_id)
            pre_red_mode = modes["pre_red_mode"] or (
                "fused" if target_config.fused_pre_red else "legacy"
            )
            resolution_mode = modes["resolution_mode"] or (
                "fused" if target_config.fused_resolution else "legacy"
            )
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
                    pre_red_mode=pre_red_mode,
                    resolution_mode=resolution_mode,
                )
            )
            evidence = []
            if resolution_mode == "fused":
                evidence.append("fused resolution, self-reviewed")
            elif resolution_mode == "fused_fallback":
                evidence.append("fused fallback → per-chunk resolver")
            if pre_red_mode in ("fused", "fused_fallback"):
                evidence.append("fused pre-RED")
            suffix = f" ({', '.join(evidence)})" if evidence else ""
            print(
                f"FINISHED target {spec.branch}: outcome={outcome}{suffix}, "
                f"duration={duration}s, cost=${cost_usd:.4f}"
            )

            # Preserve partial and unverified backports too — an incomplete or
            # unverified port is exactly what a human needs to inspect and
            # finish.
            if (
                outcome in ("success", "partial_success", "unverified")
                and worktree_created
            ):
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
