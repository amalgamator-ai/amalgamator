"""Phase 0-5 orchestrator for the auto_bug_fix pipeline.

Usage:
    Edit ``auto_bug_fix/bug_fix_config.py`` to configure your project, then run:

        python -m auto_bug_fix.run_bug_fix
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import asyncio
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone

from auto_bug_fix.utils import Tee
from auto_bug_fix.claude_utils import ClaudeConfig, PipelineStep, claude_run as _claude_run


async def _run_llm(claude_config, prompts, tracker=None):
    """Compatibility wrapper: call claude_run and feed timings into tracker."""
    batch_start = time.time()
    step_timings = await _claude_run(claude_config, prompts)

    if tracker and step_timings:
        cursor = batch_start
        for st in step_timings:
            dur = st.get("duration", 0)
            sdk_cost = st.get("cost_usd")
            tracker.record_query(
                prompt_name=st.get("name", "query"),
                start_time=cursor,
                end_time=cursor + dur,
                input_tokens=st.get("input_tokens", 0),
                output_tokens=st.get("output_tokens", 0),
                sdk_cost_usd=sdk_cost if sdk_cost else None,
            )
            log.info("LLM query %s: duration=%.1fs, tokens=%d/%d, cost=$%.4f (sdk=$%s)",
                     st.get("name", "query"), dur,
                     st.get("input_tokens", 0), st.get("output_tokens", 0),
                     sdk_cost or 0, sdk_cost)
            cursor += dur

    return step_timings


claude_run = _run_llm

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.git_tools import (
    git_cherry_pick,
    git_cherry_pick_abort,
    git_status_porcelain,
    git_diff_name_only,
    git_diff_name_only_range,
    git_format_patch,
    git_merge_base,
    git_reset_hard,
    git_commit as _git_commit_raw,
    git_rev_parse,
    resolve_merge_commit,
    git_is_ancestor,
    git_cat_file_exists,
    is_merge_commit,
    git_worktree_add,
)
from auto_bug_fix.patch_id import forward_patch_id_check, backward_patch_id_check
from auto_bug_fix.signature import capture_output
from auto_bug_fix.mechanical import resolve_mechanically
from auto_bug_fix.snapback import (
    snap_back_files,
    _git_blob,
)
from auto_bug_fix.evidence import (
    added_lines_by_file,
    audit_added_tokens,
    audit_line_provenance,
    annotate_dead_config,
    target_config_state,
    port_line_changes,
    mine_substitutions,
    apply_substitutions,
    find_pure_rename_restorations,
    token_exists_on_ref,
    scan_followup_fixes,
    run_file_checks,
    render_evidence,
    validate_triage_prerequisites,
)
from auto_bug_fix.allowlist import derive_seed, resolve_renames, enforce_allowlist, enforce_test_file_veto
from auto_bug_fix.baseline import check_regression, run_test_suite, parse_test_failures, TIMEOUT_EXIT_CODE
from auto_bug_fix.worktree import cleanup_worktree
from auto_bug_fix.config_loader import load_multi_target_config, TargetSpec, _sanitize_ref
from auto_bug_fix.range_diff import run_range_diff, parse_equivalence
from auto_bug_fix.dossier import Dossier, build_trailers, format_dossier
from auto_bug_fix.tracker import Tracker
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    SemanticTriagePrompt,
    TestPortAgentPrompt,
    NarrowResolutionAgentPrompt,
    PostResolutionCheckPrompt,
    TriageBlindReviewPrompt,
    BuildErrorRecoveryPrompt,
    RedTriagePrompt,
    SEMANTIC_TRIAGE_FILE,
    TEST_PORT_MANIFEST_FILE,
    RED_TRIAGE_FILE,
)
from auto_bug_fix.phase_4b import (
    gen_RunAndFixPrompt,
    RUN_AND_FIX_FAILURE_FILE,
    RUN_AND_FIX_SUCCESS_FILE,
)

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

_COAUTHOR_TRAILERS = {"Co-authored-by": "amalgamator <amalgamator@redhat.com>"}


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
            cwd=repo_path, capture_output=True,
        )


def git_commit(repo_path: str, message: str, trailers: dict[str, str] | None = None) -> None:
    """Wrapper around git_commit that unstages pipeline artifacts first."""
    _unstage_pipeline_artifacts(repo_path)
    _git_commit_raw(repo_path, message, trailers=trailers)


def git_amend(repo_path: str) -> None:
    """Amend HEAD with the currently staged changes, keeping the message."""
    _unstage_pipeline_artifacts(repo_path)
    result = subprocess.run(
        ["git", "commit", "--amend", "--no-edit"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning("git commit --amend failed: %s",
                    (result.stderr or result.stdout)[:300])


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
    return [l[3:] for l in status_lines if any(l.startswith(p) for p in _CONFLICT_PREFIXES)]


def _has_conflicts(status_lines: list[str]) -> list[str]:
    """Return status lines that represent merge conflicts."""
    return [l for l in status_lines if any(l.startswith(p) for p in _CONFLICT_PREFIXES)]


@dataclass
class PipelineState:
    seed: list[str] = field(default_factory=list)
    allowed_modules: list[str] = field(default_factory=list)
    baseline: set[str] = field(default_factory=set)
    ported_test_files: list[str] = field(default_factory=list)
    cherry_pick_path: str = ""
    dossier: Dossier | None = None
    triage_assessment: dict = field(default_factory=dict)
    priority_files: list[str] = field(default_factory=list)
    baseline_build_exit: int = 0
    baseline_build_output: str = ""
    baseline_verification: list[dict] = field(default_factory=list)
    red_passed_output: str = ""
    followup_fixes: list[str] = field(default_factory=list)
    followup_fixes_commit: str = ""


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
        seed_from_diff = subprocess.run(
            ["git", "diff", "--name-only", f"{config.source_fix_commit}^1", config.source_fix_commit],
            cwd=repo, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        state.seed = [f for f in seed_from_diff if f.strip()]
        if state.seed:
            log.info("Seed derived from diff range (%d files)", len(state.seed))

    fork_point = git_merge_base(repo, config.source_branch, target)
    resolved, _ = resolve_renames(
        repo, state.seed, target, fork_point,
        cap=config.allowlist_rename_expansion_cap,
    )
    state.allowed_modules = resolved

    if config.allowed_modules is not None:
        state.allowed_modules = config.allowed_modules

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
            raise PipelineStop("Target branch is not affected (no ancestry, no patch-id match, no seed files)")
        ancestry_status = "seed_files_only"
        log.info("Ancestry/patch-id checks failed but seed files exist on target — proceeding with caution")

    return {"forward_patch_id": "not_found", "ancestry": ancestry_status}


def _compute_fix_diff(config: BugFixConfig, commit: str | None = None, for_files: list[str] | None = None) -> str:
    """Compute the fix diff with truncation for large commits.

    When for_files is provided, only include diffs for those specific files
    (used during chunked resolution to keep prompt size manageable).
    """
    sha = commit or config.source_fix_commit


    if for_files:
        if is_merge_commit(config.repo_path, sha):
            stat = subprocess.run(
                ["git", "diff", "--stat", f"{sha}^1..{sha}"],
                cwd=config.repo_path, capture_output=True, text=True,
            ).stdout
            partial_diff = subprocess.run(
                ["git", "diff", f"{sha}^1..{sha}", "--"] + for_files,
                cwd=config.repo_path, capture_output=True, text=True,
            ).stdout
        else:
            stat = subprocess.run(
                ["git", "show", "--stat", sha],
                cwd=config.repo_path, capture_output=True, text=True,
            ).stdout
            partial_diff = subprocess.run(
                ["git", "show", sha, "--"] + for_files,
                cwd=config.repo_path, capture_output=True, text=True,
            ).stdout
        fix_diff = stat + "\n\n--- Diff for chunk files ---\n\n" + partial_diff
        MAX_CHUNK_CHARS = 50_000
        if len(fix_diff) > MAX_CHUNK_CHARS:
            fix_diff = stat + "\n\n[Chunk diffs too large — stat only. Agent must read files directly.]"
        return fix_diff


    if is_merge_commit(config.repo_path, sha):
        fix_diff = subprocess.run(
            ["git", "diff", f"{sha}^1..{sha}"],
            cwd=config.repo_path, capture_output=True, text=True,
        ).stdout
    else:
        fix_diff = subprocess.run(
            ["git", "show", sha],
            cwd=config.repo_path, capture_output=True, text=True,
        ).stdout

    MAX_DIFF_CHARS = 200_000
    if len(fix_diff) > MAX_DIFF_CHARS:
        log.warning("fix_diff too large (%d chars), using --stat only", len(fix_diff))
        if is_merge_commit(config.repo_path, sha):
            fix_diff = subprocess.run(
                ["git", "diff", "--stat", f"{sha}^1..{sha}"],
                cwd=config.repo_path, capture_output=True, text=True,
            ).stdout
        else:
            fix_diff = subprocess.run(
                ["git", "show", "--stat", sha],
                cwd=config.repo_path, capture_output=True, text=True,
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
                cmd, shell=True, cwd=config.repo_path,
                capture_output=True, text=True, timeout=config.command_timeout,
            )
        except subprocess.TimeoutExpired:
            log.warning("Phase 3b: regeneration timed out (30min): %s", cmd)
            git_reset_hard(config.repo_path, safe_point)
            state.dossier.add(f"Regeneration timed out: {cmd}", "Timed out after 30 minutes", "Phase 3b")
            continue

        if result.returncode == 0:
            log.info("Phase 3b: regeneration succeeded: %s", cmd)
            any_succeeded = True
        else:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            log.warning("Phase 3b: regeneration exited %d: %s", result.returncode, cmd)
            try:
                build_check = subprocess.run(
                    config.build_command, shell=True, cwd=config.build_dir,
                    capture_output=True, text=True, timeout=config.command_timeout,
                )
            except subprocess.TimeoutExpired:
                log.warning("Phase 3b: build check timed out, rolling back")
                git_reset_hard(config.repo_path, safe_point)
                continue
            if build_check.returncode == 0:
                log.info("Phase 3b: codegen exited non-zero but build still passes — keeping partial regeneration")
                any_succeeded = True
                state.dossier.add(
                    f"Regeneration partial (kept): {cmd}",
                    f"Codegen exited {result.returncode} but build passes.\n{stderr_tail}",
                    "Phase 3b",
                )
            else:
                log.warning("Phase 3b: build broken after codegen, rolling back to %s", safe_point[:12])
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
            cwd=config.repo_path, capture_output=True,
        )
        git_commit(config.repo_path, f"Regenerate files for {config.issue_id}\n\nCommands: {', '.join(regen_cmds)}")
        state.dossier.add(
            "Regenerated files",
            "\n".join(changed[:50]) + (f"\n... and {len(changed)-50} more" if len(changed) > 50 else ""),
            "Phase 3b",
        )
    else:
        log.info("Phase 3b: no successful regeneration — keeping auto-resolved versions")


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
            cwd=config.repo_path, capture_output=True, text=True,
        ).stdout
    else:
        fix_diff = subprocess.run(
            ["git", "show", "--stat", sha],
            cwd=config.repo_path, capture_output=True, text=True,
        ).stdout

    context = create_context_str(claude_config, config)
    prompt = SemanticTriagePrompt(
        context=context,
        fix_diff=fix_diff,
        source_branch=config.source_branch,
        target_branch=config.target_branch,
        seed_files=state.seed,
    )

    claude_config_triage = ClaudeConfig(
        model=claude_config.model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
    )

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
            log.info("Semantic triage: difficulty=%s, recommendation=%s",
                     assessment.get("estimated_difficulty", "?"),
                     assessment.get("recommendation", "?"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to parse semantic triage output: %s", e)

    state.triage_assessment = assessment
    state.priority_files = assessment.get("priority_files", state.allowed_modules)

    return assessment


_TEST_PATTERNS = ("_test.", "test_", "tests/", "/test/", "testing/", "spec/", "_spec.", "testdata/")


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

    log.info("Phase 0.5c: found %d test files in fix commit: %s",
             len(test_files), ", ".join(test_files))

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

    claude_config_test = ClaudeConfig(
        model=claude_config.model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
    )

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
            [config.build_command], config.build_dir, timeout=config.command_timeout,
        )
        state.baseline_build_exit = build_exit
        state.baseline_build_output = build_output
        if build_exit == 0:
            log.info("Baseline build: PASSED")
        else:
            log.warning("Baseline build: FAILED (exit %d) — target branch has pre-existing build issues", build_exit)
    else:
        log.info("Baseline build: skipped (no build command)")

    # Step 2: Test baseline (existing logic)
    if config.test_command:
        red_exit, red_output = capture_output(
            [config.test_command], config.build_dir, timeout=config.command_timeout,
        )

        state.baseline = parse_test_failures(red_output, "")
        is_bugfix = state.triage_assessment.get("commit_type", "").lower() in _BUGFIX_TYPES
        if red_exit == 0:
            if state.ported_test_files and is_bugfix:
                log.warning("RED check passed — deferring to Phase 1.5 triage")
                state.red_passed_output = red_output
            else:
                log.info("Baseline tests pass — %s",
                         "feature/refactor commit, RED check not applicable" if not is_bugfix
                         else "no ported vulnerability test available — proceeding")
    else:
        log.info("Baseline test: skipped (no test command)")

    # Step 3: Extended verification baseline (lint + optionally triage-discovered tests)
    lint_cmds = [c for c in state.triage_assessment.get("lint_commands", []) if _validate_shell_command(c)]
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
                vcmd, shell=True, cwd=config.build_dir,
                capture_output=True, text=True, timeout=config.command_timeout,
            )
            entry = {"cmd": vcmd, "exit_code": vresult.returncode, "output": (vresult.stdout + "\n" + vresult.stderr)[-3000:]}
        except subprocess.TimeoutExpired:
            entry = {"cmd": vcmd, "exit_code": TIMEOUT_EXIT_CODE, "output": f"Timed out after {config.command_timeout}s"}
        state.baseline_verification.append(entry)
        if entry["exit_code"] == 0:
            log.info("Baseline verification passed: %s", vcmd)
        else:
            log.info("Baseline verification failed (exit %d, pre-existing): %s", entry["exit_code"], vcmd)


def phase_1_5_red_triage(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> dict:
    """LLM triage when RED check passes — determine if vulnerability exists via a different vector."""
    context = create_context_str(claude_config, config)
    triage_json = json.dumps(state.triage_assessment, indent=2) if state.triage_assessment else "{}"

    prompt = RedTriagePrompt(
        context=context,
        bug_description=config.bug_description,
        test_output=state.red_passed_output,
        ported_test_files=state.ported_test_files,
        triage_assessment=triage_json,
    )

    claude_config_triage = ClaudeConfig(
        model=claude_config.model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
    )

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
        cwd=repo_path, capture_output=True,
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
        cwd=repo_path, capture_output=True,
    )
    try:
        stat_text = stat.stdout.decode("utf-8")
    except UnicodeDecodeError:
        stat_text = stat.stdout.decode("utf-8", errors="replace")
    return text[:MAX_DIFF_PROMPT_CHARS] + f"\n\n[TRUNCATED — full stat below]\n{stat_text}"


def _resolve_lint_command(config: BugFixConfig, state: PipelineState) -> str:
    """Return the lint command: YAML field takes priority, then triage-discovered.

    Returns empty string if the lint command is identical to the test command
    (already runs as part of the test step).
    """
    if config.lint_command:
        cmd = config.lint_command
    else:
        triage_lints = [c for c in state.triage_assessment.get("lint_commands", []) if _validate_shell_command(c)]
        cmd = triage_lints[0] if triage_lints else ""
    if cmd and cmd == config.test_command:
        return ""
    return cmd


def _run_build_and_test(config: BugFixConfig, state: PipelineState) -> tuple[bool, str, str, str, str]:
    """Run build, test, and lint commands. Return (passed, build_out, build_err, test_out, test_err).

    Empty build/test commands are skipped (for interpreted languages).
    """
    build_stdout = build_stderr = ""
    if config.build_command:
        build_exit, build_stdout, build_stderr = run_test_suite(
            config.build_command, config.build_dir, timeout=config.command_timeout)
        if build_exit != 0:
            return False, build_stdout, build_stderr, "", ""

    test_stdout = test_stderr = ""
    if config.test_command:
        test_exit, test_stdout, test_stderr = run_test_suite(
            config.test_command, config.build_dir, timeout=config.command_timeout)
        if test_exit == TIMEOUT_EXIT_CODE:
            log.warning("Test command timed out — treating as verification failure: %s", config.test_command)
            return False, build_stdout, build_stderr, test_stdout, test_stderr
        failures = parse_test_failures(test_stdout, test_stderr)
        passed, _ = check_regression(failures, state.baseline)
    else:
        passed = True

    if not passed:
        return False, build_stdout, build_stderr, test_stdout, test_stderr

    lint_cmd = _resolve_lint_command(config, state)
    if lint_cmd:
        lint_exit, lint_stdout, lint_stderr = run_test_suite(
            lint_cmd, config.build_dir, timeout=config.command_timeout)
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
    ok, extra_files = enforce_allowlist(repo, state.allowed_modules, config.disallowed_modules)
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


def _commit_recovery_edits(config: BugFixConfig, phase_label: str) -> bool:
    """Commit agent recovery edits so they reach the deliverables.

    The candidate patch (git format-patch), the preserved branch, and the
    Phase 4.5 range-diff are all commit-based — recovery edits left in the
    working tree by the 4a/4b agents would silently vanish from every one
    of them. Returns True if a commit was created.
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
    if git_rev_parse(config.repo_path, "HEAD") == head_before:
        # Either only unstageable pipeline artifacts were dirty, or the
        # commit itself failed — in both cases nothing landed on HEAD.
        log.warning("%s: no recovery commit created", phase_label)
        return False
    log.info("%s: recovery edits committed", phase_label)
    return True


def phase_4a_build_recovery(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
    fix_diff: str,
) -> None:
    """Agent-assisted build/test error recovery. Adapts code to the target branch API."""
    context = create_context_str(claude_config, config)

    claude_config_4a = ClaudeConfig(
        model=claude_config.model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
    )

    for attempt in range(1, config.max_build_test_retries + 1):
        build_exit, build_stdout, build_stderr = run_test_suite(
            config.build_command, config.build_dir, timeout=config.command_timeout)
        if build_exit == 0:
            test_exit, test_stdout, test_stderr = run_test_suite(
                config.test_command, config.build_dir, timeout=config.command_timeout)

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

        log.info("Phase 4a attempt %d/%d — agent diagnosing build/test errors",
                 attempt, config.max_build_test_retries)
        asyncio.run(claude_run(claude_config_4a, [prompt.prompt()], tracker=tracker))

        post_fix_files = set(git_diff_name_only(config.repo_path))
        if post_fix_files == pre_fix_files:
            raise PipelineEscalation(
                "Phase 4a: agent made no changes — likely CANNOT_FIX"
            )

    build_exit, _, _ = run_test_suite(
        config.build_command, config.build_dir, timeout=config.command_timeout)
    if build_exit == 0:
        test_exit, test_stdout, test_stderr = run_test_suite(
            config.test_command, config.build_dir, timeout=config.command_timeout)

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
) -> None:
    """Escalate to the autonomous build-test-fix loop (last resort).

    Assembles a rich context with ALL pipeline data (dossier, triage, config)
    and runs the RunAndFixPrompt as a single long agent session that iterates
    build → test → targeted fix until clean or retries are exhausted.
    """
    dossier_text = format_dossier(state.dossier) if state.dossier else ""
    triage_json = json.dumps(state.triage_assessment, indent=2) if state.triage_assessment else "{}"

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

<pipeline_dossier>
{dossier_text}
</pipeline_dossier>

<semantic_triage>
{triage_json}
</semantic_triage>
"""

    ext_claude_config = ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
        thinking={"type": "adaptive"},
        effort="max",
        max_thinking_tokens=1048576,
    )
    log.info("Phase 4b: escalating to Opus with deep thinking (last resort)")

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

    phase_4b_json_path = None
    if tracker:
        phase_4b_data = {
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
            "semantic_triage": state.triage_assessment if state.triage_assessment else {},
        }
        phase_4b_json_path = os.path.join(tracker._output_dir, "phase_4b_input.json")
        with open(phase_4b_json_path, "w") as f:
            json.dump(phase_4b_data, f, indent=2)
        log.info("Phase 4b input written to %s", phase_4b_json_path)

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

    phase_4b_log_path = os.path.join(tracker._output_dir, "phase_4b_log.txt") if tracker else None
    if phase_4b_log_path:

        _4b_log_file = open(phase_4b_log_path, "w")
        _prev_stdout = sys.stdout
        sys.stdout = Tee(_prev_stdout, _4b_log_file)

    try:
        timings = asyncio.run(_claude_run(ext_claude_config, steps))
    finally:
        if phase_4b_log_path:
            sys.stdout = _prev_stdout
            _4b_log_file.close()
            log.info("Phase 4b log written to %s", phase_4b_log_path)

    _4b_end = time.time()

    if tracker and timings:
        for st in timings:
            dur = st.get("duration", _4b_end - _4b_start)
            sdk_cost = st.get("cost_usd")
            tracker.record_query(
                prompt_name=st.get("name", "phase_4b"),
                start_time=_4b_start,
                end_time=_4b_start + dur,
                input_tokens=st.get("input_tokens", 0),
                output_tokens=st.get("output_tokens", 0),
                sdk_cost_usd=sdk_cost if sdk_cost else None,
            )

    if tracker and phase_4b_json_path:
        phase_4b_data["duration_seconds"] = _4b_end - _4b_start
        if timings:
            phase_4b_data["cost_usd"] = sum(st.get("cost_usd", 0) for st in timings)
            phase_4b_data["input_tokens"] = sum(st.get("input_tokens", 0) for st in timings)
            phase_4b_data["output_tokens"] = sum(st.get("output_tokens", 0) for st in timings)

        # Failure first — must match the decision precedence below, so a
        # session that somehow writes both files is recorded as a failure.
        for result_file in (failure_file, success_file):
            if os.path.exists(result_file):
                dest = os.path.join(tracker._output_dir, os.path.basename(result_file))
                shutil.copy2(result_file, dest)
                log.info("Phase 4b result copied to %s", dest)
                with open(result_file) as f:
                    phase_4b_data["outcome"] = "success" if result_file == success_file else "failure"
                    phase_4b_data["result_summary"] = f.read()
                break
        with open(phase_4b_json_path, "w") as f:
            json.dump(phase_4b_data, f, indent=2)
        log.info("Phase 4b JSON updated with results: %s", phase_4b_json_path)

    if os.path.exists(failure_file):
        with open(failure_file) as f:
            failure_text = f.read()
        state.dossier.add("Run-and-fix failure", failure_text, "Phase 4b")
        raise PipelineEscalation("Phase 4b: run-and-fix loop exhausted retries")
    elif os.path.exists(success_file):
        state.dossier.add("Run-and-fix loop", "Build-test-fix loop succeeded", "Phase 4b")
        log.info("Phase 4b: run-and-fix loop succeeded")
        if _commit_recovery_edits(config, "Phase 4b"):
            state.dossier.add(
                "Run-and-fix loop",
                "Recovery edits committed (fixup)",
                "Phase 4b",
            )
    else:
        # The agent session ended without writing either marker (crash, early
        # exit, ignored instructions) — success may not be assumed.
        state.dossier.add(
            "Run-and-fix loop",
            "Agent ended without writing a success or failure file",
            "Phase 4b",
        )
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


    strategies = [
        {"strategy": None, "strategy_option": None, "name": "default"},
        {"strategy": None, "strategy_option": "patience", "name": "patience"},
        {"strategy": "ort", "strategy_option": None, "name": "ort"},
    ]

    best_strategy = None
    for strat in strategies:
        result = git_cherry_pick(config.repo_path, commit,
                                 strategy=strat["strategy"],
                                 strategy_option=strat["strategy_option"])
        status = git_status_porcelain(config.repo_path)
        uu_files = _conflicted_files(status)

        log.info("%s: strategy %s: exit=%d, conflicts=%d, success=%s",
                 label, strat["name"], result.exit_code, len(uu_files), result.success)

        if result.success and not uu_files:
            log.info("%s: clean cherry-pick (strategy: %s)", label, strat["name"])
            if state.dossier:
                state.dossier.add_strategy(strat["name"], "clean")
            return "clean"

        if uu_files:
            if best_strategy is None or len(uu_files) < best_strategy["count"]:
                best_strategy = {"strat": strat, "count": len(uu_files)}
            if state.dossier:
                state.dossier.add_strategy(strat["name"], f"conflict ({len(uu_files)} files)")
        else:
            log.warning("%s: strategy %s: no conflicts but exit=%d, stderr=%s",
                        label, strat["name"], result.exit_code, result.stderr[:500])

        git_cherry_pick_abort(config.repo_path)

    if best_strategy is None:
        raise PipelineEscalation(f"{label}: unmappable cherry-pick (all strategies failed)")

    log.info("%s: using strategy %s (%d conflicts, fewest)",
             label, best_strategy["strat"]["name"], best_strategy["count"])
    strat = best_strategy["strat"]
    git_cherry_pick(config.repo_path, commit,
                    strategy=strat["strategy"],
                    strategy_option=strat["strategy_option"])
    status = git_status_porcelain(config.repo_path)
    uu_files = _conflicted_files(status)

    if not uu_files:
        log.info("%s: clean cherry-pick on retry (strategy: %s)", label, strat["name"])
        return "clean"

    conflict_summary_parts = []
    skip_set = set(files_to_skip or [])
    if skip_set:
        # Reset ALL skip files to target version — both conflicting and cleanly-applied
        all_changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=config.repo_path, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        skip_applied_clean = [f for f in all_changed if f in skip_set and f not in uu_files]
        if skip_applied_clean:
            log.info("%s: reverting %d skip files that applied cleanly", label, len(skip_applied_clean))
            for f in skip_applied_clean:
                subprocess.run(["git", "checkout", "HEAD", "--", f],
                               cwd=config.repo_path, capture_output=True)

        auto_resolved = [f for f in uu_files if f in skip_set]
        if auto_resolved:
            log.info("%s: reverting %d conflicting skip files to target version", label, len(auto_resolved))
            for f in auto_resolved:
                subprocess.run(["git", "checkout", "HEAD", "--", f],
                               cwd=config.repo_path, capture_output=True)
                subprocess.run(["git", "add", "--", f],
                               cwd=config.repo_path, capture_output=True)

        total_skipped = len(skip_applied_clean) + len(auto_resolved)
        if total_skipped:
            conflict_summary_parts.append(
                f"Reverted {total_skipped} files to target version (files_to_skip)"
            )
            uu_files = [f for f in uu_files if f not in skip_set]

    if not uu_files:
        msg = f"Port {label}\n\n[ Conflict summary: {'; '.join(conflict_summary_parts)} ]"
        git_commit(config.repo_path, msg)
        return "resolved"

    original_uu_files = list(uu_files)
    base_head = git_rev_parse(config.repo_path, "HEAD")
    fix_diff = _compute_fix_diff(config, commit)
    context = create_context_str(claude_config, config)
    priority = state.priority_files if state.priority_files else state.allowed_modules

    claude_config_resolve = ClaudeConfig(
        model=claude_config.model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
    )

    if config.mechanical_resolution:
        try:
            mech = resolve_mechanically(config.repo_path, commit, uu_files)
        except Exception as e:
            log.warning("%s: mechanical resolution failed (continuing to LLM): %s", label, e)
            mech = {}
        if mech:
            log.info("%s: mechanically re-applied %d/%d conflicted files",
                     label, len(mech), len(uu_files))
            if state.dossier:
                for f, rung in sorted(mech.items()):
                    state.dossier.add_strategy(f"mechanical:{rung}", f"resolved {f}")
            conflict_summary_parts.append(
                f"Mechanically re-applied {len(mech)} files (git apply, relaxed context)"
            )
            uu_files = [f for f in uu_files if f not in mech]
            if tracker:
                tracker.record_gate(
                    "mechanical_resolution", f"{len(mech)}/{len(original_uu_files)}")

    if not uu_files:
        return _finalize_resolution(
            config, claude_config_resolve, state, tracker, commit, label,
            conflict_summary_parts, base_head, fix_diff, context,
        )

    followup_note = ""
    if state.followup_fixes and _is_followup_target(config.repo_path, commit, state):
        followup_note = "\n".join(f"  - {f}" for f in state.followup_fixes)

    CHUNK_SIZE = 30

    for attempt in range(1, config.max_resolution_retries + 1):
        if len(uu_files) > CHUNK_SIZE:
            if len(uu_files) > 100:
                chunk_size = 10
            elif len(fix_diff) > 50_000:
                chunk_size = 15
            else:
                chunk_size = CHUNK_SIZE
            log.info("%s attempt %d/%d — resolving %d conflicts in chunks of %d",
                     label, attempt, config.max_resolution_retries, len(uu_files), chunk_size)
            for chunk_start in range(0, len(uu_files), chunk_size):
                chunk = uu_files[chunk_start:chunk_start + chunk_size]
                log.info("%s: chunk %d-%d of %d",
                         label, chunk_start + 1, chunk_start + len(chunk), len(uu_files))
                chunk_diff = _compute_fix_diff(config, commit, for_files=chunk)
                prompt = NarrowResolutionAgentPrompt(
                    fix_diff=chunk_diff,
                    conflicted_files=chunk,
                    context=context,
                    allowed_modules=priority,
                    triage_assessment=json.dumps(state.triage_assessment, indent=2) if state.triage_assessment else "",
                    target_branch=config.target_branch,
                    baseline_ref=base_head,
                    followup_fixes=followup_note,
                )
                asyncio.run(claude_run(claude_config_resolve, [prompt.prompt()], tracker=tracker))
        else:
            prompt = NarrowResolutionAgentPrompt(
                fix_diff=fix_diff,
                conflicted_files=uu_files,
                context=context,
                allowed_modules=priority,
                triage_assessment=json.dumps(state.triage_assessment, indent=2) if state.triage_assessment else "",
                target_branch=config.target_branch,
                baseline_ref=base_head,
                followup_fixes=followup_note,
            )
            log.info("%s attempt %d/%d — resolving %d conflicts",
                     label, attempt, config.max_resolution_retries, len(uu_files))
            asyncio.run(claude_run(claude_config_resolve, [prompt.prompt()], tracker=tracker))

        remaining = git_status_porcelain(config.repo_path)
        still_conflicted = _has_conflicts(remaining)
        if not still_conflicted:
            conflict_summary_parts.append(
                f"LLM resolved {len(original_uu_files)} source conflicts ({attempt} attempt(s))"
            )
            return _finalize_resolution(
                config, claude_config_resolve, state, tracker, commit, label,
                conflict_summary_parts, base_head, fix_diff, context,
            )
        log.warning("%s: attempt %d: %d conflicts remain", label, attempt, len(still_conflicted))
        uu_files = _conflicted_files(still_conflicted)

    git_cherry_pick_abort(config.repo_path)
    raise PipelineEscalation(f"{label}: failed after {config.max_resolution_retries} attempts")


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
        summary = "; ".join(conflict_summary_parts) if conflict_summary_parts else "resolved"
        git_commit(repo, f"Port {label}\n\n[ Conflict summary: {summary} ]", trailers=_COAUTHOR_TRAILERS)
        if git_rev_parse(repo, "HEAD") == base_head:
            # Empty resolution or a rejecting commit hook — never amend or
            # post-process base_head itself.
            log.warning("%s: resolution commit did not move HEAD — skipping post-passes", label)
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
                cwd=repo, capture_output=True, text=True,
            ).stdout
            added_by_file = added_lines_by_file(port_diff)
            # Mining references base_head — the tree the port actually lands
            # on (target tip PLUS ported prerequisites/tests). Using the bare
            # target branch would mine a ported prerequisite's own rename
            # backwards and mechanically revert it.
            mined = mine_substitutions(
                repo, f"{commit}^1", base_head, sorted(added_by_file))
            guarded = {
                old: new for old, new in mined.items()
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
                log.info("Auto-applied downstream substitutions: %s",
                         ", ".join(f"{o}->{n}" for o, n in sorted(applied_subs.items())))
                if state.dossier:
                    state.dossier.add(
                        "Auto-applied downstream substitutions",
                        "\n".join(f"{o} -> {n}" for o, n in sorted(applied_subs.items())),
                        "Phase 3a post-pass (mined from upstream-base vs target delta)",
                    )

            # Deterministically restore fix bytes for pure token-renames: the
            # clearest three-way violations must not depend on the verifier's
            # obedience. Guards inside the finder ensure every restored token
            # exists on the target.
            restorations = find_pure_rename_restorations(
                repo, base_head, commit, port_diff)
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
                log.info("Auto-restored fix bytes for %d pure-renamed lines in %d files",
                         sum(auto_restored.values()), len(auto_restored))
                if state.dossier:
                    state.dossier.add(
                        "Fix lines auto-restored (pure renames)",
                        "\n".join(f"{f}: {n} lines" for f, n in sorted(auto_restored.items())),
                        "Phase 3a post-pass",
                    )
        except Exception as e:
            log.warning("Evidence checks failed (continuing): %s", e)

    if config.snapback:
        try:
            snap_counts = snap_back_files(repo, changed_files, base_head, commit)
            if snap_counts:
                log.info("Snap-back: restored source bytes in %d files (%d lines)",
                         len(snap_counts), sum(snap_counts.values()))
                if state.dossier:
                    state.dossier.add(
                        "Snap-back (drifted formatting restored to source bytes)",
                        "\n".join(f"{f}: {n} lines" for f, n in sorted(snap_counts.items())),
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
            f for f, text in current_texts.items()
            if _MARKER_START_RE.search(text) and _MARKER_END_RE.search(text)
        ]
        if marker_files:
            log.warning("Conflict markers committed in %d files: %s",
                        len(marker_files), ", ".join(marker_files))
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
                cwd=repo, capture_output=True, text=True,
            ).stdout
            added_by_file = added_lines_by_file(port_diff)
            findings = audit_added_tokens(repo, base_head, commit, added_by_file)
            port_changes = port_line_changes(port_diff)
            annotate_dead_config(
                findings, port_changes, current_texts,
                lambda s: target_config_state(repo, base_head, s),
            )
            if findings:
                log.info("Token audit: %d identifiers flagged (%s)",
                         len(findings), ", ".join(f.token for f in findings[:8]))
            if tracker:
                tracker.record_gate("token_audit", f"{len(findings)} flagged")

            line_findings = audit_line_provenance(
                repo, base_head, commit, port_diff, current_texts,
                substitutions=applied_subs)
            if line_findings:
                log.info("Three-way discipline: %d line findings (%s)",
                         len(line_findings),
                         ", ".join(sorted({lf.kind for lf in line_findings})))
            if tracker:
                tracker.record_gate("line_provenance", f"{len(line_findings)} findings")
        except Exception as e:
            log.warning("Evidence audits failed (continuing): %s", e)

    if config.file_check_command:
        try:
            file_check_failures = run_file_checks(
                repo, changed_files, config.file_check_command, config.command_timeout)
            if file_check_failures:
                log.warning("Per-file checks failed for %d files", len(file_check_failures))
                if state.dossier:
                    state.dossier.add(
                        "Per-file check failures",
                        "\n".join(file_check_failures),
                        "Phase 3a post-pass",
                    )
        except Exception as e:
            log.warning("Per-file checks failed to run (continuing): %s", e)

    followups = (
        state.followup_fixes
        if _is_followup_target(repo, commit, state) else []
    )
    evidence_report = render_evidence(
        findings, applied_subs, advisory_subs, followups, file_check_failures,
        line_findings=line_findings, conflict_marker_files=marker_files,
        auto_restored=auto_restored)

    current_diff = _safe_git_diff(repo, base_head)
    if config.post_resolution_check and current_diff.strip():
        log.info("Phase 3a.1 — post-resolution adaptation check")
        check_prompt = PostResolutionCheckPrompt(
            context=context,
            fix_diff=fix_diff,
            current_diff=current_diff,
            target_branch=config.target_branch,
            triage_assessment=json.dumps(state.triage_assessment, indent=2) if state.triage_assessment else "",
            evidence_report=evidence_report,
            baseline_ref=base_head,
        )
        try:
            asyncio.run(claude_run(claude_config_resolve, [check_prompt.prompt()], tracker=tracker))
        except Exception:
            log.warning("Phase 3a.1 — post-resolution check failed, continuing with existing resolution")
        else:
            # No snap-back after the verifier: its deliberate corrections must
            # never be rewritten, and every pre-verifier line was already
            # snapped with the same sources — a second pass could only undo
            # the verifier's work (observed on real runs) or do nothing.
            post_status = git_status_porcelain(repo)
            if post_status:
                subprocess.run(["git", "add", "-u"], cwd=repo, capture_output=True)
                git_commit(repo, "fixup: adapt backport to target branch API", trailers=_COAUTHOR_TRAILERS)
                log.info("Phase 3a.1 — post-resolution check committed adaptations")
            if config.evidence_checks and findings:
                try:
                    port_diff = subprocess.run(
                        ["git", "diff", base_head, "HEAD"],
                        cwd=repo, capture_output=True, text=True,
                    ).stdout
                    remaining = audit_added_tokens(
                        repo, base_head, commit, added_lines_by_file(port_diff))
                    if tracker:
                        tracker.record_gate("token_audit_post", f"{len(remaining)} remaining")
                    if remaining:
                        log.warning("Token audit: %d identifiers still flagged after Phase 3a.1",
                                    len(remaining))
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
                asyncio.run(claude_run(
                    claude_config_resolve, [blind_prompt.prompt()],
                    tracker=tracker))
            except Exception:
                log.warning("Phase 3a.2 — blind review failed, continuing")
            else:
                blind_status = git_status_porcelain(repo)
                if blind_status:
                    subprocess.run(
                        ["git", "add", "-u"], cwd=repo, capture_output=True)
                    git_commit(
                        repo,
                        "fixup: independent review corrections (Phase 3a.2)",
                        trailers=_COAUTHOR_TRAILERS)
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
    portable = [p for p in prerequisites if p.get("portable", False) and p.get("introduced_by")]

    if not portable:
        return []

    log.info("Phase 0.5 identified %d portable prerequisites to port first", len(portable))
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
                config, claude_config, state, tracker,
                commit=sha,
                label=label,
                files_to_skip=state.triage_assessment.get("files_to_skip", []),
            )
            ported.append({"sha": sha, "symbol": symbol, "result": result})
            log.info("Prerequisite %s ported: %s", symbol, result)
        except PipelineEscalation as e:
            log.warning("Prerequisite %s failed: %s — continuing without it", symbol, e)
            state.dossier.add(
                f"Prerequisite failed: {symbol}",
                str(e),
                "Phase 0.5 prerequisite chain",
            )
        finally:
            config.source_fix_commit = old_commit

    return ported


def _squashed_backport_range(config: BugFixConfig) -> str | None:
    """Synthetic single-commit range spanning the whole backport.

    Pipeline commits (test port, resolution, review fixups, 4a/4b recovery)
    make target..HEAD multi-commit, and range-diff pairs commits one-to-one,
    so every extra commit shows up as an unmatched row regardless of content.
    Squashing the backport into one synthetic commit (tree = HEAD's tree,
    parent = the target base) makes the comparison about content, not
    commit count.
    """
    repo = config.repo_path
    try:
        target_base = git_merge_base(repo, config.target_branch, "HEAD")
        head = git_rev_parse(repo, "HEAD")
        if not target_base or not head or target_base == head:
            return None
        result = subprocess.run(
            ["git", "commit-tree", f"{head}^{{tree}}", "-p", target_base,
             "-m", f"squashed backport for {config.issue_id}"],
            cwd=repo, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        squash = result.stdout.strip()
        return f"{squash}^..{squash}" if squash else None
    except Exception:
        return None


def phase_4_5_semantic_equivalence(
    config: BugFixConfig,
    state: PipelineState,
) -> str:
    """Run range-diff to classify the port as identical, modified, or unmatched."""
    repo = config.repo_path
    try:
        backport_range = _squashed_backport_range(config)
        # The squash is a deliberate one-to-one comparison — force range-diff
        # to pair the two commits so the verdict reflects content, not the
        # pairing heuristic's similarity threshold.
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
    return equivalence


def run_pipeline(
    config: BugFixConfig | None = None,
    claude_config: ClaudeConfig | None = None,
    output_dir: str = "runs",
) -> tuple[Dossier, Tracker]:
    """Orchestrate the full pipeline: triage, baseline, cherry-pick, resolve, verify, equivalence."""
    if config is None:
        from auto_bug_fix.bug_fix_config import bug_fix_config as _default
        config = _default
    if claude_config is None:
        from auto_bug_fix.bug_fix_config import claude_config as _default
        claude_config = _default

    tracker = Tracker(
        issue_id=config.issue_id,
        config_dict=asdict(config),
        model=claude_config.model,
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
                if git_rev_parse(config.repo_path, config.source_branch) != \
                        git_rev_parse(config.repo_path, config.target_branch):
                    state.followup_fixes = scan_followup_fixes(
                        config.repo_path, config.source_fix_commit, config.source_branch)
                if state.followup_fixes:
                    state.followup_fixes_commit = git_rev_parse(
                        config.repo_path, config.source_fix_commit)
            except Exception as e:
                log.warning("Follow-up fix scan failed: %s", e)
            if state.followup_fixes:
                log.info("Upstream has %d follow-up fixes to the ported commit",
                         len(state.followup_fixes))
                state.dossier.add(
                    "Upstream follow-up fixes to the ported commit",
                    "\n".join(state.followup_fixes),
                    "Phase 0 — git log --grep 'Fixes:'",
                )
                tracker.record_gate("followup_fixes", str(len(state.followup_fixes)))

    with tracker.phase("Phase 0.5 — Semantic triage"):
    
        assessment = phase_0_5_semantic_triage(config, claude_config, state, tracker)
        state.dossier.add("Semantic triage", json.dumps(assessment, indent=2) if assessment else "no assessment", "Phase 0.5")
        tracker.record_gate("difficulty", assessment.get("estimated_difficulty", "unknown"))
        tracker.record_gate("recommendation", assessment.get("recommendation", "unknown"))
        if assessment:
            triage_dest = os.path.join(tracker._output_dir, "semantic_triage.json")
            with open(triage_dest, "w") as f:
                json.dump(assessment, f, indent=2)

        if config.evidence_checks:
            prereqs = state.triage_assessment.get("prerequisites", [])
            if prereqs:
                try:
                    corrections = validate_triage_prerequisites(
                        config.repo_path, config.target_branch, prereqs)
                except Exception as e:
                    log.warning("Triage prerequisite validation failed: %s", e)
                    corrections = []
                for corr in corrections:
                    log.warning(
                        "Triage correction: %s — %s",
                        corr.symbol, corr.correction)
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
                    tracker.record_gate(
                        "triage_corrections", str(len(corrections)))
                    if assessment:
                        with open(triage_dest, "w") as f:
                            json.dump(assessment, f, indent=2)

    cherry_pick_succeeded = False
    try:
        prereqs = state.triage_assessment.get("prerequisites", [])
        portable_prereqs = [p for p in prereqs if p.get("portable", False) and p.get("introduced_by")]
        is_bugfix = state.triage_assessment.get("commit_type", "").lower() in _BUGFIX_TYPES
        if portable_prereqs and not is_bugfix:
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
                ported_tests = phase_0_5c_test_port(config, claude_config, state, tracker)
                state.dossier.add(
                    "Test port",
                    "\n".join(ported_tests) if ported_tests else "no tests ported",
                    "Phase 0.5c",
                )
                tracker.record_gate("test_files_ported", str(len(ported_tests)))
                if ported_tests:
                    subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
                    git_commit(config.repo_path, f"Port test files for {config.issue_id} (Phase 0.5c)\n\nPorted before fix to enable RED/GREEN checking")

        with tracker.phase("Phase 1 — Baseline + RED"):
            phase_1_baseline_and_red(config, claude_config, state)

            baseline_summary = f"Build: {'PASSED' if state.baseline_build_exit == 0 else 'FAILED (exit ' + str(state.baseline_build_exit) + ')'}"
            baseline_summary += f"\nTest ({config.test_command}): {'PASSED' if not state.baseline else str(len(state.baseline)) + ' pre-existing failures'}"
            for entry in state.baseline_verification:
                status = "PASSED" if entry["exit_code"] == 0 else f"FAILED (exit {entry['exit_code']}, pre-existing)"
                baseline_summary += f"\n{entry['cmd']}: {status}"
            state.dossier.add("Baseline (clean target branch)", baseline_summary, "Phase 1")

        if state.red_passed_output:
            with tracker.phase("Phase 1.5 — RED triage"):
                triage_result = phase_1_5_red_triage(config, claude_config, state, tracker)
                verdict = triage_result.get("verdict", "inconclusive").strip().lower()
                tracker.record_gate("red_triage_verdict", verdict)
                state.dossier.add(
                    "RED triage",
                    f"Verdict: {verdict}\n{triage_result.get('reasoning', 'no reasoning')}",
                    "Phase 1.5",
                )

                if verdict == "not_vulnerable":
                    raise PipelineStop(f"RED triage: target branch is not vulnerable — {triage_result.get('reasoning', '')[:200]}")

                if verdict == "mitigated":
                    log.info("RED triage: root cause present but mitigated — porting fix for defense-in-depth")
                    state.red_passed_output = ""

                elif verdict == "vulnerable":
                    replacement = triage_result.get("replacement_test_file")
                    if replacement and os.path.exists(replacement):
                        log.info("RED triage produced replacement test — rerunning RED check")
                        subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
                        git_commit(config.repo_path, f"RED triage replacement test for {config.issue_id} (Phase 1.5)")
                        rerun_exit, rerun_output = capture_output(
                            [config.test_command], config.build_dir, timeout=config.command_timeout,
                        )
                        if rerun_exit != 0:
                            log.info("RED triage replacement test confirms vulnerability (exit %d) — continuing", rerun_exit)
                            state.baseline = parse_test_failures(rerun_output, "")
                            state.red_passed_output = ""
                        else:
                            log.warning("RED triage replacement test also passes — escalating to Phase 4b")
                            raise PipelineEscalation("RED triage: replacement test also passes on unmodified code")
                    else:
                        log.warning("RED triage says vulnerable but no valid replacement test — escalating")
                        raise PipelineEscalation("RED triage: vulnerable verdict but no replacement test produced")
                else:
                    raise PipelineEscalation("RED triage inconclusive — escalating to Phase 4b")

        subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
        status = git_status_porcelain(config.repo_path)
        if status:
            git_commit(config.repo_path, f"Pre-cherry-pick cleanup for {config.issue_id}")
        git_reset_hard(config.repo_path, "HEAD")

        with tracker.phase("Phase 2+3a — Cherry-pick and resolve"):
            files_to_skip = state.triage_assessment.get("files_to_skip", [])
            cp_result = cherry_pick_and_resolve(
                config, claude_config, state, tracker,
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

        regen_cmds = [c for c in state.triage_assessment.get("regeneration_commands", []) if _validate_shell_command(c)]
        files_skipped = state.triage_assessment.get("files_to_skip", [])
        if regen_cmds and files_skipped:
            with tracker.phase("Phase 3b — Regeneration"):
                phase_3b_regeneration(config, state, regen_cmds)

        cherry_pick_succeeded = True

    except (PipelineEscalation, RuntimeError, OSError, subprocess.SubprocessError) as e:
        log.warning("Cherry-pick resolution failed (%s: %s) — force-resolving and escalating to Phase 4b",
                     type(e).__name__, str(e)[:200])
        state.dossier.add("Cherry-pick resolution failed", f"{type(e).__name__}: {e}", "Phase 2+3a")

        status = git_status_porcelain(config.repo_path)
        remaining_conflicts = _conflicted_files(status)
        if remaining_conflicts:
            log.info("Force-resolving %d remaining conflicts with --theirs before Phase 4b", len(remaining_conflicts))
            for f in remaining_conflicts:
                subprocess.run(["git", "checkout", "--theirs", "--", f],
                               cwd=config.repo_path, capture_output=True)
                subprocess.run(["git", "add", "--", f],
                               cwd=config.repo_path, capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
            git_commit(config.repo_path,
                       f"Port {config.issue_id} (partial — force-resolved for Phase 4b)\n\n"
                       f"[ {len(remaining_conflicts)} conflicts force-resolved with --theirs ]",
                       trailers=_COAUTHOR_TRAILERS)
        else:
            cherry_pick_in_progress = subprocess.run(
                ["git", "rev-parse", "--verify", "CHERRY_PICK_HEAD"],
                cwd=config.repo_path, capture_output=True).returncode == 0
            if cherry_pick_in_progress:
                subprocess.run(["git", "cherry-pick", "--abort"], cwd=config.repo_path,
                                capture_output=True)
            git_reset_hard(config.repo_path, config.target_branch)
            git_cherry_pick(config.repo_path, config.source_fix_commit)
            status = git_status_porcelain(config.repo_path)
            force_conflicts = _conflicted_files(status)
            if force_conflicts:
                log.info("Re-applied cherry-pick, force-resolving %d conflicts with --theirs", len(force_conflicts))
                for f in force_conflicts:
                    subprocess.run(["git", "checkout", "--theirs", "--", f],
                                   cwd=config.repo_path, capture_output=True)
                    subprocess.run(["git", "add", "--", f],
                                   cwd=config.repo_path, capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=config.repo_path, capture_output=True)
            git_commit(config.repo_path,
                       f"Port {config.issue_id} (force-resolved for Phase 4b)\n\n"
                       f"[ All conflicts force-resolved with --theirs ]",
                       trailers=_COAUTHOR_TRAILERS)

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
                state.dossier.add("Build recovery", "Agent-assisted fix applied", "Phase 4a")

            with tracker.phase("Phase 4 — Verify (post-recovery)"):
                phase_4_verify(config, state)
                tracker.record_gate("verify", "passed")
        except PipelineEscalation:
            with tracker.phase("Phase 4b — Run-and-fix loop (build-test-fix)"):
                phase_4b_run_and_fix(config, claude_config, state, tracker)

    phase4_lint = _resolve_lint_command(config, state)
    lint_cmds = [c for c in state.triage_assessment.get("lint_commands", []) if _validate_shell_command(c) and c != phase4_lint]
    verification_cmds = list(lint_cmds)
    if config.auto_discover_tests:
        for c in state.triage_assessment.get("verification_commands", []):
            if _validate_shell_command(c):
                verification_cmds.append(c)

    baseline_failures = {e["cmd"]: e["exit_code"] for e in state.baseline_verification if e["exit_code"] != 0}

    if verification_cmds:
        ext_failures = []
        with tracker.phase("Phase 4.1 — Extended verification"):
            for vcmd in verification_cmds:
                log.info("Phase 4.1: running verification command: %s", vcmd)
                try:
                    vresult = subprocess.run(
                        vcmd, shell=True, cwd=config.build_dir,
                        capture_output=True, text=True, timeout=config.command_timeout,
                    )
                except subprocess.TimeoutExpired:
                    # A timeout is never treated as a pre-existing failure — we
                    # cannot conclude anything from a command that didn't finish.
                    log.warning("Phase 4.1: verification timed out — treating as failure: %s", vcmd)
                    ext_failures.append({"cmd": vcmd, "output": f"Timed out after {config.command_timeout}s"})
                    state.dossier.add(f"Extended verification timed out: {vcmd}", f"Timed out after {config.command_timeout}s", "Phase 4.1")
                    continue
                if vresult.returncode == 0:
                    log.info("Phase 4.1: verification passed: %s", vcmd)
                elif vcmd in baseline_failures:
                    log.info("Phase 4.1: verification failed (pre-existing, exit %d): %s", vresult.returncode, vcmd)
                else:
                    output = (vresult.stdout + "\n" + vresult.stderr)[-3000:]
                    log.warning("Phase 4.1: verification failed (NEW, exit %d): %s", vresult.returncode, vcmd)
                    ext_failures.append({"cmd": vcmd, "output": output})
                    state.dossier.add(
                        f"Extended verification failed: {vcmd}",
                        output,
                        "Phase 4.1",
                    )

        if ext_failures:
            log.info("Phase 4.1: %d verification commands failed — triggering Phase 4a recovery", len(ext_failures))
            try:
                with tracker.phase("Phase 4a — Build error recovery (from 4.1)"):
                    phase_4a_build_recovery(config, claude_config, state, tracker, fix_diff)
                    state.dossier.add("Build recovery (from extended verification)", "Agent-assisted fix applied", "Phase 4a")

                with tracker.phase("Phase 4.1 — Extended verification (post-recovery)"):
                    still_failing = []
                    for vcmd in verification_cmds:
                        try:
                            vresult = subprocess.run(
                                vcmd, shell=True, cwd=config.build_dir,
                                capture_output=True, text=True, timeout=config.command_timeout,
                            )
                        except subprocess.TimeoutExpired:
                            still_failing.append(vcmd)
                            continue
                        if vresult.returncode != 0 and vcmd not in baseline_failures:
                            still_failing.append(vcmd)
                    if still_failing:
                        log.warning("Phase 4.1: %d commands still failing after recovery: %s",
                                    len(still_failing), ", ".join(still_failing))
                        state.dossier.add(
                            "Extended verification still failing after Phase 4a",
                            "\n".join(still_failing),
                            "Phase 4.1 (post-recovery)",
                        )
                        raise PipelineEscalation(
                            f"Extended verification still failing: {', '.join(still_failing)}"
                        )
                    else:
                        log.info("Phase 4.1: all verification commands pass after recovery")
            except PipelineEscalation as e:
                log.warning("Phase 4a recovery failed — escalating to run-and-fix loop: %s", e)

                with tracker.phase("Phase 4b — Run-and-fix loop (build-test-fix)"):
                    phase_4b_run_and_fix(config, claude_config, state, tracker)

    with tracker.phase("Phase 4.5 — Semantic equivalence"):
        eq_result = phase_4_5_semantic_equivalence(config, state)
        state.dossier.add("Semantic equivalence", eq_result, "Phase 4.5 range-diff")
        tracker.record_gate("semantic_equivalence", eq_result)

        try:
            squashed = _squashed_backport_range(config)
            rd_output = run_range_diff(
                config.repo_path,
                f"{config.source_fix_commit}^..{config.source_fix_commit}",
                squashed or "HEAD^..HEAD",
                creation_factor=999 if squashed else None,
            )
            if rd_output.strip():
                state.dossier.add(
                    "Range-diff (upstream vs backport)",
                    rd_output[:10000] + ("\n[TRUNCATED]" if len(rd_output) > 10000 else ""),
                    "Phase 4.5 — for reviewer: shows how the backport differs from the original fix",
                )
        except Exception:
            pass

    state.dossier.trailers = build_trailers(
        phase=state.cherry_pick_path,
        model=claude_config.model,
    )

    log.info("Phase 5 — Dossier ready for human review")
    patch_text = git_format_patch(config.repo_path, base_ref=config.target_branch)
    state.dossier.add("Candidate patch", patch_text, "git format-patch")
    os.makedirs(tracker._output_dir, exist_ok=True)
    patch_path = os.path.join(tracker._output_dir, PATCH_FILE)
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)
    log.info("Patch written to %s", patch_path)
    state.dossier.add(
        "Allowlist conformance",
        "\n".join(state.allowed_modules),
        "git diff --name-only",
    )

    tracker.set_outcome("success")
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
            build_command=spec.build_command if spec.build_command is not None else base_config.build_command,
            test_command=spec.test_command if spec.test_command is not None else base_config.test_command,
            lint_command=spec.lint_command if spec.lint_command is not None else base_config.lint_command,
        )
        sanitized = _sanitize_ref(spec.branch)
        target_run_dir = os.path.join(parent_run_dir, sanitized)
        os.makedirs(target_run_dir, exist_ok=True)

        log.info("=== Target %d/%d: %s (issue_id=%s) ===",
                 i, len(targets), spec.branch, target_config.issue_id)

        target_start = time.time()
        worktree_created = False
        worktree_path = os.path.join(workdir, f"{target_config.issue_id}_{sanitized}")

        try:
            target_log_file = open(os.path.join(target_run_dir, "run_log.txt"), "w")
        except OSError as e:
            log.warning("Target %s: cannot open run log: %s", spec.branch, e)
            results.append(TargetResult(
                target_branch=spec.branch,
                issue_id=target_config.issue_id,
                outcome="error",
                duration_s=round(time.time() - target_start, 1),
                cost_usd=0.0,
                dossier_path="",
                tracker_path="",
                error=f"cannot open run log: {e}",
            ))
            continue

        saved_stdout = sys.stdout
        sys.stdout = Tee(saved_stdout, target_log_file)

        outcome = "error"
        error_msg = ""
        dossier_path = ""

        try:
            # --- Worktree setup ---
            cleanup_worktree(source_path, worktree_path)
            if os.path.exists(worktree_path):
                shutil.rmtree(worktree_path)

            print(f"Creating worktree at {worktree_path} on {spec.branch}...")
            wt_result = git_worktree_add(source_path, worktree_path, spec.branch)
            if not wt_result.success:
                print(f"ERROR: git worktree add failed: {wt_result.stderr.strip()}")
                log.warning("Target %s: worktree creation failed: %s", spec.branch, wt_result.stderr)
                results.append(TargetResult(
                    target_branch=spec.branch,
                    issue_id=target_config.issue_id,
                    outcome="worktree_error",
                    duration_s=round(time.time() - target_start, 1),
                    cost_usd=0.0,
                    dossier_path="",
                    tracker_path="",
                    error=f"git worktree add failed: {wt_result.stderr.strip()}",
                ))
                continue
            worktree_created = True
            target_config.repo_path = worktree_path
            target_config.build_dir = os.path.join(worktree_path, build_subdir)
            print("Worktree created successfully")

            # --- Pipeline ---
            try:
                dossier, _ = run_pipeline(target_config, claude_config, output_dir=target_run_dir)
                outcome = "success"
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
                    with open(dp, "w") as f:
                        f.write(dossier_text)
                    dossier_path = os.path.join(sanitized, DOSSIER_FILE)
                    print(f"Dossier written to {dp}")
                except Exception as e:
                    log.warning("Failed to write dossier: %s", e)

            # --- Cost retrieval (unified, both paths) ---
            cost_usd, tracker_json = _read_tracker_cost(target_run_dir, target_config.issue_id)
            tracker_rel = os.path.join(sanitized, os.path.basename(tracker_json)) if tracker_json else ""

            if not dossier_path:
                dp = os.path.join(target_run_dir, DOSSIER_FILE)
                if os.path.exists(dp):
                    dossier_path = os.path.join(sanitized, DOSSIER_FILE)

            duration = round(time.time() - target_start, 1)
            results.append(TargetResult(
                target_branch=spec.branch,
                issue_id=target_config.issue_id,
                outcome=outcome,
                duration_s=duration,
                cost_usd=cost_usd,
                dossier_path=dossier_path,
                tracker_path=tracker_rel,
                error=error_msg,
            ))
            print(f"FINISHED target {spec.branch}: outcome={outcome}, duration={duration}s, cost=${cost_usd:.4f}")

            if outcome == "success" and worktree_created:
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
                        cwd=source_path, capture_output=True, text=True,
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
        log.info("  %s [%s]: %s  (%.1fs, $%.4f)", r.issue_id, r.target_branch, status, r.duration_s, r.cost_usd)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

        targets, base_config, claude_config, workdir, source_path = load_multi_target_config(config_path)

        yaml_basename = os.path.splitext(os.path.basename(config_path))[0]
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        parent_run_dir = os.path.join("runs", f"{yaml_basename}_{run_timestamp}")
        os.makedirs(parent_run_dir, exist_ok=True)

        log_path = os.path.join(parent_run_dir, "run_log.txt")
        log_file = open(log_path, "w")
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        try:
            print(f"Run directory: {parent_run_dir}")

            print("Fetching latest refs from origin...")
            fetch_result = subprocess.run(
                ["git", "fetch", "--all", "--quiet"],
                cwd=source_path, capture_output=True, text=True,
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
            results = run_targets(targets, base_config, claude_config, source_path, workdir, parent_run_dir, build_subdir, run_warnings=run_warnings)
        finally:
            sys.stdout = original_stdout
            log_file.close()

        sys.exit(0 if all(r.outcome == "success" for r in results) else 1)
    else:
        from auto_bug_fix.bug_fix_config import claude_config, bug_fix_config
        run_dir = "runs"
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run_log.txt")
        log_file = open(log_path, "w")
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        start_time = time.time()
        try:
            dossier, tracker = run_pipeline(bug_fix_config, claude_config, output_dir=run_dir)

            dossier_text = format_dossier(dossier)
            dossier_path = os.path.join(run_dir, DOSSIER_FILE)
            with open(dossier_path, "w") as f:
                f.write(dossier_text)
            print(f"Dossier written to {dossier_path}")

            slug = _sanitize_branch_slug(
                bug_fix_config.issue_id
                or f"{bug_fix_config.source_fix_commit[:12]}-to-{_sanitize_ref(bug_fix_config.target_branch)}"
            )
            branch_name = f"amalgamator/{slug}"
            result = subprocess.run(
                ["git", "branch", "-f", branch_name, "HEAD"],
                cwd=bug_fix_config.repo_path, capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"Commits preserved on branch {branch_name}")
            else:
                log.warning("git branch failed: %s", result.stderr.strip())

        except PipelineStop as e:
            print(f"PIPELINE STOP: {e}")
            exit_code = 1
        except PipelineEscalation as e:
            print(f"PIPELINE ESCALATION: {e}")
            exit_code = 1
        except Exception as e:
            print(f"PIPELINE ERROR: {type(e).__name__}: {e}")
            exit_code = 1
        else:
            exit_code = 0
        finally:
            duration = time.time() - start_time
            print(f"FINISHED: total_duration = {duration:.1f}s")
            print(f"All outputs in: {run_dir}")
            if sys.stdout is not original_stdout:
                sys.stdout = original_stdout
            log_file.close()

        sys.exit(exit_code)
