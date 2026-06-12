"""Phase 0-5 orchestrator for the auto_bug_fix pipeline.

Usage:
    Edit ``auto_bug_fix/bug_fix_config.py`` to configure your project, then run:

        python -m auto_bug_fix.run_bug_fix
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any

from common.utils import Tee
from common.claude_utils import ClaudeConfig, claude_run as _claude_run


async def _run_llm(claude_config, prompts, tracker=None):
    """Compatibility wrapper: call claude_run and feed timings into tracker."""
    import time
    batch_start = time.time()
    step_timings = await _claude_run(claude_config, prompts)

    if tracker and step_timings:
        cursor = batch_start
        for st in step_timings:
            dur = st.get("duration", 0)
            tracker.record_query(
                prompt_name=st.get("name", "query"),
                start_time=cursor,
                end_time=cursor + dur,
                input_tokens=st.get("input_tokens", 0),
                output_tokens=st.get("output_tokens", 0),
            )
            cursor += dur

    return step_timings


claude_run = _run_llm

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.git_tools import (
    git_show_files,
    git_cherry_pick,
    git_cherry_pick_abort,
    git_status_porcelain,
    git_diff_name_only,
    git_format_patch,
    git_merge_base,
    git_reset_hard,
    git_commit,
    git_rev_parse,
)
from auto_bug_fix.patch_id import forward_patch_id_check, backward_patch_id_check
from auto_bug_fix.signature import capture_output, normalize_signature
from auto_bug_fix.allowlist import derive_seed, resolve_renames, enforce_allowlist, enforce_test_file_veto
from auto_bug_fix.baseline import capture_baseline, check_regression, run_test_suite
from auto_bug_fix.range_diff import run_range_diff, parse_equivalence
from auto_bug_fix.dossier import Dossier, build_trailers, format_dossier
from auto_bug_fix.tracker import Tracker
from auto_bug_fix.bug_fix_prompts import (
    create_context_str,
    SemanticTriagePrompt,
    TestPortAgentPrompt,
    NarrowResolutionAgentPrompt,
    BuildErrorRecoveryPrompt,
    SEMANTIC_TRIAGE_FILE,
    TEST_PORT_MANIFEST_FILE,
)

LOG_FILE = "__run_log_bug_fix.txt"
DOSSIER_FILE = "dossier.md"

log = logging.getLogger(__name__)


class PipelineEscalation(Exception):
    pass


class PipelineStop(Exception):
    pass


_CONFLICT_PREFIXES = ("UU ", "AA ", "DU ", "UD ", "DD ", "AU ", "UA ")

_DANGEROUS_SHELL_PATTERNS = (";", "&&", "||", "|", "`", "$(", "${", ">", "<", "\n")


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
    s_target: str = ""
    bisect_sha: str | None = None
    cherry_pick_path: str = ""
    dossier: Dossier | None = None
    triage_assessment: dict = field(default_factory=dict)
    priority_files: list[str] = field(default_factory=list)


def phase_0_triage(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
) -> dict[str, str]:
    """Run deterministic triage gates (patch-id, ancestry, seed files) and derive the allowlist."""
    repo = config.repo_path
    fix = config.source_fix_commit
    target = config.target_branch

    state.seed = derive_seed(repo, fix)

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

    from auto_bug_fix.git_tools import git_is_ancestor, git_cat_file_exists
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

    fix_diff = subprocess.run(
        ["git", "show", sha],
        cwd=config.repo_path, capture_output=True, text=True,
    ).stdout

    MAX_DIFF_CHARS = 200_000
    if len(fix_diff) > MAX_DIFF_CHARS:
        log.warning("fix_diff too large (%d chars), using --stat only", len(fix_diff))
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
                capture_output=True, text=True, timeout=1800,
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
                    capture_output=True, text=True, timeout=300,
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
    import json as _json

    fix_diff = subprocess.run(
        ["git", "show", "--stat", config.source_fix_commit],
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
                assessment = _json.load(f)
            log.info("Semantic triage: difficulty=%s, recommendation=%s",
                     assessment.get("estimated_difficulty", "?"),
                     assessment.get("recommendation", "?"))
        except (_json.JSONDecodeError, OSError) as e:
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
    test_files = identify_test_files(state.seed)
    if not test_files:
        log.info("Phase 0.5c: no test files found in fix commit — skipping test port")
        return []

    log.info("Phase 0.5c: found %d test files in fix commit: %s",
             len(test_files), ", ".join(test_files))

    manifest_abs = os.path.join(config.repo_path, TEST_PORT_MANIFEST_FILE)
    context = create_context_str(claude_config, config)
    prompt = TestPortAgentPrompt(
        context=context,
        source_fix_commit=config.source_fix_commit,
        target_branch=config.target_branch,
        test_files=test_files,
        output_manifest_file=manifest_abs,
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


def phase_1_baseline_red_bisect(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
) -> None:
    """Capture test baseline and run RED check in a single test suite invocation."""
    repo = config.repo_path

    red_exit, red_output = capture_output(
        [config.test_command], config.build_dir,
    )
    from auto_bug_fix.baseline import parse_test_failures
    state.baseline = parse_test_failures(red_output, "")
    is_bugfix = state.triage_assessment.get("commit_type", "").lower() in ("bugfix", "bug_fix", "fix", "security")
    if red_exit == 0:
        if state.ported_test_files and is_bugfix:
            raise PipelineEscalation("RED check passed — target may already be fixed or test is inapplicable")
        log.info("Baseline tests pass — %s",
                 "feature/refactor commit, RED check not applicable" if not is_bugfix
                 else "no ported vulnerability test available — proceeding")
    else:
        state.s_target = normalize_signature(red_output)

    # Positive control and bisect are documented in UNUSED_PIPELINE_CODE.md.
    # They require fixture_cache which is not yet wired to Phase 0.5c test port.


def phase_1_5_failure_mode(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
) -> str:
    """Confirm the failure mode matches via signature comparison.

    Currently always returns "skip" — requires a parent signature source
    (e.g. from a reproducer or upstream test) to be wired in.
    """
    if not state.s_target:
        return "skip"
    return "skip"



MAX_ERROR_CHARS = 50_000


def _run_build_and_test(config: BugFixConfig, state: PipelineState) -> tuple[bool, str, str, str, str]:
    """Run build and test commands, return (passed, build_out, build_err, test_out, test_err)."""
    build_exit, build_stdout, build_stderr = run_test_suite(config.build_command, config.build_dir)
    if build_exit != 0:
        return False, build_stdout, build_stderr, "", ""
    test_exit, test_stdout, test_stderr = run_test_suite(config.test_command, config.build_dir)
    from auto_bug_fix.baseline import parse_test_failures
    failures = parse_test_failures(test_stdout, test_stderr)
    passed, _ = check_regression(failures, state.baseline)

    if passed:
        repo = config.repo_path
        ok, extra_files = enforce_allowlist(repo, state.allowed_modules, config.disallowed_modules)
        if not ok:
            log.info("Files modified outside priority list (advisory): %s", extra_files)
        test_ok, vetoed = enforce_test_file_veto(repo, state.seed)
        if not test_ok:
            log.info("Test files modified outside seed (advisory): %s", vetoed)
        return True, build_stdout, build_stderr, test_stdout, test_stderr

    return False, build_stdout, build_stderr, test_stdout, test_stderr


def phase_4_verify(
    config: BugFixConfig,
    state: PipelineState,
) -> bool:
    """Build, test, and check allowlist conformance (single attempt)."""
    passed, _, _, _, _ = _run_build_and_test(config, state)
    if passed:
        return True
    raise PipelineEscalation("Phase 4 verify failed")


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
        build_exit, build_stdout, build_stderr = run_test_suite(config.build_command, config.build_dir)
        if build_exit == 0:
            test_exit, test_stdout, test_stderr = run_test_suite(config.test_command, config.build_dir)
            from auto_bug_fix.baseline import parse_test_failures
            failures = parse_test_failures(test_stdout, test_stderr)
            build_test_passed, _ = check_regression(failures, state.baseline)
            if build_test_passed:
                log.info("Phase 4a: build+test passed on attempt %d", attempt)
                return
        else:
            test_stdout, test_stderr = "", ""

        build_errors = (build_stdout + "\n" + build_stderr).strip()
        test_errors = (test_stdout + "\n" + test_stderr).strip()

        if len(build_errors) > MAX_ERROR_CHARS:
            build_errors = build_errors[:MAX_ERROR_CHARS] + "\n\n[TRUNCATED]"
        if len(test_errors) > MAX_ERROR_CHARS:
            test_errors = test_errors[:MAX_ERROR_CHARS] + "\n\n[TRUNCATED]"

        current_diff = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=config.repo_path, capture_output=True, text=True,
        ).stdout

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

    build_exit, _, _ = run_test_suite(config.build_command, config.build_dir)
    if build_exit == 0:
        _, test_stdout, test_stderr = run_test_suite(config.test_command, config.build_dir)
        from auto_bug_fix.baseline import parse_test_failures
        failures = parse_test_failures(test_stdout, test_stderr)
        final_passed, _ = check_regression(failures, state.baseline)
        if final_passed:
            log.info("Phase 4a: build+test passed after final attempt")
            return

    raise PipelineEscalation(
        f"Phase 4a: build/test still failing after {config.max_build_test_retries} recovery attempts"
    )


def phase_4b_external_pipeline(
    config: BugFixConfig,
    claude_config: ClaudeConfig,
    state: PipelineState,
    tracker,
) -> None:
    """Hand off to the external new_ai_auto_analyze pipeline's autonomous build-test-fix loop.

    Writes a rich context file with ALL pipeline data (dossier, triage, fix_diff,
    prerequisite analysis) then calls the external pipeline's RunAndFixPrompt which
    drives an LLM-heavy iterative fix loop.
    """
    import json as _json
    ext_pipeline_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "new_ai_auto_analyze")
    ext_pipeline_root = os.path.abspath(ext_pipeline_root)

    if not os.path.isdir(ext_pipeline_root):
        raise FileNotFoundError(f"External pipeline not found at {ext_pipeline_root}")

    import sys
    if ext_pipeline_root not in sys.path:
        sys.path.insert(0, ext_pipeline_root)

    from auto_code_gen.use_cases.bug_fix import gen_RunAndFixPrompt, RUN_AND_FIX_FAILURE_FILE
    from common.claude_utils import PipelineStep
    from common.claude_utils import claude_run as ext_claude_run
    from common.claude_utils import ClaudeConfig as ExtClaudeConfig

    dossier_text = format_dossier(state.dossier) if state.dossier else ""
    triage_json = _json.dumps(state.triage_assessment, indent=2) if state.triage_assessment else "{}"

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

    ext_claude_config = ExtClaudeConfig(
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

    log.info("Phase 4b: handing off to external pipeline (build-test-fix loop)")
    import time as _time
    _4b_start = _time.time()
    timings = asyncio.run(ext_claude_run(ext_claude_config, steps))
    _4b_end = _time.time()

    if tracker and timings:
        for st in timings:
            dur = st.get("duration", _4b_end - _4b_start)
            tracker.record_query(
                prompt_name=st.get("name", "phase_4b"),
                start_time=_4b_start,
                end_time=_4b_start + dur,
                input_tokens=st.get("input_tokens", 0),
                output_tokens=st.get("output_tokens", 0),
            )

    failure_file = os.path.join(config.repo_path, RUN_AND_FIX_FAILURE_FILE)
    if os.path.exists(failure_file):
        with open(failure_file) as f:
            failure_text = f.read()
        state.dossier.add("External pipeline failure", failure_text, "Phase 4b")
        raise PipelineEscalation("Phase 4b: external pipeline exhausted retries")
    else:
        state.dossier.add("External pipeline", "Build-test-fix loop succeeded", "Phase 4b")
        log.info("Phase 4b: external pipeline succeeded")


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
    """Cherry-pick a single commit and resolve conflicts. Returns 'clean', 'resolved', or raises."""
    from auto_bug_fix.git_tools import git_cherry_pick, git_cherry_pick_abort, resolve_merge_commit

    commit = resolve_merge_commit(config.repo_path, commit)
    result = git_cherry_pick(config.repo_path, commit)
    status = git_status_porcelain(config.repo_path)
    uu_files = _conflicted_files(status)

    if result.success and not uu_files:
        log.info("%s: clean cherry-pick", label)
        return "clean"

    if not uu_files:
        git_cherry_pick_abort(config.repo_path)
        raise PipelineEscalation(f"{label}: unmappable cherry-pick")

    conflict_summary_parts = []
    skip_set = set(files_to_skip or [])
    if skip_set:
        auto_resolved = [f for f in uu_files if f in skip_set]
        if auto_resolved:
            log.info("%s: auto-resolving %d generated/binary files", label, len(auto_resolved))
            for f in auto_resolved:
                subprocess.run(["git", "checkout", "--theirs", "--", f],
                               cwd=config.repo_path, capture_output=True)
                subprocess.run(["git", "add", "--", f],
                               cwd=config.repo_path, capture_output=True)
            conflict_summary_parts.append(
                f"Auto-resolved {len(auto_resolved)} generated/binary files (--theirs)"
            )
            uu_files = [f for f in uu_files if f not in skip_set]

    if not uu_files:
        msg = f"Port {label}\n\n[ Conflict summary: {'; '.join(conflict_summary_parts)} ]"
        git_commit(config.repo_path, msg)
        return "resolved"

    original_uu_files = list(uu_files)
    fix_diff = _compute_fix_diff(config, commit)
    context = create_context_str(claude_config, config)
    priority = state.priority_files if state.priority_files else state.allowed_modules

    claude_config_resolve = ClaudeConfig(
        model=claude_config.model,
        allowed_tools=claude_config.allowed_tools,
        perm_mode=claude_config.perm_mode,
        cwd=config.repo_path,
    )

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
                )
                asyncio.run(claude_run(claude_config_resolve, [prompt.prompt()], tracker=tracker))
        else:
            prompt = NarrowResolutionAgentPrompt(
                fix_diff=fix_diff,
                conflicted_files=uu_files,
                context=context,
                allowed_modules=priority,
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
            summary = "; ".join(conflict_summary_parts)
            msg = f"Port {label}\n\n[ Conflict summary: {summary} ]"
            git_commit(config.repo_path, msg)
            return "resolved"
        log.warning("%s: attempt %d: %d conflicts remain", label, attempt, len(still_conflicted))
        uu_files = _conflicted_files(still_conflicted)

    from auto_bug_fix.git_tools import git_cherry_pick_abort as _abort
    _abort(config.repo_path)
    raise PipelineEscalation(f"{label}: failed after {config.max_resolution_retries} attempts")


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


def phase_4_5_semantic_equivalence(
    config: BugFixConfig,
    state: PipelineState,
) -> str:
    """Run range-diff to classify the port as identical, modified, or unmatched."""
    repo = config.repo_path
    try:
        target_base = git_merge_base(repo, config.target_branch, "HEAD")
        rd_output = run_range_diff(
            repo,
            f"{config.source_fix_commit}^..{config.source_fix_commit}",
            f"{target_base}..HEAD",
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

    from dataclasses import asdict
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
    except Exception as e:
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

    with tracker.phase("Phase 0.5 — Semantic triage"):
        import json as _json
        assessment = phase_0_5_semantic_triage(config, claude_config, state, tracker)
        state.dossier.add("Semantic triage", _json.dumps(assessment, indent=2) if assessment else "no assessment", "Phase 0.5")
        tracker.record_gate("difficulty", assessment.get("estimated_difficulty", "unknown"))
        tracker.record_gate("recommendation", assessment.get("recommendation", "unknown"))

    cherry_pick_succeeded = False
    try:
        prereqs = state.triage_assessment.get("prerequisites", [])
        portable_prereqs = [p for p in prereqs if p.get("portable", False) and p.get("introduced_by")]
        if portable_prereqs:
            with tracker.phase("Phase 0.5b — Prerequisite chain"):
                ported = port_prerequisite_chain(config, claude_config, state, tracker)
                state.dossier.add(
                    "Prerequisites ported",
                    _json.dumps(ported, indent=2) if ported else "none",
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

        with tracker.phase("Phase 1 — Baseline + RED + Bisect"):
            phase_1_baseline_red_bisect(config, claude_config, state)
            if state.bisect_sha:
                state.dossier.add("Bisect result", state.bisect_sha, "Phase 1 git bisect")
                tracker.record_gate("bisect", state.bisect_sha)

        with tracker.phase("Phase 1.5 — Failure-mode confirmation"):
            sig_result = phase_1_5_failure_mode(config, claude_config, state)
            state.dossier.add("Signature comparison", sig_result, "Phase 1.5")
            tracker.record_gate("signature_comparison", sig_result)

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

    except (PipelineEscalation, Exception) as e:
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
                       f"[ {len(remaining_conflicts)} conflicts force-resolved with --theirs ]")
        else:
            cherry_pick_in_progress = subprocess.run(
                ["git", "rev-parse", "--verify", "CHERRY_PICK_HEAD"],
                cwd=config.repo_path, capture_output=True).returncode == 0
            if cherry_pick_in_progress:
                subprocess.run(["git", "cherry-pick", "--abort"], cwd=config.repo_path,
                                capture_output=True)
            git_reset_hard(config.repo_path, config.target_branch)
            result = git_cherry_pick(config.repo_path, config.source_fix_commit)
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
                       f"[ All conflicts force-resolved with --theirs ]")

        with tracker.phase("Phase 4b — External pipeline (build-test-fix loop)"):
            phase_4b_external_pipeline(config, claude_config, state, tracker)
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
            with tracker.phase("Phase 4b — External pipeline (build-test-fix loop)"):
                phase_4b_external_pipeline(config, claude_config, state, tracker)

    verification_cmds = state.triage_assessment.get("verification_commands", [])
    verification_cmds = [c for c in verification_cmds if _validate_shell_command(c)]
    if verification_cmds:
        ext_failures = []
        with tracker.phase("Phase 4.1 — Extended verification"):
            for vcmd in verification_cmds:
                log.info("Phase 4.1: running verification command: %s", vcmd)
                try:
                    vresult = subprocess.run(
                        vcmd, shell=True, cwd=config.build_dir,
                        capture_output=True, text=True, timeout=600,
                    )
                except subprocess.TimeoutExpired:
                    log.warning("Phase 4.1: verification timed out (10min): %s", vcmd)
                    ext_failures.append({"cmd": vcmd, "output": "Timed out after 10 minutes"})
                    state.dossier.add(f"Extended verification timed out: {vcmd}", "Timed out after 10 minutes", "Phase 4.1")
                    continue
                if vresult.returncode == 0:
                    log.info("Phase 4.1: verification passed: %s", vcmd)
                else:
                    output = (vresult.stdout + "\n" + vresult.stderr)[-3000:]
                    log.warning("Phase 4.1: verification failed (exit %d): %s", vresult.returncode, vcmd)
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
                                capture_output=True, text=True, timeout=600,
                            )
                        except subprocess.TimeoutExpired:
                            still_failing.append(vcmd)
                            continue
                        if vresult.returncode != 0:
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
                log.warning("Phase 4a recovery failed — escalating to external pipeline: %s", e)

                with tracker.phase("Phase 4b — External pipeline (build-test-fix loop)"):
                    phase_4b_external_pipeline(config, claude_config, state, tracker)

    with tracker.phase("Phase 4.5 — Semantic equivalence"):
        eq_result = phase_4_5_semantic_equivalence(config, state)
        state.dossier.add("Semantic equivalence", eq_result, "Phase 4.5 range-diff")
        tracker.record_gate("semantic_equivalence", eq_result)

        try:
            rd_output = run_range_diff(
                config.repo_path,
                f"{config.source_fix_commit}^..{config.source_fix_commit}",
                "HEAD^..HEAD",
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
        bisect_sha=state.bisect_sha,
    )

    log.info("Phase 5 — Dossier ready for human review")
    state.dossier.add("Candidate patch", git_format_patch(config.repo_path), "git format-patch")
    state.dossier.add(
        "Allowlist conformance",
        "\n".join(state.allowed_modules),
        "git diff --name-only",
    )

    tracker.set_outcome("success")
    return state.dossier, tracker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load config from YAML if provided, otherwise use defaults
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        from auto_bug_fix.config_loader import load_pipeline_config
        from datetime import datetime, timezone
        import shutil

        bug_fix_config, claude_config, workdir, source_path = load_pipeline_config(config_path)

        # Create run directory: <yaml_basename>_<datetime>/
        yaml_basename = os.path.splitext(os.path.basename(config_path))[0]
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"{yaml_basename}_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Redirect stdout to run log
        log_path = os.path.join(run_dir, "run_log.txt")
        log_file = open(log_path, "w")
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        print(f"Run directory: {run_dir}")

        # Set up worktree for this CVE (avoids copying multi-GB repos)
        os.makedirs(workdir, exist_ok=True)
        worktree_path = os.path.join(workdir, bug_fix_config.issue_id)

        if os.path.exists(worktree_path):
            print(f"Worktree already exists at {worktree_path}, removing...")
            from auto_bug_fix.worktree import cleanup_worktree
            cleanup_worktree(source_path, worktree_path)
            if os.path.exists(worktree_path):
                shutil.rmtree(worktree_path)

        target_branch = bug_fix_config.target_branch
        print(f"Creating worktree at {worktree_path} on {target_branch}...")
        from auto_bug_fix.git_tools import git_worktree_add
        result = git_worktree_add(source_path, worktree_path, target_branch)
        if not result.success:
            print(f"ERROR: git worktree add failed: {result.stderr}")
            sys.exit(1)
        print(f"Worktree created successfully")

        build_subdir = os.path.relpath(bug_fix_config.build_dir, bug_fix_config.repo_path)
        bug_fix_config.repo_path = worktree_path
        bug_fix_config.build_dir = os.path.join(worktree_path, build_subdir)
    else:
        from auto_bug_fix.bug_fix_config import claude_config, bug_fix_config
        run_dir = "runs"
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run_log.txt")
        log_file = open(log_path, "w")
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

    # Override tracker output_dir to use run_dir
    start_time = time.time()
    try:
        dossier, tracker = run_pipeline(bug_fix_config, claude_config, output_dir=run_dir)

        # Save dossier to run directory
        dossier_text = format_dossier(dossier)
        dossier_path = os.path.join(run_dir, DOSSIER_FILE)
        with open(dossier_path, "w") as f:
            f.write(dossier_text)
        print(f"Dossier written to {dossier_path}")

        # Copy semantic triage if it exists
        triage_src = os.path.join(bug_fix_config.repo_path, "semantic_triage.json")
        if os.path.exists(triage_src):
            import shutil
            shutil.copy2(triage_src, os.path.join(run_dir, "semantic_triage.json"))

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
