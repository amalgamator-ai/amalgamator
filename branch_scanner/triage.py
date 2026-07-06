"""
branch_scanner/triage.py — shared branch triage logic.

Single check_branch() used by both:
  - branch_scanner: loops over all branches to find which need a fix
  - auto_bug_fix Phase 0: checks one branch before attempting a backport

Detection cascade (stops at first conclusive result):
  1. git_is_ancestor        — exact commit already on branch
  2. forward_patch_id_check — equivalent patch (different SHA, same content)
  3. backward_patch_id_check— branch predates the bug (not affected, skip)
  4. file diff heuristic    — 85%+ of fix lines present in branch files
  5. LLM semantic check     — resolves AMBIGUOUS cases (optional, caller-driven)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Reuse existing detection functions from the pipeline — no reimplementation
from auto_bug_fix.git_tools import git_is_ancestor, git_cat_file_exists
from auto_bug_fix.patch_id import forward_patch_id_check, backward_patch_id_check

log = logging.getLogger(__name__)


class FixStatus(str, Enum):
    EXACT_MATCH    = "exact_match"       # same commit SHA on branch
    EQUIVALENT     = "equivalent_patch"  # same patch content, different SHA
    NOT_AFFECTED   = "not_affected"      # branch predates the bug — no fix needed
    LIKELY_PRESENT = "likely_present"    # file diff heuristic says fix is there
    NEEDS_FIX      = "needs_fix"         # fix is absent — backport needed
    AMBIGUOUS      = "ambiguous"         # inconclusive — enable LLM to resolve
    ERROR          = "error"             # git command failed


@dataclass
class BranchCheckResult:
    branch: str
    status: FixStatus
    method: str                # which level detected this
    detail: str                # human-readable explanation
    changed_files: list[str]   # files touched by the fix commit


# ── Helpers not already in the pipeline ───────────────────────────────────────

def _run(cmd: list[str], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def _get_changed_files(repo_path: Path, commit: str) -> list[str]:
    """Files touched by a commit."""
    rc, out, _ = _run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit],
        cwd=repo_path,
    )
    return [f for f in out.splitlines() if f] if rc == 0 else []


def _get_remote_branches(repo_path: Path, pattern: str | None = None) -> list[str]:
    """List remote branches, optionally filtered by a glob pattern."""
    import fnmatch
    rc, out, _ = _run(["git", "branch", "-r", "--format=%(refname:short)"], cwd=repo_path)
    if rc != 0:
        return []
    branches = []
    for line in out.splitlines():
        name = line.strip().removeprefix("origin/")
        if name in ("HEAD", ""):
            continue
        branches.append(name)
    if pattern:
        branches = [b for b in branches if fnmatch.fnmatch(b, pattern)]
    return sorted(set(branches))


def _check_file_diff(
    repo_path: Path, branch: str, fix_commit: str, changed_files: list[str]
) -> FixStatus:
    """
    Level 4: file diff heuristic.
    Compare lines added by the fix against the current branch file content.
    85%+ match → LIKELY_PRESENT, ≤15% → NEEDS_FIX, middle → AMBIGUOUS.
    """
    if not changed_files:
        return FixStatus.AMBIGUOUS

    rc, fix_diff, _ = _run(
        ["git", "show", "--unified=0", fix_commit, "--", *changed_files],
        cwd=repo_path,
    )
    if rc != 0:
        return FixStatus.AMBIGUOUS

    added_lines = [
        line[1:] for line in fix_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not added_lines:
        return FixStatus.AMBIGUOUS

    rc, branch_content, _ = _run(
        ["git", "show", f"origin/{branch}:{changed_files[0]}"],
        cwd=repo_path,
    )
    if rc != 0:
        return FixStatus.AMBIGUOUS

    matched = sum(1 for line in added_lines if line.strip() in branch_content)
    ratio = matched / len(added_lines)

    if ratio >= 0.85:
        return FixStatus.LIKELY_PRESENT
    elif ratio <= 0.15:
        return FixStatus.NEEDS_FIX
    return FixStatus.AMBIGUOUS


# ── Main entry point ──────────────────────────────────────────────────────────

def check_branch(
    repo_path: Path,
    branch: str,
    fix_commit: str,
    vuln_commit: str | None = None,
) -> BranchCheckResult:
    """
    Full detection cascade for one branch.

    Args:
        repo_path:   Local path to the git repository.
        branch:      Short branch name (without 'origin/').
        fix_commit:  SHA of the commit that introduced the fix on main.
        vuln_commit: SHA of the commit that introduced the bug (optional).
                     When provided, branches that predate this commit are
                     marked NOT_AFFECTED instead of NEEDS_FIX.
    """
    repo = str(repo_path)
    target = f"origin/{branch}"
    changed_files = _get_changed_files(repo_path, fix_commit)

    # Level 1: exact commit — reuses git_is_ancestor from auto_bug_fix
    if git_is_ancestor(repo, fix_commit, target):
        return BranchCheckResult(
            branch=branch,
            status=FixStatus.EXACT_MATCH,
            method="git_is_ancestor",
            detail=f"Commit {fix_commit[:10]} is a direct ancestor of {target}",
            changed_files=changed_files,
        )

    # Level 2: equivalent patch — reuses forward_patch_id_check from auto_bug_fix
    match = forward_patch_id_check(repo, fix_commit, target)
    if match:
        return BranchCheckResult(
            branch=branch,
            status=FixStatus.EQUIVALENT,
            method="forward_patch_id_check",
            detail=f"Equivalent patch already applied on {target} as {match[:10]}",
            changed_files=changed_files,
        )

    # Level 3: backward patch-id — reuses backward_patch_id_check from auto_bug_fix
    # If vuln_commit is not on the branch, the branch predates the bug entirely.
    if vuln_commit:
        vuln_on_branch = backward_patch_id_check(repo, vuln_commit, target)
        seed_exists = any(
            git_cat_file_exists(repo, target, f) for f in changed_files[:3]
        )
        if not vuln_on_branch and not seed_exists:
            return BranchCheckResult(
                branch=branch,
                status=FixStatus.NOT_AFFECTED,
                method="backward_patch_id_check",
                detail=f"Branch {target} predates the bug — no fix needed",
                changed_files=changed_files,
            )

    # Level 4: file diff heuristic (new — not in pipeline Phase 0)
    file_status = _check_file_diff(repo_path, branch, fix_commit, changed_files)
    detail_map = {
        FixStatus.LIKELY_PRESENT: "85%+ of fix lines found in branch files",
        FixStatus.NEEDS_FIX:      "Fix lines not found in branch files",
        FixStatus.AMBIGUOUS:      "Inconclusive — enable llm.enabled: true to resolve",
    }
    return BranchCheckResult(
        branch=branch,
        status=file_status,
        method="file diff heuristic",
        detail=detail_map.get(file_status, ""),
        changed_files=changed_files,
    )
