# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
backport_scanner/triage.py — shared ref triage logic.

check_ref() used by backport_scanner to loop over all refs (branches AND
tags) and determine which need a fix backported.

Detection cascade (stops at first conclusive result):
  1. git_is_ancestor        — exact commit already reachable from ref
  2. forward_patch_id_check — equivalent patch (different SHA, same content)
  3. backward_patch_id_check— ref predates the bug (not affected, skip)
  4. file diff heuristic    — 85%+ of fix lines present in ref's files
  5. LLM semantic check     — resolves AMBIGUOUS cases (optional, caller-driven)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# Reuse existing detection functions from the pipeline — no reimplementation
from auto_bug_fix.git_tools import git_cat_file_exists, git_is_ancestor, git_show_files
from auto_bug_fix.patch_id import backward_patch_id_check, forward_patch_id_check

log = logging.getLogger(__name__)


class FixStatus(StrEnum):
    EXACT_MATCH = "exact_match"  # same commit SHA on branch
    EQUIVALENT = "equivalent_patch"  # same patch content, different SHA
    NOT_AFFECTED = "not_affected"  # branch predates the bug — no fix needed
    LIKELY_PRESENT = "likely_present"  # file diff heuristic says fix is there
    NEEDS_FIX = "needs_fix"  # fix is absent — backport needed
    AMBIGUOUS = "ambiguous"  # inconclusive — LLM will attempt to resolve
    ERROR = "error"  # git command failed


@dataclass
class BranchCheckResult:
    branch: str
    status: FixStatus
    method: str  # which level detected this
    detail: str  # human-readable explanation
    changed_files: list[str]  # files touched by the fix commit
    ref_type: str = "branch"  # "branch" or "tag"


# ── Helpers not already in the pipeline ───────────────────────────────────────


def _run(cmd: list[str], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def get_remote_branches(repo_path: Path, pattern: str | None = None) -> list[str]:
    """List remote branches, optionally filtered by a glob pattern."""
    import fnmatch

    rc, out, _ = _run(
        ["git", "branch", "-r", "--format=%(refname:short)"], cwd=repo_path
    )
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


def get_tags(repo_path: Path, pattern: str | None = None) -> list[str]:
    """List tags (local and fetched), optionally filtered by a glob pattern."""
    import fnmatch

    rc, out, _ = _run(["git", "tag", "--list"], cwd=repo_path)
    if rc != 0:
        return []
    tags = [line.strip() for line in out.splitlines() if line.strip()]
    if pattern:
        tags = [t for t in tags if fnmatch.fnmatch(t, pattern)]
    return sorted(set(tags))


def _resolve_ref(name: str, ref_type: str) -> str:
    """Return the git ref string for a branch or tag name."""
    if ref_type == "tag":
        return f"refs/tags/{name}"
    return f"origin/{name}"


def _check_file_diff(
    repo_path: Path,
    name: str,
    fix_commit: str,
    changed_files: list[str],
    ref_type: str = "branch",
) -> FixStatus:
    """
    Level 4: file diff heuristic.
    Compare lines added by the fix against the current ref's file content.
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
        line[1:]
        for line in fix_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    if not added_lines:
        return FixStatus.AMBIGUOUS

    ref = _resolve_ref(name, ref_type)
    content_all = ""
    files_found = 0
    for f in changed_files:
        rc, content, _ = _run(
            ["git", "show", f"{ref}:{f}"],
            cwd=repo_path,
        )
        if rc == 0:
            files_found += 1
            content_all += content

    if files_found == 0:
        return FixStatus.AMBIGUOUS

    matched = sum(1 for line in added_lines if line.strip() in content_all)
    ratio = matched / len(added_lines)

    if ratio >= 0.85:
        return FixStatus.LIKELY_PRESENT
    elif ratio <= 0.15:
        return FixStatus.NEEDS_FIX
    return FixStatus.AMBIGUOUS


# ── Main entry point ──────────────────────────────────────────────────────────


def check_ref(
    repo_path: Path,
    name: str,
    fix_commit: str,
    vuln_commit: str | None = None,
    ref_type: str = "branch",
) -> BranchCheckResult:
    """
    Full detection cascade for one git ref (branch or tag).

    Args:
        repo_path:   Local path to the git repository.
        name:        Short ref name — branch name (without 'origin/') or tag name.
        fix_commit:  SHA of the commit that introduced the fix on main.
        vuln_commit: SHA of the commit that introduced the bug (optional).
                     When provided, refs that predate this commit are
                     marked NOT_AFFECTED instead of NEEDS_FIX.
        ref_type:    "branch" or "tag". Tags are resolved directly; branches
                     are prefixed with 'origin/'.
    """
    repo = str(repo_path)
    target = _resolve_ref(name, ref_type)
    changed_files = git_show_files(repo, fix_commit)

    # Validate that the ref actually exists before proceeding
    rc, _, _ = _run(
        ["git", "rev-parse", "--verify", "--quiet", f"{target}^{{commit}}"],
        cwd=repo_path,
    )
    if rc != 0:
        return BranchCheckResult(
            branch=name,
            status=FixStatus.ERROR,
            method="ref validation",
            detail=f"Ref {target} does not exist or cannot be resolved",
            changed_files=changed_files,
            ref_type=ref_type,
        )

    # Level 1: exact commit — reuses git_is_ancestor from auto_bug_fix
    if git_is_ancestor(repo, fix_commit, target):
        return BranchCheckResult(
            branch=name,
            status=FixStatus.EXACT_MATCH,
            method="git_is_ancestor",
            detail=f"Commit {fix_commit[:10]} is reachable from {target}",
            changed_files=changed_files,
            ref_type=ref_type,
        )

    # Level 2: equivalent patch — reuses forward_patch_id_check from auto_bug_fix
    try:
        match = forward_patch_id_check(repo, fix_commit, target)
    except Exception as e:
        log.debug(
            "forward_patch_id_check failed for %s: %s — skipping to Level 4", name, e
        )
        match = None
    if match:
        return BranchCheckResult(
            branch=name,
            status=FixStatus.EQUIVALENT,
            method="forward_patch_id_check",
            detail=f"Equivalent patch already applied on {target} as {match[:10]}",
            changed_files=changed_files,
            ref_type=ref_type,
        )

    # Level 3: backward patch-id — reuses backward_patch_id_check from auto_bug_fix
    if vuln_commit:
        try:
            vuln_on_ref = backward_patch_id_check(repo, vuln_commit, target)
        except Exception as e:
            log.debug("backward_patch_id_check failed for %s: %s — skipping", name, e)
            vuln_on_ref = None
        seed_exists = any(
            git_cat_file_exists(repo, target, f) for f in changed_files[:3]
        )
        if not vuln_on_ref and not seed_exists:
            return BranchCheckResult(
                branch=name,
                status=FixStatus.NOT_AFFECTED,
                method="backward_patch_id_check",
                detail=f"{target} predates the bug — no fix needed",
                changed_files=changed_files,
                ref_type=ref_type,
            )

    # Level 4: file diff heuristic
    file_status = _check_file_diff(
        repo_path, name, fix_commit, changed_files, ref_type=ref_type
    )
    detail_map = {
        FixStatus.LIKELY_PRESENT: f"85%+ of fix lines found in {target} files",
        FixStatus.NEEDS_FIX: f"Fix lines not found in {target} files",
        FixStatus.AMBIGUOUS: "Inconclusive — LLM semantic check will resolve",
    }
    return BranchCheckResult(
        branch=name,
        status=file_status,
        method="file diff heuristic",
        detail=detail_map.get(file_status, ""),
        changed_files=changed_files,
        ref_type=ref_type,
    )
