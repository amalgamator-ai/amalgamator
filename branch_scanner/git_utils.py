"""
Git utility functions for branch scanning.

Provides three levels of detection:
  1. Exact commit presence  (git log)
  2. Equivalent patch       (git cherry)
  3. File-level diff        (git diff on touched files)
"""

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


class FixStatus(str, Enum):
    EXACT_MATCH = "exact_match"        # same commit SHA on branch
    EQUIVALENT  = "equivalent_patch"   # same patch content, different SHA
    LIKELY_PRESENT = "likely_present"  # changed files look similar
    NEEDS_FIX   = "needs_fix"          # fix not found
    AMBIGUOUS   = "ambiguous"          # inconclusive — use LLM
    ERROR       = "error"              # git command failed


@dataclass
class BranchCheckResult:
    branch: str
    status: FixStatus
    method: str          # which method detected this
    detail: str          # human-readable explanation
    changed_files: list[str]  # files touched by the fix commit


def _run(cmd: list[str], cwd: Path, timeout: int = 30) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def get_commit_changed_files(repo_path: Path, commit: str) -> list[str]:
    """Return list of files changed by a given commit."""
    rc, out, err = _run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit],
        cwd=repo_path,
    )
    if rc != 0:
        log.warning("Could not get changed files for %s: %s", commit, err)
        return []
    return [f for f in out.splitlines() if f]


def get_remote_branches(repo_path: Path, pattern: str | None = None) -> list[str]:
    """
    List all remote branches, optionally filtered by a glob pattern.
    Returns short branch names (without 'origin/').
    """
    rc, out, _ = _run(
        ["git", "branch", "-r", "--format=%(refname:short)"],
        cwd=repo_path,
    )
    if rc != 0:
        return []

    branches = []
    for line in out.splitlines():
        name = line.strip()
        # strip 'origin/' prefix
        if name.startswith("origin/"):
            name = name[len("origin/"):]
        if name in ("HEAD", ""):
            continue
        branches.append(name)

    if pattern:
        import fnmatch
        branches = [b for b in branches if fnmatch.fnmatch(b, pattern)]

    return sorted(set(branches))


def check_exact_commit(repo_path: Path, branch: str, commit: str) -> bool:
    """
    Method 1: Check if the exact commit SHA exists on a branch.
    Uses 'git merge-base --is-ancestor'.
    """
    rc, _, _ = _run(
        ["git", "merge-base", "--is-ancestor", commit, f"origin/{branch}"],
        cwd=repo_path,
    )
    return rc == 0


def check_cherry(repo_path: Path, branch: str, commit: str) -> bool | None:
    """
    Method 2: Use 'git cherry' to check if an equivalent patch is on branch.
    Returns True  = equivalent patch already on branch (line starts with '-')
    Returns False = patch NOT on branch (line starts with '+')
    Returns None  = inconclusive
    """
    rc, out, err = _run(
        ["git", "cherry", f"origin/{branch}", commit],
        cwd=repo_path,
    )
    if rc != 0:
        log.debug("git cherry failed for %s on %s: %s", commit, branch, err)
        return None

    for line in out.splitlines():
        if line.startswith("- ") and commit[:7] in line or line.startswith("- "):
            return True   # equivalent patch already applied
        if line.startswith("+ "):
            return False  # patch not yet on branch

    return None  # no output = inconclusive


def check_file_diff(
    repo_path: Path, branch: str, commit: str, changed_files: list[str]
) -> FixStatus:
    """
    Method 3: Compare the diff introduced by the fix commit against what
    currently exists on the target branch for the same files.

    Heuristic: if the branch already has all lines added by the fix,
    it's likely present. If none of them are there, it needs the fix.
    """
    if not changed_files:
        return FixStatus.AMBIGUOUS

    # Get what the fix commit added (positive diff lines)
    rc, fix_diff, _ = _run(
        ["git", "show", "--unified=0", commit, "--", *changed_files],
        cwd=repo_path,
    )
    if rc != 0:
        return FixStatus.AMBIGUOUS

    added_lines = [
        line[1:]  # strip leading '+'
        for line in fix_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    if not added_lines:
        return FixStatus.AMBIGUOUS

    # Check how many of those lines exist in the branch version of those files
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
    else:
        return FixStatus.AMBIGUOUS


def check_branch(
    repo_path: Path, branch: str, commit: str
) -> BranchCheckResult:
    """
    Run all three detection methods in order and return a BranchCheckResult.
    """
    changed_files = get_commit_changed_files(repo_path, commit)

    # Method 1: exact commit
    if check_exact_commit(repo_path, branch, commit):
        return BranchCheckResult(
            branch=branch,
            status=FixStatus.EXACT_MATCH,
            method="git merge-base --is-ancestor",
            detail=f"Commit {commit[:10]} is a direct ancestor of origin/{branch}",
            changed_files=changed_files,
        )

    # Method 2: equivalent patch (git cherry)
    cherry = check_cherry(repo_path, branch, commit)
    if cherry is True:
        return BranchCheckResult(
            branch=branch,
            status=FixStatus.EQUIVALENT,
            method="git cherry",
            detail=f"Equivalent patch already applied on origin/{branch} (different SHA)",
            changed_files=changed_files,
        )
    if cherry is False:
        # Confirmed not present — but still run file diff for confidence
        file_status = check_file_diff(repo_path, branch, commit, changed_files)
        if file_status == FixStatus.LIKELY_PRESENT:
            # cherry says no, file diff says maybe — ambiguous
            return BranchCheckResult(
                branch=branch,
                status=FixStatus.AMBIGUOUS,
                method="git cherry + file diff",
                detail="git cherry says not present but file diff shows similar changes — needs LLM review",
                changed_files=changed_files,
            )
        return BranchCheckResult(
            branch=branch,
            status=FixStatus.NEEDS_FIX,
            method="git cherry",
            detail=f"Fix not found on origin/{branch}",
            changed_files=changed_files,
        )

    # Method 3: file-level diff (git cherry was inconclusive)
    file_status = check_file_diff(repo_path, branch, commit, changed_files)
    detail_map = {
        FixStatus.LIKELY_PRESENT: "85%+ of fix lines found in branch files",
        FixStatus.NEEDS_FIX:      "Fix lines not found in branch files",
        FixStatus.AMBIGUOUS:      "Inconclusive — recommend LLM semantic check",
    }
    return BranchCheckResult(
        branch=branch,
        status=file_status,
        method="file diff",
        detail=detail_map.get(file_status, ""),
        changed_files=changed_files,
    )
