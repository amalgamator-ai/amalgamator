"""Content-stable patch-id hashing and duplicate-patch detection for CVE backporting triage."""
from __future__ import annotations

from auto_bug_fix.git_tools import git_log_commits, git_patch_id


def compute_patch_id(repo_path: str, commit: str) -> str:
    """Compute the content-stable patch-id for a commit."""
    return git_patch_id(repo_path, commit)


def forward_patch_id_check(
    repo_path: str,
    fix_commit: str,
    target_branch: str,
    lookback: str = "12 months",
) -> str | None:
    """Check if the fix is already present on target_branch. Returns the matching SHA or None."""
    fix_pid = compute_patch_id(repo_path, fix_commit)
    if not fix_pid:
        # Fix commit is a merge or has no diff - can't check
        return None

    commits = git_log_commits(repo_path, target_branch, since=lookback)
    for commit in commits:
        pid = compute_patch_id(repo_path, commit)
        if pid and pid == fix_pid:
            return commit
    return None


def backward_patch_id_check(
    repo_path: str,
    vuln_commit: str,
    target_branch: str,
    lookback: str = "12 months",
) -> str | None:
    """Check if the vulnerability-introducing commit exists on target_branch. Returns SHA or None."""
    vuln_pid = compute_patch_id(repo_path, vuln_commit)
    if not vuln_pid:
        # Vuln commit is a merge or has no diff - can't check
        return None

    commits = git_log_commits(repo_path, target_branch, since=lookback)
    for commit in commits:
        pid = compute_patch_id(repo_path, commit)
        if pid and pid == vuln_pid:
            return commit
    return None
