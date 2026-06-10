"""Allowlist derivation and enforcement for the bug-fix porting pipeline."""
from __future__ import annotations

from auto_bug_fix.git_tools import (
    git_cat_file_exists,
    git_diff_find_renames,
    git_diff_name_only,
    git_log_follow_renames,
    git_show_files,
)

_DEFAULT_TEST_PATTERNS: list[str] = ["test", "tests", "testing", "test_", "spec"]


def derive_seed(repo_path: str, fix_commit: str) -> list[str]:
    """Return the list of files touched by the fix commit."""
    return git_show_files(repo_path, fix_commit)


def resolve_renames(
    repo_path: str,
    seed: list[str],
    target_branch: str,
    fork_point: str,
    cap: int = 3,
) -> tuple[list[str], list[str]]:
    """Map seed paths to their target-branch equivalents, following renames.

    Returns (resolved_paths, escalated_paths) where escalated paths exceeded the rename cap.
    """
    resolved: list[str] = []
    escalated: list[str] = []

    for path in seed:
        if git_cat_file_exists(repo_path, target_branch, path):
            resolved.append(path)
            continue

        new_names: list[str] = []
        follow_shas = git_log_follow_renames(
            repo_path, path, fork_point, target_branch,
        )
        if follow_shas:
            rename_pairs = git_diff_find_renames(
                repo_path, fork_point, target_branch, path,
            )
            new_names = [dest for _, dest in rename_pairs]
        else:
            rename_pairs = git_diff_find_renames(
                repo_path, fork_point, target_branch, path,
            )
            new_names = [dest for _, dest in rename_pairs]

        if len(new_names) > cap:
            escalated.extend(new_names)
        else:
            resolved.extend(new_names)

    return resolved, escalated


def enforce_allowlist(
    repo_path: str,
    allowed_modules: list[str],
    disallowed: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Check that all changed files fall within the allowlist. Returns (ok, violations)."""
    changed = git_diff_name_only(repo_path)
    disallowed = disallowed or []
    violations: list[str] = []

    for f in changed:
        allowed = any(f.startswith(prefix) for prefix in allowed_modules)
        blocked = any(f.startswith(prefix) for prefix in disallowed)
        if not allowed or blocked:
            violations.append(f)

    return (len(violations) == 0, violations)


def enforce_test_file_veto(
    repo_path: str,
    allowed_seed: list[str],
    test_patterns: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Veto changes to test files not in the original seed. Returns (ok, vetoed_files)."""
    patterns = test_patterns if test_patterns is not None else _DEFAULT_TEST_PATTERNS
    changed = git_diff_name_only(repo_path)
    vetoed: list[str] = []

    for f in changed:
        if f in allowed_seed:
            continue
        parts = f.split("/")
        if any(pattern in part for part in parts for pattern in patterns):
            vetoed.append(f)

    return (len(vetoed) == 0, vetoed)
