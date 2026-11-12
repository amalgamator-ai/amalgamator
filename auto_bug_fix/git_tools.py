# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin wrappers around git commands for the bug-fix porting pipeline."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class GitResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0


# Trailer identifying commits authored by the pipeline. Single source of
# truth for run_bug_fix, dossier.build_trailers, and ingest.
COAUTHOR_TRAILERS = {"Co-authored-by": "amalgamator <amalgamator@redhat.com>"}


def _run_git(repo_path: str, args: list[str], **kwargs) -> GitResult:
    """Execute a git command in repo_path and return a GitResult.

    Decodes as UTF-8 with replacement. Real codebases carry non-UTF-8 bytes
    (latin-1 author names and comments are common in long-lived C trees), and
    strict decoding turns any diff touching such a file into a
    UnicodeDecodeError that aborts the run.

    Replacement is safe *here* because this output drives analysis, verdicts,
    prompts and logs. Anywhere byte-exactness is required — patch generation,
    patch-ids, blob content for splicing — read bytes instead. See
    git_format_patch_bytes, git_patch_id, mechanical._git_bytes and
    snapback._git_blob.
    """
    result = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )
    return GitResult(result.returncode, result.stdout, result.stderr)


def git_show_files(repo_path: str, commit: str) -> list[str]:
    """Return the list of file paths modified by a commit."""
    r = _run_git(repo_path, ["show", commit, "--name-only", "--pretty=format:"])
    return [
        line for line in r.stdout.splitlines() if line.strip() and line.strip() != '""'
    ]


def git_patch_id(repo_path: str, commit: str) -> str:
    """Compute stable patch-id for a commit. Returns empty string for merge commits.

    Piped as bytes end to end. A patch-id computed over lossily-decoded text
    would not match the same patch elsewhere, and these ids gate the Phase 0
    already-applied and predates-the-bug decisions — a wrong answer there is
    worse than a crash.
    """
    show = subprocess.run(
        ["git", "show", commit],
        cwd=repo_path,
        capture_output=True,
    )
    pid = subprocess.run(
        ["git", "patch-id", "--stable"],
        cwd=repo_path,
        input=show.stdout,
        capture_output=True,
    )
    # The id itself is hex, so decoding the result is always safe.
    output = pid.stdout.decode("utf-8", "replace").strip()
    if not output:
        return ""
    return output.split()[0]


def git_is_ancestor(repo_path: str, commit: str, branch: str) -> bool:
    """Return True if commit is an ancestor of branch."""
    r = _run_git(repo_path, ["merge-base", "--is-ancestor", commit, branch])
    return r.success


def git_cat_file_exists(repo_path: str, ref: str, path: str) -> bool:
    """Return True if path exists in the given ref."""
    r = _run_git(repo_path, ["cat-file", "-e", f"{ref}:{path}"])
    return r.success


def git_log_follow_renames(
    repo_path: str, path: str, fork_point: str, target: str
) -> list[str]:
    """Return commit SHAs where path was renamed between fork_point and target."""
    r = _run_git(
        repo_path,
        [
            "log",
            "--follow",
            "--diff-filter=R",
            "--format=%H",
            f"{fork_point}..{target}",
            "--",
            path,
        ],
    )
    return [line for line in r.stdout.splitlines() if line.strip()]


def git_diff_find_renames(
    repo_path: str,
    ref_a: str,
    ref_b: str,
    path: str | None = None,
    threshold: int = 80,
) -> list[tuple[str, str]]:
    """Detect file renames between two refs. Returns list of (old_path, new_path) pairs."""
    args = [
        "diff",
        f"--find-renames={threshold}%",
        "--diff-filter=R",
        "--name-status",
        ref_a,
        ref_b,
    ]
    if path is not None:
        args += ["--", path]
    r = _run_git(repo_path, args)
    pairs: list[tuple[str, str]] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            pairs.append((parts[1], parts[2]))
    return pairs


def is_merge_commit(repo_path: str, commit: str) -> bool:
    """Return True if commit has more than one parent (is a merge)."""
    r = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^2"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return r.returncode == 0


def git_cherry_pick(
    repo_path: str,
    commit: str,
    strategy: str | None = None,
    strategy_option: str | None = None,
) -> GitResult:
    """Cherry-pick a commit with optional merge strategy.

    For merge commits, uses -m 1 to apply the combined diff relative
    to the first parent (the mainline), which captures all changes
    from the entire topic branch.
    """
    args = ["cherry-pick", "-x"]
    if is_merge_commit(repo_path, commit):
        args.append("-m1")
    args.append(commit)
    if strategy is not None:
        args.append(f"--strategy={strategy}")
    if strategy_option is not None:
        args.append(f"--strategy-option={strategy_option}")
    return _run_git(repo_path, args)


def git_cherry_pick_abort(repo_path: str) -> GitResult:
    """Abort an in-progress cherry-pick."""
    return _run_git(repo_path, ["cherry-pick", "--abort"])


def git_status_porcelain(repo_path: str) -> list[str]:
    """Return porcelain status lines (e.g. 'UU file.go' for conflicts)."""
    r = _run_git(repo_path, ["status", "--porcelain"])
    return [line for line in r.stdout.splitlines() if line.strip()]


def git_status_porcelain_all(repo_path: str) -> GitResult:
    """Porcelain status with untracked directories expanded file-by-file.

    Plain ``git status`` collapses an untracked directory to a single
    ``?? dir/`` entry; callers that match individual file paths (artifact
    allowlists, verification-dirt baselines) need the expanded form. Returns
    the full result so callers can fail closed on a nonzero exit instead of
    reading an empty list as an empty tree.
    """
    return _run_git(repo_path, ["status", "--porcelain", "--untracked-files=all"])


def git_diff_name_only(repo_path: str, ref: str = "HEAD") -> list[str]:
    """Return list of file paths that differ from ref."""
    r = _run_git(repo_path, ["diff", "--name-only", ref])
    return [line for line in r.stdout.splitlines() if line.strip()]


def git_diff_name_only_range(repo_path: str, ref_a: str, ref_b: str) -> list[str]:
    """Return list of file paths changed between two refs (ref_a..ref_b)."""
    r = _run_git(repo_path, ["diff", "--name-only", ref_a, ref_b])
    return [line for line in r.stdout.splitlines() if line.strip()]


def git_range_diff(
    repo_path: str,
    upstream_range: str,
    ported_range: str,
    creation_factor: int | None = None,
) -> str:
    """Run git range-diff and return stdout.

    creation_factor: percentage passed to --creation-factor. Range-diff only
    pairs commits whose diffs are similar enough; a high factor forces
    pairing, which is what a deliberate one-to-one comparison wants.
    """
    args = ["range-diff"]
    if creation_factor is not None:
        args.append(f"--creation-factor={creation_factor}")
    args += [upstream_range, ported_range]
    r = _run_git(repo_path, args)
    return r.stdout


def git_format_patch_bytes(
    repo_path: str, n: int = 1, base_ref: str | None = None
) -> bytes:
    """Generate a patch for the last n commits as raw bytes.

    The patch is a deliverable that has to apply cleanly, so it must never be
    round-tripped through a lossy decode — a replacement character would
    silently produce a patch git cannot apply.
    """
    args = (
        ["format-patch", "--stdout", base_ref]
        if base_ref
        else ["format-patch", f"-{n}", "--stdout"]
    )
    result = subprocess.run(["git"] + args, cwd=repo_path, capture_output=True)
    if result.returncode != 0:
        # An empty result is how both "failed" and "no commits in range" look
        # to the caller; name the cause so an empty deliverable on a failure
        # path is diagnosable (Phase 5 escalates on empty bytes either way).
        log.warning(
            "git format-patch %s failed (exit %d): %s",
            base_ref or f"-{n}",
            result.returncode,
            result.stderr.decode("utf-8", "replace").strip(),
        )
    return result.stdout


def git_format_patch(repo_path: str, n: int = 1, base_ref: str | None = None) -> str:
    """Patch text for display (dossier). Use git_format_patch_bytes to write it."""
    return git_format_patch_bytes(repo_path, n, base_ref).decode("utf-8", "replace")


def git_reset_hard(repo_path: str, ref: str = "HEAD") -> GitResult:
    """Hard-reset the working tree and index to ref."""
    return _run_git(repo_path, ["reset", "--hard", ref])


def git_worktree_add(repo_path: str, path: str, branch: str) -> GitResult:
    """Create a new git worktree at path, detached at the given branch/commit."""
    return _run_git(repo_path, ["worktree", "add", "--detach", path, branch])


def git_worktree_remove(repo_path: str, path: str) -> GitResult:
    """Remove a git worktree."""
    return _run_git(repo_path, ["worktree", "remove", path])


def git_log_commits(
    repo_path: str,
    branch: str,
    since: str | None = None,
    max_count: int | None = None,
) -> list[str]:
    """Return commit SHAs on branch, optionally filtered by since/max_count."""
    args = ["log", "--format=%H"]
    if since is not None:
        args.append(f"--since={since}")
    if max_count is not None:
        args += ["-n", str(max_count)]
    args.append(branch)
    r = _run_git(repo_path, args)
    return [line for line in r.stdout.splitlines() if line.strip()]


def git_merge_base(repo_path: str, a: str, b: str) -> str:
    """Return the best common ancestor (merge base) of two refs."""
    r = _run_git(repo_path, ["merge-base", a, b])
    sha = r.stdout.strip()
    if not r.success or not sha:
        raise RuntimeError(f"git merge-base failed for {a}..{b}: {r.stderr.strip()}")
    return sha


def git_commit(
    repo_path: str,
    message: str,
    trailers: dict[str, str] | None = None,
) -> GitResult:
    """Create a commit with optional trailer lines appended to the message."""
    if trailers:
        trailer_lines = "\n".join(f"{k}: {v}" for k, v in trailers.items())
        message = f"{message}\n\n{trailer_lines}"
    return _run_git(repo_path, ["commit", "-m", message])


def git_rev_parse(repo_path: str, ref: str) -> str:
    """Resolve a ref to its full SHA."""
    r = _run_git(repo_path, ["rev-parse", ref])
    sha = r.stdout.strip()
    if not r.success or not sha:
        raise RuntimeError(f"git rev-parse failed for {ref}: {r.stderr.strip()}")
    return sha


def git_log_count(repo_path: str, range_spec: str) -> int:
    """Count the number of commits in a range (e.g. 'A..B')."""
    r = _run_git(repo_path, ["rev-list", "--count", range_spec])
    output = r.stdout.strip()
    return int(output) if output else 0


def resolve_merge_commit(repo_path: str, commit: str) -> str:
    """If commit is a merge, return the second parent (topic head); else resolve to SHA.

    Used for patch-id and ancestry checks where the topic head is the
    meaningful reference.  Seed derivation uses ``derive_seed`` which
    handles merge commits directly via ``git diff --name-only M^1..M``.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^2"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode == 0:
        # It's a merge - return the second parent (topic branch head)
        return result.stdout.strip()

    # Not a merge - resolve to full SHA
    return git_rev_parse(repo_path, commit)
