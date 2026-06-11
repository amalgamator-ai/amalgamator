"""Thin wrappers around git commands for the bug-fix porting pipeline."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GitResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def _run_git(repo_path: str, args: list[str], **kwargs) -> GitResult:
    """Execute a git command in repo_path and return a GitResult."""
    result = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        **kwargs,
    )
    return GitResult(result.returncode, result.stdout, result.stderr)


def git_show_files(repo_path: str, commit: str) -> list[str]:
    """Return the list of file paths modified by a commit."""
    r = _run_git(repo_path, ["show", commit, "--name-only", "--pretty=format:"])
    return [line for line in r.stdout.splitlines() if line.strip() and line.strip() != '""']


def git_patch_id(repo_path: str, commit: str) -> str:
    """Compute stable patch-id for a commit. Returns empty string for merge commits."""
    show = _run_git(repo_path, ["show", commit])
    pid = subprocess.run(
        ["git", "patch-id", "--stable"],
        cwd=repo_path,
        input=show.stdout,
        capture_output=True,
        text=True,
    )
    output = pid.stdout.strip()
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
            "log", "--follow", "--diff-filter=R", "--format=%H",
            f"{fork_point}..{target}", "--", path,
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
        "diff", f"--find-renames={threshold}%", "--diff-filter=R",
        "--name-status", ref_a, ref_b,
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



def git_cherry_pick(
    repo_path: str,
    commit: str,
    strategy: str | None = None,
    strategy_option: str | None = None,
) -> GitResult:
    """Cherry-pick a commit with optional merge strategy."""
    args = ["cherry-pick", "-x", commit]
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


def git_diff_name_only(repo_path: str, ref: str = "HEAD") -> list[str]:
    """Return list of file paths that differ from ref."""
    r = _run_git(repo_path, ["diff", "--name-only", ref])
    return [line for line in r.stdout.splitlines() if line.strip()]


def git_range_diff(repo_path: str, upstream_range: str, ported_range: str) -> str:
    """Run git range-diff and return stdout."""
    r = _run_git(repo_path, ["range-diff", upstream_range, ported_range])
    return r.stdout


def git_format_patch(repo_path: str, n: int = 1) -> str:
    """Generate a patch for the last n commits and return it as a string."""
    r = _run_git(repo_path, ["format-patch", f"-{n}", "--stdout"])
    return r.stdout


def git_reset_hard(repo_path: str, ref: str = "HEAD") -> GitResult:
    """Hard-reset the working tree and index to ref."""
    return _run_git(repo_path, ["reset", "--hard", ref])



def git_worktree_add(repo_path: str, path: str, branch: str) -> GitResult:
    """Create a new git worktree at path checked out to branch."""
    return _run_git(repo_path, ["worktree", "add", path, branch])


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

    Many CVE fixes land as PR merge commits. The actual fix code is on the topic branch,
    so we need to resolve the merge to the topic head for seed derivation and cherry-pick.

    Returns the full SHA in both cases for consistency.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^2"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        # It's a merge - return the second parent (topic branch head)
        return result.stdout.strip()

    # Not a merge - resolve to full SHA
    return git_rev_parse(repo_path, commit)
