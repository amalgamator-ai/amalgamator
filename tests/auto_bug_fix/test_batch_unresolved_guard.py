# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A batch may not be committed with unresolved conflicts."""

from __future__ import annotations

import subprocess

import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import PipelineEscalation, _refuse_unresolved_batch


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _sha(cwd, ref="HEAD"):
    return _git(cwd, "rev-parse", ref).stdout.strip()


def _commit(repo, rel, content, message):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _sha(repo)


def _config(repo):
    return BugFixConfig(
        repo_path=str(repo),
        build_dir=str(repo),
        source_branch="main",
        target_branch="main",
        source_fix_commit="abc123",
        bug_description="x",
        issue_id="TEST-1",
    )


def _conflicted_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    base = _commit(repo, "src/a.c", "a\nb\nc\n", "base")

    _git(repo, "checkout", "-q", "-b", "theirs")
    _commit(repo, "src/a.c", "a\nTHEIRS\nc\n", "their change")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "src/a.c", "a\nOURS\nc\n", "our change")

    merge = _git(repo, "merge", "theirs")
    assert merge.returncode != 0, "merge should conflict"
    return repo, base


def test_unmerged_path_blocks_the_commit(tmp_path):
    repo, base = _conflicted_repo(tmp_path)

    with pytest.raises(PipelineEscalation) as err:
        _refuse_unresolved_batch(_config(repo), base, "batch")

    assert "src/a.c" in str(err.value)
    assert "markers" in str(err.value) or "unresolved" in str(err.value)


def test_staged_conflict_markers_block_the_commit(tmp_path):
    repo, base = _conflicted_repo(tmp_path)
    (repo / "src/a.c").write_text(
        "a\n<<<<<<< ours\nOURS\n=======\nTHEIRS\n>>>>>>> theirs\nc\n"
    )
    _git(repo, "add", "src/a.c")

    with pytest.raises(PipelineEscalation) as err:
        _refuse_unresolved_batch(_config(repo), base, "batch")

    assert "unresolved" in str(err.value) or "markers" in str(err.value)


def test_resolved_batch_passes(tmp_path):
    repo, base = _conflicted_repo(tmp_path)
    (repo / "src/a.c").write_text("a\nRESOLVED\nc\n")
    _git(repo, "add", "src/a.c")
    _git(repo, "commit", "-m", "resolved")

    _refuse_unresolved_batch(_config(repo), base, "batch")


def test_resolved_but_unstaged_paths_are_staged_and_allowed(tmp_path):
    repo, base = _conflicted_repo(tmp_path)
    # The session rewrote the file correctly but never staged it.
    (repo / "src/a.c").write_text("a\nRESOLVED\nc\n")

    _refuse_unresolved_batch(_config(repo), base, "batch")

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "UU" not in status
    assert "src/a.c" in status


def test_unresolved_paths_reports_markers_and_clears_clean_ones(tmp_path):
    from auto_bug_fix.run_bug_fix import _batch_unresolved_paths

    repo, base = _conflicted_repo(tmp_path)

    # Still unresolved: markers present.
    assert _batch_unresolved_paths(_config(repo), base) == ["src/a.c"]

    # Resolved content but never staged: staged by the helper, nothing left.
    (repo / "src/a.c").write_text("a\nRESOLVED\nc\n")
    assert _batch_unresolved_paths(_config(repo), base) == []
