# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pruning skip-listed files from series commits (commit-series mode only)."""

from __future__ import annotations

import subprocess

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import (
    PipelineState,
    _prune_skipped_after_pick,
    _prune_skipped_from_commit,
)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _write(repo, rel, content="x\n"):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    _write(repo, "src/base.c", "int main(void){return 0;}\n")
    _write(repo, "docs/guide.md", "v1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return str(repo), repo


def _commit_all(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _sha(repo)


def test_prune_deletes_files_the_target_never_had(tmp_path):
    rp, repo = _repo(tmp_path)
    base = _sha(repo)
    _write(repo, "docs/new.md", "doc\n")
    _write(repo, "src/fix.c", "fix\n")
    commit = _commit_all(repo, "add docs and fix")

    pruned, emptied = _prune_skipped_from_commit(rp, base, ["docs/"])

    assert pruned == ["docs/new.md"]
    assert emptied is False
    assert _sha(repo) != commit
    assert (repo / "src/fix.c").exists()
    assert not (repo / "docs/new.md").exists()
    assert (
        "docs/new.md"
        not in subprocess.run(
            ["git", "show", "--name-only", "--format=", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )


def test_prune_drops_a_commit_that_only_touched_skipped_files(tmp_path):
    rp, repo = _repo(tmp_path)
    base = _sha(repo)
    _write(repo, "docs/only.md", "doc\n")
    _commit_all(repo, "docs only")

    pruned, emptied = _prune_skipped_from_commit(rp, base, ["docs/"])

    assert pruned == ["docs/only.md"]
    assert emptied is True
    assert _sha(repo) == base


def test_prune_restores_a_tracked_skipped_file_to_its_target_version(tmp_path):
    rp, repo = _repo(tmp_path)
    base = _sha(repo)
    _write(repo, "docs/guide.md", "v2 changed by upstream\n")
    _commit_all(repo, "rewrite guide")

    pruned, emptied = _prune_skipped_from_commit(rp, base, ["docs/guide.md"])

    assert pruned == ["docs/guide.md"]
    assert emptied is True
    assert _sha(repo) == base
    assert (repo / "docs/guide.md").read_text() == "v1\n"


def test_prune_is_a_noop_when_nothing_matches(tmp_path):
    rp, repo = _repo(tmp_path)
    base = _sha(repo)
    _write(repo, "src/fix.c", "fix\n")
    commit = _commit_all(repo, "fix only")

    pruned, emptied = _prune_skipped_from_commit(rp, base, ["docs/"])

    assert pruned == []
    assert emptied is False
    assert _sha(repo) == commit


def test_prune_after_pick_does_nothing_outside_commit_series_mode(tmp_path):
    rp, repo = _repo(tmp_path)
    base = _sha(repo)
    _write(repo, "docs/new.md", "doc\n")
    commit = _commit_all(repo, "docs only")

    config = BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="main",
        source_fix_commit=commit,
        bug_description="x",
        issue_id="TEST-1",
    )
    state = PipelineState()
    state.triage_assessment = {"files_to_skip": ["docs/"]}

    dropped = _prune_skipped_after_pick(config, state, "main commit", base)

    assert dropped is False
    assert _sha(repo) == commit


def test_prune_after_pick_prunes_in_commit_series_mode(tmp_path):
    rp, repo = _repo(tmp_path)
    base = _sha(repo)
    _write(repo, "docs/new.md", "doc\n")
    _write(repo, "src/fix.c", "fix\n")
    commit = _commit_all(repo, "docs and fix")
    series = [commit]

    config = BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="main",
        source_fix_commit=commit,
        bug_description="x",
        issue_id="TEST-1",
        source_fix_commits=series,
    )
    state = PipelineState()
    state.triage_assessment = {"files_to_skip": ["docs/"]}

    dropped = _prune_skipped_after_pick(config, state, "commit 1/1", base)

    assert dropped is False
    assert not (repo / "docs/new.md").exists()
    assert (repo / "src/fix.c").exists()
