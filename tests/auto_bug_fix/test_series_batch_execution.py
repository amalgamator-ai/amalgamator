# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch execution: one apply and one commit for a run of conflicts."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import PipelineState, cherry_pick_batch_and_resolve
from common.claude_utils import ClaudeConfig


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _write(repo, rel, content):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _commit(repo, rel, content, message):
    _write(repo, rel, content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _sha(repo)


def _setup(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    base = _commit(repo, "src/a.c", "one\ntwo\nthree\n", "base")

    _git(repo, "checkout", "-q", "-b", "upstream")
    first = _commit(repo, "src/a.c", "ONE\ntwo\nthree\n", "upstream one")
    second = _commit(repo, "src/a.c", "ONE\ntwo\nTHREE\n", "upstream two")
    _git(repo, "checkout", "-q", "main")
    return str(repo), repo, base, [first, second]


def _config(rp, series):
    return BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="main",
        source_fix_commit=series[-1],
        bug_description="x",
        issue_id="TEST-1",
        source_fix_commits=series,
    )


def _run(rp, series, tmp_path, files_to_skip=None):
    state = PipelineState()
    state.allowed_modules = ["src/a.c"]
    state.triage_assessment = {"files_to_skip": list(files_to_skip or [])}
    tracker = SimpleNamespace(output_dir=str(tmp_path))
    return cherry_pick_batch_and_resolve(
        _config(rp, series),
        claude_config=ClaudeConfig(
            model="test", allowed_tools=[], perm_mode="acceptEdits", cwd=None
        ),
        state=state,
        tracker=tracker,
        commits=series,
        label="test batch",
        files_to_skip=files_to_skip,
    )


def test_clean_batch_becomes_one_commit(tmp_path):
    rp, repo, base, series = _setup(tmp_path)

    outcome = _run(rp, series, tmp_path)

    assert outcome == "clean"
    assert _sha(repo, "HEAD^") == base
    assert (repo / "src/a.c").read_text() == "ONE\ntwo\nTHREE\n"
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert subject == "Port test batch"


def test_batch_of_only_skipped_files_is_dropped(tmp_path):
    rp, repo, base, _ = _setup(tmp_path)
    _git(repo, "checkout", "-q", "-b", "docs-only")
    first = _commit(repo, "docs/a.md", "a\n", "docs one")
    second = _commit(repo, "docs/b.md", "b\n", "docs two")
    _git(repo, "checkout", "-q", "main")

    outcome = _run(rp, [first, second], tmp_path, files_to_skip=["docs/"])

    assert outcome == "clean"
    assert _sha(repo) == base


def test_skip_listed_files_never_enter_a_batch(tmp_path):
    rp, repo, base, _ = _setup(tmp_path)
    _git(repo, "checkout", "-q", "-b", "with-docs")
    first = _commit(repo, "docs/guide.md", "doc\n", "docs one")
    second = _commit(repo, "src/a.c", "ONE\ntwo\nthree\n", "code two")
    _git(repo, "checkout", "-q", "main")

    outcome = _run(rp, [first, second], tmp_path, files_to_skip=["docs/"])

    assert outcome == "clean"
    assert not (repo / "docs/guide.md").exists()
    assert (repo / "src/a.c").read_text() == "ONE\ntwo\nthree\n"
