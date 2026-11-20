# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.git_tools — thin git command wrappers."""

from __future__ import annotations

import subprocess

from auto_bug_fix import git_tools
from auto_bug_fix.git_tools import (
    GitResult,
    git_cat_file_exists,
    git_cherry_pick,
    git_diff_name_only,
    git_diff_name_only_range,
    git_is_ancestor,
    git_log_commits,
    git_log_count,
    git_merge_base,
    git_rev_parse,
    git_show_files,
    git_status_porcelain,
    is_merge_commit,
    resolve_merge_commit,
)


def test_git_show_files(tmp_git_repo):
    files = git_show_files(tmp_git_repo.repo_path, tmp_git_repo.fix_sha)
    assert "lib/url.c" in files
    assert "tests/test_fix.c" in files


def test_git_is_ancestor_true(tmp_git_repo):
    # The initial commit (parent of fix) is an ancestor of the source branch.
    rp = tmp_git_repo.repo_path
    src = tmp_git_repo.source_branch
    initial_sha = git_rev_parse(rp, f"{tmp_git_repo.fix_sha}~1")
    assert git_is_ancestor(rp, initial_sha, src)


def test_git_is_ancestor_false(tmp_git_repo):
    # The fix commit is NOT an ancestor of target (target branched before it).
    assert not git_is_ancestor(
        tmp_git_repo.repo_path,
        tmp_git_repo.fix_sha,
        tmp_git_repo.target_branch,
    )


def test_git_cat_file_exists_true(tmp_git_repo):
    tgt = tmp_git_repo.target_branch
    assert git_cat_file_exists(tmp_git_repo.repo_path, tgt, "lib/url.c")


def test_git_cat_file_exists_false(tmp_git_repo):
    # tests/test_fix.c was added in the fix commit, so it does not exist on target.
    tgt = tmp_git_repo.target_branch
    assert not git_cat_file_exists(tmp_git_repo.repo_path, tgt, "tests/test_fix.c")


def test_git_diff_name_only(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    # Checkout target and modify lib/url.c so it shows up in the diff.
    subprocess.run(
        ["git", "checkout", tmp_git_repo.target_branch],
        check=True,
        cwd=rp,
    )
    url_path = f"{rp}/lib/url.c"
    with open(url_path, "a") as fh:
        fh.write("\n// local edit\n")

    changed = git_diff_name_only(rp, "HEAD")
    assert "lib/url.c" in changed

    # Reset so the repo stays clean for other assertions (fixture is function-scoped
    # anyway, but be explicit).
    subprocess.run(["git", "checkout", "--", "."], check=True, cwd=rp)


def test_git_status_porcelain(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    untracked = f"{rp}/untracked.txt"
    with open(untracked, "w") as fh:
        fh.write("hello\n")

    lines = git_status_porcelain(rp)
    assert any("untracked.txt" in line for line in lines)


def test_git_merge_base(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    src = tmp_git_repo.source_branch
    tgt = tmp_git_repo.target_branch
    base = git_merge_base(rp, src, tgt)
    # Must be a valid 40-char SHA.
    assert len(base) == 40
    assert all(c in "0123456789abcdef" for c in base)


def test_git_rev_parse(tmp_git_repo):
    sha = git_rev_parse(tmp_git_repo.repo_path, "HEAD")
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_git_log_commits(tmp_git_repo):
    shas = git_log_commits(tmp_git_repo.repo_path, tmp_git_repo.source_branch)
    # At least the initial commit and the fix commit.
    assert len(shas) >= 2


def test_git_log_count(tmp_git_repo):
    tgt = tmp_git_repo.target_branch
    src = tmp_git_repo.source_branch
    count = git_log_count(tmp_git_repo.repo_path, f"{tgt}..{src}")
    assert count == 1


def test_resolve_merge_commit_not_a_merge(tmp_git_repo):
    # HEAD is a regular commit
    result = resolve_merge_commit(tmp_git_repo.repo_path, "HEAD")
    expected = git_rev_parse(tmp_git_repo.repo_path, "HEAD")
    assert result == expected


def test_resolve_merge_commit_is_merge(tmp_git_repo):
    from pathlib import Path

    rp = tmp_git_repo.repo_path

    # Create a merge
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=rp, check=True)
    Path(rp, "feature.txt").write_text("feature")
    subprocess.run(["git", "add", "feature.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "feature"], cwd=rp, check=True)

    subprocess.run(["git", "checkout", "main"], cwd=rp, check=True)
    Path(rp, "main.txt").write_text("main")
    subprocess.run(["git", "add", "main.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "main"], cwd=rp, check=True)

    subprocess.run(
        ["git", "merge", "feature", "--no-ff", "-m", "merge feature"],
        cwd=rp,
        check=True,
    )

    # resolve_merge_commit should return the feature branch head (^2)
    merge_sha = git_rev_parse(rp, "HEAD")
    topic_head = git_rev_parse(rp, "HEAD^2")

    result = resolve_merge_commit(rp, merge_sha)
    assert result == topic_head
    assert result != merge_sha


def test_is_merge_commit_regular(tmp_git_repo):
    assert not is_merge_commit(tmp_git_repo.repo_path, tmp_git_repo.fix_sha)


def test_is_merge_commit_merge(tmp_git_repo):
    from pathlib import Path

    rp = tmp_git_repo.repo_path

    subprocess.run(["git", "checkout", "-b", "feat"], cwd=rp, check=True)
    Path(rp, "feat.txt").write_text("feat")
    subprocess.run(["git", "add", "feat.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "feat"], cwd=rp, check=True)

    subprocess.run(["git", "checkout", "main"], cwd=rp, check=True)
    subprocess.run(
        ["git", "merge", "feat", "--no-ff", "-m", "merge feat"], cwd=rp, check=True
    )

    assert is_merge_commit(rp, "HEAD")


def test_git_diff_name_only_range(tmp_git_repo):
    rp = tmp_git_repo.repo_path
    parent = f"{tmp_git_repo.fix_sha}^"
    files = git_diff_name_only_range(rp, parent, tmp_git_repo.fix_sha)
    assert "lib/url.c" in files
    assert "tests/test_fix.c" in files


def test_git_diff_name_only_range_merge(tmp_git_repo):
    """For a merge commit, diff M^1..M captures all files from the topic branch."""
    from pathlib import Path

    rp = tmp_git_repo.repo_path

    subprocess.run(["git", "checkout", "-b", "topic"], cwd=rp, check=True)
    Path(rp, "topic_a.txt").write_text("a")
    subprocess.run(["git", "add", "topic_a.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "topic commit 1"], cwd=rp, check=True)

    Path(rp, "topic_b.txt").write_text("b")
    subprocess.run(["git", "add", "topic_b.txt"], cwd=rp, check=True)
    subprocess.run(["git", "commit", "-m", "topic commit 2"], cwd=rp, check=True)

    subprocess.run(["git", "checkout", "main"], cwd=rp, check=True)
    subprocess.run(
        ["git", "merge", "topic", "--no-ff", "-m", "merge topic"], cwd=rp, check=True
    )

    merge_sha = git_rev_parse(rp, "HEAD")
    files = git_diff_name_only_range(rp, f"{merge_sha}^1", merge_sha)
    assert "topic_a.txt" in files, "must include file from first topic commit"
    assert "topic_b.txt" in files, "must include file from second topic commit"


# --- non-UTF-8 repository content ---------------------------------------
# Long-lived C trees carry latin-1 bytes in comments and author names. Strict
# decoding turns any diff touching such a file into a UnicodeDecodeError that
# aborts the whole run, so git output must degrade rather than raise — except
# where byte-exactness is the point.

import pytest

from auto_bug_fix.git_tools import (
    _run_git,
    git_format_patch,
    git_format_patch_bytes,
    git_patch_id,
    git_range_diff,
)

LATIN1_SOURCE = b"/* Copyright Bj\xf8rn Hansen - caf\xe9 */\nint b = 2;\n"


@pytest.fixture
def latin1_repo(tmp_path):
    """A repo whose fix commit touches a file that is not valid UTF-8."""

    def g(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True
        ).stdout.strip()

    g("init", "-q", "-b", "main", ".")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (tmp_path / "ok.c").write_text("int a = 1;\n")
    g("add", ".")
    g("commit", "-qm", "base")
    base = g("rev-parse", "HEAD")
    (tmp_path / "legacy.c").write_bytes(LATIN1_SOURCE)
    g("add", ".")
    g("commit", "-qm", "fix: touch a latin-1 file")
    return tmp_path, base, g("rev-parse", "HEAD")


@pytest.mark.parametrize(
    "args",
    [
        ["show", "HEAD"],
        ["diff", "HEAD~1", "HEAD"],
        ["log", "-p", "-1"],
    ],
)
def test_run_git_does_not_raise_on_non_utf8_content(latin1_repo, args):
    repo, _, _ = latin1_repo
    _run_git(str(repo), args)  # must not raise UnicodeDecodeError


def test_range_diff_survives_non_utf8_content(latin1_repo):
    repo, base, head = latin1_repo
    git_range_diff(str(repo), f"{base}..{head}", f"{base}..{head}")


def test_format_patch_bytes_is_byte_exact(latin1_repo):
    """The candidate patch is a deliverable — it has to apply."""
    repo, base, _ = latin1_repo
    expected = subprocess.run(
        ["git", "format-patch", "--stdout", base], cwd=repo, capture_output=True
    ).stdout
    produced = git_format_patch_bytes(str(repo), base_ref=base)
    assert produced == expected
    assert b"Bj\xf8rn" in produced, "raw bytes must survive, not be replaced"


def test_format_patch_text_is_lossy_but_safe(latin1_repo):
    """The text view is for the dossier, so replacement there is intended."""
    repo, base, _ = latin1_repo
    assert "�" in git_format_patch(str(repo), base_ref=base)


def test_format_patch_bytes_names_a_git_failure(latin1_repo, caplog):
    """An empty result is how both "command failed" and "empty range" look to
    callers; the failure path must at least be diagnosable from the log."""
    import logging

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.git_tools")
    repo, _, _ = latin1_repo

    produced = git_format_patch_bytes(str(repo), base_ref="does-not-exist")

    assert produced == b""
    assert "git format-patch does-not-exist failed" in caplog.text


def test_format_patch_bytes_count_form_is_not_a_warning(latin1_repo, caplog):
    """The default ``-<n>`` form (and a falsy ``base_ref`` falling back to it)
    must produce a patch without tripping the failure warning."""
    import logging

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.git_tools")
    repo, _, _ = latin1_repo

    for produced in (
        git_format_patch_bytes(str(repo), n=1),
        git_format_patch_bytes(str(repo), n=1, base_ref=""),
    ):
        assert produced.startswith(b"From ")
        assert b"Bj\xf8rn" in produced

    assert "format-patch" not in caplog.text


def test_patch_id_matches_gits_own_on_non_utf8_content(latin1_repo):
    """A patch-id over lossily-decoded text would gate Phase 0 on a wrong value."""
    repo, _, head = latin1_repo
    show = subprocess.run(["git", "show", head], cwd=repo, capture_output=True).stdout
    expected = (
        subprocess.run(
            ["git", "patch-id", "--stable"], cwd=repo, input=show, capture_output=True
        )
        .stdout.decode()
        .split()[0]
    )
    assert git_patch_id(str(repo), head) == expected


# --- cherry-pick --empty=drop portability ---------------------------------
#
# `--empty=drop` exists only in git >= 2.45. RHEL 8/9, Ubuntu 22.04/24.04,
# Debian 12 and Fedora <= 40 ship older git, where the option is rejected
# while parsing arguments: the pick exits non-zero with no conflicts, which
# series mode reads as "unmappable cherry-pick (all strategies failed)".


def test_cherry_pick_passes_empty_drop_when_supported(monkeypatch, tmp_git_repo):
    calls = []

    def fake_run_git(repo_path, args, **kwargs):
        calls.append(args)
        return GitResult(0, "", "")

    monkeypatch.setattr(git_tools, "_git_cherry_pick_supports_empty_drop", lambda: True)
    monkeypatch.setattr(git_tools, "is_merge_commit", lambda *_: False)
    monkeypatch.setattr(git_tools, "_run_git", fake_run_git)

    result = git_cherry_pick(tmp_git_repo.repo_path, "abc123", drop_empty=True)

    assert result.success
    assert "--empty=drop" in calls[0]


def test_cherry_pick_omits_empty_drop_on_old_git(monkeypatch, tmp_git_repo):
    calls = []

    def fake_run_git(repo_path, args, **kwargs):
        calls.append(args)
        return GitResult(0, "", "")

    monkeypatch.setattr(
        git_tools, "_git_cherry_pick_supports_empty_drop", lambda: False
    )
    monkeypatch.setattr(git_tools, "is_merge_commit", lambda *_: False)
    monkeypatch.setattr(git_tools, "_run_git", fake_run_git)

    result = git_cherry_pick(tmp_git_repo.repo_path, "abc123", drop_empty=True)

    assert result.success
    assert "--empty=drop" not in calls[0]


def test_cherry_pick_skips_empty_pick_on_old_git(monkeypatch, tmp_git_repo):
    """An empty pick must be dropped, exactly as --empty=drop would."""
    calls = []

    def fake_run_git(repo_path, args, **kwargs):
        calls.append(args)
        if args[:2] == ["cherry-pick", "--skip"]:
            return GitResult(0, "", "")
        return GitResult(
            1,
            "",
            "The previous cherry-pick is now empty, possibly due to conflict "
            "resolution.\nIf you wish to commit it anyway, use:\n\n"
            "    git commit --allow-empty\n",
        )

    monkeypatch.setattr(
        git_tools, "_git_cherry_pick_supports_empty_drop", lambda: False
    )
    monkeypatch.setattr(git_tools, "is_merge_commit", lambda *_: False)
    monkeypatch.setattr(git_tools, "_run_git", fake_run_git)

    result = git_cherry_pick(tmp_git_repo.repo_path, "abc123", drop_empty=True)

    assert result.success
    assert calls[1] == ["cherry-pick", "--skip"]


def test_cherry_pick_does_not_skip_a_real_failure(monkeypatch, tmp_git_repo):
    calls = []

    def fake_run_git(repo_path, args, **kwargs):
        calls.append(args)
        return GitResult(128, "", "fatal: bad object abc123")

    monkeypatch.setattr(
        git_tools, "_git_cherry_pick_supports_empty_drop", lambda: False
    )
    monkeypatch.setattr(git_tools, "is_merge_commit", lambda *_: False)
    monkeypatch.setattr(git_tools, "_run_git", fake_run_git)

    result = git_cherry_pick(tmp_git_repo.repo_path, "abc123", drop_empty=True)

    assert not result.success
    assert len(calls) == 1
    assert calls[0][:2] != ["cherry-pick", "--skip"]


def test_cherry_pick_drops_empty_commit_on_real_repo(monkeypatch, tmp_git_repo):
    """End to end against real git: re-applying a commit is dropped, not fatal."""
    monkeypatch.setattr(
        git_tools, "_git_cherry_pick_supports_empty_drop", lambda: False
    )
    rp = tmp_git_repo.repo_path
    subprocess.run(
        ["git", "checkout", "-q", tmp_git_repo.target_branch], check=True, cwd=rp
    )

    assert git_cherry_pick(rp, tmp_git_repo.fix_sha, drop_empty=True).success

    # Second application leaves nothing to commit — the emulated drop keeps
    # the series moving instead of failing the cluster.
    assert git_cherry_pick(rp, tmp_git_repo.fix_sha, drop_empty=True).success

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=rp, capture_output=True, text=True
    ).stdout
    assert status.strip() == ""
