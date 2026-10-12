# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.ingest — Phase -1 patch materialization."""

import os
import subprocess
import textwrap

import pytest

from auto_bug_fix.ingest import (
    _REF_PREFIX,
    IngestError,
    IngestResult,
    _mbox_message_count,
    _sha256_file,
    _slug,
    ingest_patch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git"] + list(args),
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )


def _sha(repo, ref="HEAD"):
    return _git(repo, "rev-parse", ref).stdout.strip()


def _commit(repo, message, files=None):
    """Stage files dict {path: content} and commit."""
    if files:
        for path, content in files.items():
            full = os.path.join(repo, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
            _git(repo, "add", path)
    _git(repo, "commit", "-m", message, "--allow-empty")
    return _sha(repo)


def _make_lore_mbox(format_patch_bytes: bytes) -> bytes:
    """Convert git format-patch output to lore-style mbox (From mboxrd@z ...)."""
    lines = format_patch_bytes.split(b"\n")
    out = []
    for line in lines:
        if line.startswith(b"From ") and len(line.split()) >= 3:
            parts = line.split(b" ", 2)
            out.append(b"From mboxrd@z " + parts[2])
        else:
            out.append(line)
    return b"\n".join(out)


def _make_series_mbox(patch_repo, lore=False) -> str:
    """Create a 2-commit series mbox in patches_dir; restores main to base_sha."""
    repo = patch_repo["repo"]
    _git(repo, "reset", "--hard", patch_repo["fix_sha"])
    _commit(
        repo,
        "followup fix",
        {
            "lib/utils.c": "int safe_copy() { /* bounds check v2 */ return 2; }\n",
        },
    )
    raw = subprocess.run(
        ["git", "format-patch", "-2", "--stdout"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    if lore:
        raw = _make_lore_mbox(raw)
    path = os.path.join(
        patch_repo["patches_dir"],
        "lore-series.mbox" if lore else "series.mbox",
    )
    with open(path, "wb") as f:
        f.write(raw)
    _git(repo, "reset", "--hard", patch_repo["base_sha"])
    return path


@pytest.fixture
def patch_repo(tmp_path):
    """Create a repo with main (2 commits) and a patch file for the second commit."""
    repo = str(tmp_path / "repo")
    workdir = str(tmp_path / "workdir")
    patches = tmp_path / "patches"
    patches.mkdir()

    os.makedirs(repo)
    os.makedirs(workdir)

    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")

    _commit(repo, "initial", {"lib/utils.c": "int safe_copy() { return 0; }\n"})
    base_sha = _sha(repo)

    _git(repo, "branch", "release-1.0")

    fix_sha = _commit(
        repo,
        "fix: CVE-2026-1234 buffer overflow",
        {
            "lib/utils.c": "int safe_copy() { /* bounds check */ return 1; }\n",
        },
    )

    diff_patch = str(patches / "fix.patch")
    with open(diff_patch, "w") as f:
        result = _git(repo, "diff", f"{fix_sha}^", fix_sha)
        f.write(result.stdout)

    mbox_patch = str(patches / "fix.mbox")
    format_patch_raw = subprocess.run(
        ["git", "format-patch", "-1", "--stdout", fix_sha],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    with open(mbox_patch, "wb") as f:
        f.write(format_patch_raw)

    lore_patch = str(patches / "fix.lore.mbox")
    with open(lore_patch, "wb") as f:
        f.write(_make_lore_mbox(format_patch_raw))

    _git(repo, "reset", "--hard", base_sha)

    return {
        "repo": repo,
        "workdir": workdir,
        "base_sha": base_sha,
        "fix_sha": fix_sha,
        "diff_patch": diff_patch,
        "mbox_patch": mbox_patch,
        "lore_patch": lore_patch,
        "format_patch_raw": format_patch_raw,
        "patches_dir": str(patches),
    }


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestSlug:
    def test_simple(self):
        assert _slug("CVE-2026-1234") == "CVE-2026-1234"

    def test_special_chars(self):
        assert _slug("issue/foo:bar") == "issue-foo-bar"

    def test_empty(self):
        assert _slug("") == "patch"

    def test_collapses_dashes(self):
        assert _slug("a::b") == "a-b"


class TestSha256:
    def test_computes_hash(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello world\n")
        h = _sha256_file(str(p))
        assert len(h) == 64
        assert h == _sha256_file(str(p))


class TestMboxMessageCount:
    def test_format_patch_single(self, patch_repo):
        assert _mbox_message_count(patch_repo["repo"], patch_repo["mbox_patch"]) == 1

    def test_bare_diff(self, patch_repo):
        assert _mbox_message_count(patch_repo["repo"], patch_repo["diff_patch"]) == 0

    def test_lore_style_single(self, patch_repo):
        assert _mbox_message_count(patch_repo["repo"], patch_repo["lore_patch"]) == 1

    def test_body_from_line_not_counted(self, patch_repo):
        """A commit body containing 'From now on...' must not be counted as a second message."""
        tricky = os.path.join(patch_repo["patches_dir"], "tricky.mbox")
        raw = patch_repo["format_patch_raw"]
        # Inject a body line starting with "From " before the diff
        parts = raw.split(b"\n---\n", 1)
        injected = parts[0] + b"\n\nFrom now on, bounds are checked.\n---\n" + parts[1]
        with open(tricky, "wb") as f:
            f.write(injected)
        assert _mbox_message_count(patch_repo["repo"], tricky) == 1

    def test_multi_patch_format_patch(self, patch_repo):
        """A 2-commit format-patch series should count as 2."""
        series = _make_series_mbox(patch_repo)
        assert _mbox_message_count(patch_repo["repo"], series) == 2

    def test_multi_patch_lore_style(self, patch_repo):
        """A 2-commit lore-style series should count as 2."""
        series = _make_series_mbox(patch_repo, lore=True)
        assert _mbox_message_count(patch_repo["repo"], series) == 2


# ---------------------------------------------------------------------------
# Integration tests — bare diff
# ---------------------------------------------------------------------------


class TestIngestDiff:
    def test_basic_ingest(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )

        assert isinstance(result, IngestResult)
        assert len(result.synthetic_commit) == 40
        assert result.base_ref == "main"
        assert result.base_sha == patch_repo["base_sha"]
        assert result.method == "apply"
        assert len(result.patch_sha256) == 64
        assert result.warnings == []

    def test_commit_has_trailer(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        msg = _git(
            patch_repo["repo"], "log", "-1", "--format=%B", result.synthetic_commit
        ).stdout
        assert "X-Amalgamator-Patch:" in msg
        assert f"sha256:{result.patch_sha256}" in msg

    def test_ref_pinned(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        ref_sha = _sha(patch_repo["repo"], f"{_REF_PREFIX}/CVE-2026-1234")
        assert ref_sha == result.synthetic_commit

    def test_worktree_cleaned_up(self, patch_repo):
        ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        ingest_dir = os.path.join(patch_repo["workdir"], "ingest_CVE-2026-1234")
        assert not os.path.exists(ingest_dir)

    def test_idempotent_reingest(self, patch_repo):
        r1 = ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        r2 = ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        assert r1.synthetic_commit == r2.synthetic_commit
        assert r2.method == "reuse"

    def test_whitespace_retry_records_warning(self, tmp_path):
        """A patch that only applies with --ignore-whitespace should succeed with a warning."""
        repo = str(tmp_path / "ws_repo")
        workdir = str(tmp_path / "ws_workdir")
        os.makedirs(repo)
        os.makedirs(workdir)

        _git(repo, "init", "-b", "main")
        _git(repo, "config", "user.name", "Test User")
        _git(repo, "config", "user.email", "test@example.com")

        # Base file uses tabs for indentation
        base_file = os.path.join(repo, "file.c")
        with open(base_file, "w") as f:
            f.write("#include <stdio.h>\n" "\n" "int foo() {\n" "\treturn 0;\n" "}\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "initial")

        # Patch uses spaces — strict apply fails, --ignore-whitespace succeeds
        ws_patch = str(tmp_path / "ws.patch")
        with open(ws_patch, "w") as f:
            f.write(textwrap.dedent("""\
                diff --git a/file.c b/file.c
                --- a/file.c
                +++ b/file.c
                @@ -1,5 +1,5 @@
                 #include <stdio.h>

                 int foo() {
                -    return 0;
                +    return 1;
                 }
            """))

        result = ingest_patch(repo, ws_patch, "main", "WS-TEST", "ws fix", workdir)
        assert result.method == "apply-ws"
        assert "applied with --ignore-whitespace" in result.warnings

    def test_dirty_source_unaffected(self, patch_repo):
        dirty_file = os.path.join(patch_repo["repo"], "dirty.txt")
        with open(dirty_file, "w") as f:
            f.write("uncommitted work")

        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["diff_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        assert os.path.exists(dirty_file)
        assert len(result.synthetic_commit) == 40


# ---------------------------------------------------------------------------
# Integration tests — mbox
# ---------------------------------------------------------------------------


class TestIngestMbox:
    def test_mbox_preserves_author(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["mbox_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        assert result.method == "am"
        author = _git(
            patch_repo["repo"],
            "log",
            "-1",
            "--format=%an",
            result.synthetic_commit,
        ).stdout.strip()
        assert author == "Test User"

    def test_mbox_preserves_subject(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["mbox_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        subject = _git(
            patch_repo["repo"],
            "log",
            "-1",
            "--format=%s",
            result.synthetic_commit,
        ).stdout.strip()
        assert "CVE-2026-1234" in subject or "buffer overflow" in subject.lower()

    def test_mbox_has_trailer(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["mbox_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        msg = _git(
            patch_repo["repo"], "log", "-1", "--format=%B", result.synthetic_commit
        ).stdout
        assert "X-Amalgamator-Patch:" in msg
        assert f"sha256:{result.patch_sha256}" in msg

    def test_lore_style_mbox_uses_am(self, patch_repo):
        """lore.kernel.org-style mbox (From mboxrd@z ...) must route to git am."""
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["lore_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        assert result.method == "am"

    def test_lore_style_preserves_author(self, patch_repo):
        result = ingest_patch(
            patch_repo["repo"],
            patch_repo["lore_patch"],
            "main",
            "CVE-2026-1234",
            "buffer overflow fix",
            patch_repo["workdir"],
        )
        author = _git(
            patch_repo["repo"],
            "log",
            "-1",
            "--format=%an",
            result.synthetic_commit,
        ).stdout.strip()
        assert author == "Test User"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestIngestErrors:
    def test_patch_not_found(self, patch_repo):
        with pytest.raises(IngestError, match="not found"):
            ingest_patch(
                patch_repo["repo"],
                "/nonexistent/patch.diff",
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )

    def test_bad_base_ref(self, patch_repo):
        with pytest.raises(IngestError, match="Cannot resolve"):
            ingest_patch(
                patch_repo["repo"],
                patch_repo["diff_patch"],
                "nonexistent-branch",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )

    def test_patch_does_not_apply(self, patch_repo):
        bad_patch = os.path.join(patch_repo["patches_dir"], "bad.patch")
        with open(bad_patch, "w") as f:
            f.write(textwrap.dedent("""\
                diff --git a/nonexistent.c b/nonexistent.c
                --- a/nonexistent.c
                +++ b/nonexistent.c
                @@ -1,3 +1,3 @@
                 context that
                -does not
                +will not
                 match anything
            """))
        with pytest.raises(IngestError, match="does not apply"):
            ingest_patch(
                patch_repo["repo"],
                bad_patch,
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )

    def test_multi_patch_format_patch_rejected(self, patch_repo):
        series = _make_series_mbox(patch_repo)
        with pytest.raises(IngestError, match="Multi-patch"):
            ingest_patch(
                patch_repo["repo"],
                series,
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )

    def test_multi_patch_lore_style_rejected(self, patch_repo):
        """A lore-style multi-patch series must be rejected, not silently squashed."""
        series = _make_series_mbox(patch_repo, lore=True)
        with pytest.raises(IngestError, match="Multi-patch"):
            ingest_patch(
                patch_repo["repo"],
                series,
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )

    def test_worktree_cleaned_on_failure(self, patch_repo):
        bad_patch = os.path.join(patch_repo["patches_dir"], "bad.patch")
        with open(bad_patch, "w") as f:
            f.write("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n")

        with pytest.raises(IngestError):
            ingest_patch(
                patch_repo["repo"],
                bad_patch,
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )
        ingest_dir = os.path.join(patch_repo["workdir"], "ingest_CVE-2026-1234")
        assert not os.path.exists(ingest_dir)

    def test_already_applied_diff(self, patch_repo):
        """A diff that's already on the base should produce a clear error."""
        _git(patch_repo["repo"], "reset", "--hard", patch_repo["fix_sha"])
        with pytest.raises(IngestError, match="already applied|no changes"):
            ingest_patch(
                patch_repo["repo"],
                patch_repo["diff_patch"],
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )

    def test_failure_message_includes_stderr(self, patch_repo):
        """Non-applying patch error should include git's diagnostic output."""
        bad_patch = os.path.join(patch_repo["patches_dir"], "bad.patch")
        with open(bad_patch, "w") as f:
            f.write(textwrap.dedent("""\
                diff --git a/nonexistent.c b/nonexistent.c
                --- a/nonexistent.c
                +++ b/nonexistent.c
                @@ -1,3 +1,3 @@
                 context that
                -does not
                +will not
                 match anything
            """))
        with pytest.raises(IngestError, match="git apply:"):
            ingest_patch(
                patch_repo["repo"],
                bad_patch,
                "main",
                "CVE-2026-1234",
                "desc",
                patch_repo["workdir"],
            )
