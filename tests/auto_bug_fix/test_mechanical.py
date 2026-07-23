"""Tests for auto_bug_fix.mechanical — git-native lenient conflict resolution."""
from __future__ import annotations

import subprocess

import pytest

from auto_bug_fix.mechanical import _is_ws_sensitive, resolve_mechanically

BASE = """\
void foo(void)
{
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    return;
}
"""

FIXED = BASE.replace("    int c = 3;", "    int c = 30; /* fixed */")

# Target reindented the body with tabs — a whitespace-only divergence that
# still conflicts the cherry-pick because the fix touches a reindented line.
REINDENTED = """\
void foo(void)
{
\tint a = 1;
\tint b = 2;
\tint c = 3;
\tint d = 4;
\treturn;
}
"""


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _status(cwd):
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout


def _make_conflicted_repo(tmp_path, target_content):
    """Base -> fix on main; divergent target; cherry-pick fix onto target."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    (repo / "base.c").write_text(BASE)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base_sha = _sha(rp)

    (repo / "base.c").write_text(FIXED)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix c")
    fix_sha = _sha(rp)

    _git(rp, "branch", "target", base_sha)
    _git(rp, "checkout", "-q", "target")
    (repo / "base.c").write_text(target_content)
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "target divergence")

    result = subprocess.run(
        ["git", "cherry-pick", fix_sha], cwd=rp, capture_output=True, text=True,
    )
    assert result.returncode != 0, "fixture must produce a conflict"
    assert "UU base.c" in _status(rp)
    return rp, fix_sha, repo / "base.c"


def test_whitespace_drift_conflict_is_resolved(tmp_path):
    rp, fix_sha, path = _make_conflicted_repo(tmp_path, REINDENTED)
    resolved = resolve_mechanically(rp, fix_sha, ["base.c"])
    assert resolved == {"base.c": "apply-ignore-ws"}
    content = path.read_text()
    assert "int c = 30; /* fixed */" in content
    assert "<<<<<<<" not in content
    assert "UU base.c" not in _status(rp)  # staged as resolved


def test_content_conflict_is_left_conflicted(tmp_path):
    diverged = BASE.replace("    int c = 3;", "    int c = 99;")
    rp, fix_sha, path = _make_conflicted_repo(tmp_path, diverged)
    resolved = resolve_mechanically(rp, fix_sha, ["base.c"])
    assert resolved == {}
    # The conflicted working-tree state must be restored for the LLM.
    assert "<<<<<<<" in path.read_text()
    assert "UU base.c" in _status(rp)


def test_crlf_line_endings_are_preserved(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    _git(rp, "config", "core.autocrlf", "false")

    (repo / "base.c").write_bytes(BASE.replace("\n", "\r\n").encode())
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base_sha = _sha(rp)

    (repo / "base.c").write_bytes(FIXED.replace("\n", "\r\n").encode())
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix c")
    fix_sha = _sha(rp)

    _git(rp, "branch", "target", base_sha)
    _git(rp, "checkout", "-q", "target")
    (repo / "base.c").write_bytes(REINDENTED.replace("\n", "\r\n").encode())
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "target divergence")

    result = subprocess.run(
        ["git", "cherry-pick", fix_sha], cwd=rp, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "UU base.c" in _status(rp)

    resolved = resolve_mechanically(rp, fix_sha, ["base.c"])
    assert resolved == {"base.c": "apply-ignore-ws"}
    content = (repo / "base.c").read_bytes()
    assert b"int c = 30; /* fixed */\r\n" in content
    # Every line keeps its CRLF ending — nothing was silently normalized.
    assert content.count(b"\n") == content.count(b"\r\n")


@pytest.mark.parametrize("path,sensitive", [
    ("pkg/module.py", True),
    ("deploy/app.yaml", True),
    ("Makefile", True),
    ("build/Makefile.inc", True),
    ("rules.mk", True),
    ("drivers/misc/sample_driver.c", False),
    ("include/linux/mm.h", False),
])
def test_ws_sensitive_detection(path, sensitive):
    assert _is_ws_sensitive(path) is sensitive
