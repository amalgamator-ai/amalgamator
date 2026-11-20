# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packaging-carried changes (dist-git, RPM/Debian patch stacks)."""

from __future__ import annotations

import subprocess

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.packaging import (
    carrier_paths,
    carrier_payload,
    is_packaging_path,
)
from auto_bug_fix.run_bug_fix import (
    _is_test_top_dir,
    _packaging_build_output_paths,
    _packaging_only,
    _pre_red_packaging_mode,
    _session_test_files,
    _shares_content_with_fix,
    _validate_fused_pre_red,
)

TEST_PATCH = """\
diff --git a/tests/foo_test.c b/tests/foo_test.c
new file mode 100644
--- /dev/null
+++ b/tests/foo_test.c
@@ -0,0 +1,2 @@
+int test_foo(void) { return 0; }
+/* long enough to be payload */
"""

PROD_PATCH = TEST_PATCH.replace("tests/foo_test.c", "src/foo.c")


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def _config(rp, fix):
    return BugFixConfig(
        repo_path=rp,
        build_dir=rp,
        source_branch="main",
        target_branch="target",
        source_fix_commit=fix,
        bug_description="test",
        issue_id="TEST-1",
    )


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "base.c").write_text("int x = 1;\n")
    (repo / "pkg.spec").write_text("Name: test\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)
    (repo / "base.c").write_text("int x = 2;\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix = _sha(rp)
    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "target")
    return rp, fix


def test_carrier_paths_and_payload():
    assert carrier_paths(TEST_PATCH) == ["tests/foo_test.c"]
    payload = carrier_payload(TEST_PATCH, 6)
    assert ("+", "int test_foo(void) { return 0; }") in payload
    assert not any(line.startswith("+++") for _sign, line in payload)


def test_carrier_payload_keeps_content_starting_with_diff_markers():
    content = """\
diff --git a/x.c b/x.c
--- a/x.c
+++ b/x.c
@@ -1,2 +1,1 @@
---counter;
-    old_counter;
+++i += 1;
"""
    payload = carrier_payload(content, 6)
    assert ("+", "++i += 1;") in payload
    assert ("-", "--counter;") in payload


def test_packaging_path_classification():
    assert is_packaging_path("pkg.spec")
    assert is_packaging_path("debian/patches/fix.patch")
    assert is_packaging_path("fix.patch")
    assert not is_packaging_path("src/foo.c")
    assert _packaging_only(["pkg.spec", "fix.patch"])
    assert not _packaging_only(["pkg.spec", "src/foo.c"])
    assert not _packaging_only([])


def test_source_repo_with_untracked_patch_is_not_packaging(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "base.c").write_text("int x = 1;\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    (repo / "notes.patch").write_text("just a fixture\n")
    (repo / "new_source.c").write_text("int added;\n")
    untracked = ["notes.patch", "new_source.c"]

    assert _pre_red_packaging_mode(rp, _sha(rp), untracked) is False
    assert _packaging_build_output_paths(rp, _sha(rp), untracked) == set()


def test_packaging_build_output_keeps_top_level_carriers_and_tests(tmp_path):
    rp, _fix = _repo(tmp_path)
    paths = [
        "pkg.spec",
        "fix.patch",
        "debian/patches/fix.patch",
        "regression-tests/new_test.c",
        "libexpat-R_2_2_5/tests/fixture.patch",
        "libexpat-R_2_2_5/expat/tests/runtests.c",
        "aarch64/expat.rpm",
    ]

    assert _pre_red_packaging_mode(rp, _sha(rp), paths) is True
    assert _packaging_build_output_paths(rp, _sha(rp), paths) == {
        "libexpat-R_2_2_5/tests/fixture.patch",
        "libexpat-R_2_2_5/expat/tests/runtests.c",
        "aarch64/expat.rpm",
    }
    assert _is_test_top_dir("regression-tests")
    assert _is_test_top_dir("unittests")
    assert not _is_test_top_dir("latest")


def test_validate_fused_pre_red_accepts_spec_plus_test_patch(tmp_path):
    rp, fix = _repo(tmp_path)
    base_head = _sha(rp)
    (tmp_path / "repo" / "foo-tests.patch").write_text(TEST_PATCH)
    (tmp_path / "repo" / "pkg.spec").write_text("Name: test\nPatch1: foo-tests.patch\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "test-port: TEST-1")

    assert _validate_fused_pre_red(_config(rp, fix), base_head) == []
    assert _session_test_files(rp, base_head) == ["tests/foo_test.c"]


def test_validate_fused_pre_red_rejects_production_patch_payload(tmp_path):
    rp, fix = _repo(tmp_path)
    base_head = _sha(rp)
    (tmp_path / "repo" / "prod.patch").write_text(PROD_PATCH)
    (tmp_path / "repo" / "pkg.spec").write_text("Name: test\nPatch1: prod.patch\n")
    _git(rp, "add", "-A")
    _git(rp, "commit", "-m", "test-port: TEST-1")

    problems = _validate_fused_pre_red(_config(rp, fix), base_head)
    assert any("non-test paths" in p for p in problems)


def test_shares_content_with_fix_unwraps_added_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")
    (repo / "a.c").write_text("int f(void) {\n    return 1;\n}\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")
    base = _sha(rp)

    (repo / "a.c").write_text(
        "int f(void) {\n"
        "    validate_bounds(ptr, length);\n"
        "    audit_the_caller(ctx);\n"
        "    check_capacity(capacity, length);\n"
        "    return 1;\n"
        "}\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix = _sha(rp)

    _git(rp, "branch", "target", base)
    _git(rp, "checkout", "-q", "--detach", "target")
    diff = subprocess.run(
        ["git", "diff", f"{fix}^", fix],
        cwd=rp,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (repo / "fix.patch").write_text(diff)
    _git(rp, "add", "fix.patch")
    _git(rp, "commit", "-m", "packaging backport")

    assert _shares_content_with_fix(_config(rp, fix)) is True
