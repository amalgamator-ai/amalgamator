"""Tests for auto_bug_fix.snapback — whitespace snap-back to source bytes."""
from __future__ import annotations

import subprocess

from auto_bug_fix.snapback import normalize_ws, snap_back_files, snap_back_text


def test_normalize_ws_collapses_runs():
    assert normalize_ws("\t\tint  a =\t1;  ") == "int a = 1;"


def test_indent_drift_snaps_to_upstream_bytes():
    upstream = "\n".join([
        "int core_glock_get(struct core_sbd *sdp, u64 number,",
        "\t\t   const struct core_glock_operations *glops,",
        "\t\t   int create, struct core_glock **glp);",
    ])
    current = "\n".join([
        "int core_glock_get(struct core_sbd *sdp, u64 number,",
        "           const struct core_glock_operations *glops,",
        "           int create, struct core_glock **glp);",
    ])
    result, snapped = snap_back_text(current, [upstream])
    assert result == upstream
    assert snapped == 2  # each drifted line snaps to its unique source form


def test_byte_equal_source_line_is_never_touched():
    upstream = "\tint x = compute(a, b);"
    preimage = "    int x = compute(a, b);"
    # Current uses the preimage's exact bytes — a real source form, keep it.
    result, snapped = snap_back_text(preimage, [upstream, preimage])
    assert result == preimage
    assert snapped == 0


def test_ambiguous_byte_forms_are_skipped():
    source = "\n".join([
        "\tcall_something(a, b, c);",
        "\t\tcall_something(a, b, c);",
    ])
    current = "  call_something(a, b, c);"
    result, snapped = snap_back_text(current, [source])
    assert result == current
    assert snapped == 0


def test_short_keys_are_not_snapped():
    source = "\treturn 0;"
    current = "  return 0;"
    result, snapped = snap_back_text(current, [source])
    assert result == current
    assert snapped == 0


def test_preimage_is_fallback_source():
    preimage = "\tstatic const struct file_operations core_fops = { .owner = THIS_MODULE };"
    current = "        static const struct file_operations core_fops = { .owner = THIS_MODULE };"
    result, snapped = snap_back_text(current, ["/* unrelated upstream */", preimage])
    assert result == preimage
    assert snapped == 1


def test_upstream_priority_over_preimage():
    upstream = "\tmutex_lock(&sdp->core_freeze_mutex);"
    preimage = "        mutex_lock(&sdp->core_freeze_mutex);"
    current = "  mutex_lock(&sdp->core_freeze_mutex);"
    result, snapped = snap_back_text(current, [upstream, preimage])
    assert result == upstream
    assert snapped == 1


def test_novel_lines_untouched():
    source = "void unrelated(void);"
    current = "\tint totally_new_code = adapt_to_target(x, y);"
    result, snapped = snap_back_text(current, [source])
    assert result == current
    assert snapped == 0


def test_no_sources_is_noop():
    current = "  anything at all goes here();"
    result, snapped = snap_back_text(current, [])
    assert result == current
    assert snapped == 0


def test_ws_sensitive_files_are_never_snapped(tmp_path):
    # Indentation is semantics in Python/YAML — snap-back must skip them
    # entirely (the sources aren't even consulted, so no repo is needed).
    drifted = "def handler(self, request, response, extra_argument_name):\n"
    (tmp_path / "app.py").write_text(drifted)
    counts = snap_back_files(str(tmp_path), ["app.py"], "HEAD", "HEAD")
    assert counts == {}
    assert (tmp_path / "app.py").read_text() == drifted


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_snap_back_files_uses_git_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    rp = str(repo)
    _git(rp, "init", "-b", "main")
    _git(rp, "config", "user.name", "t")
    _git(rp, "config", "user.email", "t@t")

    (repo / "f.c").write_text("void foo(void);\n")
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "base")

    (repo / "f.c").write_text(
        "void foo(void);\n"
        "int core_glock_get(struct core_sbd *sdp,\n"
        "\t\t   const struct core_glock_operations *glops);\n"
    )
    _git(rp, "add", ".")
    _git(rp, "commit", "-m", "fix")
    fix_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rp, capture_output=True, text=True, check=True,
    ).stdout.strip()
    _git(rp, "checkout", "-q", "HEAD~1")

    # Working tree holds an LLM-drifted resolution: spaces instead of tabs.
    (repo / "f.c").write_text(
        "void foo(void);\n"
        "int core_glock_get(struct core_sbd *sdp,\n"
        "           const struct core_glock_operations *glops);\n"
    )
    counts = snap_back_files(rp, ["f.c"], "HEAD", fix_sha)
    assert counts == {"f.c": 1}
    assert "\t\t   const struct core_glock_operations *glops);" in (repo / "f.c").read_text()
