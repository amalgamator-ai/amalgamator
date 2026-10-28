# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build output must never reach a commit boundary, and the guard must never
be able to hide agent work.

The two failure modes pulled in opposite directions: an exclude that fenced
``repository.build_subdir`` unconditionally silently dropped fixes ported
into a *source* subdirectory (the checked-in example uses ``build_subdir:
src``), while no guard at all swept agent-run build output into the
candidate patch. These tests pin the resolution: fence only a build
directory git knows nothing about, learn what the orchestrator's own
verifications create, and never trust ``.git/info/exclude``.
"""

import os
import pathlib
import subprocess

import pytest

from auto_bug_fix import run_bug_fix as rbf
from auto_bug_fix.git_tools import GitResult, git_status_porcelain
from auto_bug_fix.run_bug_fix import (
    PipelineEscalation,
    StageResult,
    _branch_checkout_holders,
    _commit_recovery_edits,
    _commit_worktree_edits,
    _conflicted_files,
    _discard_verification_dirt,
    _empty_dirs,
    _fence_build_dir,
    _finish_verification,
    _force_resolve_for_phase_4b,
    _non_artifact_dirt,
    _porcelain_line_paths,
    _prune_emptied_dirs,
    _restore_target_baseline,
    _stage_agent_work,
    _stage_or_escalate,
    _worktree_snapshot,
)
from tests.auto_bug_fix.test_run_bug_fix import _git_4b, _phase_4b_fixtures


def _staged(tmp_path):
    return _git_4b(tmp_path, "diff", "--cached", "--name-only")


def _learn(config, state, *paths):
    """Record generated paths the way a discard does: name plus content."""
    state.generated_output |= set(paths)
    rbf._record_generated_digests(config, state, set(paths))


# ---------- The fence: untracked build dirs only ----------


def _fail_ls_tree(monkeypatch):
    """Make the tree lookup that decides the fence unreadable."""

    def fake(repo, args, *a, **k):
        if args and args[0] == "ls-tree":
            return GitResult(1, "", "simulated ls-tree failure")
        return real_run_git(repo, args, *a, **k)

    real_run_git = rbf._run_git
    monkeypatch.setattr(rbf, "_run_git", fake)


def test_fence_build_dir_fails_closed_only_where_it_can_restage(tmp_path, monkeypatch):
    """An unreadable tree must not decide the fence for the strip — there the
    fence only ever removes more — but the staging boundary re-stages what it
    cannot account for, so it can afford to fence."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.rpm").write_text("build output\n")
    _fail_ls_tree(monkeypatch)

    assert _fence_build_dir(config, fail_closed=True) == "rpm-build"
    assert _fence_build_dir(config) is None


def test_staging_fences_an_unreadable_build_tree_without_dropping_source(
    tmp_path, monkeypatch
):
    """Fencing on an unknown tree keeps output out; the restage keeps the
    source the same unknown tree might hold."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.rpm").write_text("build output\n")
    _learn(config, state, "rpm-build/artifact.rpm")
    (tmp_path / "rpm-build" / "ported.c").write_text("int b; /* source */\n")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")
    _fail_ls_tree(monkeypatch)

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "rpm-build"
    assert result.ok
    staged = _staged(tmp_path).splitlines()
    assert "src.c" in staged
    assert "rpm-build/ported.c" in staged
    assert "rpm-build/artifact.rpm" not in staged


def test_stage_agent_work_fences_an_untracked_build_dir(tmp_path):
    """Build output in a directory git knows nothing about stays unstaged, once
    the orchestrator's own verification has been seen to create it — which is
    the state every agent boundary runs in, Phase 1 having built first."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.rpm").write_text("build output\n")
    state.generated_output = {"rpm-build/artifact.rpm"}
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "rpm-build"
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_stage_agent_work_fences_a_gitignored_build_dir_wholesale(tmp_path):
    """A gitignored build tree needs no learning at all: `git status` never
    reports it, so the subtree exclusion keeps its full effect and the
    thousand-file sweep an agent's `git add -A` would cause is impossible."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / ".gitignore").write_text("rpm-build/\n")
    _git_4b(tmp_path, "add", ".gitignore")
    _git_4b(tmp_path, "commit", "-m", "ignore the build tree")
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    for n in range(5):
        (tmp_path / "rpm-build" / f"artifact{n}.o").write_text("output\n")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "rpm-build"
    assert result.restaged == frozenset()
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_stage_agent_work_keeps_new_source_in_an_empty_build_subdir(tmp_path):
    """`build_subdir` means "where the commands run", and the checked-in example
    uses `src`. An *empty* one satisfies the fence's "nothing tracked at HEAD"
    proxy exactly like a build tree does, so fencing it on that proxy alone
    silently dropped the fix ported there. Content the pipeline cannot account
    for as build output is staged instead."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "src")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new_helper.c").write_text("int n = 1; /* the fix */\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "src"
    assert result.restaged == frozenset({"src/new_helper.c"})
    assert "src/new_helper.c" in _staged(tmp_path).splitlines()


def test_stage_agent_work_separates_learned_output_from_new_source(tmp_path):
    """The two live side by side in the same fenced directory: the learned
    artifact stays out, the unaccounted source goes in."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "build")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "artifact.o").write_text("output\n")
    (tmp_path / "build" / "ported_fix.c").write_text("int n = 1;\n")
    state.generated_output = {"build/artifact.o"}

    result = _stage_agent_work(config, state)

    assert result.restaged == frozenset({"build/ported_fix.c"})
    assert _staged(tmp_path).splitlines() == ["build/ported_fix.c"]


def test_stage_agent_work_fails_closed_when_the_fence_cannot_be_checked(tmp_path):
    """An unreadable status cannot show the fence is dropping build output and
    nothing else, so the boundary reports an incomplete stage."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.o").write_text("output\n")

    result = rbf._restage_unaccounted_fenced(config, "rpm-build", None, set())

    assert result == (frozenset(), False)


def test_stage_agent_work_clears_a_build_dir_the_agent_staged_in(tmp_path):
    """The fence decision is about HEAD, not the index: an agent that stages a
    build artifact itself must not switch the fence off (the next `git add -A`
    would sweep the whole build tree in), and whatever it staged there is
    unstaged again — the pipeline owns every commit."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.rpm").write_text("build output\n")
    _git_4b(tmp_path, "add", "rpm-build/artifact.rpm")
    (tmp_path / "rpm-build" / "unrelated.o").write_text("more output\n")
    state.generated_output = {"rpm-build/artifact.rpm", "rpm-build/unrelated.o"}
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "rpm-build"
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_stage_agent_work_never_fences_a_tracked_build_subdir(tmp_path):
    """`build_subdir` means "where the commands run", not "generated": when it
    holds tracked source (the checked-in example uses `src`), a file the
    agent adds there is a fix and must reach the patch."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "existing.c").write_text("int e = 1;\n")
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "tracked source under build_subdir")
    config.build_dir = str(tmp_path / "src")
    (tmp_path / "src" / "new_helper.c").write_text("int n = 1;\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir is None
    assert "src/new_helper.c" in _staged(tmp_path).splitlines()


def test_stage_agent_work_learned_output_at_the_repo_root(tmp_path):
    """The default build_dir IS the repo root: nothing can be fenced there,
    so output an earlier verification produced is left unstaged by path."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    assert config.build_dir == str(tmp_path)
    (tmp_path / "gen.o").write_text("agent-run build output\n")
    _learn(config, state, "gen.o")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.generated == {"gen.o"}
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_stage_agent_work_keeps_an_unlearned_untracked_file(tmp_path):
    """Only paths an orchestrator verification actually created are learned;
    a new source file the agent wrote is not build output."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "generated.c").write_text("int g = 1;\n")
    (tmp_path / "gen.o").write_text("output\n")
    _learn(config, state, "gen.o")

    _stage_agent_work(config, state)

    assert _staged(tmp_path).splitlines() == ["generated.c"]


def test_stage_agent_work_unstages_learned_output_the_agent_staged(tmp_path):
    """An agent session that runs `git add` turns its build output into staged
    additions, which the untracked set no longer names. Learned paths must be
    unstaged from the index too, or the output rides into the commit."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("agent-run build output\n")
    _learn(config, state, "gen.o")
    _git_4b(tmp_path, "add", "gen.o")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.generated == {"gen.o"}
    assert _staged(tmp_path).splitlines() == ["src.c"]
    assert (tmp_path / "gen.o").exists(), "unstaged, never deleted"


def test_stage_agent_work_never_unstages_a_learned_path_head_tracks(tmp_path):
    """A learned path was untracked when it was recorded. If HEAD now tracks
    it, it is source: unstaging a modification there would drop a real edit
    from the commit."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("checked in\n")
    _git_4b(tmp_path, "add", "gen.o")
    _git_4b(tmp_path, "commit", "-m", "track gen.o")
    state.generated_output = {"gen.o"}
    (tmp_path / "gen.o").write_text("agent edit to a tracked file\n")

    result = _stage_agent_work(config, state)

    assert result.generated == set()
    assert _staged(tmp_path).splitlines() == ["gen.o"]


def test_stage_agent_work_keeps_source_the_agent_wrote_at_a_learned_path(tmp_path):
    """`generated_output` records a *path*, and a path is not an identity. A
    verification that creates `pkg/generated.go` teaches the pipeline that
    name; when the agent later authors the ported fix at the same name,
    unstaging it by name dropped the deliverable with nothing but a log line
    to show for it. Content the orchestrator never saw a build write is
    treated as work."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    (tmp_path / "gen.o").write_text("verification build output\n")
    _discard_verification_dirt(config, baseline, state)
    assert state.generated_output == {"gen.o"}
    assert state.generated_digests["gen.o"]

    # The agent now authors real source at the very same path.
    (tmp_path / "gen.o").write_text("int ported_fix = 1;\n")

    result = _stage_agent_work(config, state)

    assert result.generated == set()
    assert _staged(tmp_path).splitlines() == ["gen.o"]


def test_stage_agent_work_still_drops_rebuilt_output_at_a_learned_path(tmp_path):
    """The flip side: the agent runs the same build, the same bytes reappear
    at the learned path, and the boundary still leaves them out."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    (tmp_path / "gen.o").write_text("verification build output\n")
    _discard_verification_dirt(config, baseline, state)

    # The agent's own run of the same build command recreates it.
    (tmp_path / "gen.o").write_text("verification build output\n")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.generated == {"gen.o"}
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_discard_records_what_the_build_wrote_at_each_learned_path(tmp_path):
    """Hashes are taken while the files are still on disk — the clean that
    follows removes them, and nothing could be recorded afterwards."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    (tmp_path / "gen.o").write_text("output\n")
    expected = _git_4b(tmp_path, "hash-object", "gen.o")

    assert _discard_verification_dirt(config, baseline, state) is True

    assert not (tmp_path / "gen.o").exists()
    assert state.generated_digests == {"gen.o": {expected}}


def test_discard_does_not_hash_inside_a_fenced_build_dir(tmp_path):
    """A build tree can hold tens of thousands of files, and the fence
    excludes the whole subtree, so no boundary ever decides about them one by
    one. Hashing them would be pure cost."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    baseline = _worktree_snapshot(config)
    (tmp_path / "rpm-build" / "artifact.o").write_text("output\n")
    (tmp_path / "root-gen.o").write_text("output\n")

    _discard_verification_dirt(config, baseline, state)

    assert "rpm-build/artifact.o" in state.generated_output
    assert set(state.generated_digests) == {"root-gen.o"}


def test_recognized_build_output_requires_content_outside_a_fence(tmp_path):
    """No recorded content is not evidence: hashing can fail, and then a name
    is all the path has. Outside a fenced build directory that is not proof
    enough — the file is kept as work. Inside one the name is all there ever
    is, so the path rule stands there."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("output\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.o").write_text("output\n")
    state.generated_output = {"gen.o", "build/out.o"}

    assert rbf._recognized_build_output(config, state, {"gen.o"}) == set()
    assert rbf._recognized_build_output(config, state, {"build/out.o"}, "build") == {
        "build/out.o"
    }


def test_comparison_snapshot_does_not_capture_content(tmp_path):
    """Only the starting snapshot restores content; the comparison snapshot
    must not write a second unreachable stash commit on every discard."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* dirty */\n")

    baseline = _worktree_snapshot(config)
    assert baseline is not None and baseline.dirty_ref is not None
    comparison = _worktree_snapshot(config, capture_content=False)
    assert comparison is not None and comparison.dirty_ref is None


def test_stage_agent_work_in_a_linked_worktree_leaves_the_source_repo(tmp_path):
    """`.git/info/exclude` is shared by every linked worktree, so using it as
    the fence polluted the source repo and outlived cleanup. Staging must be
    purely worktree-local: pathspecs, no persistent ignore state."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    wt = tmp_path.parent / f"{tmp_path.name}-linked"
    _git_4b(tmp_path, "worktree", "add", str(wt), "HEAD")
    exclude = tmp_path / ".git" / "info" / "exclude"
    exclude_before = exclude.read_text() if exclude.exists() else None

    config.repo_path = str(wt)
    config.build_dir = str(wt / "build")
    (wt / "build").mkdir()
    (wt / "build" / "artifact.o").write_text("output\n")
    state.generated_output = {"build/artifact.o"}
    (wt / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "build"
    assert _git_4b(wt, "diff", "--cached", "--name-only").splitlines() == ["src.c"]
    assert (exclude.read_text() if exclude.exists() else None) == exclude_before


# ---------- The discard: baseline content and learned output ----------


def test_discard_restores_a_file_that_was_already_dirty_at_the_baseline(tmp_path):
    """A path dirty before the verification is not untouchable: its baseline
    content is captured, so the verification's further edit is reverted
    without the pre-existing modification being lost."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* agent work */\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None and baseline.dirty_ref

    # The verification edits the same file again and drops its own output.
    (tmp_path / "src.c").write_text("int a = 3; /* verification dirt */\n")
    (tmp_path / "gen.o").write_text("output\n")

    assert _discard_verification_dirt(config, baseline, state) is True

    assert (tmp_path / "src.c").read_text() == "int a = 2; /* agent work */\n"
    assert not (tmp_path / "gen.o").exists()
    assert state.generated_output == {"gen.o"}


def test_discard_restores_a_baseline_dirty_file_the_verification_reset(tmp_path):
    """A verification can reset a baseline-dirty file to HEAD content. It is
    then no longer "modified", so intersecting the two snapshots missed it and
    the baseline content was silently lost."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 99; /* local work */\n")
    baseline = _worktree_snapshot(config)

    # The verification restores the file to its HEAD content.
    _git_4b(tmp_path, "checkout", "--", "src.c")

    assert _discard_verification_dirt(config, baseline, state) is True
    assert (tmp_path / "src.c").read_text() == "int a = 99; /* local work */\n"


def test_discard_reports_failure_when_a_dirty_baseline_was_not_captured(
    tmp_path, monkeypatch
):
    """Without the baseline content, a verification's edit to an already-dirty
    file cannot be reverted — that must not pass as a clean discard."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(run_bug_fix, "_capture_dirty_ref", lambda *a: None)
    (tmp_path / "src.c").write_text("int a = 2; /* agent work */\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None and baseline.dirty_ref is None

    (tmp_path / "src.c").write_text("int a = 3; /* verification dirt */\n")

    assert _discard_verification_dirt(config, baseline, state) is False
    assert (tmp_path / "src.c").read_text() == "int a = 3; /* verification dirt */\n"


def test_porcelain_line_paths_splits_renames_and_copies():
    """Only a rename moves the source off the worktree; a copy leaves it in
    place, so its source is not part of the change. Both forms print
    ``src -> dst`` and both must still yield the destination path."""
    assert _porcelain_line_paths("R  src.c -> src_moved.c") == (
        "src_moved.c",
        "src.c",
    )
    assert _porcelain_line_paths("C  src.c -> src_copy.c") == ("src_copy.c", None)
    # A file may legally be *named* "a -> b"; only R/C lines carry two paths.
    assert _porcelain_line_paths("?? a -> b") == ("a -> b", None)
    assert _porcelain_line_paths('R  "\\303\\251.c" -> "b.c"') == ("b.c", "é.c")


def test_discard_keeps_a_preexisting_untracked_file_the_verification_staged(
    tmp_path,
):
    """A verification that runs ``git add`` on a file that predates it must
    not get that file deleted: it was never in git, so the discard's
    "staged as a new file" rule would destroy it irrecoverably."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "evidence.txt").write_text("local notes\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None and baseline.untracked == {"evidence.txt"}

    _git_4b(tmp_path, "add", "evidence.txt")  # verification stages it

    assert _discard_verification_dirt(config, baseline, state) is True

    assert (tmp_path / "evidence.txt").read_text() == "local notes\n"
    assert _git_4b(tmp_path, "status", "--porcelain") == "?? evidence.txt"


def test_discard_reports_a_preexisting_untracked_file_the_verification_deleted(
    tmp_path, caplog
):
    """The discard only subtracts forward, so a file that was untracked before
    the verification and gone after it (``make clean`` is the common case) has
    no restore path. Its content cannot come back, but the discard must not
    pass as clean without naming it."""
    import logging

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.run_bug_fix")
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "evidence.txt").write_text("local notes\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None and baseline.untracked == {"evidence.txt"}

    (tmp_path / "evidence.txt").unlink()

    assert _discard_verification_dirt(config, baseline, state) is True
    assert "Verification deleted 1 pre-existing untracked file(s)" in caplog.text
    assert "evidence.txt" in caplog.text


def test_discard_does_not_report_a_staged_preexisting_file_as_removed(tmp_path, caplog):
    """A verification that only *stages* a pre-existing untracked file moves it
    from ``untracked`` to ``modified``; it is still on disk and must not be
    reported as lost."""
    import logging

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.run_bug_fix")
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "evidence.txt").write_text("local notes\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None

    _git_4b(tmp_path, "add", "evidence.txt")

    assert _discard_verification_dirt(config, baseline, state) is True
    assert "Verification deleted" not in caplog.text
    assert (tmp_path / "evidence.txt").exists()


def test_discard_does_not_report_build_output_cleaned_from_a_build_subdir(
    tmp_path, caplog
):
    """Everything under a non-root build directory is pipeline-disposable, so
    a clean-before-build verification removing its artifacts is expected —
    reporting it as irrecoverable work would cry wolf on every rebuild."""
    import logging

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.run_bug_fix")
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "build")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "lib.o").write_text("object\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None and "build/lib.o" in baseline.untracked

    (tmp_path / "build" / "lib.o").unlink()

    assert _discard_verification_dirt(config, baseline, state) is True
    assert "Verification deleted" not in caplog.text


def test_discard_names_how_many_vanished_files_it_did_not_list(tmp_path, caplog):
    """The log names at most five paths; the rest must not disappear silently."""
    import logging

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.run_bug_fix")
    config, _, state = _phase_4b_fixtures(tmp_path)
    for i in range(6):
        (tmp_path / f"note{i}.txt").write_text("local\n")
    baseline = _worktree_snapshot(config)
    assert baseline is not None

    for i in range(6):
        (tmp_path / f"note{i}.txt").unlink()

    assert _discard_verification_dirt(config, baseline, state) is True
    assert "Verification deleted 6 pre-existing untracked file(s)" in caplog.text
    assert "(+1 more)" in caplog.text


def test_discard_restores_a_file_deleted_at_the_baseline(tmp_path):
    """The baseline can hold a tracked *deletion* — pre-existing dirt the run
    does not own. A verification that recreates the file must not leave its
    content behind: the next ``git add -A`` boundary would commit it into
    candidate.patch."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").unlink()  # pre-existing deletion, not staged
    baseline = _worktree_snapshot(config)
    assert baseline is not None and "src.c" in baseline.modified

    (tmp_path / "src.c").write_text("verification recreated me\n")
    _git_4b(tmp_path, "add", "src.c")  # and the verification staged it

    assert _discard_verification_dirt(config, baseline, state) is True

    assert not (tmp_path / "src.c").exists()
    # Unstaged deletion, matching the baseline state — nothing staged by the
    # verification survives either.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout
    assert status == " D src.c\n"


def test_discard_undoes_a_verification_rename_on_both_sides(tmp_path):
    """A rename is two paths. Restoring only the destination leaves the
    source's staged deletion behind, which a later commit boundary commits
    and ships in candidate.patch."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    assert baseline is not None

    _git_4b(tmp_path, "mv", "src.c", "src_moved.c")

    assert _discard_verification_dirt(config, baseline, state) is True

    assert (tmp_path / "src.c").read_text() == "int a = 1;\n"
    assert not (tmp_path / "src_moved.c").exists()
    assert _git_4b(tmp_path, "status", "--porcelain") == ""


def test_discard_removes_a_new_file_the_verification_staged(tmp_path):
    """The flip side of the rule above: a file the verification created and
    staged is not pre-existing work, so it must still be removed."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    assert baseline is not None

    (tmp_path / "generated.c").write_text("verification output\n")
    _git_4b(tmp_path, "add", "generated.c")

    assert _discard_verification_dirt(config, baseline, state) is True

    assert not (tmp_path / "generated.c").exists()
    assert _git_4b(tmp_path, "status", "--porcelain") == ""


def test_discard_prunes_directories_its_clean_left_empty(tmp_path):
    """git clean removes files, not directories; a tree the verification
    created must not survive as empty scaffolding."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    (tmp_path / "builddir" / "nested").mkdir(parents=True)
    (tmp_path / "builddir" / "nested" / "x.o").write_text("output\n")

    assert _discard_verification_dirt(config, baseline, state) is True

    assert not (tmp_path / "builddir").exists()


def test_discard_removes_a_nested_repository_created_by_a_verification(tmp_path):
    """`git clean -fd` exits 0 on a nested repository without removing it, so
    the discard reported a clean tree while the content survived."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    nested = tmp_path / "nested"
    nested.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=nested, check=True, capture_output=True)
    (nested / "artifact.txt").write_text("verification content\n")

    assert _discard_verification_dirt(config, baseline, state) is True
    assert not nested.exists(), "the nested repo and its content are gone"


def test_discard_fails_when_a_path_survives_git_clean(tmp_path, monkeypatch):
    """Whatever the exit code says, the discard only reports clean when the
    paths it asked git to remove are actually gone."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    (tmp_path / "gen.o").write_text("output\n")

    monkeypatch.setattr(rbf, "_run_git_pathspec_batches", lambda *a, **k: True)

    assert _discard_verification_dirt(config, baseline, state) is False


def test_discard_reports_failure_when_clean_fails(tmp_path, monkeypatch):
    """A discard that could not finish must say so — the caller escalates."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    (tmp_path / "gen.o").write_text("output\n")

    monkeypatch.setattr(run_bug_fix, "_run_git_pathspec_batches", lambda *a, **k: False)

    assert _discard_verification_dirt(config, baseline, state) is False


def test_empty_dirs_reports_failure_on_a_permission_error(tmp_path):
    """`os.walk` swallows errors by default; an unreadable directory must
    disable the prune rather than look empty and get removed."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "x").write_text("x")
    os.chmod(blocked, 0)
    try:
        if os.geteuid() == 0:  # root ignores the mode bits
            pytest.skip("permission bits do not apply to root")
        assert _empty_dirs(config) is None
    finally:
        os.chmod(blocked, 0o755)


# ---------- Staging failures are not logged and forgotten ----------


def test_stage_or_escalate_raises_when_staging_failed(tmp_path, monkeypatch):
    """A success-path boundary cannot continue past a failed stage: the
    commit, the candidate patch and the preserved branch would all be built
    from an index that dropped this phase's work."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(
        run_bug_fix,
        "_stage_agent_work",
        lambda *a: StageResult(fenced_build_dir=None, generated=set(), ok=False),
    )

    with pytest.raises(PipelineEscalation, match="Phase 3b.*could not stage"):
        _stage_or_escalate(config, state, "Phase 3b (regenerated files)")


def test_stage_or_escalate_returns_a_successful_stage(tmp_path):
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    staged = _stage_or_escalate(config, state, "Phase 3b (regenerated files)")

    assert staged.ok is True
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_stage_or_escalate_escalates_on_a_real_locked_index(tmp_path):
    """End-to-end against real git: a locked index makes ``git add`` fail, and
    the boundary must escalate instead of committing without its work."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* phase work */\n")
    (tmp_path / ".git" / "index.lock").write_text("")

    with pytest.raises(PipelineEscalation, match="could not stage"):
        _stage_or_escalate(config, state, "Phase 3b (regenerated files)")

    # Nothing was committed, so the work is still visible to a human.
    assert "phase work" in (tmp_path / "src.c").read_text()


def test_non_artifact_dirt_fails_closed_on_a_real_git_failure(tmp_path):
    """Same contract as the mocked test above, but through real git: a
    directory that is not a repository makes ``git status`` exit nonzero."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    # Outside the fixture repo: git walks up, so a subdirectory would still
    # find the repository and report a clean tree.
    not_a_repo = tmp_path.parent / f"{tmp_path.name}-not-a-repo"
    not_a_repo.mkdir()
    config.repo_path = str(not_a_repo)

    assert _non_artifact_dirt(config) is None


def test_non_artifact_dirt_fails_closed_when_git_status_fails(tmp_path, monkeypatch):
    """An unreadable status is not an empty one: returning [] would let the
    caller report a clean boundary while work may have been dropped."""
    from auto_bug_fix import run_bug_fix
    from auto_bug_fix.git_tools import GitResult

    config, _, _ = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(
        run_bug_fix,
        "git_status_porcelain_all",
        lambda *a, **k: GitResult(exit_code=128, stdout="", stderr="boom"),
    )

    assert _non_artifact_dirt(config) is None


def test_commit_recovery_edits_escalates_when_git_status_fails(tmp_path, monkeypatch):
    """An unreadable status is not a clean tree here either.

    The early emptiness check must not return False ("nothing to commit") when
    ``git status`` failed: Phase 4b reads False as "no work to commit",
    verifies, and reports success with the agent's edits missing from
    candidate.patch and the preserved branch.
    """
    from auto_bug_fix import run_bug_fix
    from auto_bug_fix.git_tools import GitResult

    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(
        run_bug_fix,
        "git_status_porcelain_all",
        lambda *a, **k: GitResult(exit_code=128, stdout="", stderr="boom"),
    )

    with pytest.raises(PipelineEscalation, match="could not read the worktree"):
        _commit_recovery_edits(config, "Phase 4b")

    assert _git_4b(tmp_path, "rev-parse", "HEAD") == head_before


def test_commit_recovery_edits_escalates_when_staging_partially_failed(
    tmp_path, monkeypatch
):
    """A landed commit with a failed staging step is not a success.

    ``_stage_or_escalate`` treats any staging failure as fatal before the
    commit; this boundary commits first so the failure-path export keeps the
    work, but it must still escalate instead of reporting the phase clean.
    """
    from auto_bug_fix import run_bug_fix

    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")
    real = run_bug_fix._stage_agent_work
    monkeypatch.setattr(
        run_bug_fix,
        "_stage_agent_work",
        lambda *a, **k: real(*a, **k)._replace(ok=False),
    )

    with pytest.raises(PipelineEscalation, match="committed, but staging"):
        _commit_recovery_edits(config, "Phase 4b")

    # The commit still landed, so the work survives into candidate.patch.
    assert "int a = 2" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_best_effort_commit_proceeds_when_git_status_fails(
    tmp_path, caplog, monkeypatch
):
    """The rescue path must not read an unreadable status as "nothing to
    commit": trying the commit is the only way the work reaches
    candidate.patch at all."""
    import logging

    from auto_bug_fix import run_bug_fix
    from auto_bug_fix.git_tools import GitResult

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.run_bug_fix")
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(
        run_bug_fix,
        "git_status_porcelain_all",
        lambda *a, **k: GitResult(exit_code=128, stdout="", stderr="boom"),
    )

    assert _commit_worktree_edits(config, "Phase 4b abort") is True

    assert _git_4b(tmp_path, "rev-parse", "HEAD") != head_before
    assert "int a = 2" in _git_4b(tmp_path, "show", "HEAD:src.c")
    assert "attempting the commit without the emptiness check" in caplog.text


def test_best_effort_commit_names_an_incomplete_stage(tmp_path, caplog, monkeypatch):
    """A landed rescue commit with a failed stage is kept (losing the work is
    worse), but build output may now be inside a patch that gets exported, so
    the log must say so."""
    import logging

    from auto_bug_fix import run_bug_fix

    caplog.set_level(logging.WARNING, logger="auto_bug_fix.run_bug_fix")
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")
    real = run_bug_fix._stage_agent_work
    monkeypatch.setattr(
        run_bug_fix,
        "_stage_agent_work",
        lambda *a, **k: real(*a, **k)._replace(ok=False),
    )

    assert _commit_worktree_edits(config, "Phase 4b abort", state) is True

    assert "incomplete stage" in caplog.text
    assert "int a = 2" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_non_artifact_dirt_names_learned_output_it_leaves_behind(tmp_path, caplog):
    """Build output left unstaged on purpose is not dirt, but the log must
    name it: an agent-authored file at a learned path is dropped here."""
    import logging

    caplog.set_level(logging.INFO, logger="auto_bug_fix.run_bug_fix")
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("output\n")
    _learn(config, state, "gen.o")

    staged = _stage_agent_work(config, state)

    assert _non_artifact_dirt(config, staged) == []
    assert "learned build output" in caplog.text


# ---------- Pruning emptied directories from the removal set ----------


def test_prune_emptied_dirs_only_touches_ancestors_of_removed_paths(tmp_path):
    """The prune is candidate-driven: it must not walk (or remove) anything
    a removal did not leave empty."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "build" / "nested").mkdir(parents=True)
    (tmp_path / "unrelated").mkdir()

    _prune_emptied_dirs(config, set(), {"build/nested/x.o"})

    assert not (tmp_path / "build").exists()
    assert (tmp_path / "unrelated").is_dir()


def test_prune_emptied_dirs_spares_dirs_empty_at_the_baseline(tmp_path):
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "scaffold").mkdir()

    _prune_emptied_dirs(config, {"scaffold"}, {"scaffold/x.o"})

    assert (tmp_path / "scaffold").is_dir()


def test_prune_emptied_dirs_is_disabled_without_a_baseline(tmp_path):
    """Without the baseline set, an empty directory may predate the run —
    removing it would destroy state the pipeline does not own."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    (tmp_path / "logs").mkdir()

    _prune_emptied_dirs(config, None, {"logs/x.o"})

    assert (tmp_path / "logs").is_dir()


# ---------- Target baseline restoration reports why it was skipped ----------


def test_restore_target_baseline_is_silent_at_the_baseline(tmp_path):
    config, _, _ = _phase_4b_fixtures(tmp_path)
    head = _git_4b(tmp_path, "rev-parse", "HEAD")

    assert _restore_target_baseline(str(tmp_path), str(tmp_path), "main", head) is None


def test_restore_target_baseline_reports_a_ref_it_cannot_move(tmp_path):
    """A tag is not refs/heads/*: moving it would create a bogus branch, so
    the reason must reach the run summary instead of a log line."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "src.c").write_text("int a = 2;\n")
    _git_4b(tmp_path, "commit", "-am", "moved")
    _git_4b(tmp_path, "tag", "v1")

    reason = _restore_target_baseline(str(tmp_path), str(tmp_path), "v1", base)

    assert reason is not None and "not a local branch" in reason


def test_restore_target_baseline_reports_a_branch_held_elsewhere(tmp_path):
    config, _, _ = _phase_4b_fixtures(tmp_path)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "src.c").write_text("int a = 2;\n")
    _git_4b(tmp_path, "commit", "-am", "moved")
    _git_4b(tmp_path, "branch", "release")
    holder = tmp_path.parent / f"{tmp_path.name}-holder"
    _git_4b(tmp_path, "worktree", "add", str(holder), "release")

    reason = _restore_target_baseline(
        str(tmp_path), str(tmp_path / "pipeline"), "release", base
    )

    assert reason is not None and "checked out in another worktree" in reason
    assert _git_4b(tmp_path, "rev-parse", "release") != base


def test_branch_checkout_holders_unquotes_porcelain_paths(tmp_path, monkeypatch):
    """Porcelain C-quotes paths git considers non-printable; comparing the
    quoted string to a realpath misses the holder and lets the branch be
    repointed under a live checkout."""
    from auto_bug_fix import run_bug_fix

    holder = tmp_path / "wörk"
    holder.mkdir()
    quoted = '"' + str(holder).replace("ö", "\\303\\266") + '"'
    porcelain = f"worktree {quoted}\nbranch refs/heads/main\n\n"
    monkeypatch.setattr(
        run_bug_fix.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=a[0], returncode=0, stdout=porcelain, stderr=""
        ),
    )

    holders = _branch_checkout_holders(str(tmp_path), "refs/heads/main", str(tmp_path))

    assert holders == [str(holder)]


def test_branch_checkout_holders_unquotes_a_real_worktree_path(tmp_path):
    """Same contract against real ``git worktree list --porcelain``: a non-ASCII
    holder path must still be recognized, or the branch is repointed under a
    live checkout. Compared by inode, which survives filesystem unicode
    normalization."""
    config, _, _ = _phase_4b_fixtures(tmp_path)
    _git_4b(tmp_path, "branch", "release")
    holder = tmp_path.parent / f"{tmp_path.name}-wörk"
    _git_4b(tmp_path, "worktree", "add", str(holder), "release")

    holders = _branch_checkout_holders(
        str(tmp_path), "refs/heads/release", str(tmp_path / "pipeline")
    )

    assert holders is not None
    assert len(holders) == 1
    assert os.path.samefile(holders[0], str(holder))


# ---------- Escalation plumbing ----------


def test_finish_verification_escalates_when_the_discard_fails(tmp_path, monkeypatch):
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(run_bug_fix, "_discard_verification_dirt", lambda *a: False)

    with pytest.raises(PipelineEscalation, match="could not be discarded"):
        _finish_verification(config, None, state, "test phase", in_flight=False)


def test_finish_verification_defers_when_an_exception_is_in_flight(
    tmp_path, monkeypatch
):
    """Inside a `finally`, raising would replace the real failure."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(run_bug_fix, "_discard_verification_dirt", lambda *a: False)

    _finish_verification(config, None, state, "test phase", in_flight=True)


def test_phase_4b_pre_verification_commit_excludes_learned_output(tmp_path):
    """Default-root layout, the reviewer's P1: the agent runs the same build
    the orchestrator already saw, so its output reappears at the learned
    paths and must not be committed with the fix."""
    from unittest.mock import patch

    from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix

    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("agent-run build output\n")
    _learn(config, state, "gen.o")

    async def session(cfg, steps):
        (tmp_path / "src.c").write_text("int a = 2; /* fixed by 4b */\n")
        (tmp_path / "gen.o").write_text("agent-run build output\n")
        (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\nRetries: 1/3\n")
        return []

    with patch("auto_bug_fix.run_bug_fix._agent_run", new=session):
        phase_4b_run_and_fix(config, claude_cfg, state, None)

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src.c" in tree
    assert "gen.o" not in tree
    assert "fixed by 4b" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_phase_3b_regeneration_commit_uses_the_staging_guard(tmp_path):
    """The regeneration boundary runs shell commands with agent-level freedom;
    their build output must not ride along with the regenerated file."""
    from auto_bug_fix.run_bug_fix import phase_3b_regeneration

    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.o").write_text("output\n")
    state.generated_output = {"rpm-build/artifact.o"}

    phase_3b_regeneration(config, state, ["echo regenerated > src.c"])

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src.c" in tree
    assert "rpm-build/artifact.o" not in tree
    assert "regenerated" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_commit_worktree_edits_on_an_abort_path_uses_the_staging_guard(tmp_path):
    """The abort-path best-effort commit must not sweep learned output in."""
    from auto_bug_fix.run_bug_fix import _commit_worktree_edits

    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* fixed */\n")
    (tmp_path / "gen.o").write_text("output\n")
    _learn(config, state, "gen.o")

    assert _commit_worktree_edits(config, "Phase 4b", state) is True

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src.c" in tree
    assert "gen.o" not in tree


def test_pre_cherry_pick_cleanup_fails_closed_when_status_is_unreadable(
    tmp_path, monkeypatch
):
    """A failed status used to read as "nothing to commit", and the
    `git reset --hard` that follows then discarded the work the boundary had
    just staged."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* staged work */\n")
    resets = []

    class _Failed:
        success = False
        exit_code = 128
        stdout = ""
        stderr = "fatal: index file corrupt"

    monkeypatch.setattr(rbf, "git_status_porcelain_all", lambda *a, **k: _Failed())
    monkeypatch.setattr(rbf, "git_reset_hard", lambda *a, **k: resets.append(a))

    with pytest.raises(PipelineEscalation, match="Pre-cherry-pick cleanup"):
        rbf._pre_cherry_pick_cleanup(config, state)

    assert resets == [], "the reset must not run on an unreadable status"
    assert "staged work" in (tmp_path / "src.c").read_text()


def test_pre_cherry_pick_cleanup_commits_staged_work_and_resets(tmp_path):
    """The happy path still commits the work first, then resets the tree."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "src.c").write_text("int a = 2; /* staged work */\n")

    rbf._pre_cherry_pick_cleanup(config, state)

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src.c" in tree
    assert "staged work" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_cleanup_of_an_unrelated_verification_still_reports_clean(tmp_path):
    """Sanity check on the success return: a discard with nothing to do is
    not a failure."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    assert _discard_verification_dirt(config, baseline, state) is True
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_discard_removes_a_staged_new_symlink_with_a_missing_target(tmp_path):
    """A dangling symlink must not survive the discard.

    A build that creates ``generated.h -> build/generated.h`` before the
    target exists, and runs ``git add -A``, leaves a *staged new* dangling
    link. ``os.path.isfile`` is False for it, so an isfile-only removal left
    it in the tree — unlearned (it was staged, so it never enters the
    untracked set) and therefore staged again by the next commit boundary.
    """
    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    os.symlink("build/generated.h", tmp_path / "generated.h")
    _git_4b(tmp_path, "add", "generated.h")

    assert _discard_verification_dirt(config, baseline, state) is True

    assert not os.path.lexists(tmp_path / "generated.h")
    _stage_agent_work(config, state)
    assert _staged(tmp_path) == ""


# ---------- in_flight must describe *this* block, not the call stack ----------


def test_discarding_scope_escalates_from_inside_an_enclosing_except(
    tmp_path, monkeypatch
):
    """The regression: ``sys.exc_info()`` is per-thread and keeps reporting an
    enclosing handler's exception in every frame below it. Phase 4b and both
    post-recovery re-verifies are only ever reached from inside an ``except``,
    and those paths carry on to Phase 5 — so a failed discard there has to
    escalate, or the dirt it left reaches the next ``git add -A``."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(run_bug_fix, "_discard_verification_dirt", lambda *a: False)

    try:
        raise PipelineEscalation("Phase 4a gave up")
    except PipelineEscalation:
        # Nothing escapes the block below, so the run would continue here.
        with (
            pytest.raises(PipelineEscalation, match="could not be discarded"),
            run_bug_fix._discarding_verification_dirt(
                config, None, state, "Phase 4b re-verify"
            ),
        ):
            pass


def test_discarding_scope_defers_to_the_exception_leaving_its_own_body(
    tmp_path, monkeypatch
):
    """When the body itself fails the run really is aborting: the body's
    exception must survive rather than be replaced by the discard's."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    monkeypatch.setattr(run_bug_fix, "_discard_verification_dirt", lambda *a: False)

    with (
        pytest.raises(RuntimeError, match="build blew up"),
        run_bug_fix._discarding_verification_dirt(
            config, None, state, "Phase 4 verify"
        ),
    ):
        raise RuntimeError("build blew up")


def test_discarding_scope_discards_on_the_happy_path(tmp_path):
    """A clean body still gets its dirt swept."""
    from auto_bug_fix import run_bug_fix

    config, _, state = _phase_4b_fixtures(tmp_path)
    baseline = _worktree_snapshot(config)
    with run_bug_fix._discarding_verification_dirt(
        config, baseline, state, "Phase 4 verify"
    ):
        (tmp_path / "build.log").write_text("verification output\n")

    assert not os.path.lexists(tmp_path / "build.log")


def test_no_verification_site_derives_in_flight_from_exc_info():
    """Pins the contract repo-wide: a new verification site that hand-rolls
    the flag reintroduces the bug silently, so none may."""
    import inspect

    from auto_bug_fix import run_bug_fix

    src = inspect.getsource(run_bug_fix)
    assert "in_flight=sys.exc_info()" not in src


# ---------- Phase 4a's recovery prompt must still see the port ----------


def test_phase_4a_prompt_diff_survives_the_per_attempt_commit(tmp_path):
    """Attempt N now commits the agent's edits at its own boundary and the
    discard wipes the rest, so a worktree-vs-HEAD diff is empty from attempt 2
    on. The prompt's "what was actually applied" section has to range from the
    frozen baseline instead, or the agent retries blind to its own work."""
    from unittest.mock import patch

    from auto_bug_fix.run_bug_fix import phase_4a_build_recovery

    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.max_build_test_retries = 2
    config.frozen_target_sha = _git_4b(tmp_path, "rev-parse", "HEAD").strip()

    seen = []

    def fake_prompt(**kwargs):
        seen.append(kwargs["current_diff"])

        class _P:
            def prompt(self):
                return "x"

        return _P()

    attempt = {"n": 0}

    async def session(cfg, steps):
        attempt["n"] += 1
        (tmp_path / "src.c").write_text(f"int a = {attempt['n'] + 1}; /* try */\n")
        return []

    with (
        patch("auto_bug_fix.run_bug_fix._agent_run", new=session),
        patch("auto_bug_fix.run_bug_fix.BuildErrorRecoveryPrompt", new=fake_prompt),
        patch(
            "auto_bug_fix.run_bug_fix.run_test_suite",
            new=lambda *a, **k: (1, "build broke", ""),
        ),
        patch("auto_bug_fix.run_bug_fix.parse_test_failures", new=lambda *a: []),
        patch("auto_bug_fix.run_bug_fix.check_regression", new=lambda *a: (True, [])),
        pytest.raises(PipelineEscalation),
    ):
        phase_4a_build_recovery(config, claude_cfg, state, None, "fix diff")

    # Attempt 1 has nothing applied yet; attempt 2 must see attempt 1's commit.
    assert len(seen) >= 2, seen
    assert "/* try */" in seen[1], seen[1]


# ---------- The force-resolve boundary: commit, then escalate ----------


def _conflicted_repo(tmp_path):
    """Leave the repo mid-merge with a real ``UU`` entry for src.c."""
    _git_4b(tmp_path, "checkout", "-b", "side")
    (tmp_path / "src.c").write_text("int a = 99;\n")
    _git_4b(tmp_path, "commit", "-am", "side edit")
    _git_4b(tmp_path, "checkout", "main")
    (tmp_path / "src.c").write_text("int a = 2;\n")
    _git_4b(tmp_path, "commit", "-am", "main edit")
    subprocess.run(["git", "merge", "side"], cwd=tmp_path, capture_output=True)
    assert _conflicted_files(git_status_porcelain(str(tmp_path)))


def test_force_resolve_escalates_when_staging_failed(tmp_path, monkeypatch):
    """The force-resolve boundary used to drop ``StageResult.ok`` on the floor:
    a failed ``git add`` produced an incomplete commit, Phase 4b ran on it, and
    nothing ever reported that the fix might be missing."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _conflicted_repo(tmp_path)

    def _staged_but_failing(cfg, st=None):
        subprocess.run(["git", "add", "-A"], cwd=cfg.repo_path, capture_output=True)
        return StageResult(None, set(), ok=False)

    monkeypatch.setattr(rbf, "_stage_agent_work", _staged_but_failing)

    with pytest.raises(PipelineEscalation, match="force-resolve"):
        _force_resolve_for_phase_4b(config, state)


def test_force_resolve_commits_before_it_escalates(tmp_path, monkeypatch):
    """Ordering is the whole point: Phase 4b and the failure-path export both
    read commits, and ``_emit_candidate_patch_on_failure`` never commits. An
    escalation raised ahead of ``git_commit`` would throw away the conflicts
    this boundary just force-resolved."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _conflicted_repo(tmp_path)
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")

    def _staged_but_failing(cfg, st=None):
        subprocess.run(["git", "add", "-A"], cwd=cfg.repo_path, capture_output=True)
        return StageResult(None, set(), ok=False)

    monkeypatch.setattr(rbf, "_stage_agent_work", _staged_but_failing)

    with pytest.raises(PipelineEscalation):
        _force_resolve_for_phase_4b(config, state)

    assert _git_4b(tmp_path, "rev-parse", "HEAD") != head_before
    assert "int a = 99;" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_force_resolve_is_silent_when_staging_succeeded(tmp_path):
    """The normal path still commits the force-resolved tree and says nothing."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _conflicted_repo(tmp_path)
    head_before = _git_4b(tmp_path, "rev-parse", "HEAD")

    _force_resolve_for_phase_4b(config, state)

    assert _git_4b(tmp_path, "rev-parse", "HEAD") != head_before


# ---------- Agent-created commits: the guard runs after them too ----------


def _learn_gen_o(config, state, content="verification build output\n"):
    """Run one orchestrator verification that writes gen.o, and discard it."""
    baseline = _worktree_snapshot(config)
    with open(os.path.join(config.repo_path, "gen.o"), "w") as f:
        f.write(content)
    assert _discard_verification_dirt(config, baseline, state) is True
    return state


def test_agent_committed_build_output_is_untracked_again(tmp_path):
    """An agent that runs `git add -A && git commit` sweeps its own build
    output into the tree. The staging guard never saw that commit, and from
    then on HEAD tracks the output — which is exactly the signal both guards
    read as "this is source". The guard therefore runs after the agent too."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _learn_gen_o(config, state)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")

    # The agent commits its fix *and* its build output, as agents do.
    (tmp_path / "gen.o").write_text("verification build output\n")
    (tmp_path / "src.c").write_text("int a = 2; /* the fix */\n")
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "agent commit")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    assert "src.c" in tree
    assert (tmp_path / "gen.o").exists(), "kept on disk — later builds expect it"
    # The deliverable ranges over base..HEAD, and the path was added inside
    # that range, so it is gone from the patch entirely.
    net = _git_4b(tmp_path, "diff", "--name-only", base, "HEAD").splitlines()
    assert net == ["src.c"]


def test_agent_commit_no_longer_disables_the_build_dir_fence(tmp_path):
    """The sting in the original failure: _fence_build_dir asks whether HEAD
    tracks anything under the build directory, so one agent commit switched
    the fence off for the whole run."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    base = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "rpm-build" / "artifact.o").write_text("output\n")
    state.generated_output = {"rpm-build/artifact.o"}
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "agent commit with build output")
    assert _fence_build_dir(config) is None, "fence disabled by the agent commit"

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert _fence_build_dir(config) == "rpm-build"


def test_agent_committed_ignored_output_is_untracked_again(tmp_path):
    """A force-added artifact is still recognizable: `git check-ignore
    --no-index` answers from the ignore rules, not from the index."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / ".gitignore").write_text("*.o\n")
    _git_4b(tmp_path, "add", ".gitignore")
    _git_4b(tmp_path, "commit", "-m", "ignore object files")
    base = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "stray.o").write_text("output\n")
    _git_4b(tmp_path, "add", "-f", "stray.o")
    _git_4b(tmp_path, "commit", "-m", "agent force-added build output")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert (
        "stray.o"
        not in _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    )


def test_agent_committed_source_is_never_untracked(tmp_path):
    """Accounting runs the same way here as at a staging boundary: anything
    the pipeline cannot show is build output stays in the commit. A noisier
    patch is recoverable, removing the fix from the tree is not."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "new_helper.c").write_text("int n = 1;\n")
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "agent commit")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert _git_4b(tmp_path, "rev-parse", "HEAD") != base
    assert (
        "new_helper.c"
        in _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    )


def test_agent_committed_source_at_a_learned_path_is_never_untracked(tmp_path):
    """The name-collision case again, on the commit side: the agent authored
    the fix at a path some verification once generated."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _learn_gen_o(config, state)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "gen.o").write_text("int ported_fix = 1;\n")
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "agent commit")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert (
        "gen.o"
        in _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    )


def test_strip_committed_build_output_is_a_no_op_without_an_agent_commit(tmp_path):
    """HEAD never moved: nothing to inspect, and no empty commit created."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert _git_4b(tmp_path, "rev-parse", "HEAD") == base


def test_resolution_skip_path_strips_build_output_from_its_commit(
    tmp_path, monkeypatch
):
    """The ``files_to_skip`` path commits and returns without reaching
    ``_finalize_resolution``, so the strip there never saw it. Whatever the
    index carried at that commit — including build output staged alongside the
    conflict — went straight into candidate.patch."""
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    _conflicted_repo(tmp_path)  # real ``UU src.c``, mid-merge
    (tmp_path / "gen.o").write_text("compile output\n")
    _learn(config, state, "gen.o")
    _git_4b(tmp_path, "add", "gen.o")

    class _Failed:
        exit_code = 1
        success = False
        stderr = "conflict"

    # The conflict is already in the worktree; keep it there instead of
    # re-running (and aborting) a cherry-pick.
    monkeypatch.setattr(rbf, "git_cherry_pick", lambda *a, **k: _Failed())
    monkeypatch.setattr(rbf, "git_cherry_pick_abort", lambda *a, **k: None)

    rbf.cherry_pick_and_resolve(
        config,
        claude_cfg,
        state,
        None,
        commit="0" * 40,
        label="test",
        files_to_skip=["src.c"],
    )

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    assert (tmp_path / "gen.o").exists(), "kept on disk for the build that expects it"


def test_semantic_triage_strips_build_output_from_an_agent_commit(
    tmp_path, monkeypatch
):
    """The Phase 0.5 triage agent was the one agent boundary nothing guarded:
    it runs before Phase 1 learns any output, so a `git add -A` there put its
    own build tree into HEAD — which in turn disabled the fence for the whole
    run. Ignore rules are the only evidence available this early."""
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / ".gitignore").write_text("*.o\n")
    _git_4b(tmp_path, "add", ".gitignore")
    _git_4b(tmp_path, "commit", "-m", "ignore object files")
    config.source_fix_commit = _git_4b(tmp_path, "rev-parse", "HEAD")

    async def session(cfg, steps, tracker=None):
        (tmp_path / "gen.o").write_text("compile output\n")
        (tmp_path / "triage.json").write_text('{"recommendation": "proceed"}\n')
        # -f: the agent force-adds its build tree over the ignore rules.
        _git_4b(tmp_path, "add", "-A", "-f")
        _git_4b(tmp_path, "commit", "-m", "agent commit")
        return []

    monkeypatch.setattr(rbf, "claude_run", session)

    rbf.phase_0_5_semantic_triage(config, claude_cfg, state, None)

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    assert "triage.json" in tree, "the agent's analysis is work and stays"
    assert (tmp_path / "gen.o").exists(), "kept on disk for the build that expects it"


def test_red_triage_strips_build_output_from_an_agent_commit(tmp_path, monkeypatch):
    """Phase 1.5's triage session is a full agent with a shell. Its own
    commit was outside every boundary, and by then Phase 1 has run, so the
    paths a build was seen to write are known and can be taken back out."""
    from auto_bug_fix.bug_fix_prompts import RED_TRIAGE_FILE

    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("compile output\n")
    _learn(config, state, "gen.o")
    state.red_passed_output = "PASSED tests/test_x.py"

    async def session(cfg, steps, tracker=None):
        (tmp_path / "gen.o").write_text("compile output\n")
        (tmp_path / RED_TRIAGE_FILE).write_text('{"verdict": "vulnerable"}\n')
        _git_4b(tmp_path, "add", "-A")
        _git_4b(tmp_path, "commit", "-m", "agent commit")
        return []

    monkeypatch.setattr(rbf, "claude_run", session)

    result = rbf.phase_1_5_red_triage(config, claude_cfg, state, None)

    assert result["verdict"] == "vulnerable"
    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    # Pipeline artifacts never enter a pipeline commit either, but the file
    # has to survive on disk for the function to read its verdict.
    assert RED_TRIAGE_FILE not in tree
    assert (tmp_path / "gen.o").exists(), "kept on disk for the build that expects it"
    assert (tmp_path / RED_TRIAGE_FILE).exists()


def test_a_session_that_commits_and_then_raises_is_still_stripped(
    tmp_path, monkeypatch
):
    """The failure-path export formats frozen..HEAD, so a commit left behind
    by a session that then errors would ship its build output exactly when
    the run gives up on the phase. The strip has to run in ``finally``."""
    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("compile output\n")
    _learn(config, state, "gen.o")
    state.red_passed_output = "PASSED tests/test_x.py"

    async def session(cfg, steps, tracker=None):
        (tmp_path / "gen.o").write_text("compile output\n")
        _git_4b(tmp_path, "add", "-A")
        _git_4b(tmp_path, "commit", "-m", "agent commit")
        raise RuntimeError("session crashed")

    monkeypatch.setattr(rbf, "claude_run", session)

    with pytest.raises(RuntimeError, match="session crashed"):
        rbf.phase_1_5_red_triage(config, claude_cfg, state, None)

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    assert (tmp_path / "gen.o").exists(), "kept on disk for the build that expects it"


def test_phase_4a_survives_and_strips_a_self_committing_agent(tmp_path):
    """A recovery agent that commits for itself looks exactly like one that did
    nothing — a clean worktree — so the run aborted with CANNOT_FIX while the
    work sat in HEAD. And the commit it made was never guarded."""
    from unittest.mock import patch

    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    config.max_build_test_retries = 1
    (tmp_path / "gen.o").write_text("compile output\n")
    _learn(config, state, "gen.o")

    async def session(cfg, steps):
        (tmp_path / "src.c").write_text("int a = 3; /* fix */\n")
        (tmp_path / "gen.o").write_text("compile output\n")
        _git_4b(tmp_path, "add", "-A")
        _git_4b(tmp_path, "commit", "-m", "agent commit")
        return []

    with (
        patch("auto_bug_fix.run_bug_fix._agent_run", new=session),
        patch(
            "auto_bug_fix.run_bug_fix.run_test_suite",
            new=lambda *a, **k: (1, "build broke", ""),
        ),
        patch("auto_bug_fix.run_bug_fix.parse_test_failures", new=lambda *a: []),
        patch("auto_bug_fix.run_bug_fix.check_regression", new=lambda *a: (True, [])),
        pytest.raises(PipelineEscalation, match="still failing"),
    ):
        rbf.phase_4a_build_recovery(config, claude_cfg, state, None, "fix diff")

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    assert "int a = 3;" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_phase_4b_strips_a_self_committing_agent(tmp_path):
    """Phase 4b's session is a full agent with a shell; its own `git add -A &&
    git commit` was outside every boundary the phase sets up afterwards."""
    from unittest.mock import patch

    config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
    (tmp_path / "gen.o").write_text("compile output\n")
    _learn(config, state, "gen.o")

    async def session(cfg, steps):
        (tmp_path / "src.c").write_text("int a = 4; /* fixed by 4b */\n")
        (tmp_path / "gen.o").write_text("compile output\n")
        _git_4b(tmp_path, "add", "-A")
        _git_4b(tmp_path, "commit", "-m", "agent commit")
        (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\nRetries: 1/1\n")
        return []

    with patch("auto_bug_fix.run_bug_fix._agent_run", new=session):
        rbf.phase_4b_run_and_fix(config, claude_cfg, state, None)

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "gen.o" not in tree
    assert "int a = 4;" in _git_4b(tmp_path, "show", "HEAD:src.c")
    assert (tmp_path / "gen.o").exists(), "kept on disk for the build that expects it"


def _learn_rewritten_tracked_file(config, state, content="/* build stamped this */\n"):
    """Run one verification that rewrites the tracked src.c, then discard it."""
    baseline = _worktree_snapshot(config)
    pathlib.Path(config.repo_path, "src.c").write_text(content)
    assert _discard_verification_dirt(config, baseline, state) is True
    return state


def test_agent_commit_overwriting_a_tracked_file_is_reverted(tmp_path):
    """A build that rewrites a tracked file leaves a modification, not an
    addition, so the addition pass never saw it and the build's version went
    into the patch."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _learn_rewritten_tracked_file(config, state)
    assert state.generated_digests.get("src.c"), "the build's content was recorded"
    base = _git_4b(tmp_path, "rev-parse", "HEAD")

    pathlib.Path(tmp_path, "src.c").write_text("/* build stamped this */\n")
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "agent commit")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert _git_4b(tmp_path, "show", "HEAD:src.c") == "int a = 1;"
    # Net effect: the build's version is gone from the range entirely.
    assert "build stamped" not in _git_4b(tmp_path, "diff", base, "HEAD")


def test_agent_edit_to_a_learned_tracked_file_is_kept(tmp_path):
    """The path rule is not evidence for a modification: the agent's own edit
    to a file the build also regenerates must survive the strip."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    _learn_rewritten_tracked_file(config, state)
    base = _git_4b(tmp_path, "rev-parse", "HEAD")

    pathlib.Path(tmp_path, "src.c").write_text("int a = 7; /* the fix */\n")
    _git_4b(tmp_path, "add", "-A")
    _git_4b(tmp_path, "commit", "-m", "agent commit")

    assert rbf._strip_committed_build_output(config, state, base, "Phase 3a") is True

    assert _git_4b(tmp_path, "show", "HEAD:src.c") == "int a = 7; /* the fix */"
