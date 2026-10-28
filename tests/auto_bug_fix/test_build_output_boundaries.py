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
import subprocess

import pytest

from auto_bug_fix import run_bug_fix as rbf
from auto_bug_fix.git_tools import git_status_porcelain
from auto_bug_fix.run_bug_fix import (
    PipelineEscalation,
    StageResult,
    _branch_checkout_holders,
    _conflicted_files,
    _force_resolve_for_phase_4b,
    _commit_recovery_edits,
    _commit_worktree_edits,
    _discard_verification_dirt,
    _empty_dirs,
    _finish_verification,
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


# ---------- The fence: untracked build dirs only ----------


def test_stage_agent_work_fences_an_untracked_build_dir(tmp_path):
    """Build output in a directory git knows nothing about stays unstaged."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    config.build_dir = str(tmp_path / "rpm-build")
    (tmp_path / "rpm-build").mkdir()
    (tmp_path / "rpm-build" / "artifact.rpm").write_text("build output\n")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.fenced_build_dir == "rpm-build"
    assert _staged(tmp_path).splitlines() == ["src.c"]


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
    state.generated_output = {"gen.o"}
    (tmp_path / "gen.o").write_text("agent-run build output\n")
    (tmp_path / "src.c").write_text("int a = 2; /* fix */\n")

    result = _stage_agent_work(config, state)

    assert result.generated == {"gen.o"}
    assert _staged(tmp_path).splitlines() == ["src.c"]


def test_stage_agent_work_keeps_an_unlearned_untracked_file(tmp_path):
    """Only paths an orchestrator verification actually created are learned;
    a new source file the agent wrote is not build output."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    state.generated_output = {"gen.o"}
    (tmp_path / "generated.c").write_text("int g = 1;\n")
    (tmp_path / "gen.o").write_text("output\n")

    _stage_agent_work(config, state)

    assert _staged(tmp_path).splitlines() == ["generated.c"]


def test_stage_agent_work_unstages_learned_output_the_agent_staged(tmp_path):
    """An agent session that runs `git add` turns its build output into staged
    additions, which the untracked set no longer names. Learned paths must be
    unstaged from the index too, or the output rides into the commit."""
    config, _, state = _phase_4b_fixtures(tmp_path)
    state.generated_output = {"gen.o"}
    (tmp_path / "gen.o").write_text("agent-run build output\n")
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
    state.generated_output = {"gen.o"}
    (tmp_path / "gen.o").write_text("output\n")

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
    state.generated_output = {"gen.o"}

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

    phase_3b_regeneration(config, state, ["echo regenerated > src.c"])

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src.c" in tree
    assert "rpm-build/artifact.o" not in tree
    assert "regenerated" in _git_4b(tmp_path, "show", "HEAD:src.c")


def test_commit_worktree_edits_on_an_abort_path_uses_the_staging_guard(tmp_path):
    """The abort-path best-effort commit must not sweep learned output in."""
    from auto_bug_fix.run_bug_fix import _commit_worktree_edits

    config, _, state = _phase_4b_fixtures(tmp_path)
    state.generated_output = {"gen.o"}
    (tmp_path / "src.c").write_text("int a = 2; /* fixed */\n")
    (tmp_path / "gen.o").write_text("output\n")

    assert _commit_worktree_edits(config, "Phase 4b", state) is True

    tree = _git_4b(tmp_path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    assert "src.c" in tree
    assert "gen.o" not in tree


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
