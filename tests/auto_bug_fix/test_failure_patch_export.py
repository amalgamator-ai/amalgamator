"""Failure-path deliverables: candidate.patch and dossier.md on abort.

Regression tests for the frozen-baseline failure path: when the pipeline
escalates or errors, the agent's committed *and uncommitted* work must still
reach candidate.patch, dossier.md must still be written, the frozen baseline
(not the movable branch name) must define every range, and a PipelineStop
after the Phase 0.5c test port must not submit a test-only patch.
"""

import os
import subprocess

import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.dossier import Dossier
from auto_bug_fix.run_bug_fix import (
    PipelineEscalation,
    PipelineState,
    _emit_candidate_patch_on_failure,
    _export_candidate_patch_bytes,
    _restore_target_baseline,
    _squashed_backport_range,
    _write_dossier_on_failure,
)
from auto_bug_fix.tracker import Tracker


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture
def target_worktree(tmp_git_repo, tmp_path):
    """A worktree checked out on the target branch, like run_targets creates."""
    worktree = tmp_path / "wt"
    _git(tmp_git_repo.repo_path, "worktree", "add", str(worktree), "target")
    frozen = _git(tmp_git_repo.repo_path, "rev-parse", "target").stdout.strip()
    return tmp_git_repo.repo_path, str(worktree), frozen


def _config(repo_path: str, frozen: str) -> BugFixConfig:
    return BugFixConfig(
        repo_path=repo_path,
        build_dir=repo_path,
        source_branch="main",
        target_branch="target",
        source_fix_commit="abc1234",
        bug_description="test failure path",
        issue_id="CVE-2024-1234",
        frozen_target_sha=frozen,
    )


def _tracker(tmp_path) -> Tracker:
    return Tracker(
        issue_id="CVE-2024-1234",
        config_dict={},
        output_dir=str(tmp_path / "out"),
    )


def _commit(repo, msg, files):
    for name, text in files.items():
        path = os.path.join(repo, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", msg)


class TestEmitCandidatePatchOnFailure:
    def test_committed_and_uncommitted_work_both_reach_the_patch(
        self, target_worktree, tmp_path
    ):
        repo, worktree, frozen = target_worktree
        _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})
        # Phase 4b raises on re-verify failure *before* committing its edits.
        with open(os.path.join(worktree, "lib/url.c"), "a") as f:
            f.write("\nvoid recovered() { 4b edit }\n")

        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(_config(worktree, frozen), tracker)

        patch = os.path.join(tracker._output_dir, "candidate.patch")
        assert os.path.exists(patch)
        with open(patch) as f:
            content = f.read()
        assert "fixed" in content
        assert "recovered" in content  # the uncommitted 4b edit, snapshotted

    def test_no_work_means_no_patch(self, target_worktree, tmp_path):
        repo, worktree, frozen = target_worktree
        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(_config(worktree, frozen), tracker)
        assert not os.path.exists(os.path.join(tracker._output_dir, "candidate.patch"))

    def test_pipeline_stop_with_test_only_changes_emits_nothing(
        self, target_worktree, tmp_path
    ):
        """Phase 0.5c ported tests, then RED triage stopped the pipeline."""
        repo, worktree, frozen = target_worktree
        _commit(
            worktree,
            "port test files (Phase 0.5c)",
            {"tests/test_url.c": "void test_fixed() {}"},
        )

        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(
            _config(worktree, frozen), tracker, allow_test_only=False
        )
        assert not os.path.exists(os.path.join(tracker._output_dir, "candidate.patch"))

    def test_pipeline_stop_with_real_changes_still_emits(
        self, target_worktree, tmp_path
    ):
        repo, worktree, frozen = target_worktree
        _commit(
            worktree,
            "port the fix",
            {"lib/url.c": "fixed", "tests/test_url.c": "void t() {}"},
        )

        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(
            _config(worktree, frozen), tracker, allow_test_only=False
        )
        assert os.path.exists(os.path.join(tracker._output_dir, "candidate.patch"))

    def test_escalation_with_test_only_changes_still_emits(
        self, target_worktree, tmp_path
    ):
        """Suppression is reserved for PipelineStop; escalations always emit."""
        repo, worktree, frozen = target_worktree
        _commit(worktree, "tests only", {"tests/test_url.c": "void t() {}"})

        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(
            _config(worktree, frozen), tracker, allow_test_only=True
        )
        assert os.path.exists(os.path.join(tracker._output_dir, "candidate.patch"))


class TestWriteDossierOnFailure:
    def test_dossier_written_on_failure(self, tmp_path):
        state = PipelineState()
        state.dossier = Dossier(
            issue_id="CVE-2024-1234",
            source_branch="main",
            target_branch="target",
            fix_commit="abc1234",
            bug_description="b",
        )
        state.dossier.add("Triage result", "gate output", "Phase 0")

        tracker = _tracker(tmp_path)
        _write_dossier_on_failure(state, tracker)

        path = os.path.join(tracker._output_dir, "dossier.md")
        assert os.path.exists(path)
        with open(path) as f:
            assert "Triage result" in f.read()

    def test_fresh_state_still_writes(self, tmp_path):
        """A state whose phases never ran still produces an (empty) dossier."""
        state = PipelineState()
        tracker = _tracker(tmp_path)
        _write_dossier_on_failure(state, tracker)
        assert os.path.exists(os.path.join(tracker._output_dir, "dossier.md"))


class TestRestoreTargetBaseline:
    def _move_branch(self, repo, branch, worktree):
        _commit(worktree, "agent moved the branch", {"lib/moved.c": "moved"})
        return _git(repo, "rev-parse", branch).stdout.strip()

    def test_moved_local_branch_is_restored(self, target_worktree):
        repo, worktree, frozen = target_worktree
        moved = self._move_branch(repo, "target", worktree)
        assert moved != frozen

        _restore_target_baseline(repo, worktree, "target", frozen)

        assert _git(repo, "rev-parse", "target").stdout.strip() == frozen

    def test_remote_tracking_ref_is_left_alone(self, target_worktree):
        repo, worktree, frozen = target_worktree
        remote_ref = "refs/remotes/origin/target"
        moved = self._move_branch(repo, "target", worktree)
        _git(repo, "update-ref", remote_ref, moved)

        _restore_target_baseline(repo, worktree, "origin/target", frozen)

        # Unchanged, and no bogus refs/heads/origin/target was created.
        assert _git(repo, "rev-parse", "origin/target").stdout.strip() == moved
        assert (
            _git(
                repo, "rev-parse", "--verify", "refs/heads/origin/target", check=False
            ).returncode
            != 0
        )

    def test_tag_is_left_alone(self, target_worktree):
        repo, worktree, frozen = target_worktree
        moved = self._move_branch(repo, "target", worktree)
        _git(repo, "tag", "v9.9", moved)

        _restore_target_baseline(repo, worktree, "v9.9", frozen)

        # Tag unchanged; no same-named local branch spawned from it.
        assert _git(repo, "rev-parse", "v9.9").stdout.strip() == moved
        assert (
            _git(
                repo, "rev-parse", "--verify", "refs/heads/v9.9", check=False
            ).returncode
            != 0
        )

    def test_unmoved_branch_is_a_noop(self, target_worktree):
        repo, worktree, frozen = target_worktree
        _restore_target_baseline(repo, worktree, "target", frozen)
        assert _git(repo, "rev-parse", "target").stdout.strip() == frozen


class TestFrozenShaRanges:
    def test_empty_patch_escalates(self, target_worktree):
        """Phase 5 refuses to report success without a deliverable."""
        repo, worktree, frozen = target_worktree
        with pytest.raises(PipelineEscalation, match="empty"):
            _export_candidate_patch_bytes(_config(worktree, frozen))

    def test_squash_ranges_from_frozen_sha_when_branch_moved(self, target_worktree):
        """Agent commits landed on the target branch: branch-name merge-base
        collapses to HEAD (no squash); the frozen SHA must still yield one."""
        repo, worktree, frozen = target_worktree
        _commit(
            worktree,
            "agent port on the target branch",
            {"lib/url.c": "void f() { fixed }"},
        )

        cfg = _config(worktree, frozen)
        state = PipelineState()
        rng = _squashed_backport_range(cfg, state)

        assert rng is not None
        # The synthetic commit's parent must be the frozen baseline.
        parent = _git(worktree, "rev-parse", f"{rng.split('^..')[0]}^").stdout.strip()
        assert parent == frozen


class TestRunPipelineFailureWiring:
    """run_pipeline's except handlers, not just the helpers they call."""

    def _abort(self, target_worktree, tmp_path, monkeypatch, claude_cfg, exc, prepare):
        repo, worktree, frozen = target_worktree

        import auto_bug_fix.run_bug_fix as rbf

        def fake_phases(config, claude_config, state, tracker):
            # Phases 0-4 ran: the dossier has evidence, and the agent left
            # work behind — during the pipeline, i.e. after run_pipeline
            # froze the baseline, as in a real run.
            prepare(worktree)
            state.dossier.add("Triage result", "gate output", "Phase 0")
            raise exc("simulated abort")

        monkeypatch.setattr(rbf, "_run_pipeline_phases", fake_phases)
        out = tmp_path / "out"
        with pytest.raises(exc):
            rbf.run_pipeline(_config(worktree, frozen), claude_cfg, output_dir=str(out))
        return out

    def test_escalation_emits_patch_with_uncommitted_work_and_dossier(
        self, target_worktree, tmp_path, monkeypatch, claude_cfg
    ):
        def prepare(worktree):
            _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})
            with open(os.path.join(worktree, "lib/url.c"), "a") as f:
                f.write("\nvoid recovered() { 4b edit }\n")

        out = self._abort(
            target_worktree,
            tmp_path,
            monkeypatch,
            claude_cfg,
            PipelineEscalation,
            prepare,
        )
        patch = out / "candidate.patch"
        assert patch.exists()
        content = patch.read_text()
        assert "fixed" in content
        assert "recovered" in content
        assert "gate output" in (out / "dossier.md").read_text()

    def test_pipeline_stop_with_test_only_work_emits_no_patch(
        self, target_worktree, tmp_path, monkeypatch, claude_cfg
    ):
        from auto_bug_fix.run_bug_fix import PipelineStop

        def prepare(worktree):
            _commit(
                worktree,
                "port test files (Phase 0.5c)",
                {"tests/test_url.c": "void test_fixed() {}"},
            )

        out = self._abort(
            target_worktree, tmp_path, monkeypatch, claude_cfg, PipelineStop, prepare
        )
        assert not (out / "candidate.patch").exists()
        assert (out / "dossier.md").exists()

    def test_unexpected_error_emits_patch(
        self, target_worktree, tmp_path, monkeypatch, claude_cfg
    ):
        def prepare(worktree):
            _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})

        out = self._abort(
            target_worktree, tmp_path, monkeypatch, claude_cfg, RuntimeError, prepare
        )
        assert (out / "candidate.patch").exists()
