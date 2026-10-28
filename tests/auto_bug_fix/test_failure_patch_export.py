# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Failure-path deliverables: candidate.patch and dossier.md on abort.

Regression tests for the frozen-baseline failure path: when the pipeline
escalates or errors, the agent's *committed* work must still reach
candidate.patch, dossier.md must still be written, the frozen baseline (not
the movable branch name) must define every range, and a PipelineStop after
the Phase 0.5c test port must not submit a test-only patch.

The export is deliberately commit-based. An earlier revision snapshotted the
working tree at abort time; on the motivating expat case that tree is what
``centpkg prep`` left behind, so the snapshot committed build corruption into
the deliverable. The phases now commit at their own pre-verification and
abort boundaries instead, and these tests pin that the export adds nothing.
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
    def test_committed_work_reaches_the_patch(self, target_worktree, tmp_path):
        repo, worktree, frozen = target_worktree
        _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})

        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(_config(worktree, frozen), tracker)

        patch = os.path.join(tracker._output_dir, "candidate.patch")
        assert os.path.exists(patch)
        with open(patch) as f:
            content = f.read()
        assert "fixed" in content

    def test_export_never_commits_verification_dirt(self, target_worktree, tmp_path):
        """The export is commit-based and must add nothing of its own.

        On the expat path the uncommitted tree at abort time is what
        ``centpkg prep`` left: a re-patched tracked file and an unpacked
        source tree. An export that snapshotted the worktree would ship that
        corruption to the verifier.
        """
        repo, worktree, frozen = target_worktree
        _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})
        # Verification side effects, tracked and untracked.
        with open(os.path.join(worktree, "lib/url.c"), "a") as f:
            f.write("\n/* re-patched by centpkg prep */\n")
        os.makedirs(os.path.join(worktree, "curl-8.9.1"), exist_ok=True)
        with open(os.path.join(worktree, "curl-8.9.1", "unpacked.c"), "w") as f:
            f.write("unpacked source tree\n")

        head_before = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(_config(worktree, frozen), tracker)

        # Nothing was committed by the export itself.
        assert _git(worktree, "rev-parse", "HEAD").stdout.strip() == head_before
        with open(os.path.join(tracker._output_dir, "candidate.patch")) as f:
            content = f.read()
        assert "fixed" in content
        assert "centpkg prep" not in content
        assert "unpacked" not in content

    def test_exported_patch_applies_to_the_frozen_baseline(
        self, target_worktree, tmp_path
    ):
        """A patch that cannot be applied is not a deliverable."""
        repo, worktree, frozen = target_worktree
        _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})
        _commit(worktree, "add a helper", {"lib/helper.c": "void h() {}"})
        # Verification dirt that must not make it into the applied result.
        with open(os.path.join(worktree, "lib/url.c"), "a") as f:
            f.write("\n/* dirt */\n")

        tracker = _tracker(tmp_path)
        _emit_candidate_patch_on_failure(_config(worktree, frozen), tracker)
        patch = os.path.join(tracker._output_dir, "candidate.patch")

        pristine = tmp_path / "pristine"
        _git(repo, "worktree", "add", "--detach", str(pristine), frozen)
        am = _git(pristine, "am", patch, check=False)
        assert am.returncode == 0, am.stdout + am.stderr
        with open(pristine / "lib" / "url.c") as f:
            applied = f.read()
        assert "fixed" in applied
        assert "dirt" not in applied
        assert (pristine / "lib" / "helper.c").exists()

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

    def test_escalation_emits_committed_work_but_not_verify_dirt(
        self, target_worktree, tmp_path, monkeypatch, claude_cfg
    ):
        def prepare(worktree):
            _commit(worktree, "port the fix", {"lib/url.c": "void f() { fixed }"})
            # What the orchestrator's re-verify leaves behind when it fails.
            with open(os.path.join(worktree, "lib/url.c"), "a") as f:
                f.write("\n/* re-patched by centpkg prep */\n")

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
        assert "centpkg prep" not in content
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


class TestRestoreTargetBaselineSafety:
    """The restore repoints a *shared* ref, so every guard must fail closed."""

    def _move_branch(self, repo, branch, worktree):
        _commit(worktree, "agent commit on the target branch", {"x.c": "moved"})
        return _git(worktree, "rev-parse", "HEAD").stdout.strip()

    def test_moved_tip_is_preserved_before_the_branch_is_reset(self, target_worktree):
        """Restoring must never be the only copy of the agent's commits."""
        repo, worktree, frozen = target_worktree
        moved = self._move_branch(repo, "target", worktree)

        assert _restore_target_baseline(repo, worktree, "target", frozen) is None

        assert _git(repo, "rev-parse", "target").stdout.strip() == frozen
        preserved = _git(repo, "rev-parse", "amalgamator-failed/target").stdout.strip()
        assert preserved == moved

    def test_branch_held_by_another_worktree_is_left_alone(
        self, target_worktree, tmp_path
    ):
        """The ref is shared state; repointing it corrupts the other checkout."""
        repo, worktree, frozen = target_worktree
        moved = self._move_branch(repo, "target", worktree)
        # A second worktree that also holds the branch. git refuses two
        # checkouts of one branch, so hand the pipeline a *different*
        # worktree path and let `target` stay held by `worktree`.
        other = tmp_path / "decoy"
        _git(repo, "worktree", "add", "--detach", str(other), frozen)

        reason = _restore_target_baseline(repo, str(other), "target", frozen)

        assert reason is not None and "another worktree" in reason
        assert _git(repo, "rev-parse", "target").stdout.strip() == moved

    def test_unreadable_worktree_list_fails_closed(self, target_worktree, monkeypatch):
        """'Cannot tell' must never be read as 'no holders'."""
        import auto_bug_fix.run_bug_fix as rbf

        repo, worktree, frozen = target_worktree
        moved = self._move_branch(repo, "target", worktree)
        monkeypatch.setattr(rbf, "_branch_checkout_holders", lambda *a, **k: None)

        reason = _restore_target_baseline(repo, worktree, "target", frozen)

        assert reason is not None and "worktree list failed" in reason
        assert _git(repo, "rev-parse", "target").stdout.strip() == moved

    def test_concurrent_move_loses_the_cas_and_is_left_alone(
        self, target_worktree, monkeypatch
    ):
        """A branch that moved again after we read it must not be clobbered."""
        import auto_bug_fix.run_bug_fix as rbf

        repo, worktree, frozen = target_worktree
        self._move_branch(repo, "target", worktree)
        real_rev_parse = rbf.git_rev_parse

        def stale(path, ref):
            # Report a tip that is not the branch's real value, so the
            # compare-and-swap old-value check must fail.
            if ref == "target":
                return "0" * 40
            return real_rev_parse(path, ref)

        monkeypatch.setattr(rbf, "git_rev_parse", stale)
        reason = _restore_target_baseline(repo, worktree, "target", frozen)

        assert reason is not None and "CAS" in reason
        assert _git(repo, "rev-parse", "target").stdout.strip() != frozen

    def test_no_frozen_baseline_is_reported_not_silently_skipped(self, target_worktree):
        repo, worktree, frozen = target_worktree
        reason = _restore_target_baseline(repo, worktree, "target", "")
        assert reason is not None and "no frozen baseline" in reason


class TestIsTestPath:
    """The heuristic can suppress a deliverable, so false positives cost real fixes."""

    @pytest.mark.parametrize(
        "path",
        [
            "src/test/java/com/x/FooTest.java",
            "src/test/java/com/x/FooTests.java",
            "tests/test_url.c",
            "lib/url_test.go",
            "lib/url_test.c",
            "spec/url_spec.rb",
            "testdata/corpus.bin",
            "web/url.test.js",
        ],
    )
    def test_recognised_as_test(self, path):
        from auto_bug_fix.run_bug_fix import _is_test_path

        assert _is_test_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            # A lowercased `endswith("test.java")` misclassified these.
            "src/main/java/com/x/Contest.java",
            "src/main/java/com/x/Latest.java",
            "lib/url.c",
            "src/protest.go",
            "docs/testing.md",
        ],
    )
    def test_not_a_test(self, path):
        from auto_bug_fix.run_bug_fix import _is_test_path

        assert _is_test_path(path) is False


class TestCommitBoundaries:
    """Agent work is committed before verification can dirty the tree."""

    def test_phase_4b_commits_before_reverification(self, tmp_path, monkeypatch):
        """The pre-verification commit is what carries 4b's work to the patch.

        Committing after re-verification (as this code used to) loses the
        work on exactly the path this PR exists for: re-verify fails, the
        phase raises, and format-patch sees no commit.
        """
        from unittest.mock import patch as mock_patch

        from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
        from tests.auto_bug_fix.test_run_bug_fix import (
            _git_4b,
            _phase_4b_fixtures,
        )

        config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
        head_before = _git_4b(tmp_path, "rev-parse", "HEAD")

        async def session(cfg, steps):
            (tmp_path / "src.c").write_text("int a = 2; /* 4b fix */\n")
            (tmp_path / "run_and_fix_success.txt").write_text("SUCCESS\n")
            return []

        def failing_verify(cfg, st):
            raise PipelineEscalation("re-verify failed (centpkg prep)")

        with (
            mock_patch("auto_bug_fix.run_bug_fix._agent_run", new=session),
            mock_patch("auto_bug_fix.run_bug_fix.phase_4_verify", new=failing_verify),
            pytest.raises(PipelineEscalation),
        ):
            phase_4b_run_and_fix(config, claude_cfg, state, None)

        assert _git_4b(tmp_path, "rev-parse", "HEAD") != head_before
        assert "4b fix" in _git_4b(tmp_path, "show", "HEAD")

    def test_phase_4b_agent_failure_commits_its_work(self, tmp_path):
        """An agent that reports failure may still have produced a real fix."""
        from unittest.mock import patch as mock_patch

        from auto_bug_fix.run_bug_fix import phase_4b_run_and_fix
        from tests.auto_bug_fix.test_run_bug_fix import (
            _git_4b,
            _phase_4b_fixtures,
        )

        config, claude_cfg, state = _phase_4b_fixtures(tmp_path)
        head_before = _git_4b(tmp_path, "rev-parse", "HEAD")

        async def session(cfg, steps):
            (tmp_path / "src.c").write_text("int a = 2; /* partial fix */\n")
            (tmp_path / "run_and_fix_failure.txt").write_text("FAILED\n")
            return []

        with (
            mock_patch("auto_bug_fix.run_bug_fix._agent_run", new=session),
            pytest.raises(PipelineEscalation),
        ):
            phase_4b_run_and_fix(config, claude_cfg, state, None)

        assert _git_4b(tmp_path, "rev-parse", "HEAD") != head_before
        shown = _git_4b(tmp_path, "show", "HEAD")
        assert "partial fix" in shown
        # The marker file is pipeline bookkeeping, not part of the fix.
        assert "run_and_fix_failure" not in shown


class TestEmptyDeliverable:
    def test_commits_with_no_net_diff_escalate(self, target_worktree):
        """format-patch renders a message-only commit as headers with no diff."""
        repo, worktree, frozen = target_worktree
        _git(worktree, "commit", "--allow-empty", "-m", "message-only fixup")

        with pytest.raises(PipelineEscalation, match="candidate patch is empty"):
            _export_candidate_patch_bytes(_config(worktree, frozen))
