# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cluster plans: parsing, skip-safe dropping, and verify gates."""

from __future__ import annotations

import subprocess

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import (
    _commit_only_touches_skipped,
    _merge_series_assessment,
    _prepare_series_clusters,
    _verify_cluster_plan,
)
from auto_bug_fix.series_clusters import parse_cluster_plan


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _sha(cwd):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@t")
    (repo / "src").mkdir()
    (repo / "src/base.c").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return str(repo), repo


def _commit(repo, rel, content, message):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _sha(repo)


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


def test_parse_plan_accepts_a_valid_document():
    plan, errors = parse_cluster_plan(
        {
            "cluster": "tracker",
            "actions": [
                {
                    "symbol": "XML_GE",
                    "action": "synthesize",
                    "verify": "grep -q XML_GE lib/internal.h",
                }
            ],
            "risks": ["parser struct prefix"],
        }
    )

    assert errors == []
    assert plan is not None
    assert plan.cluster == "tracker"
    assert plan.actions[0].action == "synthesize"
    assert plan.risks == ["parser struct prefix"]


def test_parse_plan_rejects_missing_verify_and_unknown_action():
    _, missing = parse_cluster_plan(
        {"cluster": "c", "actions": [{"symbol": "s", "action": "export"}]}
    )
    assert any("verify" in error for error in missing)

    _, unknown = parse_cluster_plan(
        {
            "cluster": "c",
            "actions": [{"symbol": "s", "action": "port", "verify": "true"}],
        }
    )
    assert any("unknown action" in error for error in unknown)


def test_parse_plan_rejects_empty_and_non_object():
    for raw in ({}, [], None, {"cluster": "", "actions": []}):
        plan, errors = parse_cluster_plan(raw)
        assert plan is None
        assert errors


def test_commit_only_touches_skipped_distinguishes_source_from_docs(tmp_path):
    rp, repo = _repo(tmp_path)
    docs = _commit(repo, "docs/guide.md", "doc\n", "docs only")
    source = _commit(repo, "src/fix.c", "fix\n", "source change")

    assert _commit_only_touches_skipped(rp, docs, ["docs/"])
    assert not _commit_only_touches_skipped(rp, source, ["docs/"])
    assert not _commit_only_touches_skipped(rp, docs, [])


def test_prepare_clusters_uses_triage_partition(tmp_path):
    rp, repo = _repo(tmp_path)
    first = _commit(repo, "src/base.c", "a\n", "refactor")
    second = _commit(repo, "src/base.c", "b\n", "feature")
    docs = _commit(repo, "docs/note.md", "n\n", "docs")
    series = [first, second, docs]

    from auto_bug_fix.run_bug_fix import PipelineState

    state = PipelineState()
    state.triage_assessment = {
        "clusters": [{"id": "all", "commits": [first, second]}],
        "dropped": [{"commit": docs, "reason": "docs"}],
        "files_to_skip": ["docs/"],
    }

    clusters = _prepare_series_clusters(_config(rp, series), state)

    assert [cluster.id for cluster in clusters] == ["all"]
    assert [cluster.commits for cluster in clusters] == [[first, second]]
    assert state.dropped_commits


def test_prepare_clusters_keeps_a_dropped_commit_that_touches_source(tmp_path):
    rp, repo = _repo(tmp_path)
    first = _commit(repo, "src/base.c", "a\n", "fix")
    source = _commit(repo, "src/fix.c", "fix\n", "source")
    series = [first, source]

    from auto_bug_fix.run_bug_fix import PipelineState

    state = PipelineState()
    state.triage_assessment = {
        "clusters": [{"id": "all", "commits": series}],
        "dropped": [{"commit": source, "reason": "claimed unrelated"}],
        "files_to_skip": ["docs/"],
    }

    clusters = _prepare_series_clusters(_config(rp, series), state)

    covered = [sha for cluster in clusters for sha in cluster.commits]
    assert source in covered
    assert not state.dropped_commits


def test_verify_cluster_plan_reports_only_failures(tmp_path):
    rp, repo = _repo(tmp_path)
    sha = _commit(repo, "src/base.c", "b\n", "change")
    config = _config(rp, [sha])

    failures = _verify_cluster_plan(
        config,
        {
            "actions": [
                {"symbol": "ok", "action": "export", "verify": "test -f src/base.c"},
                {"symbol": "bad", "action": "export", "verify": "test -f missing.c"},
                {"symbol": "none", "action": "export", "verify": ""},
            ]
        },
    )

    assert len(failures) == 1
    assert "bad" in failures[0]


def test_verify_cluster_plan_accepts_grep_L_for_absent_symbol(tmp_path):
    rp, repo = _repo(tmp_path)
    sha = _sha(repo)
    config = _config(rp, [sha])

    failures = _verify_cluster_plan(
        config,
        {
            "actions": [
                {
                    "symbol": "ABSENT",
                    "action": "translate",
                    "verify": "grep -L ABSENT src/base.c",
                }
            ]
        },
    )

    assert failures == []


def test_verify_cluster_plan_rejects_grep_L_for_present_symbol(tmp_path):
    rp, repo = _repo(tmp_path)
    sha = _commit(repo, "src/base.c", "ABSENT\n", "change")
    config = _config(rp, [sha])

    failures = _verify_cluster_plan(
        config,
        {
            "actions": [
                {
                    "symbol": "ABSENT",
                    "action": "translate",
                    "verify": "grep -L ABSENT src/base.c",
                }
            ]
        },
    )

    assert len(failures) == 1


def test_verify_cluster_plan_rejects_grep_L_on_missing_path(tmp_path):
    rp, repo = _repo(tmp_path)
    sha = _sha(repo)
    config = _config(rp, [sha])

    failures = _verify_cluster_plan(
        config,
        {
            "actions": [
                {
                    "symbol": "ABSENT",
                    "action": "translate",
                    "verify": "grep -L ABSENT does-not-exist.c",
                }
            ]
        },
    )

    assert len(failures) == 1
    assert "does-not-exist.c" in failures[0]


def test_merge_keeps_clusters_from_triage():
    merged = _merge_series_assessment(
        {"files_to_skip": [], "test_files": [], "priority_files": []},
        {"clusters": [{"id": "a", "commits": ["abc"]}], "dropped": []},
    )

    assert merged["clusters"] == [{"id": "a", "commits": ["abc"]}]
