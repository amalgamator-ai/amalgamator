# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic clustering: measurement, invariants, caps, fallback."""

from __future__ import annotations

import subprocess

from auto_bug_fix.series_clusters import (
    Cluster,
    ClusterCaps,
    batch_groups,
    build_partition,
    cluster_mode,
    measure_cluster,
    predict_conflict_files,
    resolve_sha,
    single_commit_clusters,
    validate_partition,
)


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
    return str(repo), repo


def _commit(repo, rel, content, message):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _sha(repo)


def _linear_series(tmp_path):
    """Base plus three commits; c1 and c2 rewrite the same line."""
    rp, repo = _repo(tmp_path)
    _commit(repo, "src/base.c", "a\nb\nc\n", "base")
    first = _commit(repo, "src/base.c", "a1\nb\nc\n", "refactor 1")
    second = _commit(repo, "src/base.c", "a2\nb\nc\n", "refactor 2")
    third = _commit(repo, "src/other.c", "x\ny\n", "feature")
    return rp, repo, [first, second, third]


def test_measure_uses_net_range_not_sum_of_commits(tmp_path):
    rp, _, series = _linear_series(tmp_path)
    first_two = measure_cluster(rp, series[:2])
    per_commit = sum(measure_cluster(rp, [sha]).changed_lines for sha in series[:2])

    # Base had "a"; the net change to that line is one insertion and one
    # deletion, while the two commits touched it twice.
    assert first_two.changed_lines == 2
    assert per_commit == 4
    assert first_two.files == 1
    assert first_two.paths == ("src/base.c",)


def test_measure_of_single_commit_matches_its_own_diff(tmp_path):
    rp, _, series = _linear_series(tmp_path)
    size = measure_cluster(rp, [series[2]])

    assert size.files == 1
    assert size.changed_lines == 2  # two added lines in a new file


def test_resolve_sha_matches_prefixes_within_the_series_only(tmp_path):
    rp, repo, series = _linear_series(tmp_path)
    unrelated = _commit(repo, "src/base.c", "a2\nb\nc\nd\n", "not in series")

    assert resolve_sha(series[0][:7], series) == series[0]
    assert resolve_sha(unrelated[:7], series) is None
    assert resolve_sha("", series) is None


def test_validate_accepts_contiguous_partition(tmp_path):
    rp, _, series = _linear_series(tmp_path)
    clusters = [
        Cluster(id="a", commits=series[:2]),
        Cluster(id="b", commits=series[2:]),
    ]
    assert validate_partition(clusters, series, rp) == []


def test_validate_rejects_gaps_overlaps_and_reordering(tmp_path):
    rp, _, series = _linear_series(tmp_path)

    gap = validate_partition(
        [Cluster(id="a", commits=[series[0], series[2]])], series, rp
    )
    assert any("contiguous" in error or "cover" in error for error in gap)

    overlap = validate_partition(
        [
            Cluster(id="a", commits=[series[0], series[1]]),
            Cluster(id="b", commits=[series[1], series[2]]),
        ],
        series,
        rp,
    )
    assert any("twice" in error for error in overlap)

    reordered = validate_partition(
        [
            Cluster(id="a", commits=[series[1], series[0]]),
            Cluster(id="b", commits=[series[2]]),
        ],
        series,
        rp,
    )
    assert any("reorder" in error or "contiguous" in error for error in reordered)


def test_validate_rejects_empty_cluster_and_unknown_sha(tmp_path):
    rp, _, series = _linear_series(tmp_path)

    empty = validate_partition([Cluster(id="a", commits=[])], series, rp)
    assert any("empty" in error for error in empty)

    unknown = validate_partition([Cluster(id="a", commits=["deadbeef"])], series, rp)
    assert any("not in the series" in error for error in unknown)


def test_validate_rejects_a_merge_inside_a_multi_commit_cluster(tmp_path):
    rp, repo, series = _linear_series(tmp_path)
    _git(repo, "checkout", "-q", "-b", "side", series[0])
    side = _commit(repo, "side.txt", "s\n", "side work")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "--no-ff", "-m", "merge side", "side")
    merge = _sha(repo)
    extended = [*series, side, merge]

    errors = validate_partition([Cluster(id="a", commits=extended)], extended, rp)
    assert any("merge commit" in error for error in errors)

    ok = validate_partition([Cluster(id="merge", commits=[merge])], [merge], rp)
    assert ok == []


def test_caps_split_at_a_commit_boundary_and_never_split_one_commit(tmp_path):
    rp, _, series = _linear_series(tmp_path)
    caps = ClusterCaps(max_net_lines=3, max_files=1, max_commits=1)

    partition = build_partition(
        [{"id": "all", "commits": series}], series, rp, caps=caps
    )

    assert [cluster.commits for cluster in partition.clusters] == [
        [series[0]],
        [series[1]],
        [series[2]],
    ]
    assert partition.splits and not partition.fell_back


def test_single_commit_over_caps_is_kept_whole(tmp_path):
    rp, _, series = _linear_series(tmp_path)
    caps = ClusterCaps(max_net_lines=1, max_files=1, max_commits=1)

    partition = build_partition(
        [{"id": "one", "commits": [series[2]]}], series[2:], rp, caps=caps
    )

    assert len(partition.clusters) == 1
    assert partition.clusters[0].commits == [series[2]]


def test_mode_is_mechanical_only_for_one_small_non_public_file(tmp_path):
    rp, repo, series = _linear_series(tmp_path)

    small = Cluster(id="small", commits=[series[2]])
    small.size = measure_cluster(rp, small.commits)
    assert small.size.files == 1
    assert cluster_mode(small, []) == "mechanical"
    assert cluster_mode(small, ["src/other.c"]) == "semantic"

    (repo / "src/m1.c").write_text("m\n")
    (repo / "src/m2.c").write_text("n\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "two files")
    two_files = _sha(repo)
    multi = Cluster(id="multi", commits=[two_files])
    multi.size = measure_cluster(rp, multi.commits)
    assert multi.size.files == 2
    assert cluster_mode(multi, []) == "semantic"

    (repo / "src/large.c").write_text("\n".join(f"line {i}" for i in range(200)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "large")
    large = Cluster(id="large", commits=[_sha(repo)])
    large.size = measure_cluster(rp, large.commits)
    assert large.size.changed_lines > 120
    assert cluster_mode(large, []) == "semantic"


def test_unknown_sha_falls_back_to_one_cluster_per_commit(tmp_path):
    rp, _, series = _linear_series(tmp_path)

    partition = build_partition([{"id": "bad", "commits": ["deadbeef"]}], series, rp)

    assert partition.fell_back
    assert partition.rejected
    assert [cluster.commits for cluster in partition.clusters] == [
        [sha] for sha in series
    ]


def test_dropped_commits_leave_the_partition(tmp_path):
    rp, _, series = _linear_series(tmp_path)

    partition = build_partition(
        [{"id": "kept", "commits": series[:2]}],
        series,
        rp,
        dropped=[{"commit": series[2][:8], "reason": "docs"}],
    )

    assert not partition.fell_back
    assert [cluster.commits for cluster in partition.clusters] == [series[:2]]
    assert partition.dropped == [{"commit": series[2][:8], "reason": "docs"}]


def test_single_commit_clusters_preserve_order(tmp_path):
    rp, _, series = _linear_series(tmp_path)
    fallback = single_commit_clusters(series)

    assert [cluster.commits for cluster in fallback] == [[sha] for sha in series]


def test_commit_omitted_from_the_partition_gets_its_own_cluster(tmp_path):
    rp, _, series = _linear_series(tmp_path)

    partition = build_partition(
        [{"id": "first-two", "commits": series[:2]}], series, rp
    )

    assert not partition.fell_back
    assert [cluster.commits for cluster in partition.clusters] == [
        [series[0], series[1]],
        [series[2]],
    ]
    assert partition.clusters[-1].id.startswith("unclustered")


def _predicted(commits: list[str], mapping: dict) -> Cluster:
    cluster = Cluster(id="c", commits=commits)
    cluster.predicted_conflicts = mapping
    return cluster


def test_batch_groups_joins_commits_with_the_same_conflict_set():
    a, b = "a" * 40, "b" * 40
    cluster = _predicted([a, b], {a: ("src/x.c",), b: ("src/x.c",)})

    assert batch_groups(cluster) == [[a, b]]


def test_batch_groups_refuses_a_wider_conflict_set():
    a, b = "a" * 40, "b" * 40
    cluster = _predicted([a, b], {a: ("src/x.c",), b: ("src/x.c", "src/y.h")})

    assert batch_groups(cluster) == [[a], [b]]


def test_batch_groups_breaks_on_a_clean_commit():
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    cluster = _predicted(
        [a, b, c],
        {a: ("src/x.c",), b: (), c: ("src/x.c",)},
    )

    assert batch_groups(cluster) == [[a], [c]]


def test_predict_conflict_files_matches_a_real_conflict(tmp_path):
    rp, repo = _repo(tmp_path)
    _commit(repo, "src/base.c", "a\nb\nc\n", "base")

    _git(repo, "checkout", "-q", "-b", "upstream")
    _commit(repo, "src/base.c", "a\nB\nc\n", "upstream rewrites line")
    conflicting = _sha(repo)
    _commit(repo, "src/extra.c", "new\n", "upstream adds a file")
    clean = _sha(repo)

    _git(repo, "checkout", "-q", "main")
    base = _commit(repo, "src/base.c", "a\nT\nc\n", "target rewrites line")

    assert predict_conflict_files(rp, base, conflicting) == ("src/base.c",)
    assert predict_conflict_files(rp, base, clean) == ()
