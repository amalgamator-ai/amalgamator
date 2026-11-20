# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cluster a fix commit series for planning, sizing and verification.

The semantic triage proposes where the boundaries are; this module owns the
arithmetic: it resolves the proposed SHAs against the configured series,
enforces the partition invariants, measures each cluster with git, splits
oversized clusters at commit boundaries, and classifies each cluster as
mechanical or semantic.

Nothing here calls an LLM. See docs/series-clusters.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from auto_bug_fix.git_tools import _run_git

log = logging.getLogger(__name__)

MECHANICAL_MAX_CHANGED_LINES = 120

# Plan actions the resolver can act on. `port` is deliberately absent:
# prerequisites are never ported in series mode.
PLAN_ACTIONS = frozenset(
    {"translate", "synthesize", "export", "adapt_call_site", "skip"}
)


@dataclass(frozen=True)
class ClusterCaps:
    """Limits that decide whether commits may be merged into one cluster."""

    max_net_lines: int = 300
    max_files: int = 8
    max_commits: int = 8


@dataclass(frozen=True)
class ClusterSize:
    """Net diff of a cluster, measured from git rather than estimated."""

    insertions: int = 0
    deletions: int = 0
    files: int = 0
    paths: tuple[str, ...] = ()

    @property
    def changed_lines(self) -> int:
        return self.insertions + self.deletions

    def exceeds(self, caps: ClusterCaps, commits: int) -> bool:
        return (
            self.changed_lines > caps.max_net_lines
            or self.files > caps.max_files
            or commits > caps.max_commits
        )


@dataclass
class Cluster:
    """A contiguous run of series commits handled as one unit."""

    id: str
    commits: list[str]
    intent: str = ""
    kind: str = ""
    expected_conflict_files: list[str] = field(default_factory=list)
    size: ClusterSize = field(default_factory=ClusterSize)
    mode: str = "semantic"
    split_from: str = ""
    predicted_conflicts: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class Partition:
    """The accepted cluster list plus everything that was decided on the way."""

    clusters: list[Cluster]
    dropped: list[dict] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    fell_back: bool = False


@dataclass(frozen=True)
class PlanAction:
    """One binding instruction for a cluster, with its own gate."""

    symbol: str
    action: str
    verify: str
    target_state: str = ""
    use_instead: str = ""
    where: str = ""
    note: str = ""


@dataclass
class ClusterPlan:
    """The per-cluster plan, generated lazily against the current tree."""

    cluster: str
    actions: list[PlanAction] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


def parse_cluster_plan(raw: object) -> tuple[ClusterPlan | None, list[str]]:
    """Validate a plan document; return the plan and the reasons it is unusable.

    Structural only. Shell-command safety is enforced by the pipeline, which
    owns the validator the rest of the code uses.
    """
    if not isinstance(raw, dict):
        return None, ["plan is not a JSON object"]
    cluster = raw.get("cluster")
    if not isinstance(cluster, str) or not cluster.strip():
        return None, ["plan has no cluster id"]
    actions_raw = raw.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        return None, ["plan has no actions"]

    errors: list[str] = []
    actions: list[PlanAction] = []
    for index, entry in enumerate(actions_raw, start=1):
        if not isinstance(entry, dict):
            errors.append(f"action {index}: not an object")
            continue
        symbol = str(entry.get("symbol") or "").strip()
        action = str(entry.get("action") or "").strip()
        verify = str(entry.get("verify") or "").strip()
        if not symbol:
            errors.append(f"action {index}: no symbol")
        if action not in PLAN_ACTIONS:
            errors.append(f"action {index}: unknown action {action!r}")
        if not verify:
            errors.append(f"action {index}: no verify command")
        actions.append(
            PlanAction(
                symbol=symbol,
                action=action,
                verify=verify,
                target_state=str(entry.get("target_state") or ""),
                use_instead=str(entry.get("use_instead") or ""),
                where=str(entry.get("where") or ""),
                note=str(entry.get("note") or ""),
            )
        )

    risks = [str(risk) for risk in raw.get("risks") or [] if str(risk).strip()]
    if errors:
        return None, errors
    return ClusterPlan(cluster=cluster.strip(), actions=actions, risks=risks), []


def measure_cluster(repo_path: str, commits: list[str]) -> ClusterSize:
    """Net numstat between the parent of the first commit and the last.

    Summing per-commit diffs overstates a cluster: its commits touch the same
    files, so the net range is the honest size. Returns a zeroed size when the
    range cannot be measured; callers treat that as "unsized", never as "small".
    """
    if not commits:
        return ClusterSize()
    parent = _run_git(repo_path, ["rev-parse", "--verify", f"{commits[0]}^"])
    if not parent.success:
        return ClusterSize()
    result = _run_git(repo_path, ["diff", "--numstat", f"{commits[0]}^", commits[-1]])
    if not result.success:
        return ClusterSize()
    insertions = deletions = files = 0
    paths: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files += 1
        paths.append(parts[2])
        # Binary files report "-" instead of a count; they still count as files.
        insertions += int(parts[0]) if parts[0].isdigit() else 0
        deletions += int(parts[1]) if parts[1].isdigit() else 0
    return ClusterSize(
        insertions=insertions,
        deletions=deletions,
        files=files,
        paths=tuple(paths),
    )


def predict_conflict_files(
    repo_path: str, base_ref: str, commit: str
) -> tuple[str, ...]:
    """Files a cherry-pick of ``commit`` would conflict in on ``base_ref``.

    ``git merge-tree --merge-base=<commit>^`` reproduces cherry-pick's 3-way
    merge exactly (the commit's parent is the merge base), without touching the
    worktree. Predicting each commit against a fixed base overestimates later
    commits in a series, because their predecessors have not landed — that is
    acceptable for grouping, and it never underestimates.
    """
    result = _run_git(
        repo_path,
        [
            "merge-tree",
            "--write-tree",
            f"--merge-base={commit}^",
            "--name-only",
            base_ref,
            commit,
        ],
    )
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        return ()
    paths: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            break
        paths.append(line.strip())
    return tuple(paths)


def batch_groups(cluster: Cluster, max_commits: int = 7) -> list[list[str]]:
    """Contiguous runs of commits that conflict in at least one shared file.

    A run is worth batching because each of its commits would otherwise open
    its own resolution session, and they collide with the same target code.
    Commits that apply cleanly end a run: they cost nothing alone.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    shared: set[str] = set()
    for sha in cluster.commits:
        files = set(cluster.predicted_conflicts.get(sha) or ())
        if not files:
            if current:
                groups.append(current)
            current, shared = [], set()
            continue
        if not current:
            current, shared = [sha], files
        elif files == shared and len(current) < max_commits:
            current.append(sha)
        else:
            groups.append(current)
            current, shared = [sha], files
    if current:
        groups.append(current)
    return groups


def is_merge_commit(repo_path: str, sha: str) -> bool:
    """True when the commit has more than one parent."""
    result = _run_git(repo_path, ["rev-list", "--parents", "-n", "1", sha])
    if not result.success:
        return False
    return len(result.stdout.split()) > 2


def resolve_sha(raw: str, series: list[str]) -> str | None:
    """Match a model-supplied SHA against the configured series by prefix.

    Only the series may be named — a commit outside it must not silently widen
    the backport — and the match must be unambiguous.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    matches = [
        sha
        for sha in series
        if sha == raw or sha.startswith(raw) or raw.startswith(sha)
    ]
    return matches[0] if len(matches) == 1 else None


def validate_partition(
    clusters: list[Cluster],
    series: list[str],
    repo_path: str,
) -> list[str]:
    """Return the invariant violations of a proposed partition.

    An empty list means the partition is usable. Contiguity is checked against
    the series order so a cluster can never reorder upstream commits.
    """
    errors: list[str] = []
    if not clusters:
        return ["no clusters"]

    position = {sha: index for index, sha in enumerate(series)}
    seen: list[str] = []
    for cluster in clusters:
        if not cluster.commits:
            errors.append(f"cluster {cluster.id}: empty")
            continue
        for sha in cluster.commits:
            if sha not in position:
                errors.append(f"cluster {cluster.id}: {sha} is not in the series")
            elif sha in seen:
                errors.append(f"cluster {cluster.id}: {sha} appears twice")
        seen.extend(cluster.commits)
        if len(cluster.commits) > 1:
            for sha in cluster.commits:
                if is_merge_commit(repo_path, sha):
                    errors.append(
                        f"cluster {cluster.id}: merge commit {sha} inside a "
                        "multi-commit cluster"
                    )

    if errors:
        return errors

    if seen != series:
        if sorted(seen) == sorted(series):
            errors.append("clusters reorder the series")
        else:
            missing = [sha for sha in series if sha not in seen]
            errors.append(
                "clusters do not cover the series"
                + (f"; missing {', '.join(missing[:5])}" if missing else "")
            )
    for cluster in clusters:
        indexes = [position[sha] for sha in cluster.commits]
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            errors.append(f"cluster {cluster.id}: commits are not contiguous")
    return errors


def single_commit_clusters(series: list[str]) -> list[Cluster]:
    """The fallback partition: one cluster per commit, in order."""
    return [
        Cluster(id=f"commit-{index}", commits=[sha])
        for index, sha in enumerate(series, start=1)
    ]


def _split(cluster: Cluster, repo_path: str, caps: ClusterCaps) -> list[Cluster]:
    """Peel the largest commit off an oversized cluster until it fits.

    Caps merge commits; they never split one. A single commit that exceeds the
    caps stays whole, because a commit is atomic.
    """
    pieces = [cluster]
    out: list[Cluster] = []
    while pieces:
        current = pieces.pop(0)
        if len(current.commits) <= 1:
            out.append(current)
            continue
        current.size = measure_cluster(repo_path, current.commits)
        if not current.size.exceeds(caps, len(current.commits)):
            out.append(current)
            continue
        largest = max(
            current.commits,
            key=lambda sha: measure_cluster(repo_path, [sha]).changed_lines,
        )
        index = current.commits.index(largest)
        remainder = current.commits[:index] + current.commits[index + 1 :]
        peeled = Cluster(
            id=f"{current.id}.{largest[:8]}",
            commits=[largest],
            intent=current.intent,
            kind=current.kind,
            expected_conflict_files=list(current.expected_conflict_files),
            split_from=current.id,
        )
        out.append(peeled)
        if remainder:
            pieces.insert(
                0,
                Cluster(
                    id=current.id,
                    commits=remainder,
                    intent=current.intent,
                    kind=current.kind,
                    expected_conflict_files=list(current.expected_conflict_files),
                    split_from=current.id,
                ),
            )
    return out


def _restore_order(clusters: list[Cluster], series: list[str]) -> list[Cluster]:
    """Order clusters and their commits by series position after splitting."""
    position = {sha: index for index, sha in enumerate(series)}
    for cluster in clusters:
        cluster.commits.sort(key=lambda sha: position[sha])
    return sorted(clusters, key=lambda c: position[c.commits[0]])


def cluster_mode(cluster: Cluster, public_interface_files: list[str]) -> str:
    """Mechanical only when a single small file moves and no public surface does."""
    if cluster.size.files != 1:
        return "semantic"
    if cluster.size.changed_lines > MECHANICAL_MAX_CHANGED_LINES:
        return "semantic"
    interface = set(public_interface_files or [])
    if interface and set(cluster.size.paths) & interface:
        return "semantic"
    return "mechanical"


def build_partition(
    raw_clusters: list[dict],
    series: list[str],
    repo_path: str,
    caps: ClusterCaps | None = None,
    public_interface_files: list[str] | None = None,
    dropped: list[dict] | None = None,
    base_ref: str = "",
) -> Partition:
    """Resolve, validate, split and classify a triage-proposed partition.

    Any rejection falls back to one cluster per commit: clustering is an
    optimisation and must never turn a runnable case into a failed one.
    """
    caps = caps or ClusterCaps()
    dropped = list(dropped or [])

    dropped_shas = {
        resolved
        for resolved in (
            resolve_sha(item.get("commit", ""), series) for item in dropped
        )
        if resolved
    }
    effective = [sha for sha in series if sha not in dropped_shas]

    clusters: list[Cluster] = []
    rejections: list[str] = []
    for index, raw in enumerate(raw_clusters or [], start=1):
        resolved: list[str] = []
        for entry in raw.get("commits") or []:
            sha = resolve_sha(entry, series)
            if sha is None:
                rejections.append(f"cluster {index}: unknown commit {entry!r}")
                continue
            resolved.append(sha)
        clusters.append(
            Cluster(
                id=str(raw.get("id") or f"cluster-{index}"),
                commits=resolved,
                intent=str(raw.get("intent") or ""),
                kind=str(raw.get("kind") or ""),
                expected_conflict_files=[
                    str(path) for path in raw.get("expected_conflict_files") or []
                ],
            )
        )

    covered = {sha for cluster in clusters for sha in cluster.commits}
    position = {sha: index for index, sha in enumerate(effective)}
    for index, sha in enumerate(effective):
        if sha in covered:
            continue
        log.info(
            "Cluster partition leaves %s unclustered — keeping it as its own cluster",
            sha[:12],
        )
        clusters.append(Cluster(id=f"unclustered-{index + 1}", commits=[sha]))
    # A cluster may start with a commit the model also listed as dropped.
    # Ordering must not raise on it: validate_partition below reports the
    # overlap and the partition falls back cleanly.
    clusters.sort(
        key=lambda c: (
            position.get(c.commits[0], len(position)) if c.commits else len(position)
        )
    )

    errors = rejections + validate_partition(clusters, effective, repo_path)
    if errors:
        log.warning(
            "Cluster partition rejected (%s) — falling back to one cluster per commit",
            "; ".join(errors[:3]),
        )
        fallback = single_commit_clusters(effective)
        for cluster in fallback:
            cluster.size = measure_cluster(repo_path, cluster.commits)
            cluster.mode = cluster_mode(cluster, public_interface_files or [])
        return Partition(
            clusters=fallback,
            dropped=dropped,
            rejected=errors,
            fell_back=True,
        )

    splits: list[str] = []
    split_clusters: list[Cluster] = []
    for cluster in clusters:
        cluster.size = measure_cluster(repo_path, cluster.commits)
        if (
            cluster.size.exceeds(caps, len(cluster.commits))
            and len(cluster.commits) > 1
        ):
            splits.append(
                f"cluster {cluster.id}: {cluster.size.changed_lines} lines / "
                f"{cluster.size.files} files over caps — split"
            )
            split_clusters.extend(_split(cluster, repo_path, caps))
        else:
            split_clusters.append(cluster)

    final = _restore_order(split_clusters, effective)
    for cluster in final:
        cluster.size = measure_cluster(repo_path, cluster.commits)
        cluster.mode = cluster_mode(cluster, public_interface_files or [])
        if base_ref:
            cluster.predicted_conflicts = {
                sha: predict_conflict_files(repo_path, base_ref, sha)
                for sha in cluster.commits
            }
    return Partition(clusters=final, dropped=dropped, splits=splits)
