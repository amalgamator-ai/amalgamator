"""Local guardrails for the human-run PR scripts.

Every script calls exactly one of these before it asks for confirmation, so the
confirmation prompt is never the only safety net. Everything here is local and
read-only: git status/rev-parse/remote/diff, file inspection. No network.

Exit codes (shared contract with pr_flow/scripts/*.sh):
    0  OK
    21 state file missing/unusable
    22 change scope violation (unexpected files, or internal strings in the diff)
    23 commit/PR description invalid
    24 working tree not clean, or HEAD != state.head_sha
    25 branch name unsafe or collides with the base branch
    26 remotes do not match the state (fork missing, file://, or fork == upstream)
    27 verification verdict is not green (or is stale for the current HEAD)
    28 nothing pending (empty round inbox)
    29 gh CLI missing or unauthenticated
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import hashlib
import json
import os
import re
import subprocess

from llm.auto_code_gen.pr_flow import gates, pr_state

EXIT_OK = 0
EXIT_STATE = 21
EXIT_SCOPE = 22
EXIT_DESCS = 23
EXIT_TREE = 24
EXIT_BRANCH = 25
EXIT_REMOTE = 26
EXIT_VERDICT = 27
EXIT_DCO = 28
EXIT_NOTHING = 28
EXIT_AUTH = 29
EXIT_REBASE = 30
EXIT_LINT = 31

MAX_COMMIT_TITLE = 72

PROTECTED_BRANCHES = ("main", "master", "develop", "trunk", "next", "HEAD")
BRANCH_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{2,59}$")
BRANCH_COMPONENT_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")

INTERNAL_PATTERNS = (
    (r"/home/[A-Za-z0-9._-]+", "developer home path"),
    (r"/workspace/[A-Za-z0-9._/-]+", "cluster PVC path"),
    (r"quay\.io/\S+", "internal container image"),
    (r"\bai_auto_analyze\b|\bauto_code_gen\b", "internal tooling name"),
    (r"\bCo-Authored-By\b", "co-author trailer"),
    (r"\bClaude\b|\bClaude Code\b|\bAI assistant\b|Generated with", "tool attribution"),
)


class CheckError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _git(repo_dir, *args):
    result = subprocess.run(
        ["git", "-C", repo_dir] + list(args), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise CheckError(
            EXIT_STATE,
            "git {} failed: {}".format(
                " ".join(args), result.stderr.strip() or result.stdout.strip()
            ),
        )
    return result.stdout


def _load_state(args):
    path = getattr(args, "state", None)
    if not path or not os.path.isfile(path):
        raise CheckError(EXIT_STATE, f"state file not found: {path}")
    return pr_state.load_state(path)


def _porcelain_files(repo_dir):
    files = []
    for line in _git(repo_dir, "status", "--porcelain").splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if "->" in path.split(" ", 1)[0] or " -> " in path:
            path = path.split(" -> ")[-1]
        files.append(path.strip().strip('"'))
    return files


def _patch_files(patch_path):
    files = []
    try:
        with open(patch_path) as f:
            for line in f:
                m = re.match(r"^\+\+\+ b/(.+)$", line.rstrip("\n"))
                if m and m.group(1) != "/dev/null":
                    files.append(m.group(1))
    except OSError as exc:
        raise CheckError(EXIT_STATE, f"cannot read patch {patch_path}: {exc}") from exc
    return files


def _latest_runtime_patch(output_dir):
    from llm.auto_code_gen.code_gen_prompts import RUNTIME_DIR

    root = os.path.join(output_dir, RUNTIME_DIR)
    found = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            m = re.match(r"^V(\d+)$", name)
            candidate = os.path.join(root, name, "patch.patch")
            if m and os.path.isfile(candidate):
                found.append((int(m.group(1)), candidate))
    return max(found)[1] if found else None


def _scan_for_internal(text, extra_patterns=()):
    hits = []
    for pattern, label in list(INTERNAL_PATTERNS) + list(extra_patterns):
        m = re.search(pattern, text)
        if m:
            hits.append((label, m.group(0)[:80]))
    return hits


def _state_scrub_patterns(state):
    """Add the run's own cluster/config identifiers to the leak scan."""
    extra = []
    config_path = state.get("config_path") or ""
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except OSError, json.JSONDecodeError:
            cfg = {}
        for key in ("remote_namespace", "remote_pvc_name", "remote_model_pvc_name"):
            value = cfg.get(key)
            if isinstance(value, str) and value:
                extra.append((rf"\b{re.escape(value)}\b", f"internal {key}"))
    return extra


def check_branch_name(state, name, allow_prefixless=False):
    if not name:
        raise CheckError(EXIT_BRANCH, "branch name is empty")
    if (
        name in PROTECTED_BRANCHES
        or name.startswith("release/")
        or name.startswith("releases/")
    ):
        raise CheckError(EXIT_BRANCH, f"'{name}' is a protected branch name")
    if (
        name.startswith("-")
        or ".." in name
        or "~" in name
        or "^" in name
        or ":" in name
    ):
        raise CheckError(
            EXIT_BRANCH, f"'{name}' contains a git-revision-unsafe character"
        )
    if not BRANCH_RE.match(name):
        raise CheckError(
            EXIT_BRANCH,
            f"'{name}' violates {BRANCH_RE.pattern} (lowercase, 3-60 chars, [a-z0-9._/-])",
        )
    for component in name.split("/"):
        if not BRANCH_COMPONENT_RE.match(component):
            raise CheckError(
                EXIT_BRANCH,
                f"'{name}' has a path segment '{component}' that starts or ends with a separator "
                "character -- git and many CLIs mishandle those",
            )
    if name == state.get("upstream_branch") or name == state.get("base_branch"):
        raise CheckError(
            EXIT_BRANCH,
            f"'{name}' equals the base branch -- a PR would target itself",
        )
    if name == state.get("local_branch"):
        raise CheckError(
            EXIT_BRANCH,
            f"'{name}' equals the local branch; pick a distinct remote name",
        )
    prefix = state.get("branch_prefix") or ""
    if prefix and not name.startswith(prefix) and not allow_prefixless:
        raise CheckError(
            EXIT_BRANCH,
            f"'{name}' must start with '{prefix}' (state.branch_prefix)",
        )
    return f"branch name OK: {name}"


def check_remotes(state):
    repo_dir = state.get("repo_dir") or ""
    if not os.path.isdir(os.path.join(repo_dir, ".git")) and not os.path.isfile(
        os.path.join(repo_dir, ".git")
    ):
        raise CheckError(EXIT_REMOTE, f"not a git repository: {repo_dir}")

    remotes = {}
    for line in _git(repo_dir, "remote", "-v").splitlines():
        parts = line.split()
        if len(parts) >= 2 and "(fetch)" in line:
            remotes[parts[0]] = parts[1]
    if not remotes:
        raise CheckError(EXIT_REMOTE, f"repository has no remotes: {repo_dir}")

    fork_remote = state.get("fork_remote") or "fork"
    if fork_remote not in remotes:
        raise CheckError(
            EXIT_REMOTE,
            "remote '{}' is missing; add it:  git -C {} remote add {} {}".format(
                fork_remote, repo_dir, fork_remote, state.get("fork_url", "<url>")
            ),
        )

    fork_url = remotes[fork_remote]
    if fork_url.startswith("file://") or fork_url.startswith("/"):
        raise CheckError(
            EXIT_REMOTE,
            f"remote '{fork_remote}' resolves to a local path ({fork_url})",
        )
    fork_repo = pr_state.parse_git_url(fork_url)
    if not fork_repo:
        raise CheckError(EXIT_REMOTE, f"cannot parse remote URL: {fork_url}")
    if fork_repo == state.get("upstream_repo"):
        raise CheckError(
            EXIT_REMOTE,
            f"remote '{fork_remote}' is the upstream repo ({fork_repo}); push to a fork",
        )
    if state.get("fork_repo") and fork_repo != state.get("fork_repo"):
        raise CheckError(
            EXIT_REMOTE,
            "remote '{}' is {}, state expects {}".format(
                fork_remote, fork_repo, state.get("fork_repo")
            ),
        )

    owner = pr_state.url_owner(fork_url)
    if state.get("fork_owner") and owner != state.get("fork_owner"):
        raise CheckError(
            EXIT_REMOTE,
            "fork owner mismatch: remote says {}, state says {}".format(
                owner, state.get("fork_owner")
            ),
        )

    for name, url in remotes.items():
        if (
            pr_state.parse_git_url(url) == state.get("upstream_repo")
            and name != fork_remote
        ):
            return f"remotes OK: push target '{fork_remote}' -> {fork_repo}, upstream '{name}' is fetch-only"
    return f"remotes OK: push target '{fork_remote}' -> {fork_repo}"


def check_descs(state, extra_patterns=()):
    from llm.auto_code_gen.pr_flow.pr_state import COMMIT_DESC_FILE, PR_DESC_FILE

    directory = pr_state.descs_dir(state["output_dir"])
    commit_path = os.path.join(directory, COMMIT_DESC_FILE)
    pr_path = os.path.join(directory, PR_DESC_FILE)
    for path in (commit_path, pr_path):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise CheckError(
                EXIT_DESCS,
                f"missing or empty: {path} (run the descs-only generation first)",
            )

    with open(commit_path) as f:
        commit_text = f.read()
    with open(pr_path) as f:
        pr_text = f.read()

    lines = commit_text.splitlines()
    title = lines[0].strip() if lines else ""
    if not title:
        raise CheckError(
            EXIT_DESCS,
            f"commit title (first line of {COMMIT_DESC_FILE}) is empty",
        )
    if len(title) > MAX_COMMIT_TITLE:
        raise CheckError(
            EXIT_DESCS,
            f"commit title is {len(title)} chars (max {MAX_COMMIT_TITLE})",
        )
    if len([line for line in lines[1:] if line.strip()]) < 2:
        raise CheckError(EXIT_DESCS, "commit message has no body beyond the title")

    if "## Summary" not in pr_text and "## Summary".lower() not in pr_text.lower():
        raise CheckError(EXIT_DESCS, "PR description has no '## Summary' section")

    for label, text in ((COMMIT_DESC_FILE, commit_text), (PR_DESC_FILE, pr_text)):
        hits = _scan_for_internal(text, extra_patterns)
        if hits:
            detail = "; ".join(f"{name}: {sample}" for name, sample in hits)
            raise CheckError(EXIT_DESCS, f"{label} leaks internal strings -> {detail}")

    digest = hashlib.sha256(pr_text.encode()).hexdigest()
    return f"descriptions OK; title='{title}'; pr_desc sha256={digest[:16]}"


def check_tree_scope(state, patch_path=None, allow=(), extra_patterns=()):
    repo_dir = state["repo_dir"]
    output_dir = state["output_dir"]
    patch_path = patch_path or _latest_runtime_patch(output_dir)
    if not patch_path or not os.path.isfile(patch_path):
        raise CheckError(EXIT_SCOPE, "cannot determine the reference patch")

    allowed = set(_patch_files(patch_path)) | set(allow)
    changed = set(_porcelain_files(repo_dir))
    unexpected = sorted(changed - allowed)
    if unexpected:
        raise CheckError(
            EXIT_SCOPE,
            "working tree touches files outside {} -> {}".format(
                os.path.basename(patch_path), ", ".join(unexpected)
            ),
        )

    diff = ""
    for args in (["diff"], ["diff", "--cached"]):
        diff += _git(repo_dir, *args)
    for path in _git(
        repo_dir, "ls-files", "--others", "--exclude-standard"
    ).splitlines():
        full = os.path.join(repo_dir, path)
        try:
            if os.path.isfile(full) and os.path.getsize(full) < 1_000_000:
                with open(full, errors="replace") as f:
                    diff += f.read()
        except OSError:
            continue

    hits = _scan_for_internal(diff, extra_patterns)
    if hits:
        detail = "; ".join(f"{name}: {sample}" for name, sample in hits)
        raise CheckError(
            EXIT_SCOPE, f"the change set leaks internal strings -> {detail}"
        )

    return f"scope OK: {len(changed)} file(s) changed, all inside the patch scope"


def check_clean_tree(state, expect_ahead=False, allow_rewritten=False):
    """Tree must be clean and on the state branch.

    expect_ahead=False -- HEAD must still be exactly state.head_sha (used before
                          the first push, so nothing can sneak in unreviewed).
    expect_ahead=True  -- HEAD must differ from what the fork holds, i.e. there is
                          something worth pushing (used by step 4).
    allow_rewritten    -- accept a HEAD that is NOT a descendant of the remote
                          head: that is a rebase, and only a leased force push
                          may publish it.
    """
    repo_dir = state["repo_dir"]
    dirty = _porcelain_files(repo_dir)
    if dirty:
        raise CheckError(
            EXIT_TREE, "working tree is not clean: {}".format(", ".join(dirty[:10]))
        )
    head = _git(repo_dir, "rev-parse", "HEAD").strip()
    expected = state.get("remote_head_sha") or state.get("head_sha") or ""
    branch = _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if state.get("local_branch") and branch != state["local_branch"]:
        raise CheckError(
            EXIT_TREE,
            "checked out '{}' but state expects '{}'".format(
                branch, state["local_branch"]
            ),
        )

    if expect_ahead:
        if not expected:
            raise CheckError(EXIT_TREE, "state.head_sha is unset -- step 2 has not run")
        if head.startswith(expected) or expected.startswith(head):
            raise CheckError(
                EXIT_TREE,
                f"HEAD {head[:12]} is still state.head_sha -- nothing new to push",
            )
        result = subprocess.run(
            ["git", "-C", repo_dir, "merge-base", "--is-ancestor", expected, head],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if not allow_rewritten:
                raise CheckError(
                    EXIT_TREE,
                    f"HEAD {head[:12]} is not a descendant of the published head {expected[:12]} -- the branch "
                    "was rebased/rewritten, so only a --force-with-lease push is safe "
                    "here",
                )
            if not _git_ok(repo_dir, "cat-file", "-e", f"{expected}^{{commit}}"):
                raise CheckError(
                    EXIT_REBASE,
                    f"published head {expected[:12]} no longer exists in this clone -- cannot lease a "
                    "force push; re-fetch the fork branch",
                )
            replayed = _git(repo_dir, "rev-list", "--count", "HEAD").strip()
            return f"tree clean on {branch} @ {head[:12]} (REWRITTEN history, {replayed} commit(s); lease {expected[:12]})"
        new_commits = _git(repo_dir, "rev-list", "--count", f"{expected}..HEAD").strip()
        return f"tree clean on {branch} @ {head[:12]} ({new_commits} new commit(s) above {expected[:12]})"

    if expected and not head.startswith(expected) and not expected.startswith(head):
        raise CheckError(
            EXIT_TREE,
            f"HEAD {head[:12]} does not match state.head_sha {expected[:12]} (did someone commit or "
            "rebase outside the flow?)",
        )
    return f"tree clean on {branch} @ {head[:12]}"


def _git_ok(repo_dir, *args):
    return (
        subprocess.run(
            ["git", "-C", repo_dir] + list(args), capture_output=True, text=True
        ).returncode
        == 0
    )


def check_rebase_absent(state):
    """No rebase may be paused when a step expects a normal worktree."""
    from llm.auto_code_gen.pr_flow import rebase

    repo_dir = state["repo_dir"]
    if rebase.in_progress(repo_dir):
        raise CheckError(
            EXIT_REBASE,
            f"a rebase is paused in {repo_dir} ({rebase.current_step(repo_dir)}) -- finish it with "
            "pr_flow_step_r1_rebase.sh --continue or --abort",
        )
    markers = rebase.scan_markers(repo_dir)
    if markers:
        raise CheckError(
            EXIT_REBASE, "conflict markers left in: {}".format(", ".join(markers[:10]))
        )
    return "no rebase in progress; no conflict markers"


def check_rebase_paused(state):
    """The conflict-fix step requires a paused rebase with real conflicts."""
    from llm.auto_code_gen.pr_flow import rebase

    repo_dir = state["repo_dir"]
    if not rebase.in_progress(repo_dir):
        raise CheckError(
            EXIT_REBASE,
            f"no rebase is paused in {repo_dir} -- start it with "
            "pr_flow_step_r1_rebase.sh",
        )
    files = rebase.conflicted_files(repo_dir)
    if not files:
        raise CheckError(
            EXIT_REBASE,
            "rebase is paused but no conflicted file is reported -- "
            "run pr_flow_step_r1_rebase.sh --status",
        )
    return "paused with {} conflict(s): {}".format(len(files), ", ".join(files[:10]))


def check_rebase_resume(state):
    """'rebase --continue' needs a paused rebase, with or without conflicts left.

    Nothing unmerged is legitimate here: this script stages the resolutions
    itself, so a re-run finds them staged already. Conflict markers left in a
    tracked file are never legitimate.
    """
    from llm.auto_code_gen.pr_flow import rebase

    repo_dir = state["repo_dir"]
    if not rebase.in_progress(repo_dir):
        raise CheckError(
            EXIT_REBASE,
            f"no rebase is paused in {repo_dir} -- nothing to continue; start it "
            "with pr_flow_step_r1_rebase.sh",
        )
    markers = rebase.scan_markers(repo_dir)
    if markers:
        raise CheckError(
            EXIT_REBASE, "conflict markers remain in {}".format(", ".join(markers[:10]))
        )
    unmerged = rebase.unmerged_files(repo_dir)
    if unmerged:
        return "paused, {} conflict(s) ready to stage: {}".format(
            len(unmerged), ", ".join(unmerged[:10])
        )
    return "paused with nothing unmerged -- rebase --continue will replay the rest"


def check_rebase_resolved(state, resolution_path):
    """Conflicts are resolved when the worktree is clean and every file is covered.

    Index entries stay unmerged until the human script runs 'git add', so an
    unmerged index is expected here and is not a failure -- what must be gone is
    the markers, and every unmerged file must appear in the report so a conflict
    cannot be quietly skipped.
    """
    from llm.auto_code_gen.pr_flow import rebase

    repo_dir = state["repo_dir"]
    summary = rebase.parse_resolution(resolution_path)
    markers = rebase.scan_markers(repo_dir)
    unmerged = rebase.unmerged_files(repo_dir)
    problems = []
    if not summary["pass"]:
        problems.append(summary["reason"] or "resolution report missing")
    if markers:
        problems.append("conflict markers remain in {}".format(", ".join(markers[:10])))
    uncovered = sorted(set(unmerged) - set(summary["files"]))
    if uncovered:
        problems.append(
            "not covered by the resolution report: {}".format(", ".join(uncovered[:10]))
        )
    if problems:
        raise CheckError(EXIT_REBASE, "; ".join(problems))
    return (
        "all {} conflict(s) resolved, no markers left; {} file(s) awaiting "
        "'git add' by the human step".format(summary["total"], len(unmerged))
    )


def check_rebase_plan(state):
    """Refuse to rebase onto something we do not have, or onto the current base."""
    from llm.auto_code_gen.pr_flow import rebase

    repo_dir = state["repo_dir"]
    check_rebase_absent(state)
    data = rebase.plan(state, repo_dir)
    if not data.get("target_sha"):
        raise CheckError(
            EXIT_REMOTE,
            "rebase target '{}' is not present locally -- fetch it first "
            "(that fetch is a network operation the script does for you)".format(
                data.get("target")
            ),
        )
    if data.get("target_sha", "")[:12] == (data.get("base_commit") or "")[:12]:
        raise CheckError(
            EXIT_REBASE,
            "target {} is already the PR base -- nothing to rebase".format(
                data.get("target")
            ),
        )
    if not data.get("pr_commits"):
        raise CheckError(
            EXIT_TREE,
            "no commits above base {} -- did step 1 commit the patch?".format(
                (data.get("base_commit") or "")[:12]
            ),
        )
    return rebase.render_plan(data)


def check_verdict(state, require_head=True):
    path = gates.latest_verdict(state["output_dir"])
    verdict = gates.read_verdict(path)
    if not verdict:
        raise CheckError(
            EXIT_VERDICT, "no verify_verdict.json found -- nothing was verified"
        )
    if verdict.get("overall") != "PASS":
        raise CheckError(
            EXIT_VERDICT,
            "latest verdict is {} ({})".format(verdict.get("overall"), path),
        )
    for name in ("perf", "lm_eval", "closure"):
        if not (verdict.get(name) or {}).get("pass"):
            raise CheckError(
                EXIT_VERDICT,
                f"gate '{name}' did not pass: {json.dumps(verdict.get(name), sort_keys=True)}",
            )
    if require_head:
        head = _git(state["repo_dir"], "rev-parse", "HEAD").strip()
        recorded = verdict.get("pr_head_sha") or ""
        if recorded and not head.startswith(recorded) and not recorded.startswith(head):
            # The harness measures the worktree; the human commits those bytes
            # afterwards, so a new commit is expected. What must not change is the
            # content -- prove that directly instead of trusting commit identity.
            matched, detail = gates.measured_content_matches_head(state, path)
            if matched is None:
                raise CheckError(
                    EXIT_VERDICT,
                    f"verdict was measured for {recorded[:12]} but HEAD is {head[:12]}, and nothing recorded "
                    f"what was measured ({detail}): re-verify",
                )
            if not matched:
                raise CheckError(
                    EXIT_VERDICT,
                    f"verdict was measured for {recorded[:12]} but HEAD is {head[:12]} and {detail} -- the tree "
                    "being pushed is not the tree that was measured",
                )
            return (
                "verdict PASS (iteration V{}): HEAD {} carries exactly the "
                "measured content ({})".format(
                    verdict.get("iteration"), head[:12], detail
                )
            )
    return "verdict PASS (iteration V{}) at {}".format(verdict.get("iteration"), path)


def check_inbox(inbox_dir):
    from llm.auto_code_gen.pr_flow.review_comments import PENDING_COMMENTS_FILE

    path = os.path.join(inbox_dir, PENDING_COMMENTS_FILE)
    if not os.path.isfile(path):
        raise CheckError(
            EXIT_NOTHING,
            f"no {PENDING_COMMENTS_FILE} in {inbox_dir} (run the fetch script)",
        )
    with open(path) as f:
        payload = json.load(f)
    count = len(payload.get("pending") or [])
    if not count:
        raise CheckError(EXIT_NOTHING, f"no pending review comments in {inbox_dir}")
    return f"{count} pending review comment(s) in {inbox_dir}"


def check_pre_commit(state, files=()):
    """Run the upstream repo's own pre-commit hooks on the changed files.

    This is the same suite the PR's CI runs, so a green run here means the
    pre-commit job on GitHub will pass. Hooks that auto-fix (ruff-check
    --fix, ruff-format, typos --force-exclude, ...) rewrite the files in
    place; a hook failure without a fixer exits EXIT_LINT so the operator
    can repair the source and re-run. --no-verify is never suggested.

    The files are passed to `pre-commit run --files`, so hooks that do not
    match them are skipped; `pre-commit run` also skips hooks whose stage
    does not include the default `pre-commit` stage (e.g. manual-only mypy
    versions), which mirrors what a local `git commit` would run.
    """
    repo_dir = state["repo_dir"]
    if not os.path.isfile(os.path.join(repo_dir, ".pre-commit-config.yaml")):
        raise CheckError(EXIT_LINT, f"{repo_dir} has no .pre-commit-config.yaml")

    changed = list(files) or _porcelain_files(repo_dir)
    if not changed:
        return "pre-commit OK: nothing to lint (clean tree)"

    cmd = ["pre-commit", "run", "--files"] + changed
    try:
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise CheckError(
            EXIT_LINT, "pre-commit is not installed (pip install pre-commit)"
        ) from exc

    output = (proc.stdout or "") + (proc.stderr or "")
    print(output.rstrip())
    if proc.returncode != 0:
        raise CheckError(
            EXIT_LINT,
            "pre-commit failed (see output above). If a hook "
            "rewrote files, inspect the diff and re-run this "
            "check; if it only reports errors, fix the source "
            "and re-run.",
        )
    fixed = [
        line for line in output.splitlines() if "reformatted" in line or "Fixed" in line
    ]
    summary = f"pre-commit OK on {len(changed)} file(s)"
    if fixed:
        summary += " -- hooks rewrote: {}".format(
            ", ".join(sorted({line.strip() for line in fixed}))[:200]
        )
    return summary


def check_gh_auth():
    try:
        result = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise CheckError(EXIT_AUTH, "gh CLI not installed") from exc
    if result.returncode != 0:
        raise CheckError(
            EXIT_AUTH,
            "gh is not authenticated -- run:  gh auth login "
            f"(git protocol: ssh). Detail: {(result.stderr or result.stdout).strip().splitlines()[:1]}",
        )
    return "gh authenticated"


def check_dco(state):
    """Every commit the PR will carry must be DCO-signed before pushing.

    The DCO bot on the upstream repo rejects a PR when ANY commit lacks a
    ``Signed-off-by:`` trailer matching its author (name and email). The
    pipeline's own gates (verdict, clean-tree, fingerprints) never look at
    trailers, so without this check a signoff-less commit only surfaces as a
    red DCO status AFTER publishing -- fixable only by rewriting history and
    force-pushing again. Checking here turns that into a local refusal with
    the exact repair command.
    """
    repo_dir = state["repo_dir"]
    base = state.get("base_commit") or ""
    if not base:
        raise CheckError(EXIT_STATE, "state.base_commit is unset -- step 0 has not run")
    result = subprocess.run(
        [
            "git",
            "-C",
            repo_dir,
            "log",
            "--format=%H%x00%an%x00%ae%x00%B%x1e",
            "--reverse",
            f"{base}..HEAD",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CheckError(
            EXIT_TREE,
            "could not list commits {}..HEAD: {}".format(
                base[:12], (result.stderr or "").strip()[:200]
            ),
        )
    commits = [c for c in result.stdout.split("\x1e") if c.strip()]
    if not commits:
        raise CheckError(EXIT_STATE, f"no commits above base {base[:12]} to sign")

    bad = []
    for raw in commits:
        fields = raw.strip("\n").split("\x00", 3)
        if len(fields) != 4:
            continue  # malformed record; git's own format guarantees 4 fields
        sha, name, email, body = fields
        expected = f"Signed-off-by: {name} <{email}>"
        if expected not in body:
            bad.append((sha[:12], f"{name} <{email}>"))
    if bad:
        detail = "; ".join(
            f"{sha} needs 'Signed-off-by: {who}'" for sha, who in bad[:5]
        )
        raise CheckError(
            EXIT_DCO,
            f"{len(bad)} commit(s) missing a matching DCO signoff: {detail}. Repair with:  "
            f"git -C {repo_dir} rebase --signoff {base[:12]}  (rewrites the commit SHAs; "
            "content is unchanged, so re-publish with --force-with-lease)",
        )
    return f"DCO OK: {len(commits)} commit(s) above {base[:12]} all carry Signed-off-by"


def parse_args():
    parser = argparse.ArgumentParser(description="Local PR-flow guardrails")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_state(name):
        p = sub.add_parser(name)
        p.add_argument("--state", required=True)
        return p

    p = with_state("remotes")
    p = with_state("clean-tree")
    p.add_argument(
        "--expect-ahead",
        action="store_true",
        help="require HEAD to be a strict descendant of state.head_sha",
    )
    p.add_argument(
        "--allow-rewritten",
        action="store_true",
        help="with --expect-ahead, accept a rebased/rewritten HEAD (a leased "
        "force push is the only safe way to publish it)",
    )

    for name in ("rebase-absent", "rebase-paused", "rebase-resume", "rebase-plan"):
        with_state(name)

    p = with_state("rebase-resolved")
    p.add_argument("--resolution", required=True)
    p = with_state("descs")
    p = with_state("verdict")
    p.add_argument("--no-head-check", action="store_true")

    p = with_state("branch-name")
    p.add_argument("--name", required=True)
    p.add_argument("--allow-prefixless", action="store_true")

    p = with_state("tree-scope")
    p.add_argument("--patch", default=None)
    p.add_argument("--allow", action="append", default=[])

    p = with_state("pre-commit")
    p.add_argument(
        "--file",
        action="append",
        dest="files",
        default=[],
        help="restrict the lint to these files (default: changed files)",
    )

    p = sub.add_parser("inbox")
    p.add_argument("--inbox", required=True)

    with_state("dco")
    sub.add_parser("gh-auth")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "branch-name":
            state = _load_state(args)
            print(check_branch_name(state, args.name, args.allow_prefixless))
        elif args.command == "remotes":
            print(check_remotes(_load_state(args)))
        elif args.command == "descs":
            state = _load_state(args)
            print(check_descs(state, _state_scrub_patterns(state)))
        elif args.command == "tree-scope":
            state = _load_state(args)
            print(
                check_tree_scope(
                    state, args.patch, args.allow, _state_scrub_patterns(state)
                )
            )
        elif args.command == "pre-commit":
            print(check_pre_commit(_load_state(args), args.files))
        elif args.command == "clean-tree":
            state = _load_state(args)
            print(
                check_clean_tree(
                    state,
                    expect_ahead=args.expect_ahead,
                    allow_rewritten=args.allow_rewritten,
                )
            )
        elif args.command == "rebase-absent":
            print(check_rebase_absent(_load_state(args)))
        elif args.command == "rebase-paused":
            print(check_rebase_paused(_load_state(args)))
        elif args.command == "rebase-resume":
            print(check_rebase_resume(_load_state(args)))
        elif args.command == "rebase-plan":
            print(check_rebase_plan(_load_state(args)))
        elif args.command == "rebase-resolved":
            print(check_rebase_resolved(_load_state(args), args.resolution))
        elif args.command == "verdict":
            print(check_verdict(_load_state(args), require_head=not args.no_head_check))
        elif args.command == "inbox":
            print(check_inbox(args.inbox))
        elif args.command == "dco":
            print(check_dco(_load_state(args)))
        elif args.command == "gh-auth":
            print(check_gh_auth())
        else:
            return 2
    except CheckError as exc:
        print(f"REFUSED ({exc.code}): {exc}")
        return exc.code
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
