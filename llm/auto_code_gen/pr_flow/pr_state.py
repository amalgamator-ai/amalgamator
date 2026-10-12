# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Durable state for one PR: submission + review-fix loop.

One JSON document per PR, at <output_dir>/pr_p3/pr_state.json, is the single
source of truth shared by the human-run bash scripts and the AI-side runtime.
It exists so that every script can refuse to act on a stale assumption, a
review round can be resumed after an interruption, and the audit trail of what
was pushed and measured is never inferred from git alone.

CLI (used by the scripts, never by the agent):

    python -m llm.auto_code_gen.pr_flow.pr_state path --config <code_gen.json>
    python -m llm.auto_code_gen.pr_flow.pr_state init --config <code_gen.json> \\
        --repo-dir <clone> --fork-url git@github.com:you/vllm.git
    python -m llm.auto_code_gen.pr_flow.pr_state show --state <pr_state.json>
    python -m llm.auto_code_gen.pr_flow.pr_state get  --state <p> --field head_sha
    python -m llm.auto_code_gen.pr_flow.pr_state set  --state <p> --set head_sha=abc
    python -m llm.auto_code_gen.pr_flow.pr_state gate --state <p> --json '{"iteration":3}'
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import json
import os
import re
import tempfile
import time

STATE_VERSION = 1
PR_FLOW_DIR = "pr_p3"
STATE_FILE = "pr_state.json"
INBOX_DIR = "inbox"
PR_DESCS_DIR = "pr-generate"  # must match code_gen_prompts.PR_COMMANDS_DIR
COMMENTS_DIR = "comments"
COMMIT_DESC_FILE = "commit_desc.txt"
PR_DESC_FILE = "pr_desc.txt"

DEFAULT_UPSTREAM_REPO = "vllm-project/vllm"
DEFAULT_UPSTREAM_BRANCH = "main"
DEFAULT_BRANCH_PREFIX = "perf/"

SCALAR_FIELDS = (
    "config_path",
    "output_dir",
    "repo_dir",
    "logs_dir",
    "upstream_repo",
    "upstream_remote",
    "upstream_branch",
    "fork_repo",
    "fork_owner",
    "fork_remote",
    "fork_url",
    "local_branch",
    "remote_branch",
    "branch_prefix",
    "base_branch",
    "base_commit",
    "head_sha",
    # head_sha is what the worktree holds; remote_head_sha is what the fork
    # holds. They diverge on purpose during a rebase, which rewrites history.
    "remote_head_sha",
    "pre_rebase_head",
    "rebase_target",
    "pr_number",
    "pr_url",
    "descs_sha256",
    # Reviewer-suggestion comment (steps 2b/2c): the draft's content hash at
    # generation time and, once posted, the comment's URL/id.
    "reviewers_draft_sha256",
    "reviewers_comment_url",
    "reviewers_comment_id",
)
INT_FIELDS = ("round", "iteration")
STRUCTURED_FIELDS = ("cursor", "baselines", "gates")


def pr_flow_dir(output_dir):
    return os.path.join(output_dir, PR_FLOW_DIR)


def state_path(output_dir):
    return os.path.join(pr_flow_dir(output_dir), STATE_FILE)


def inbox_dir(output_dir, round_no):
    return os.path.join(pr_flow_dir(output_dir), INBOX_DIR, f"round_{round_no}")


def descs_dir(output_dir):
    """Where commit_desc.txt / pr_desc.txt live.

    This is the directory the existing generator writes
    (llm.auto_code_gen.run_generate_pr_commands --descs-only), so the flow reads the
    same files it validates instead of maintaining a copy that can drift.
    Round comment bodies go next to the state file, in comments/.
    """
    return os.path.join(output_dir, PR_DESCS_DIR)


def comments_dir(output_dir):
    return os.path.join(pr_flow_dir(output_dir), COMMENTS_DIR)


GIT_PATH_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")


def parse_git_url(url):
    """Return 'owner/repo' for a remote URL, or None when it is not a usable one.

    Accepts the two forms git actually stores -- scp-style SSH
    (git@github.com:owner/repo.git) and schemed URLs
    (https://host/owner/repo.git, ssh://git@host/owner/repo.git) -- plus the
    bare 'owner/repo' short form. Local paths (file://..., /srv/...) are not
    remote repositories and deliberately return None.
    """
    if not url:
        return None
    url = url.strip()
    if url.startswith("file://") or url.startswith("/"):
        return None

    path = None
    m = re.match(r"^(?:https?|git|ssh)://(?:[^@/]+@)?[^/]+/(.+)$", url)
    if m:
        path = m.group(1)
    else:
        m = re.match(r"^(?:[^@/:]+@)?[A-Za-z0-9._-]+:/{0,2}(.+)$", url)
        if m:
            path = m.group(1)
        elif "/" in url and " " not in url:
            path = url

    if path is None:
        return None
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return path if GIT_PATH_RE.match(path) else None


def url_owner(url):
    full = parse_git_url(url)
    if not full or "/" not in full:
        return None
    return full.split("/")[0]


def new_state(
    output_dir,
    repo_dir,
    fork_url,
    fork_owner,
    upstream_repo=DEFAULT_UPSTREAM_REPO,
    upstream_branch=DEFAULT_UPSTREAM_BRANCH,
    upstream_remote="upstream",
    fork_remote="fork",
    local_branch="",
    base_branch="",
    base_commit="",
    config_path="",
    branch_prefix=DEFAULT_BRANCH_PREFIX,
):
    """Build a state document. Raises ValueError on an unusable configuration."""
    if fork_url.startswith("file://") or fork_url.startswith("/"):
        raise ValueError(
            f"fork URL is a local path ({fork_url}); a PR needs a real remote"
        )
    fork_repo = parse_git_url(fork_url)
    if not fork_repo:
        raise ValueError(f"cannot parse fork URL: {fork_url}")
    if fork_repo == upstream_repo:
        raise ValueError(
            f"fork URL points at the upstream repo ({fork_repo}); push to a fork"
        )
    if not fork_owner:
        raise ValueError("fork_owner is required (GitHub login that owns the fork)")
    if not re.match(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$", fork_owner):
        raise ValueError(f"fork_owner does not look like a GitHub login: {fork_owner}")

    return {
        "version": STATE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": os.path.abspath(config_path) if config_path else "",
        "output_dir": os.path.abspath(output_dir),
        "repo_dir": os.path.abspath(repo_dir),
        "logs_dir": os.path.join(pr_flow_dir(os.path.abspath(output_dir)), "logs"),
        "upstream_repo": upstream_repo,
        "upstream_remote": upstream_remote,
        "upstream_branch": upstream_branch,
        "fork_repo": fork_repo,
        "fork_owner": fork_owner,
        "fork_remote": fork_remote,
        "fork_url": fork_url,
        "local_branch": local_branch,
        "remote_branch": "",
        "branch_prefix": branch_prefix,
        "base_branch": base_branch or upstream_branch,
        "base_commit": base_commit,
        "head_sha": "",
        "remote_head_sha": "",
        "pre_rebase_head": "",
        "rebase_target": f"{upstream_remote}/{upstream_branch}",
        "pr_number": None,
        "pr_url": "",
        "descs_sha256": "",
        "cursor": {"max_comment_id": 0, "last_seen_at": ""},
        "round": 0,
        "iteration": None,
        "baselines": {"bench_off_base_commit": "", "lm_eval_off_base_commit": ""},
        "gates": [],
    }


def load_state(path):
    with open(path) as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"state file is not a JSON object: {path}")
    version = state.get("version", 0)
    if version > STATE_VERSION:
        raise ValueError(
            f"state file version {version} is newer than this code supports ({STATE_VERSION})"
        )
    return state


def save_state(path, state):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".pr_state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def update(path, **values):
    """Merge values into the state file. Unknown keys are rejected."""
    state = load_state(path)
    for key, value in values.items():
        if (
            key not in SCALAR_FIELDS
            and key not in INT_FIELDS
            and key not in (STRUCTURED_FIELDS)
        ):
            raise ValueError(f"unknown state field: {key}")
        if key in INT_FIELDS and value is not None:
            value = int(value)
        elif key in STRUCTURED_FIELDS and isinstance(value, str):
            # Structured fields are dicts/lists; from the CLI they arrive as a
            # JSON string, and anything that is not JSON would silently break
            # every later reader -- so refuse rather than store it.
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"field {key} must be JSON ({exc}): {value}") from exc
            if not isinstance(value, (dict, list)):
                raise ValueError(f"field {key} must be a JSON object or array")
        state[key] = value
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_state(path, state)
    return state


def record_gate(path, gate):
    """Append one gate result to state['gates'] (monotonic audit trail)."""
    state = load_state(path)
    gate = dict(gate)
    gate["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state.setdefault("gates", []).append(gate)
    save_state(path, state)
    return state


def _load_config_quietly(config_path):
    """Load a code-gen config with its stdout noise swallowed.

    CodeGenConfig.from_json prints progress ("Loading trace parameters ...").
    The scripts capture the stdout of this CLI to obtain paths, so that noise
    would end up inside a path variable.
    """
    import contextlib
    import io

    from llm.auto_code_gen.code_gen_configs import CodeGenConfig

    with contextlib.redirect_stdout(io.StringIO()):
        return CodeGenConfig.from_json(config_path)


def state_path_for_config(config_path):
    """Resolve the state path from a code-gen config without loading use cases."""
    config = _load_config_quietly(config_path)
    return state_path(config.output_dir)


def _require_state(args):
    if not args.state or not os.path.isfile(args.state):
        raise SystemExit(
            f"state file not found: {args.state} (run 'pr_state init' first)"
        )
    return args.state


def parse_args():
    parser = argparse.ArgumentParser(description="Read/write pr_state.json")
    sub = parser.add_subparsers(dest="command", required=True)

    p_path = sub.add_parser("path", help="print the state path for a config")
    p_path.add_argument("--config", required=True)

    p_init = sub.add_parser("init", help="create pr_state.json")
    p_init.add_argument("--config", required=True)
    p_init.add_argument("--repo-dir", required=True)
    p_init.add_argument("--fork-url", required=True)
    p_init.add_argument("--fork-owner", default=None)
    p_init.add_argument("--upstream-repo", default=DEFAULT_UPSTREAM_REPO)
    p_init.add_argument("--upstream-branch", default=DEFAULT_UPSTREAM_BRANCH)
    p_init.add_argument("--upstream-remote", default="upstream")
    p_init.add_argument("--fork-remote", default="fork")
    p_init.add_argument("--local-branch", default="")
    p_init.add_argument("--base-branch", default="")
    p_init.add_argument("--base-commit", default="")
    p_init.add_argument("--branch-prefix", default=DEFAULT_BRANCH_PREFIX)
    p_init.add_argument("--force", action="store_true")

    p_show = sub.add_parser("show")
    p_show.add_argument("--state", required=True)

    p_get = sub.add_parser("get", help="print one field (empty when unset)")
    p_get.add_argument("--state", required=True)
    p_get.add_argument("--field", required=True)

    p_set = sub.add_parser("set", help="update fields: --set key=value")
    p_set.add_argument("--state", required=True)
    p_set.add_argument("--set", action="append", default=[], dest="assignments")

    p_gate = sub.add_parser("gate", help="append a gate result record")
    p_gate.add_argument("--state", required=True)
    p_gate.add_argument("--json", required=True, help="gate record as JSON")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "path":
        print(state_path_for_config(args.config))
        return 0

    if args.command == "init":
        config = _load_config_quietly(args.config)
        path = state_path(config.output_dir)
        if os.path.isfile(path) and not args.force:
            raise SystemExit(f"state exists: {path} (use --force to overwrite)")
        fork_owner = args.fork_owner or url_owner(args.fork_url) or ""
        state = new_state(
            output_dir=config.output_dir,
            repo_dir=args.repo_dir,
            fork_url=args.fork_url,
            fork_owner=fork_owner,
            upstream_repo=args.upstream_repo,
            upstream_branch=args.upstream_branch,
            upstream_remote=args.upstream_remote,
            fork_remote=args.fork_remote,
            local_branch=args.local_branch,
            base_branch=args.base_branch,
            base_commit=args.base_commit,
            config_path=os.path.abspath(args.config),
            branch_prefix=args.branch_prefix,
        )
        os.makedirs(descs_dir(config.output_dir), exist_ok=True)
        os.makedirs(state["logs_dir"], exist_ok=True)
        save_state(path, state)
        print(path)
        return 0

    path = _require_state(args)

    if args.command == "show":
        print(json.dumps(load_state(path), indent=2, sort_keys=True))
    elif args.command == "get":
        value = load_state(path).get(args.field, "")
        print("" if value is None else value)
    elif args.command == "set":
        values = {}
        for item in args.assignments:
            if "=" not in item:
                raise SystemExit(f"expected key=value, got: {item}")
            key, value = item.split("=", 1)
            values[key.strip()] = value.strip()
        update(path, **values)
        print("updated: {}".format(", ".join(sorted(values)) or "nothing"))
    elif args.command == "gate":
        record_gate(path, json.loads(args.json))
        print("gate recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
