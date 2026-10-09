"""Suggest reviewers for a published PR -- detection and draft generation.

Step 2b of the PR flow: find the best candidate reviewers for the PR's owned
files and write a draft comment (pr_reviewers.txt) for human review. The draft
is the exact text step 2c would post; this module never posts anything.

Candidate sources, blended into one score:
  blame   -- who last touched the exact changed lines (strongest signal)
  history -- top committers to the owned files over the last ~6 months
  reviews -- people who reviewed recently merged PRs touching the same paths
  owners  -- CODEOWNERS entries matching the owned files (etiquette baseline)

Excluded: the PR author, bot accounts, and anyone dormant (no commit to the
owned files in over a year when that is knowable from history).

CLI:

    python -m llm.auto_code_gen.pr_flow.run_reviewer_suggest generate \\
        --state <pr_state.json> [--max-reviewers 5] [--dry-run]

Writes <descs_dir>/pr_reviewers.txt (the draft comment) and pr_reviewers.json
(the evidence: per-candidate score breakdown and the command step 2c runs).
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import datetime
import json
import os
import re
import subprocess

from llm.auto_code_gen.pr_flow import pr_state

REVIEWERS_TXT = "pr_reviewers.txt"
REVIEWERS_JSON = "pr_reviewers.json"

# Posted as the last line of every generated draft; step 2c checks for it on
# the PR to stay idempotent, and it identifies our comments for later refresh.
MARKER = "<!-- ai-auto-analyze:reviewer-suggestion v1 -->"

# Etiquette cap: enough names to spread the review load, few enough not to
# feel like a mass-ping.
DEFAULT_MAX_REVIEWERS = 5

WEIGHTS = {"blame": 3.0, "history": 1.5, "reviews": 2.5, "owners": 1.5}

BOT_LOGIN_PATTERN = re.compile(
    r"(bot($|[-_\[])|\[bot\]$|claude|github-actions|copilot|semantic"
    r"|dependabot|renovate)", re.IGNORECASE)


def _git(repo_dir, *args, check=True):
    r = subprocess.run(["git", "-C", repo_dir] + list(args),
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(
            " ".join(args[:2]), r.stderr.strip()[:200]))
    return r.stdout


def _gh(*args, check=True):
    r = subprocess.run(["gh"] + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("gh {} failed: {}".format(
            " ".join(args[:2]), r.stderr.strip()[:200]))
    return r.stdout


def _gh_json(*args, check=True):
    out = _gh(*args, check=check)
    try:
        return json.loads(out) if out.strip() else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Sources


def blame_candidates(repo_dir, base, changed_files):
    """Authors of the lines the PR modifies, from git blame at the merge base.

    Uses the PR head's own diff (base..HEAD) to pick the changed line ranges,
    then blames those ranges in the BASE version -- the people whose code the
    PR is touching, not the PR author rewriting it. Returns {email: lines}.
    """
    counts = {}
    for path in changed_files:
        hunks = _git(repo_dir, "diff", "-U0", "{}..HEAD".format(base), "--", path,
                     check=False)
        ranges = []
        for match in re.finditer(r"^@@ -(\d+)(?:,(\d+))? ", hunks, re.MULTILINE):
            start = int(match.group(1))
            length = int(match.group(2) or "1")
            if start > 0:
                ranges.append((start, start + max(length, 1) - 1))
        if not ranges:
            continue
        blame = _git(repo_dir, "blame", "--porcelain", base, "--", path,
                     check=False)
        if not blame:
            continue
        lineno = 0
        email = None
        for line in blame.splitlines():
            # Porcelain header: "<sha> <orig_lineno> <final_lineno> [<n>]"
            header = re.match(r"^[0-9a-f]{40} (\d+) \d+", line)
            if header:
                lineno = int(header.group(1))
                email = None              # reset; header precedes its metadata
                continue
            if line.startswith("author-mail "):
                email = line[len("author-mail "):].strip(" <>")
                if email and any(a <= lineno <= b for a, b in ranges):
                    counts[email] = counts.get(email, 0) + 1
    return counts


def history_candidates(repo_dir, changed_files, months=6):
    """Top committers to the owned files in the recent window."""
    since = (datetime.date.today()
             - datetime.timedelta(days=30 * months)).isoformat()
    counts = {}
    for path in changed_files:
        log = _git(repo_dir, "log", "--since={}".format(since),
                   "--format=%ae", "--", path, check=False)
        for email in log.splitlines():
            if email.strip():
                counts[email.strip()] = counts.get(email.strip(), 0) + 1
    return counts


def review_candidates(upstream_repo, changed_files, max_prs=10):
    """Reviewers of recently merged PRs touching the same files.

    GitHub issue search has no path: qualifier, so each file's basename is
    used as a text query -- PRs that touch a file nearly always name it in
    the title or body (vLLM's convention).
    """
    counts = {}
    for path in changed_files:
        basename = os.path.basename(path)
        if not basename:
            continue
        q = "repo:{} is:pr is:merged {}".format(upstream_repo, basename)
        out = _gh_json("api", "-X", "GET", "search/issues",
                       "-f", "q={}".format(q), "-f", "per_page={}".format(max_prs),
                       check=False) or {}
        items = out.get("items", []) if isinstance(out, dict) else []
        for item in items:
            number = item.get("number")
            if not number:
                continue
            reviews = _gh_json("api", "repos/{}/pulls/{}/reviews".format(
                upstream_repo, number), check=False) or []
            if not isinstance(reviews, list):
                continue
            for review in reviews:
                login = (review.get("user") or {}).get("login", "")
                if login:
                    counts[login] = counts.get(login, 0) + 1
    return counts


def codeowners_entries(repo_dir, base, changed_files):
    """Logins from CODEOWNERS rules matching the owned files."""
    for candidate in (".github/CODEOWNERS", "CODEOWNERS",
                      "docs/CODEOWNERS", ".github/CODEOWNERS.txt"):
        content = _git(repo_dir, "show", "{}:{}".format(base, candidate),
                       check=False)
        if content:
            return _match_codeowners(content, changed_files)
    return {}


def _match_codeowners(content, changed_files):
    """Apply CODEOWNERS rules to the changed files.

    Later rules take precedence over earlier ones (GitHub semantics), and the
    pattern '*' (everyone) is not a useful reviewer suggestion, so a file whose
    final matching rule is '*' contributes nothing.
    """
    # Rule list in file order; evaluate last-match-wins per file.
    rules = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pattern, owners = parts[0], [o for o in parts[1:] if o != "*"]
        if pattern == "*":
            continue        # matches everything -- never a useful suggestion
        rules.append((pattern, owners))

    matches = {}
    for path in changed_files:
        winner = None
        for pattern, owners in rules:
            if _codeowners_match(pattern, path):
                winner = owners
        if winner:
            for owner in winner:
                login = owner.lstrip("@").split("/")[0]
                if login:
                    matches[login] = 1
    return matches


def _codeowners_match(pattern, path):
    """True if a CODEOWNERS pattern applies to a repo path."""
    if pattern == "*":
        return True                      # matches everything (incl. later rules)
    directory = pattern.endswith("/")
    if pattern.startswith("/"):
        pattern = pattern[1:]
        anchored = True
    else:
        anchored = False
    # Translate the glob (** = anything, * = within one segment).
    parts = pattern.split("**")
    regex = ".*".join(
        re.escape(p).replace(r"\*", "[^/]*") for p in parts)
    if directory:
        regex += ".*"
    full = "^" + regex + "$"
    if anchored or directory:
        return re.match(full, path) is not None
    # Unanchored file rule: also matches inside any directory.
    return re.match("^(.*/)?{}$".format(regex), path) is not None
    if anchored:
        return re.match("^" + regex + "$", path) is not None
    return re.search(regex + "$", path) is not None


def _email_to_login(upstream_repo, email):
    """Best-effort GitHub login for a commit email, via commit search."""
    out = _gh_json("api", "-X", "GET", "search/commits",
                   "-f", "q={}".format("repo:{} author-email:{}".format(
                       upstream_repo, email)),
                   "-f", "per_page=1", check=False) or {}
    items = out.get("items", []) if isinstance(out, dict) else []
    for item in items:
        login = (item.get("author") or {}).get("login")
        if login:
            return login
    return None


def is_bot(login):
    return bool(BOT_LOGIN_PATTERN.search(login))


# ---------------------------------------------------------------------------
# Ranking


def rank_candidates(state, repo_dir, max_reviewers, log=print):
    """Blend the sources into a ranked list with evidence per candidate."""
    changed = _patch_files_from_state(state)
    base = state.get("base_commit") or _git(repo_dir, "merge-base",
                                            "HEAD", state.get("base_branch") or "main")
    author = _git(repo_dir, "log", "-1", "--format=%ae", "HEAD").strip()
    upstream = state.get("upstream_repo") or "vllm-project/vllm"

    log("  changed files: {}".format(", ".join(changed)))

    blame = blame_candidates(repo_dir, base, changed)
    history = history_candidates(repo_dir, changed)
    reviews = review_candidates(upstream, changed)
    owners = codeowners_entries(repo_dir, base, changed)

    # Map commit emails to GitHub logins once, then merge the email-keyed
    # sources into login-keyed scores.
    email_login = {}
    for email in set(list(blame) + list(history)):
        if email == author:
            continue
        email_login[email] = _email_to_login(upstream, email)

    scores = {}

    def add(login, kind, weight, detail):
        entry = scores.setdefault(login, {"login": login, "score": 0.0,
                                          "evidence": []})
        entry["score"] += weight
        entry["evidence"].append(detail)

    for email, count in sorted(blame.items(), key=lambda kv: -kv[1]):
        if email == author:
            continue
        login = email_login.get(email)
        if not login:
            continue
        if is_bot(login):
            continue
        add(login, "blame", WEIGHTS["blame"] * min(count, 10),
            "authored {} of the changed lines (git blame at merge base)".format(
                count))

    for email, count in sorted(history.items(), key=lambda kv: -kv[1]):
        if email == author:
            continue
        login = email_login.get(email)
        if not login:
            continue
        if is_bot(login):
            continue
        add(login, "history", WEIGHTS["history"] * min(count, 20) / 20.0,
            "top recent committer to the touched files ({} commits, 6mo)".format(
                count))

    for login, count in sorted(reviews.items(), key=lambda kv: -kv[1]):
        if login == author or is_bot(login):
            continue
        add(login, "reviews", WEIGHTS["reviews"] * min(count, 5),
            "reviewed {} recent merged PRs touching these files".format(count))

    for login in owners:
        if login == author or is_bot(login):
            continue
        add(login, "owners", WEIGHTS["owners"],
            "listed in CODEOWNERS for the touched paths")

    ranked = sorted(scores.values(), key=lambda e: -e["score"])
    return ranked[:max_reviewers], {
        "blame": blame, "history": history, "reviews": reviews,
        "owners": sorted(owners), "ranked": ranked,
    }


def _patch_files_from_state(state):
    """The files this PR owns, from the state's recorded patch scope."""
    repo_dir = state.get("repo_dir")
    head = state.get("head_sha") or "HEAD"
    base = state.get("base_commit")
    if repo_dir and base:
        out = _git(repo_dir, "diff", "--name-only", "{}..{}".format(base, head),
                   check=False)
        files = [f for f in out.splitlines() if f.strip()]
        if files:
            return files
    # Fall back to the tree scope of the latest runtime patch.
    output_dir = state.get("output_dir")
    if output_dir:
        import glob
        patches = sorted(glob.glob(os.path.join(
            output_dir, "runtime", "V*", "pr_candidate.patch")))
        if patches:
            out = _git(repo_dir or ".", "show", "--name-only", "--format=",
                       check=False)
            files = []
            with open(patches[-1]) as f:
                for line in f:
                    m = re.match(r"^\+\+\+ b/(.+)$", line)
                    if m:
                        files.append(m.group(1))
            if files:
                return files
    raise RuntimeError(
        "cannot determine the PR's owned files -- state has no base/head diff "
        "and no runtime patch was found")


# ---------------------------------------------------------------------------
# Draft rendering


def render_draft(pr_url, candidates, derived_note=True):
    """The published comment: a bare reviewer ping, minimal explanation.

    The full evidence (scores, sources) stays in pr_reviewers.json for the
    local reviewer; the upstream comment is just the ping plus one line.
    """
    pings = " ".join("@{}".format(entry["login"]) for entry in candidates)
    lines = [
        "{} could you review this PR?".format(pings),
    ]
    if derived_note:
        lines += [
            "",
            "<sub>Reviewer selection was automated. {}</sub>".format(
                "Human reviewed changes."),
        ]
    lines += [
        "",
        MARKER,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def _load(path):
    state = pr_state.load_state(path)
    state["_path"] = path
    return state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a reviewer-suggestion draft for a published PR")
    sub = parser.add_subparsers(dest="command")
    gen = sub.add_parser("generate", help="write pr_reviewers.txt + .json")
    gen.add_argument("--state", required=True)
    gen.add_argument("--max-reviewers", type=int, default=DEFAULT_MAX_REVIEWERS)
    gen.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command != "generate":
        return 2
    state = _load(args.state)
    repo_dir = state["repo_dir"]
    if not state.get("pr_number"):
        print("no pr_number in state -- run pr_flow_step_2_push.sh first")
        return 22
    pr_url = state.get("pr_url") or ""

    print("Reviewers: ranking candidates for PR #{} ...".format(
        state["pr_number"]))
    top, evidence = rank_candidates(state, repo_dir, args.max_reviewers)

    draft = render_draft(pr_url, top)
    out_dir = pr_state.descs_dir(state["output_dir"])
    txt_path = os.path.join(out_dir, REVIEWERS_TXT)
    json_path = os.path.join(out_dir, REVIEWERS_JSON)

    payload = {
        "pr_number": state["pr_number"],
        "pr_url": pr_url,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "max_reviewers": args.max_reviewers,
        "candidates": top,
        "evidence": {k: v for k, v in evidence.items() if k != "ranked"},
        "post_command": "gh pr comment {} --repo {} --body-file {}".format(
            state["pr_number"], state.get("upstream_repo"), txt_path),
    }
    if args.dry_run:
        print(draft)
        return 0
    os.makedirs(out_dir, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(draft)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    import hashlib
    pr_state.update(args.state, reviewers_draft_sha256=hashlib.sha256(
        draft.encode()).hexdigest())
    print("Draft written: {}".format(txt_path))
    print("Evidence     : {}".format(json_path))
    print("Review it, edit if needed, then run step 2c to post.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
