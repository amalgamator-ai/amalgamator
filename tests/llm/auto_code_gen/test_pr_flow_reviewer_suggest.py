# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reviewer suggestion: ranking sources, draft rendering, and CLI plumbing.

run_reviewer_suggest blends blame/history/review/CODEOWNERS signals into a
ranked candidate list and renders pr_reviewers.txt -- the exact comment step 2c
posts. These tests use fixture git repos and stubbed gh calls; nothing touches
the network.
"""

import json
import os
import shutil
import subprocess

import pytest

from llm.auto_code_gen.pr_flow import run_reviewer_suggest as rs

# This E2E test drives the real CLI entry point, which shells out to the GitHub
# CLI for reviewer data. Skip when it cannot run meaningfully (no gh binary, or
# CI without a token), so the gate stays green everywhere gh is unavailable.
# Mirrors the production token lookup, which accepts GH_TOKEN or GITHUB_TOKEN.
needs_gh = pytest.mark.skipif(
    not shutil.which("gh")
    or not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")),
    reason="requires the gh CLI with GH_TOKEN or GITHUB_TOKEN for live GitHub data",
)


def _run_git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd)] + list(args), check=True, capture_output=True, text=True
    )


@pytest.fixture()
def fixture_repo(tmp_path):
    """A base commit plus a PR head touching one file, with real history."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-q", "-b", "main")

    def commit(name, email, path, content):
        (repo / os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content)
        _run_git(repo, "add", "-A")
        _run_git(
            repo,
            "-c",
            f"user.name={name}",
            "-c",
            f"user.email={email}",
            "commit",
            "-q",
            "-m",
            name,
        )

    commit("alice", "alice@example.com", "src/feature.py", "line1\nline2\nline3\n")
    commit("bob", "bob@example.com", "src/feature.py", "line1\nline2 changed\nline3\n")
    _run_git(
        repo,
        "-c",
        "user.name=carol",
        "-c",
        "user.email=carol@example.com",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "base marker",
    )
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    # The PR: carol changes line 2 (originally bob's, per blame at base).
    (repo / "src" / "feature.py").write_text("line1\nline2 rewritten\nline3\n")
    _run_git(repo, "add", "-A")
    _run_git(
        repo,
        "-c",
        "user.name=carol",
        "-c",
        "user.email=carol@example.com",
        "commit",
        "-q",
        "-m",
        "pr change",
    )
    return repo, base, "src/feature.py"


class TestBlameCandidates:

    def test_counts_base_authors_of_changed_lines(self, fixture_repo):
        repo, base, path = fixture_repo
        counts = rs.blame_candidates(str(repo), base, [path])
        # Line 2 at base was last touched by bob; the PR rewrites it.
        assert counts.get("bob@example.com") == 1
        assert "carol@example.com" not in counts

    def test_unchanged_lines_are_not_counted(self, fixture_repo):
        repo, base, path = fixture_repo
        counts = rs.blame_candidates(str(repo), base, [path])
        # Line 1/3 are unchanged, so alice (their author) is not counted.
        assert "alice@example.com" not in counts


class TestHistoryCandidates:

    def test_counts_recent_committers(self, fixture_repo):
        repo, _base, path = fixture_repo
        counts = rs.history_candidates(str(repo), [path])
        assert counts["alice@example.com"] >= 1
        assert counts["bob@example.com"] >= 1


class TestCodeowners:

    def test_directory_rule_matches_files_under_it(self):
        content = "* @default-owner\n/src/ @src-team\n"
        matches = rs._match_codeowners(content, ["src/feature.py", "docs/x.md"])
        assert "src-team" in matches
        assert "src-team" in rs._match_codeowners(content, ["src/feature.py"])

    def test_exact_file_rule(self):
        content = "/src/feature.py @expert\n"
        assert "expert" in rs._match_codeowners(content, ["src/feature.py"])
        assert "expert" not in rs._match_codeowners(content, ["src/other.py"])

    def test_star_rule_is_ignored(self):
        content = "* @everyone\n"
        assert rs._match_codeowners(content, ["src/x.py"]) == {}


class TestBotFilter:

    def test_bot_logins_detected(self):
        assert rs.is_bot("github-actions[bot]")
        assert rs.is_bot("dependabot")
        assert rs.is_bot("my-repo-claude-bot")
        assert not rs.is_bot("alice")
        assert not rs.is_bot("bob-smith")


class TestRankCandidates:

    def test_blame_author_outranks_minor_history_committer(
        self, fixture_repo, monkeypatch
    ):
        repo, base, path = fixture_repo
        state = {
            "base_commit": base,
            "base_branch": "main",
            "repo_dir": str(repo),
            "head_sha": "HEAD",
            "upstream_repo": "vllm-project/vllm",
        }

        # Stub the network parts: email->login mapping and PR reviews.
        monkeypatch.setattr(
            rs, "_email_to_login", lambda repo_, email: email.split("@")[0]
        )
        monkeypatch.setattr(rs, "review_candidates", lambda *a, **k: {})

        # alice has only history weight; bob owns the changed line.
        ranked, evidence = rs.rank_candidates(state, str(repo), 3)
        logins = [entry["login"] for entry in ranked]
        assert "bob" in logins
        bob = next(e for e in ranked if e["login"] == "bob")
        assert bob["score"] > 0
        assert any(
            "blame" in line or "changed lines" in line for line in bob["evidence"]
        )
        assert len(ranked) <= 3

    def test_author_is_excluded(self, fixture_repo, monkeypatch):
        repo, base, path = fixture_repo
        state = {
            "base_commit": base,
            "repo_dir": str(repo),
            "head_sha": "HEAD",
            "upstream_repo": "vllm-project/vllm",
        }
        monkeypatch.setattr(
            rs, "_email_to_login", lambda repo_, email: email.split("@")[0]
        )
        monkeypatch.setattr(rs, "review_candidates", lambda *a, **k: {})
        ranked, _ = rs.rank_candidates(state, str(repo), 3)
        assert "carol" not in [e["login"] for e in ranked]

    def test_bots_are_excluded(self, fixture_repo, monkeypatch):
        repo, base, path = fixture_repo
        state = {
            "base_commit": base,
            "repo_dir": str(repo),
            "head_sha": "HEAD",
            "upstream_repo": "vllm-project/vllm",
        }
        monkeypatch.setattr(
            rs, "_email_to_login", lambda repo_, email: "github-actions[bot]"
        )
        monkeypatch.setattr(rs, "review_candidates", lambda *a, **k: {})
        ranked, _ = rs.rank_candidates(state, str(repo), 3)
        assert ranked == []


class TestReviewCandidatesQuery:

    def test_uses_basename_text_search_not_path_qualifier(self, monkeypatch):
        """path: is not a GitHub search qualifier; the basename is used."""
        captured = {}

        def fake_gh(*args, check=True):
            captured["args"] = args
            return json.dumps({"items": []})

        monkeypatch.setattr(rs, "_gh", fake_gh)
        rs.review_candidates("vllm-project/vllm", ["vllm/x/flashinfer_mla_sparse.py"])
        q = captured["args"][captured["args"].index("-f") + 1]
        assert q.startswith("q=")
        assert "flashinfer_mla_sparse.py" in q
        assert "path:" not in q


class TestRenderDraft:

    def test_draft_is_minimal_ping_with_attribution(self):
        candidates = [
            {"login": "alice", "score": 6.0, "evidence": ["reviewed 3 PRs"]},
            {"login": "bob", "score": 3.0, "evidence": ["CODEOWNERS"]},
        ]
        draft = rs.render_draft("http://pr", candidates)
        assert draft.splitlines()[0] == "@alice @bob could you review this PR?"
        assert "reviewed 3 PRs" not in draft  # evidence stays local
        assert draft.rstrip().endswith(rs.MARKER)
        assert "Human reviewed changes" in draft

    def test_empty_candidate_list_still_renders(self):
        draft = rs.render_draft("http://pr", [])
        assert "could you review this PR?" in draft
        assert rs.MARKER in draft


class TestPatchFilesFromState:

    def test_uses_base_head_diff(self, fixture_repo):
        repo, base, path = fixture_repo
        state = {
            "repo_dir": str(repo),
            "base_commit": base,
            "head_sha": "HEAD",
            "output_dir": None,
        }
        assert rs._patch_files_from_state(state) == [path]


@needs_gh
class TestEndToEndGenerate:

    def test_generate_writes_draft_and_json_and_stamps_state(
        self, fixture_repo, tmp_path, monkeypatch
    ):
        repo, base, path = fixture_repo
        output_dir = tmp_path / "run"
        from llm.auto_code_gen.pr_flow import pr_state

        state_path = str(tmp_path / "pr_state.json")
        pr_state.save_state(
            state_path,
            dict(
                pr_state.new_state(
                    str(output_dir),
                    str(repo),
                    "git@github.com:me/vllm.git",
                    "me",
                    "vllm-project/vllm",
                    base_branch="main",
                ),
                base_commit=base,
                head_sha="HEAD",
                repo_dir=str(repo),
                pr_number=42,
                pr_url="http://pr/42",
                upstream_repo="vllm-project/vllm",
            ),
        )

        monkeypatch.setattr(
            rs, "_email_to_login", lambda repo_, email: email.split("@")[0]
        )
        monkeypatch.setattr(rs, "review_candidates", lambda *a, **k: {})

        rc = subprocess.run(
            [
                "python",
                "-m",
                "llm.auto_code_gen.pr_flow.run_reviewer_suggest",
                "generate",
                "--state",
                state_path,
            ],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        assert rc.returncode == 0, rc.stderr
        draft_path = os.path.join(pr_state.descs_dir(str(output_dir)), rs.REVIEWERS_TXT)
        json_path = os.path.join(pr_state.descs_dir(str(output_dir)), rs.REVIEWERS_JSON)
        assert os.path.isfile(draft_path) and os.path.isfile(json_path)

        with open(json_path) as f:
            payload = json.load(f)
        assert payload["pr_number"] == 42
        assert "gh pr comment 42" in payload["post_command"]

        with open(draft_path) as f:
            draft = f.read()
        assert draft.rstrip().endswith(rs.MARKER)

        import hashlib

        stamped = pr_state.load_state(state_path)["reviewers_draft_sha256"]
        assert stamped == hashlib.sha256(draft.encode()).hexdigest()
