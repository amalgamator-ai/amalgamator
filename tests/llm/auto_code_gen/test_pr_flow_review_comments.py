import json

import pytest

from llm.auto_code_gen.pr_flow import gh_publish, review_comments


def _comment(
    cid, body="use a constant", author="reviewer", parent=None, path="vllm/x.py"
):
    item = {
        "id": cid,
        "body": body,
        "user": {"login": author},
        "path": path,
        "line": 10 + cid,
        "created_at": f"2026-09-2{cid % 9}T10:00:00Z",
        "updated_at": f"2026-09-2{cid % 9}T11:00:00Z",
    }
    if parent is not None:
        item["in_reply_to_id"] = parent
    return item


def test_threads_group_replies_under_their_root():
    comments = [_comment(1), _comment(2, parent=1), _comment(3, parent=2), _comment(4)]
    threads = review_comments.build_threads(comments)
    assert [t[0]["id"] for t in threads] == [1, 4]
    assert [c["id"] for c in threads[0]] == [1, 2, 3]


def test_pending_excludes_our_own_replies_bots_and_cursor_backlog():
    comments = [
        _comment(10, author="reviewer"),
        _comment(11, body="fixed in the last push", author="us", parent=10),
        _comment(20, author="dependabot[bot]"),
        _comment(30, author="second-reviewer"),
    ]
    data = {"comments": comments, "threads": None}
    pending = review_comments.select_pending(data, our_logins=["us"])
    assert [item["comment_id"] for item in pending] == [10, 30]
    assert pending[0]["replies"] == [{"author": "us", "body": "fixed in the last push"}]

    after = review_comments.select_pending(
        data, cursor={"max_comment_id": 30}, our_logins=["us"]
    )
    assert after == []
    partial = review_comments.select_pending(
        data, cursor={"max_comment_id": 29}, our_logins=["us"]
    )
    assert [item["comment_id"] for item in partial] == [30]


def test_resolved_threads_are_dropped():
    comments = [_comment(10), _comment(11, author="us", parent=10), _comment(20)]
    graphql = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {
                                "isResolved": True,
                                "comments": {
                                    "nodes": [{"databaseId": 10}, {"databaseId": 11}]
                                },
                            },
                        ]
                    }
                }
            }
        }
    }
    pending = review_comments.select_pending(
        {"comments": comments, "threads": graphql}, our_logins=["us"]
    )
    assert [item["comment_id"] for item in pending] == [20]


def test_paginated_slurp_output_is_flattened():
    assert review_comments.flatten_pages([[{"id": 1}], [{"id": 2}, {"id": 3}]]) == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]
    assert review_comments.flatten_pages([{"id": 1}]) == [{"id": 1}]
    assert review_comments.flatten_pages({"a": 1}) == {"a": 1}
    assert review_comments.flatten_pages(None) is None


def test_round_directory_is_rendered_for_the_agent(tmp_path):
    inbox = tmp_path / "round_1"
    inbox.mkdir()
    (inbox / "pr_view.json").write_text(
        json.dumps(
            {
                "number": 4321,
                "url": "https://github.com/vllm-project/vllm/pull/4321",
                "state": "OPEN",
                "headRefOid": "abc",
                "mergeable": "MERGEABLE",
                "reviewDecision": "CHANGES_REQUESTED",
            }
        )
    )
    (inbox / "comments.json").write_text(
        json.dumps([[_comment(10, body="please use a named constant")], [_comment(11)]])
    )
    (inbox / "reviews.json").write_text(json.dumps([]))
    (inbox / "issue_comments.json").write_text(
        json.dumps(
            [
                {
                    "id": 99,
                    "body": "Overall this looks reasonable.",
                    "user": {"login": "maintainer"},
                }
            ]
        )
    )
    (inbox / "checks.json").write_text(
        json.dumps(
            [[{"name": "build", "status": "completed", "conclusion": "success"}]]
        )
    )

    summary = review_comments.summarize(str(inbox), our_logins=["us"], round_no=1)
    assert summary["pending_count"] == 2
    assert summary["max_comment_id"] == 11

    text = (inbox / review_comments.NEW_COMMENTS_FILE).read_text()
    assert "## Comment 10" in text and "## Comment 11" in text
    assert "please use a named constant" in text
    assert "CHANGES_REQUESTED" in text
    assert "Overall this looks reasonable." in text
    assert "CI check build: completed success" in text

    pending = json.loads((inbox / review_comments.PENDING_COMMENTS_FILE).read_text())
    assert [item["comment_id"] for item in pending["pending"]] == [10, 11]


def test_round_with_only_our_own_comments_is_empty(tmp_path):
    inbox = tmp_path / "round_2"
    inbox.mkdir()
    (inbox / "comments.json").write_text(
        json.dumps(
            [
                {
                    "id": 5,
                    "body": "done",
                    "user": {"login": "us"},
                    "path": "vllm/x.py",
                    "line": 1,
                    "created_at": "2026-09-20T10:00:00Z",
                    "updated_at": "2026-09-20T10:00:00Z",
                }
            ]
        )
    )
    summary = review_comments.summarize(str(inbox), our_logins=["us"], round_no=2)
    assert summary["pending_count"] == 0


def _state(tmp_path):
    return {
        "output_dir": str(tmp_path),
        "upstream_repo": "vllm-project/vllm",
        "pr_number": 4321,
        "gates": [
            {
                "round": 1,
                "iteration": 3,
                "overall": "PASS",
                "accepted": True,
                "perf": {"pass": True, "actual_pct": 2.86},
                "lm_eval": {"pass": True, "verdict": "PASS"},
                "closure": {
                    "pass": True,
                    "total": 4,
                    "fixed": 3,
                    "explained": 1,
                    "unresolved": 0,
                },
            }
        ],
    }


def test_round_comment_reports_the_gates(tmp_path):
    body = gh_publish.render_round_comment(_state(tmp_path))
    assert "+2.86%" in body
    assert "review comments addressed | 4 of 4" in body.lower()
    assert "review round 1" in body


def test_round_comment_fails_closed_without_a_gate(tmp_path):
    with pytest.raises(SystemExit):
        gh_publish.render_round_comment({"output_dir": str(tmp_path), "gates": []})


def test_thread_replies_are_bounded_and_cleaned(tmp_path):
    path = tmp_path / "thread_replies.json"
    path.write_text(
        json.dumps(
            {
                "replies": [
                    {"comment_id": 10, "status": "fixed", "body": "  done  "},
                    {"comment_id": 11, "status": "explained", "body": "   "},
                    {"status": "fixed", "body": "no id"},
                ]
            }
        )
    )
    assert gh_publish.render_thread_replies(str(path)) == [(10, "done")]


def test_posting_is_opt_in(tmp_path, capsys):
    state = _state(tmp_path)
    assert gh_publish.post_round_comment(state, "hello", execute=False) == 0
    assert gh_publish.post_reply(state, 10, "done", execute=False) == 0
    printed = capsys.readouterr().out
    assert "gh pr comment 4321" in printed
    assert "repos/vllm-project/vllm/pulls/4321/comments/10/replies" in printed


def test_replies_without_a_pr_number_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        gh_publish.post_reply({"upstream_repo": "vllm-project/vllm"}, 10, "done")
