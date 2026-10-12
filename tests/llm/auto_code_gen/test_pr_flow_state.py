# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from llm.auto_code_gen.pr_flow import pr_state

FORK_URL = "git@github.com:example-dev/vllm.git"


def test_parse_git_url_accepts_ssh_https_and_short_form():
    assert pr_state.parse_git_url(FORK_URL) == "example-dev/vllm"
    assert (
        pr_state.parse_git_url("https://github.com/example-dev/vllm.git")
        == "example-dev/vllm"
    )
    assert pr_state.parse_git_url("example-dev/vllm") == "example-dev/vllm"
    assert pr_state.url_owner(FORK_URL) == "example-dev"


def test_parse_git_url_rejects_garbage():
    assert pr_state.parse_git_url("") is None
    assert pr_state.parse_git_url("not a url") is None
    assert pr_state.parse_git_url("git@github.com:example-dev") is None


def test_local_fork_url_is_refused():
    with pytest.raises(ValueError, match="local path"):
        pr_state.new_state("/tmp/out", "/tmp/repo", "file:///srv/vllm", "example-dev")


def test_fork_must_differ_from_upstream():
    with pytest.raises(ValueError, match="upstream"):
        pr_state.new_state(
            "/tmp/out",
            "/tmp/repo",
            "git@github.com:vllm-project/vllm.git",
            "vllm-project",
        )


def test_fork_owner_is_required_and_validated():
    with pytest.raises(ValueError, match="fork_owner"):
        pr_state.new_state("/tmp/out", "/tmp/repo", FORK_URL, "")
    with pytest.raises(ValueError, match="GitHub login"):
        pr_state.new_state("/tmp/out", "/tmp/repo", FORK_URL, "not a login")


def test_new_state_defaults(tmp_path):
    state = pr_state.new_state(
        str(tmp_path),
        str(tmp_path / "repo"),
        FORK_URL,
        "example-dev",
        local_branch="local-perf",
    )
    assert state["upstream_repo"] == "vllm-project/vllm"
    assert state["remote_branch"] == ""
    assert state["round"] == 0
    assert state["gates"] == []
    assert state["cursor"]["max_comment_id"] == 0


def test_round_trip_update_and_gate_history(tmp_path):
    path = pr_state.state_path(str(tmp_path))
    pr_state.save_state(
        path, pr_state.new_state(str(tmp_path), str(tmp_path), FORK_URL, "example-dev")
    )
    pr_state.update(path, head_sha="abc123", round=2)
    assert pr_state.load_state(path)["head_sha"] == "abc123"
    assert pr_state.load_state(path)["round"] == 2

    pr_state.record_gate(path, {"overall": "PASS", "iteration": 3})
    pr_state.record_gate(path, {"overall": "FAIL", "iteration": 4})
    gates = pr_state.load_state(path)["gates"]
    assert [g["overall"] for g in gates] == ["PASS", "FAIL"]
    assert gates[0]["recorded_at"]

    with pytest.raises(ValueError, match="unknown state field"):
        pr_state.update(path, not_a_field="x")


def test_state_paths_are_stable_per_output_dir(tmp_path):
    assert pr_state.state_path(str(tmp_path)) == str(
        tmp_path / "pr_p3" / "pr_state.json"
    )
    assert pr_state.inbox_dir(str(tmp_path), 3) == str(
        tmp_path / "pr_p3" / "inbox" / "round_3"
    )


def test_newer_state_version_is_refused(tmp_path):
    path = tmp_path / "pr_state.json"
    path.write_text(json.dumps({"version": pr_state.STATE_VERSION + 1}))
    with pytest.raises(ValueError, match="newer"):
        pr_state.load_state(str(path))
