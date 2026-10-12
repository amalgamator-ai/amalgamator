import json
from types import SimpleNamespace

import pytest

from llm.auto_code_gen.pr_flow import checks
from llm.auto_code_gen.pr_flow.run_pr_review_iters import _repo_dir, _resolve_round


def test_state_and_config_must_agree_on_the_clone(tmp_path):
    repo = tmp_path / "vllm"
    config = SimpleNamespace(source_code_dir=str(repo))
    assert _repo_dir(config, {"repo_dir": str(repo)}) == str(repo)

    with pytest.raises(checks.CheckError) as excinfo:
        _repo_dir(config, {"repo_dir": str(tmp_path / "other-clone")})
    assert excinfo.value.code == checks.EXIT_STATE
    assert "other-clone" in str(excinfo.value)


def test_state_without_a_repo_falls_back_to_the_config(tmp_path):
    config = SimpleNamespace(source_code_dir=str(tmp_path / "vllm"))
    assert _repo_dir(config, {}) == str(tmp_path / "vllm")


def _write_pending(tmp_path, round_no, count):
    inbox = tmp_path / "pr_p3" / "inbox" / f"round_{round_no}"
    inbox.mkdir(parents=True)
    (inbox / "pending_comments.json").write_text(
        json.dumps({"round": round_no, "pending": [{"comment_id": 1}] * count})
    )


def test_round_defaults_to_the_fetched_one(tmp_path):
    _write_pending(tmp_path, 3, 2)
    assert _resolve_round({"round": 3}, str(tmp_path)) == 3


def test_round_moves_on_when_the_fetched_one_is_answered(tmp_path):
    _write_pending(tmp_path, 3, 0)
    assert _resolve_round({"round": 3}, str(tmp_path)) == 4
    assert _resolve_round({}, str(tmp_path)) == 1
    assert _resolve_round({"round": 9}, str(tmp_path)) == 10
