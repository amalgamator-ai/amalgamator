"""Credential-bearing runtime profiles must never become public records."""
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pytest

from auto_bug_fix import run_bug_fix
from auto_bug_fix.tracker import Tracker
from common.claude_utils import print_claude_config
from common.model_registry import ModelProfile


@pytest.fixture
def secret_profile():
    return ModelProfile(
        id="local", provider="litellm", model="openai/local",
        api_key="SECRET_KEY", api_base="https://user:SECRET_PASS@host/v1?key=SECRET_QUERY",
        params={"extra_body": {"headers": {"x-api-key": "SECRET_HEADER"}}},
    )


def test_pipeline_persists_only_public_profile(bug_cfg, claude_cfg, secret_profile, tmp_path, monkeypatch):
    bug_cfg.phase_4b_profile = secret_profile
    claude_cfg.profile = secret_profile
    monkeypatch.setattr(run_bug_fix, "_run_pipeline_phases", lambda c, cc, s, t: (s.dossier, t))
    run_bug_fix.run_pipeline(bug_cfg, claude_cfg, output_dir=str(tmp_path))
    text = next(tmp_path.glob("*.json")).read_text()
    assert "SECRET_" not in text
    assert json.loads(text)["config"]["phase_4b_profile"] == secret_profile.public_snapshot()
    assert secret_profile.api_key == "SECRET_KEY"
    assert secret_profile.params["extra_body"]["headers"]["x-api-key"] == "SECRET_HEADER"


@pytest.mark.parametrize("serialized", [False, True])
def test_tracker_write_boundary_sanitizes_profiles(secret_profile, tmp_path, serialized):
    tracker = Tracker("snapshot", {}, output_dir=str(tmp_path))
    # Even a caller updating config after construction is covered.
    tracker._run.config["profiles"] = [asdict(secret_profile) if serialized else secret_profile]
    text = Path(tracker.save()).read_text()
    assert "SECRET_" not in text
    assert json.loads(text)["config"]["profiles"] == [secret_profile.public_snapshot()]


def test_profile_logs_use_safe_identity(claude_cfg, secret_profile, capsys, caplog):
    claude_cfg.profile = secret_profile
    print_claude_config(claude_cfg)
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info("profile=%s", secret_profile)
        logging.getLogger(__name__).info(f"profile={secret_profile}")
    text = capsys.readouterr().out + caplog.text + repr(secret_profile)
    assert "SECRET_" not in text
    assert "local" in text
