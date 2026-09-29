"""Boundary tests for registry-backed and legacy pipeline configuration."""
import pytest
import yaml

from auto_bug_fix.config_loader import load_pipeline_config, load_multi_target_config
from common import llm_client


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(llm_client, "_api_base", None)
    monkeypatch.setattr(llm_client, "_api_key", None)
    monkeypatch.setattr(llm_client, "_model_params", {})


@pytest.fixture
def raw_config(tmp_path):
    return {
        "issue_id": "profile-test", "bug_description": "test",
        "repository": {"source_path": str(tmp_path / "repo")},
        "branches": {"source": "main", "target": "old"},
        "fix": {"commit": "abc123"}, "build": {},
    }


def write_config(tmp_path, raw_config, profiles=None, default="local"):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    if profiles is not None:
        (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": profiles, "default": default}))
    return str(path)


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("key_fields,expected", [
    ({"api_key_file": "~/test.key"}, "FILE_KEY"),
    ({"api_key_file": "~/missing.key"}, None),
    ({"api_key_file": "~/missing.key", "api_key": "INLINE_KEY"}, "INLINE_KEY"),
])
def test_credential_resolution_through_loaders(tmp_path, raw_config, loader, named, key_fields, expected, caplog):
    (tmp_path / "test.key").write_text("  FILE_KEY\n")
    raw_config["model"] = "local" if named else "openai/local"
    profiles = {"local": {"model": "openai/local", **key_fields}} if named else None
    if not named:
        raw_config.update(key_fields)
    result = loader(write_config(tmp_path, raw_config, profiles))
    assert result[-3].profile.api_key == expected
    assert ("Cannot read" in caplog.text) == (expected is None)


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("selection", [None, "local"])
@pytest.mark.parametrize("override", [None, "other", "openai/local", False])
def test_default_and_phase_profile_resolution(tmp_path, raw_config, loader, selection, override, caplog):
    profiles = {
        "local": {"model": "openai/local", "api_base": "http://localhost/v1", "api_key": "KEY"},
        "other": {"model": "gpt-4o", "api_key": "OTHER_KEY"},
    }
    if selection:
        raw_config["model"] = selection
    if override is not None:
        raw_config["advanced"] = {"phase_4b_model": override}
    bug, claude = loader(write_config(tmp_path, raw_config, profiles))[-4:-2]
    assert claude.model == "openai/local"
    if override == "other":
        assert bug.phase_4b_profile.id == "other"
        assert bug.phase_4b_model == "gpt-4o"
    else:
        assert bug.phase_4b_profile is claude.profile
        assert bug.phase_4b_model == claude.model
    if override is False:
        assert "phase_4b_model" in caplog.text


def test_no_registry_defaults_to_legacy_sonnet(tmp_path, raw_config):
    bug, claude, *_ = load_pipeline_config(write_config(tmp_path, raw_config))
    assert claude.model == "claude-sonnet-4-6"
    assert bug.phase_4b_profile is claude.profile
