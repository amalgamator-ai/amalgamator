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
