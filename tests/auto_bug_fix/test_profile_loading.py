"""Boundary tests for registry-backed and legacy pipeline configuration."""
import pytest
import yaml
import sys
from types import SimpleNamespace
from unittest.mock import Mock

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


def write_config(tmp_path, raw_config, profiles=None, default="local", *, strict=False):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    if profiles is not None:
        (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": profiles, "default": default, "strict": strict}))
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


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
def test_unrelated_registry_preserves_legacy_request(tmp_path, raw_config, loader, monkeypatch):
    raw_config.update(model="openai/bare", api_base="http://localhost/v1", api_key="KEY",
                      model_params={"temperature": 0.2})
    path = write_config(tmp_path, raw_config)
    before = loader(path)[-3].profile
    write_config(tmp_path, raw_config, {"local": {"model": "gpt-4o"}})
    bug, claude = loader(path)[-4:-2]
    assert claude.profile == before
    assert bug.phase_4b_profile is claude.profile
    request = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=request))
    llm_client.completion(claude.model, [], profile=claude.profile)
    assert request.call_args.kwargs["api_base"] == before.api_base
    assert request.call_args.kwargs["api_key"] == before.api_key
    assert request.call_args.kwargs["temperature"] == 0.2


@pytest.mark.parametrize("selection", [None, "local"])
def test_named_selection_does_not_seed_globals_or_merge_inline(tmp_path, raw_config, monkeypatch, caplog, selection):
    raw_config.update(api_base="https://SECRET_ENDPOINT", api_key="SECRET_KEY",
                      model_params={"temperature": 0.8})
    if selection:
        raw_config["model"] = selection
    configure = Mock()
    monkeypatch.setattr(llm_client, "configure", configure)
    _, claude, *_ = load_pipeline_config(write_config(tmp_path, raw_config, {
        "local": {"model": "gpt-4o"},
    }))
    configure.assert_not_called()
    assert claude.profile.api_base is None
    assert claude.profile.api_key is None
    assert claude.profile.params == {}
    assert "ignores inline fields" in caplog.text
    assert "SECRET_" not in caplog.text


def test_different_bare_phase_uses_only_yaml_inline_transport(tmp_path, raw_config, caplog):
    raw_config.update(model="local", api_key="INLINE_KEY", api_base="http://inline/v1",
                      advanced={"phase_4b_model": "openai/other"})
    bug, claude, *_ = load_pipeline_config(write_config(tmp_path, raw_config, {
        "local": {"model": "openai/local", "api_key": "NAMED_KEY", "api_base": "http://named/v1"},
    }))
    assert claude.profile.api_key == "NAMED_KEY"
    assert bug.phase_4b_profile.api_key == "INLINE_KEY"
    assert bug.phase_4b_profile.api_base == "http://inline/v1"
    assert "inherits inline fields" in caplog.text
    assert "INLINE_KEY" not in caplog.text


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("invalid", [False, True])
def test_explicit_registry_never_falls_back(tmp_path, raw_config, loader, invalid):
    from common.model_registry import RegistryError
    path = write_config(tmp_path, raw_config, {"local": {"model": "gpt-4o"}})
    explicit = tmp_path / "explicit.yaml"
    if invalid:
        explicit.write_text("models: []")
    with pytest.raises(RegistryError):
        loader(path, models_path=str(explicit))


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("explicit", ["~/explicit.yaml", "explicit.yaml"])
def test_explicit_registry_precedence_and_expansion(tmp_path, raw_config, loader, explicit, monkeypatch):
    path = write_config(tmp_path, raw_config, {"local": {"model": "gpt-4o"}})
    (tmp_path / "explicit.yaml").write_text("models:\n  chosen:\n    model: openai/explicit\ndefault: chosen\n")
    monkeypatch.chdir(tmp_path)
    assert loader(path, models_path=explicit)[-3].profile.id == "chosen"


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("phase_override", [False, True])
@pytest.mark.parametrize("selection", ["snnet", "openai/local", "o3"])
def test_strict_registry_rejects_unregistered_selections(tmp_path, raw_config, loader, phase_override, selection):
    from common.model_registry import RegistryError
    raw_config["model"] = "local" if phase_override else selection
    if phase_override:
        raw_config["advanced"] = {"phase_4b_model": selection}
    path = write_config(tmp_path, raw_config, {"local": {"model": "openai/local"}}, strict=True)
    with pytest.raises(RegistryError, match="Unknown model profile"):
        loader(path)


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("selection", [None, "local"])
def test_strict_registry_accepts_default_and_named_profile(tmp_path, raw_config, loader, selection):
    if selection:
        raw_config["model"] = selection
    # Strict mode validates references, not a hardcoded list of model families.
    path = write_config(tmp_path, raw_config, {"local": {"model": "new-provider/custom-model"}}, strict=True)
    bug, agent = loader(path)[-4:-2]
    assert agent.profile.id == "local"
    assert agent.model == "new-provider/custom-model"
    assert bug.phase_4b_profile is agent.profile


@pytest.mark.parametrize("loader", [load_pipeline_config, load_multi_target_config])
@pytest.mark.parametrize("phase_override", [False, True])
def test_permissive_registry_warns_for_bare_fallback(tmp_path, raw_config, loader, phase_override, caplog):
    raw_config["model"] = "local" if phase_override else "snnet"
    if phase_override:
        raw_config["advanced"] = {"phase_4b_model": "snnet"}
    path = write_config(tmp_path, raw_config, {"local": {"model": "gpt-4o"}})
    bug, agent = loader(path)[-4:-2]
    assert (bug.phase_4b_profile if phase_override else agent.profile).model == "snnet"
    assert "not a registered profile id" in caplog.text
