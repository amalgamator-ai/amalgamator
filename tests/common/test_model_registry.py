# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for common.model_registry — profile loading, resolution, and legacy fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.model_registry import (
    ModelProfile,
    ModelRegistry,
    RegistryError,
    _infer_provider,
    build_legacy_profile,
    find_registry,
    load_registry,
)

# ── ModelProfile ────────────────────────────────────────────────────────────


class TestModelProfile:
    def test_anthropic_model_is_not_self_hosted(self):
        p = ModelProfile(id="a", provider="anthropic", model="claude-sonnet-4-6")
        assert not p.is_self_hosted()

    def test_ollama_model_is_self_hosted(self):
        p = ModelProfile(id="a", provider="litellm", model="ollama/qwen3:14b")
        assert p.is_self_hosted()

    def test_openai_prefix_with_api_base_is_self_hosted(self):
        p = ModelProfile(
            id="a",
            provider="litellm",
            model="openai/deepseek-r1",
            api_base="http://localhost:8080/v1",
        )
        assert p.is_self_hosted()

    def test_openai_prefix_without_api_base_is_not_self_hosted(self):
        p = ModelProfile(id="a", provider="litellm", model="openai/deepseek-r1")
        assert not p.is_self_hosted()

    def test_frozen_dataclass(self):
        p = ModelProfile(id="a", provider="anthropic", model="claude-sonnet-4-6")
        with pytest.raises(AttributeError):
            p.model = "other"  # pyrefly: ignore [read-only] (deliberately exercises the frozen error)


# ── _infer_provider ─────────────────────────────────────────────────────────


class TestInferProvider:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "anthropic/claude-sonnet-4-6",
            "vertex_ai/claude-opus-4-6",
        ],
    )
    def test_claude_models_infer_anthropic(self, model):
        assert _infer_provider(model) == "anthropic"

    @pytest.mark.parametrize(
        "model",
        [
            "openai/deepseek-r1",
            "ollama/qwen3:14b",
            "gpt-4o",
            "together_ai/meta-llama/Llama-3.1-70B",
        ],
    )
    def test_non_claude_models_infer_litellm(self, model):
        assert _infer_provider(model) == "litellm"


# ── build_legacy_profile ────────────────────────────────────────────────────


class TestBuildLegacyProfile:
    def test_bare_claude_model(self):
        p = build_legacy_profile("claude-sonnet-4-6")
        assert p.provider == "anthropic"
        assert p.model == "claude-sonnet-4-6"
        assert p.id.startswith("_legacy_")

    def test_openai_prefix_with_transport(self):
        p = build_legacy_profile(
            "openai/deepseek-r1",
            api_base="http://localhost:8080/v1",
            api_key="test-key",
            params={"temperature": 0.1},
        )
        assert p.provider == "litellm"
        assert p.api_base == "http://localhost:8080/v1"
        assert p.api_key == "test-key"
        assert p.params == {"temperature": 0.1}

    def test_api_base_trailing_slash_stripped(self):
        p = build_legacy_profile("openai/m", api_base="http://host:8080/")
        assert p.api_base == "http://host:8080"


# ── load_registry ───────────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(content)
    return p


class TestLoadRegistry:
    @pytest.mark.parametrize("strict", ["true", "false", "'false'", "null", "1"])
    def test_strict_mode_requires_boolean(self, tmp_path, strict):
        path = _write_yaml(
            tmp_path,
            f"models:\n  m1:\n    model: gpt-4o\ndefault: m1\nstrict: {strict}\n",
        )
        if strict in ("true", "false"):
            assert load_registry(path).strict is (strict == "true")
        else:
            with pytest.raises(RegistryError, match="must be a boolean"):
                load_registry(path)

    def test_load_simple_registry(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    provider: anthropic
    model: claude-sonnet-4-6
  m2:
    provider: litellm
    model: openai/deepseek-r1
    api_base: http://localhost:8080/v1
default: m1
""",
        )
        reg = load_registry(path)
        assert "m1" in reg.profiles
        assert "m2" in reg.profiles
        assert reg.default_id == "m1"
        assert reg.default_profile.model == "claude-sonnet-4-6"
        assert reg.profiles["m2"].api_base == "http://localhost:8080/v1"

    def test_provider_inferred_when_absent(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: claude-opus-4-6
default: m1
""",
        )
        reg = load_registry(path)
        assert reg.profiles["m1"].provider == "anthropic"

    def test_invalid_provider_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    provider: invalid
    model: some-model
default: m1
""",
        )
        with pytest.raises(RegistryError, match="provider must be"):
            load_registry(path)

    def test_missing_model_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    provider: anthropic
default: m1
""",
        )
        with pytest.raises(RegistryError, match="non-empty 'model'"):
            load_registry(path)

    def test_default_pointing_at_undefined_id_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: claude-sonnet-4-6
default: m99
""",
        )
        with pytest.raises(RegistryError, match="does not match any defined profile"):
            load_registry(path)

    def test_missing_default_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: claude-sonnet-4-6
""",
        )
        with pytest.raises(RegistryError, match="must have a 'default'"):
            load_registry(path)

    def test_empty_models_raises(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models: {}
default: m1
""",
        )
        with pytest.raises(RegistryError, match="non-empty 'models'"):
            load_registry(path)

    def test_params_passthrough(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: openai/deepseek-r1
    params:
      temperature: 0.2
      top_p: 0.9
default: m1
""",
        )
        reg = load_registry(path)
        assert reg.profiles["m1"].params == {"temperature": 0.2, "top_p": 0.9}

    def test_api_key_file_expansion(self, tmp_path):
        key_file = tmp_path / "test.key"
        key_file.write_text("  sk-test-key-123  \n")
        path = _write_yaml(
            tmp_path,
            f"""
models:
  m1:
    model: openai/deepseek-r1
    api_key_file: {key_file}
default: m1
""",
        )
        reg = load_registry(path)
        assert reg.profiles["m1"].api_key == "sk-test-key-123"

    def test_api_key_file_missing_warns(self, tmp_path, caplog):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: openai/deepseek-r1
    api_key_file: /nonexistent/path/key.txt
default: m1
""",
        )
        reg = load_registry(path)
        assert reg.profiles["m1"].api_key is None
        assert "Cannot read" in caplog.text

    def test_invalid_params_warns(self, tmp_path, caplog):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: claude-sonnet-4-6
    params: "not a dict"
default: m1
""",
        )
        reg = load_registry(path)
        assert reg.profiles["m1"].params == {}
        assert "must be a mapping" in caplog.text


# ── ModelRegistry.resolve ───────────────────────────────────────────────────


class TestModelRegistryResolve:
    @pytest.fixture
    def registry(self):
        return ModelRegistry(
            profiles={
                "m1": ModelProfile(
                    id="m1", provider="anthropic", model="claude-sonnet-4-6"
                ),
                "m2": ModelProfile(
                    id="m2",
                    provider="litellm",
                    model="openai/deepseek-r1",
                    api_base="http://localhost:8080/v1",
                ),
            },
            default_id="m1",
        )

    def test_resolve_known_id(self, registry):
        p = registry.resolve("m2")
        assert p.model == "openai/deepseek-r1"
        assert p.api_base == "http://localhost:8080/v1"

    def test_resolve_none_returns_default(self, registry):
        p = registry.resolve(None)
        assert p.id == "m1"
        assert p.model == "claude-sonnet-4-6"

    def test_resolve_none_with_fallback(self, registry):
        fallback = ModelProfile(id="fb", provider="anthropic", model="claude-opus-4-6")
        p = registry.resolve(None, fallback=fallback)
        assert p.id == "fb"

    def test_resolve_unknown_string_builds_legacy(self, registry):
        p = registry.resolve("gpt-4o")
        assert p.id == "_legacy_gpt-4o"
        assert p.provider == "litellm"
        assert p.model == "gpt-4o"

    def test_strict_validation_precedes_legacy_factory(self, registry):
        from unittest.mock import Mock

        strict = ModelRegistry(registry.profiles, registry.default_id, strict=True)
        factory = Mock(side_effect=AssertionError("Must not read inline credentials"))
        assert strict.resolve("m1", legacy_factory=factory) is strict.default_profile
        with pytest.raises(RegistryError, match="Unknown model profile"):
            strict.resolve("typo", legacy_factory=factory)
        factory.assert_not_called()

    def test_id_match_wins_over_bare_string(self):
        reg = ModelRegistry(
            profiles={
                "claude-sonnet-4-6": ModelProfile(
                    id="claude-sonnet-4-6",
                    provider="litellm",
                    model="openai/custom-sonnet",
                    api_base="http://my-proxy:8080/v1",
                ),
            },
            default_id="claude-sonnet-4-6",
        )
        p = reg.resolve("claude-sonnet-4-6")
        assert p.model == "openai/custom-sonnet"


# ── find_registry ───────────────────────────────────────────────────────────


class TestFindRegistry:
    def test_explicit_path(self, tmp_path):
        path = _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: claude-sonnet-4-6
default: m1
""",
        )
        reg = find_registry(explicit_path=str(path))
        assert reg is not None
        assert "m1" in reg.profiles

    def test_config_dir_discovery(self, tmp_path):
        _write_yaml(
            tmp_path,
            """
models:
  m1:
    model: claude-sonnet-4-6
default: m1
""",
        )
        reg = find_registry(config_dir=str(tmp_path))
        assert reg is not None

    def test_no_file_returns_none(self, tmp_path):
        reg = find_registry(config_dir=str(tmp_path / "nonexistent"))
        assert reg is None
