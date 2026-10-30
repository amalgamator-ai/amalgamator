# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Named model profile registry.

Decouples pipeline configs from provider details. A ``models.yaml`` file
defines named profiles (id → provider + model + endpoint + credentials);
pipeline configs reference profiles by id.

When no registry is loaded, bare model strings are still accepted for
backward compatibility — a synthetic profile is built from the model name
and any legacy inline fields (api_base, api_key, model_params).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from common.model_identity import is_claude_model, is_ollama_model

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Everything needed to call a model: provider routing, endpoint, credentials."""

    id: str
    provider: str
    model: str
    api_base: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    params: dict = field(default_factory=dict, repr=False)
    # The model's true context window, when known. The agent loop uses it as
    # the compaction budget (overriding its 128K default) when the request
    # does not carry an explicit context_length.
    context_length: int | None = None

    def __post_init__(self):
        reserved = {
            "model",
            "provider",
            "api_base",
            "api_key",
            "api_key_file",
        } & self.params.keys()
        if reserved:
            raise RegistryError(
                "Profile params must use top-level fields for: "
                + ", ".join(sorted(reserved))
            )

    def public_snapshot(self) -> dict[str, str]:
        """Allowlisted identity for logs and persistence; never runtime transport."""
        return {"id": self.id, "provider": self.provider, "model": self.model}

    def is_self_hosted(self) -> bool:
        if is_ollama_model(self.model):
            return True
        return bool(self.model.startswith("openai/") and self.api_base)


class RegistryError(Exception):
    pass


def public_config_snapshot(value: Any) -> Any:
    """Copy configuration without serializing credential-bearing profiles.

    Profile-shaped dictionaries are also sanitized, defending tracker callers
    that have already used dataclasses.asdict(). Other fields keep their shape.
    """
    if isinstance(value, ModelProfile):
        return value.public_snapshot()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: public_config_snapshot(getattr(value, f.name))
            for f in fields(value)
        }
    if isinstance(value, dict):
        if {"id", "provider", "model"} <= value.keys():
            return {key: value[key] for key in ("id", "provider", "model")}
        return {key: public_config_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(public_config_snapshot(item) for item in value)
    return deepcopy(value)


def _infer_provider(model: str) -> str:
    if is_claude_model(model):
        return "anthropic"
    return "litellm"


def _resolve_api_key_file(path_str: str) -> str | None:
    resolved = os.path.expanduser(path_str)
    try:
        return Path(resolved).read_text().strip()
    except OSError as e:
        log.warning("Cannot read api_key_file %s: %s", resolved, e)
        return None


def resolve_api_key(config: dict[str, Any]) -> str | None:
    """Resolve an inline key first, then a whitespace-stripped key file."""
    if config.get("api_key"):
        return str(config["api_key"])
    if config.get("api_key_file"):
        return _resolve_api_key_file(str(config["api_key_file"]))
    return None


def _parse_profile(profile_id: str, raw: dict[str, Any]) -> ModelProfile:
    model = raw.get("model")
    if not model or not isinstance(model, str):
        raise RegistryError(
            f"Profile '{profile_id}' must have a non-empty 'model' string"
        )

    provider = raw.get("provider")
    if provider and provider not in ("anthropic", "litellm"):
        raise RegistryError(
            f"Profile '{profile_id}': provider must be 'anthropic' or 'litellm', "
            f"got '{provider}'"
        )
    if not provider:
        provider = _infer_provider(model)

    api_base = raw.get("api_base")
    if api_base:
        api_base = str(api_base).rstrip("/")

    api_key = resolve_api_key(raw)

    params = raw.get("params", {})
    if not isinstance(params, dict):
        log.warning("Profile '%s': params must be a mapping, ignoring", profile_id)
        params = {}

    context_length = None
    raw_ctx = raw.get("context_length")
    if raw_ctx is not None:
        try:
            context_length = int(raw_ctx)
        except (TypeError, ValueError):
            log.warning("Profile '%s': context_length not an int, ignoring", profile_id)

    return ModelProfile(
        id=profile_id,
        provider=provider,
        model=model.strip(),
        api_base=api_base,
        api_key=api_key,
        params=params,
        context_length=context_length,
    )


class ModelRegistry:
    """Resolves profile ids to ModelProfile instances."""

    def __init__(
        self,
        profiles: dict[str, ModelProfile],
        default_id: str,
        *,
        strict: bool = False,
        source_path: Path | None = None,
    ):
        if not isinstance(strict, bool):
            raise RegistryError("Registry 'strict' must be a boolean")
        self.profiles = dict(profiles)
        self.default_id = default_id
        self.strict = strict
        self.source_path = source_path
        if default_id not in self.profiles:
            raise RegistryError(
                f"default '{default_id}' does not match any defined profile id "
                f"(available: {', '.join(sorted(self.profiles))})"
            )

    @property
    def default_profile(self) -> ModelProfile:
        return self.profiles[self.default_id]

    def resolve(
        self,
        id_or_model: str | None,
        fallback: ModelProfile | None = None,
        *,
        legacy_factory: Callable[[str], ModelProfile] | None = None,
    ) -> ModelProfile:
        """Resolve a string to a ModelProfile.

        - Known profile id → that profile.
        - None → fallback if provided, else the default profile.
        - Unknown string → error in strict mode, otherwise a synthetic profile.

        The optional legacy_factory applies the caller's inline settings only
        when a bare-model fallback is actually selected.
        """
        if id_or_model is None:
            return fallback if fallback is not None else self.default_profile

        if id_or_model in self.profiles:
            return self.profiles[id_or_model]

        if self.strict:
            raise RegistryError(
                f"Unknown model profile '{id_or_model}' in strict registry "
                f"(available: {', '.join(sorted(self.profiles))}). "
                "Add a named profile or set strict: false to allow bare model names."
            )

        log.warning(
            "Model '%s' is not a registered profile id — building a synthetic "
            "profile. Consider adding it to models.yaml.",
            id_or_model,
        )
        return (legacy_factory or build_legacy_profile)(id_or_model)


def build_legacy_profile(
    model: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    params: dict | None = None,
) -> ModelProfile:
    """Build a synthetic profile from a bare model string and legacy inline fields.

    Used for backward compatibility when the pipeline config uses a raw model
    name instead of a profile id.
    """
    return ModelProfile(
        id=f"_legacy_{model}",
        provider=_infer_provider(model),
        model=model,
        api_base=api_base.rstrip("/") if api_base else None,
        api_key=api_key,
        params=dict(params) if params else {},
    )


def load_registry(path: Path) -> ModelRegistry:
    """Parse a models.yaml file into a ModelRegistry."""
    import yaml

    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as e:
        raise RegistryError(f"Cannot load {path}: {e}") from e

    if not isinstance(raw, dict):
        raise RegistryError(f"{path} must contain a YAML mapping")

    models_section = raw.get("models")
    if not isinstance(models_section, dict) or not models_section:
        raise RegistryError(f"{path} must have a non-empty 'models' mapping")

    default_id = raw.get("default")
    if not default_id or not isinstance(default_id, str):
        raise RegistryError(
            f"{path} must have a 'default' string pointing at a profile id"
        )

    profiles: dict[str, ModelProfile] = {}
    for pid, praw in models_section.items():
        pid = str(pid)
        if not isinstance(praw, dict):
            raise RegistryError(f"Profile '{pid}' must be a mapping")
        profiles[pid] = _parse_profile(pid, praw)

    return ModelRegistry(
        profiles=profiles,
        default_id=default_id,
        strict=raw.get("strict", False),
        source_path=path.resolve(),
    )


def find_registry(
    config_dir: str | None = None,
    explicit_path: str | None = None,
) -> ModelRegistry | None:
    """Search for a models.yaml and load it if found.

    Search order:
    1. explicit_path (--models flag)
    2. models.yaml in config_dir (same directory as the pipeline config)
    3. ~/.config/amalgamator/models.yaml

    An explicit path is mandatory when supplied: failure never falls back to
    discovery. Relative paths use the caller's working directory.
    """
    candidates: list[Path] = []

    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        log.info("Loading explicit model registry from %s", path)
        return load_registry(path)
    if config_dir:
        candidates.append(Path(config_dir) / "models.yaml")
    candidates.append(Path("~/.config/amalgamator/models.yaml").expanduser())

    for path in candidates:
        if path.is_file():
            log.info("Loading model registry from %s", path)
            return load_registry(path)

    return None
