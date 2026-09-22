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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Everything needed to call a model: provider routing, endpoint, credentials."""
    id: str
    provider: str
    model: str
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    params: dict = field(default_factory=dict)

    def is_self_hosted(self) -> bool:
        if self.model.startswith("ollama/") or self.model.startswith("ollama_chat/"):
            return True
        if self.model.startswith("openai/") and self.api_base:
            return True
        return False


class RegistryError(Exception):
    pass


_CLAUDE_PREFIXES = ("claude-", "anthropic/", "vertex_ai/claude-")


def _infer_provider(model: str) -> str:
    if any(model.startswith(p) for p in _CLAUDE_PREFIXES):
        return "anthropic"
    return "litellm"


def _resolve_api_key_file(path_str: str) -> str | None:
    resolved = os.path.expanduser(path_str)
    try:
        return Path(resolved).read_text().strip()
    except OSError as e:
        log.warning("Cannot read api_key_file %s: %s", resolved, e)
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

    api_key = None
    if raw.get("api_key"):
        api_key = str(raw["api_key"])
    elif raw.get("api_key_file"):
        api_key = _resolve_api_key_file(str(raw["api_key_file"]))

    params = raw.get("params", {})
    if not isinstance(params, dict):
        log.warning("Profile '%s': params must be a mapping, ignoring", profile_id)
        params = {}

    return ModelProfile(
        id=profile_id,
        provider=provider,
        model=model.strip(),
        api_base=api_base,
        api_key=api_key,
        params=params,
    )


class ModelRegistry:
    """Resolves profile ids to ModelProfile instances."""

    def __init__(
        self,
        profiles: dict[str, ModelProfile],
        default_id: str,
    ):
        self.profiles = dict(profiles)
        self.default_id = default_id
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
    ) -> ModelProfile:
        """Resolve a string to a ModelProfile.

        - Known profile id → that profile.
        - None → fallback if provided, else the default profile.
        - Unknown string → synthetic profile built from the bare model name.
        """
        if id_or_model is None:
            return fallback if fallback is not None else self.default_profile

        if id_or_model in self.profiles:
            return self.profiles[id_or_model]

        log.warning(
            "Model '%s' is not a registered profile id — building a synthetic "
            "profile. Consider adding it to models.yaml.",
            id_or_model,
        )
        return build_legacy_profile(id_or_model)


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
        raise RegistryError(f"{path} must have a 'default' string pointing at a profile id")

    profiles: dict[str, ModelProfile] = {}
    for pid, praw in models_section.items():
        pid = str(pid)
        if not isinstance(praw, dict):
            raise RegistryError(f"Profile '{pid}' must be a mapping")
        profiles[pid] = _parse_profile(pid, praw)

    return ModelRegistry(profiles=profiles, default_id=default_id)


def find_registry(
    config_dir: str | None = None,
    explicit_path: str | None = None,
) -> ModelRegistry | None:
    """Search for a models.yaml and load it if found.

    Search order:
    1. explicit_path (--models flag)
    2. models.yaml in config_dir (same directory as the pipeline config)
    3. ~/.config/amalgamator/models.yaml
    """
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))
    if config_dir:
        candidates.append(Path(config_dir) / "models.yaml")
    candidates.append(Path("~/.config/amalgamator/models.yaml").expanduser())

    for path in candidates:
        if path.is_file():
            log.info("Loading model registry from %s", path)
            return load_registry(path)

    return None
