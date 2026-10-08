"""Data models for llm.auto_config."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResolvedRecipe:
    """Result of resolving a recipe from recipes.vllm.ai."""
    model_id: str
    extra_flags: str
    extra_env: dict[str, str]
    docker_image: str
