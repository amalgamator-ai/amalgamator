# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free model classification for legacy routing and inference."""

_CLAUDE_PREFIXES = ("claude-", "anthropic/", "vertex_ai/claude-")
_OLLAMA_PREFIXES = ("ollama/", "ollama_chat/")


def is_claude_model(model: str) -> bool:
    return model.startswith(_CLAUDE_PREFIXES)


def is_ollama_model(model: str) -> bool:
    return model.startswith(_OLLAMA_PREFIXES)
