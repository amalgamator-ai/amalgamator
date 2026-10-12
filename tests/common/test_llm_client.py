"""Compatibility surface of common.llm_client.

The module docstring documents ``is_claude_model()`` as part of its public
surface (callers use it to choose between the Claude Agent SDK and the
generic LiteLLM loop), so the re-export must survive refactors of the
module's own imports.
"""

from common.llm_client import is_claude_model


def test_documented_compat_reexport():
    assert is_claude_model("claude-opus-4-6")
    assert is_claude_model("anthropic/claude-sonnet-4-6")
    assert is_claude_model("vertex_ai/claude-opus-4-6")
    assert not is_claude_model("gpt-4o")
    assert not is_claude_model("ollama/qwen3:14b")
