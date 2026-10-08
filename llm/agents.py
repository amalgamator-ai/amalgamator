"""Public facade for provider-neutral agent execution in the LLM subtree.

The implementation lives at ``llm.auto_code_gen.agents``; consumers inside
the LLM pipelines should import it from here so the backing module can move
without touching every call site.
"""

from llm.auto_code_gen.agents import (
    AgentConfig,
    add_agent_argument,
    agent_run,
    agent_session,
    resolve_agent_config,
)

__all__ = [
    "AgentConfig",
    "add_agent_argument",
    "agent_run",
    "agent_session",
    "resolve_agent_config",
]

