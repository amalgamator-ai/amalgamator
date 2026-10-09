"""Provider-neutral agent runtime shared by code generation and analysis."""
from contextlib import asynccontextmanager

from .config import AgentConfig, add_agent_argument, resolve_agent_config


@asynccontextmanager
async def agent_session(config):
    if getattr(config, "backend", "claude_sdk") == "claude_sdk":
        from .claude import session
        async with session(config) as client:
            yield client
    else:
        from .runner import LiteLLMSession
        yield LiteLLMSession(config)


async def agent_run(config, steps, resume=False, client=None):
    from .runner import LiteLLMSession
    if getattr(config, "backend", "claude_sdk") == "claude_sdk":
        if isinstance(client, LiteLLMSession):
            raise ValueError("A LiteLLM session cannot run a Claude SDK config")
        from .claude import run
        return await run(config, steps, resume=resume, client=client)
    if client is not None:
        if not isinstance(client, LiteLLMSession) or client.config != config:
            raise ValueError("Agent session configuration does not match the run")
        return await client.run(steps, resume=resume)
    async with agent_session(config) as session:
        return await session.run(steps, resume=resume)


__all__ = ["AgentConfig", "agent_run", "agent_session", "add_agent_argument", "resolve_agent_config"]
