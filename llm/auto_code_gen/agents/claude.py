"""Adapter preserving the target's native Claude behavior."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, fields


def native_config(config):
    """Map an AgentConfig onto the common ClaudeConfig.

    Copies only the fields the two config types share: the target's
    ClaudeConfig carries profile/transport fields AgentConfig does not (and
    vice versa), and the SDK path must keep the persona the pipeline has
    always run with.

    ``command_timeout`` is translated: the LiteLLM loop's 1800s default is a
    transport-loop guard, while the Claude SDK path has always meant "no
    timeout" (None). The SDK path ignores the field, so an explicitly
    configured timeout passes through unchanged.
    """
    from common.claude_utils import ClaudeConfig

    shared = {f.name for f in fields(ClaudeConfig)} & {
        f.name for f in fields(type(config))
    }
    values = {name: getattr(config, name) for name in shared}
    if values.get("command_timeout") == 1800:
        values["command_timeout"] = None
    return ClaudeConfig(**values)


@dataclass
class ClaudeSession:
    config: object
    client: object


@asynccontextmanager
async def session(config):
    from common.claude_utils import claude_session

    async with claude_session(native_config(config)) as client:
        yield ClaudeSession(config, client)


async def run(config, steps, resume=False, client=None):
    from common.claude_utils import claude_run

    if isinstance(client, ClaudeSession):
        if client.config != config:
            raise ValueError("Agent session configuration does not match the run")
        client = client.client
    return await claude_run(native_config(config), steps, resume=resume, client=client)
