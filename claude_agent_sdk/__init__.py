# Stub package for claude_agent_sdk to allow imports without the full SDK installed.
# Used for testing purposes only.

from dataclasses import dataclass
from typing import Any


class TextBlock:
    text: str


class ThinkingBlock:
    signature: str
    thinking: str


class ToolUseBlock:
    name: str
    input: dict


class ToolResultBlock:
    is_error: bool
    content: str


ContentBlock = Any


class UserMessage:
    content: Any


class AssistantMessage:
    content: list
    error: Any


class SystemMessage:
    subtype: str
    data: Any


class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    result: Any


@dataclass
class ClaudeAgentOptions:
    system_prompt: str = ""
    allowed_tools: list = None
    permission_mode: str = "default"
    cwd: str = None
    env: dict = None
    model: str = ""
    thinking: dict = None
    effort: str = "default"


class ClaudeSDKClient:
    def __init__(self, options=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt):
        pass

    async def receive_response(self):
        return iter([])
