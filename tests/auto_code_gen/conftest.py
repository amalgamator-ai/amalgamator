import pytest
from common.claude_utils import ClaudeConfig


@pytest.fixture
def claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
        thinking={"type": "adaptive"},
        effort="max",
        max_thinking_tokens=1048576,
    )

