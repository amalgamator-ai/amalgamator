import pytest
from auto_code_gen.claude_utils import CodeGenClaudeConfig


@pytest.fixture
def claude_cfg():
    return CodeGenClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
        thinking={"type": "adaptive"},
        effort="max",
        max_thinking_tokens=1048576,
    )

