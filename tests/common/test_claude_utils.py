"""Tests for the shared Claude Agent SDK glue and its per-package personas."""
import asyncio
import pytest

from common.claude_utils import (
    ClaudeConfig,
    PipelineStep,
    ProfilingClaudeConfig,
    DEFAULT_SYSTEM_PROMPT,
    PROFILING_SYSTEM_PROMPT,
    claude_run,
)
from auto_bug_fix.claude_utils import BugFixClaudeConfig, BUG_FIX_SYSTEM_PROMPT
from auto_code_gen.claude_utils import CodeGenClaudeConfig, CODE_GEN_SYSTEM_PROMPT


BASE_KWARGS = dict(
    model="claude-opus-4-6",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",
    cwd="/tmp",
)


class TestPersonas:
    """Each package extends the common config with its own system prompt."""

    @pytest.mark.parametrize("cls,expected", [
        (ClaudeConfig, DEFAULT_SYSTEM_PROMPT),
        (ProfilingClaudeConfig, PROFILING_SYSTEM_PROMPT),
        (BugFixClaudeConfig, BUG_FIX_SYSTEM_PROMPT),
        (CodeGenClaudeConfig, CODE_GEN_SYSTEM_PROMPT),
    ])
    def test_default_system_prompt(self, cls, expected):
        assert cls(**BASE_KWARGS).system_prompt == expected

    @pytest.mark.parametrize("cls", [
        ProfilingClaudeConfig, BugFixClaudeConfig, CodeGenClaudeConfig,
    ])
    def test_specializations_are_claude_configs(self, cls):
        assert issubclass(cls, ClaudeConfig)
        assert isinstance(cls(**BASE_KWARGS), ClaudeConfig)

    def test_explicit_prompt_overrides_the_persona(self):
        cfg = BugFixClaudeConfig(**BASE_KWARGS, system_prompt="custom")
        assert cfg.system_prompt == "custom"

    def test_personas_are_distinct(self):
        prompts = {
            DEFAULT_SYSTEM_PROMPT, PROFILING_SYSTEM_PROMPT,
            BUG_FIX_SYSTEM_PROMPT, CODE_GEN_SYSTEM_PROMPT,
        }
        assert len(prompts) == 4

    def test_specializations_inherit_common_defaults(self):
        """Subclassing must not disturb the fields it does not override."""
        base, bug_fix = ClaudeConfig(**BASE_KWARGS), BugFixClaudeConfig(**BASE_KWARGS)
        for attr in ("effort", "max_thinking_tokens", "thinking"):
            assert getattr(bug_fix, attr) == getattr(base, attr)


class _StubClient:
    """Stands in for ClaudeSDKClient; these steps never reach the model."""

    def __init__(self, options=None):
        self.options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def stub_sdk(monkeypatch):
    """Isolate claude_run from the SDK.

    tests/auto_bug_fix/conftest.py replaces claude_agent_sdk in sys.modules
    process-wide, so whether the real SDK is in scope here depends on
    collection order. Stub it explicitly instead of depending on that.
    """
    import common.claude_utils as cu
    captured = {}

    def _options(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(cu, "ClaudeAgentOptions", _options)
    monkeypatch.setattr(cu, "ClaudeSDKClient", _StubClient)
    return captured


class TestNonLLMSteps:
    """Only explicit callables are runnable as non-LLM steps."""

    def test_callable_step_runs(self, tmp_path, stub_sdk):
        calls = []
        cfg = ClaudeConfig(**{**BASE_KWARGS, "cwd": str(tmp_path)})
        step = PipelineStep(name="cb", prompt={"fn": lambda: calls.append(1)})
        asyncio.run(claude_run(cfg, [step]))
        assert calls == [1]

    def test_cmd_step_is_rejected(self, tmp_path, stub_sdk):
        """The former {"cmd": "<source>"} form was executed via eval()."""
        cfg = ClaudeConfig(**{**BASE_KWARGS, "cwd": str(tmp_path)})
        step = PipelineStep(name="danger", prompt={"cmd": "print('x')"})
        with pytest.raises(ValueError, match="only callable 'fn' steps"):
            asyncio.run(claude_run(cfg, [step]))

    def test_persona_reaches_the_agent_options(self, tmp_path, stub_sdk):
        """The config's system_prompt is what the SDK is actually given."""
        cfg = BugFixClaudeConfig(**{**BASE_KWARGS, "cwd": str(tmp_path)})
        asyncio.run(claude_run(cfg, [PipelineStep(name="noop", prompt={"fn": lambda: None})]))
        assert stub_sdk["system_prompt"] == BUG_FIX_SYSTEM_PROMPT
