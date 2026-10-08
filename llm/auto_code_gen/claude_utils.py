"""Code-generation specialization of the shared Claude Agent SDK glue.

Everything mechanical — the agent runner, resume handling, message printing,
token accounting — lives in ``common.claude_utils``. This module adds only
the choice of agent persona.

Re-exports the common types so callers inside ``auto_code_gen`` have a single
import site.
"""

from dataclasses import dataclass

from common.claude_utils import (
    PROFILING_SYSTEM_PROMPT,
    ClaudeConfig,
    PipelineStep,
    _all_steps_complete,
    agent_run,
    claude_run,
    print_claude_config,
)


@dataclass
class CodeGenClaudeConfig(ClaudeConfig):
    """Persona for the AI code generation pipeline.

    Deliberately the *profiling* persona, not a code-porting one. Before the
    persona became configurable, ``claude_run`` hardcoded
    ``profile_system_prompt()`` for every caller, so this is what the code gen
    pipeline has always run with. Its prompts and expected outputs are tuned
    against that persona, so it stays pinned.

    Changing it is a one-line edit here, but it is a behaviour change to the
    pipeline and should be validated on a real run, not assumed.
    """
    system_prompt: str = PROFILING_SYSTEM_PROMPT


__all__ = [
    "CodeGenClaudeConfig",
    "ClaudeConfig",
    "PipelineStep",
    "_all_steps_complete",
    "agent_run",
    "claude_run",
    "print_claude_config",
]
