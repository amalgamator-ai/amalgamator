"""Code-generation specialization of the shared Claude Agent SDK glue.

Everything mechanical — the agent runner, resume handling, message printing,
token accounting — lives in ``common.claude_utils``. This module adds only
what is specific to code porting: the agent persona.

Re-exports the common types so callers inside ``auto_code_gen`` have a single
import site.
"""

from dataclasses import dataclass

from common.claude_utils import (
    ClaudeConfig,
    PipelineStep,
    _all_steps_complete,
    claude_run,
    print_claude_config,
)

CODE_GEN_SYSTEM_PROMPT = (
    "You are an expert software engineer specializing in porting code between "
    "large codebases: reading unfamiliar source trees, planning a port, "
    "writing the patch and its tests, and iterating until it builds and runs."
)


@dataclass
class CodeGenClaudeConfig(ClaudeConfig):
    """Persona for the AI code generation pipeline."""
    system_prompt: str = CODE_GEN_SYSTEM_PROMPT


__all__ = [
    "CODE_GEN_SYSTEM_PROMPT",
    "CodeGenClaudeConfig",
    "ClaudeConfig",
    "PipelineStep",
    "_all_steps_complete",
    "claude_run",
    "print_claude_config",
]
