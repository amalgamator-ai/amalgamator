"""Bug-fix specialization of the shared Claude Agent SDK glue.

Everything mechanical — the agent runner, resume handling, message printing,
token accounting — lives in ``common.claude_utils``. This module adds only
what is specific to backporting: the agent persona.

Re-exports the common types so callers inside ``auto_bug_fix`` have a single
import site.
"""

from dataclasses import dataclass

from common.claude_utils import (
    ClaudeConfig,
    PipelineStep,
    claude_run,
    print_claude_config,
)

BUG_FIX_SYSTEM_PROMPT = (
    "You are an expert software engineer specializing in backporting bug "
    "fixes between branches of large codebases: git history analysis, "
    "cherry-picking, merge conflict resolution, and build/test verification."
)


@dataclass
class BugFixClaudeConfig(ClaudeConfig):
    """Persona for the git-first backport pipeline.

    Construct this rather than a bare :class:`ClaudeConfig` anywhere in
    ``auto_bug_fix`` — the persona then travels with the config instead of
    having to be remembered at each of the pipeline's construction sites.
    """
    system_prompt: str = BUG_FIX_SYSTEM_PROMPT


__all__ = [
    "BUG_FIX_SYSTEM_PROMPT",
    "BugFixClaudeConfig",
    "ClaudeConfig",
    "PipelineStep",
    "claude_run",
    "print_claude_config",
]
