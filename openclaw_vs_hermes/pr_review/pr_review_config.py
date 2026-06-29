import json
import os
from dataclasses import dataclass

from common.claude_utils import ClaudeConfig
from openclaw_vs_hermes.agent_config import _MODE_PARAMS


@dataclass
class PRReviewConfig:
    output_dir: str = ""
    openclaw_source_dir: str = ""
    openclaw_github_url: str = "https://github.com/openclaw/openclaw"
    pr_url: str = ""
    branch_name: str = ""
    build_command: str = "pnpm install && pnpm build"
    lint_command: str = ""
    test_command: str = "pnpm test"
    thinking_mode: str = "deep"
    max_fix_iterations: int = 3

    @classmethod
    def from_json(cls, path: str) -> "PRReviewConfig":
        with open(path) as f:
            data = json.load(f)

        errors = []

        output_dir = data.get("output_dir", "")
        if not output_dir:
            errors.append('"output_dir" is required.')
        elif not os.path.isdir(output_dir):
            errors.append('"output_dir" directory not found: {}'.format(output_dir))

        openclaw_source_dir = data.get("openclaw_source_dir", "")
        if not openclaw_source_dir:
            errors.append('"openclaw_source_dir" is required.')
        elif not os.path.isdir(openclaw_source_dir):
            errors.append('"openclaw_source_dir" directory not found: {}'.format(
                openclaw_source_dir
            ))

        pr_url = data.get("pr_url", "")
        if not pr_url:
            errors.append('"pr_url" is required.')

        branch_name = data.get("branch_name", "")
        if not branch_name:
            errors.append('"branch_name" is required.')

        thinking_mode = data.get("thinking_mode", "deep")
        if thinking_mode not in ("normal", "deep"):
            errors.append('"thinking_mode" must be "normal" or "deep".')

        if errors:
            raise ValueError(
                "PR review config errors:\n"
                + "\n".join("  - {}".format(e) for e in errors)
            )

        return cls(
            output_dir=os.path.abspath(output_dir),
            openclaw_source_dir=os.path.abspath(openclaw_source_dir),
            openclaw_github_url=data.get(
                "openclaw_github_url",
                "https://github.com/openclaw/openclaw",
            ),
            pr_url=pr_url,
            branch_name=branch_name,
            build_command=data.get("build_command", "pnpm install && pnpm build"),
            lint_command=data.get("lint_command", ""),
            test_command=data.get("test_command", "pnpm test"),
            thinking_mode=thinking_mode,
            max_fix_iterations=data.get("max_fix_iterations", 3),
        )

    def make_claude_config(self) -> ClaudeConfig:
        params = _MODE_PARAMS[self.thinking_mode]
        return ClaudeConfig(
            model=params["model"],
            allowed_tools=["Read", "Write", "Bash"],
            perm_mode="acceptEdits",
            cwd=self.output_dir,
            thinking=params["thinking"],
            effort=params["effort"],
            max_thinking_tokens=params["max_thinking_tokens"],
        )
