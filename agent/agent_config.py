import json
import os
from dataclasses import dataclass, field

from common.claude_utils import ClaudeConfig


_MODE_PARAMS = {
    "deep": {
        "model": "claude-opus-4-6[1m]",
        "thinking": {"type": "adaptive"},
        "effort": "max",
        "max_thinking_tokens": 1048576,
    },
    "normal": {
        "model": "claude-sonnet-4-6",
        "thinking": {"type": "adaptive"},
        "effort": "medium",
        "max_thinking_tokens": 65536,
    },
}


@dataclass
class AgentConfig:
    # Source and target repos
    hermes_source_dir: str = ""
    openclaw_source_dir: str = ""
    hermes_github_url: str = "https://github.com/nousresearch/hermes-agent"
    openclaw_github_url: str = "https://github.com/openclaw/openclaw"

    # Output
    output_dir: str = ""

    # Pipeline iteration counts
    num_code_port_plan_iterations: int = 3
    num_test_plan_iterations: int = 3
    num_code_gen_iterations: int = 3
    num_runtime_iterations: int = 5

    # Review skipping
    code_port_plan_skip_review: bool = False
    test_plan_skip_review: bool = False
    code_gen_skip_review: bool = False

    # Thinking mode
    thinking_mode: str = "deep"

    # Feature detection
    max_features_to_detect: int = 10
    feature_id: int = 0

    # Build/test commands for OpenClaw
    build_command: str = "pnpm install && pnpm build"
    test_command: str = "pnpm test"

    @classmethod
    def from_json(cls, path: str) -> "AgentConfig":
        with open(path) as f:
            data = json.load(f)

        errors = []

        hermes_source_dir = data.get("hermes_source_dir", "")
        if not hermes_source_dir:
            errors.append('"hermes_source_dir" is required.')
        elif not os.path.isdir(hermes_source_dir):
            errors.append('"hermes_source_dir" directory not found: {}'.format(
                hermes_source_dir
            ))

        openclaw_source_dir = data.get("openclaw_source_dir", "")
        if not openclaw_source_dir:
            errors.append('"openclaw_source_dir" is required.')
        elif not os.path.isdir(openclaw_source_dir):
            errors.append('"openclaw_source_dir" directory not found: {}'.format(
                openclaw_source_dir
            ))

        output_dir = data.get("output_dir", "")
        if not output_dir:
            errors.append('"output_dir" is required.')

        thinking_mode = data.get("thinking_mode", "deep")
        if thinking_mode not in ("normal", "deep"):
            errors.append('"thinking_mode" must be "normal" or "deep".')

        if errors:
            raise ValueError(
                "Agent config errors:\n" + "\n".join("  - {}".format(e) for e in errors)
            )

        return cls(
            hermes_source_dir=os.path.abspath(hermes_source_dir),
            openclaw_source_dir=os.path.abspath(openclaw_source_dir),
            hermes_github_url=data.get(
                "hermes_github_url",
                "https://github.com/nousresearch/hermes-agent",
            ),
            openclaw_github_url=data.get(
                "openclaw_github_url",
                "https://github.com/openclaw/openclaw",
            ),
            output_dir=os.path.abspath(output_dir),
            num_code_port_plan_iterations=data.get("num_code_port_plan_iterations", 3),
            num_test_plan_iterations=data.get("num_test_plan_iterations", 3),
            num_code_gen_iterations=data.get("num_code_gen_iterations", 3),
            num_runtime_iterations=data.get("num_runtime_iterations", 5),
            code_port_plan_skip_review=data.get("code_port_plan_skip_review", False),
            test_plan_skip_review=data.get("test_plan_skip_review", False),
            code_gen_skip_review=data.get("code_gen_skip_review", False),
            thinking_mode=thinking_mode,
            max_features_to_detect=data.get("max_features_to_detect", 10),
            feature_id=data.get("feature_id", 0),
            build_command=data.get("build_command", "pnpm install && pnpm build"),
            test_command=data.get("test_command", "pnpm test"),
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
