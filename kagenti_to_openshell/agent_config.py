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
    kagenti_source_dir: str = ""
    openshell_source_dir: str = ""
    kagenti_github_url: str = "https://github.com/kagenti/kagenti"
    openshell_github_url: str = "https://github.com/NVIDIA/openshell"

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

    # Build/test commands for OpenShell
    build_command: str = "cargo build"
    test_command: str = "cargo test"

    @classmethod
    def from_json(cls, path: str) -> "AgentConfig":
        with open(path) as f:
            data = json.load(f)

        errors = []

        kagenti_source_dir = data.get("kagenti_source_dir", "")
        if not kagenti_source_dir:
            errors.append('"kagenti_source_dir" is required.')
        elif not os.path.isdir(kagenti_source_dir):
            errors.append('"kagenti_source_dir" directory not found: {}'.format(
                kagenti_source_dir
            ))

        openshell_source_dir = data.get("openshell_source_dir", "")
        if not openshell_source_dir:
            errors.append('"openshell_source_dir" is required.')
        elif not os.path.isdir(openshell_source_dir):
            errors.append('"openshell_source_dir" directory not found: {}'.format(
                openshell_source_dir
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
            kagenti_source_dir=os.path.abspath(kagenti_source_dir),
            openshell_source_dir=os.path.abspath(openshell_source_dir),
            kagenti_github_url=data.get(
                "kagenti_github_url",
                "https://github.com/kagenti/kagenti",
            ),
            openshell_github_url=data.get(
                "openshell_github_url",
                "https://github.com/NVIDIA/openshell",
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
            build_command=data.get("build_command", "cargo build"),
            test_command=data.get("test_command", "cargo test"),
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
