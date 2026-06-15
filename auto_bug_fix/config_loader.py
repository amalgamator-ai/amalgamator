"""YAML/JSON configuration loader with validation for bug fix porting pipeline."""
from __future__ import annotations

import os
import yaml
import json
from pathlib import Path
from typing import Any

from auto_bug_fix.bug_fix_config import BugFixConfig
from common.claude_utils import ClaudeConfig


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load YAML or JSON config file."""
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()

    try:
        with open(path) as f:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ConfigError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_path}: {e}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML/JSON mapping, got {type(data).__name__}")
    return data


def _require_mapping(config: dict, key: str) -> dict:
    """Get a sub-section from config, raising ConfigError if it's not a mapping."""
    val = config.get(key, {})
    if not isinstance(val, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(val).__name__}")
    return val


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration structure and required fields."""
    required = ["issue_id", "bug_description", "repository", "branches", "fix", "build"]
    missing = [f for f in required if f not in config]
    if missing:
        raise ConfigError(f"Missing required fields: {', '.join(missing)}")

    repo = _require_mapping(config, "repository")
    if repo.get("source_path") is None:
        raise ConfigError("repository.source_path is required")

    branches = _require_mapping(config, "branches")
    if branches.get("source") is None:
        raise ConfigError("branches.source is required")
    if branches.get("target") is None:
        raise ConfigError("branches.target is required")

    fix = _require_mapping(config, "fix")
    if fix.get("commit") is None:
        raise ConfigError("fix.commit is required")

    build = _require_mapping(config, "build")
    if build.get("command") is None:
        raise ConfigError("build.command is required")
    if build.get("test_command") is None:
        raise ConfigError("build.test_command is required")


def expand_path(path: str | None, default: str = "") -> str:
    """Expand ~ and environment variables in path. Returns default for None."""
    if path is None:
        return default
    return os.path.expanduser(os.path.expandvars(path))


def parse_bug_fix_config(config: dict[str, Any]) -> tuple[BugFixConfig, str, str]:
    """Parse YAML/JSON config into BugFixConfig dataclass."""
    validate_config(config)

    source_path = os.path.normpath(expand_path(config["repository"]["source_path"]))
    workdir = expand_path(config["repository"].get("workdir"), "/tmp/bugfix-workdir")
    repo_name = os.path.basename(source_path)
    repo_path = os.path.join(workdir, repo_name)

    build_subdir = config["repository"].get("build_subdir", "")
    if build_subdir:
        build_dir = os.path.join(repo_path, build_subdir)
    else:
        build_dir = repo_path

    advanced = config.get("advanced", {})
    if not isinstance(advanced, dict):
        advanced = {}

    bug_fix_config = BugFixConfig(
        repo_path=repo_path,
        build_dir=build_dir,
        source_branch=str(config["branches"]["source"]),
        target_branch=str(config["branches"]["target"]),
        source_fix_commit=str(config["fix"]["commit"]),
        bug_description=str(config["bug_description"]),
        issue_id=str(config["issue_id"]),
        build_command=config["build"]["command"],
        test_command=config["build"]["test_command"],
        max_build_test_retries=advanced.get("max_build_test_retries", 3),
        max_resolution_retries=advanced.get("max_resolution_retries", 3),
        forward_patch_id_lookback=advanced.get("forward_patch_id_lookback", "12 months"),
        allowlist_rename_expansion_cap=advanced.get("allowlist_rename_expansion_cap", 3),
    )

    return bug_fix_config, workdir, source_path


def parse_claude_config(config: dict[str, Any], output_dir: str) -> ClaudeConfig:
    """Parse Claude configuration section."""
    claude = config.get("claude", {})
    if not isinstance(claude, dict):
        claude = {}
    model = claude.get("model") or config.get("model") or "claude-sonnet-4-6"

    return ClaudeConfig(
        model=model,
        allowed_tools=claude.get("allowed_tools", ["Read", "Write", "Bash"]),
        perm_mode=claude.get("perm_mode", "acceptEdits"),
        cwd=output_dir,
    )


def load_pipeline_config(config_path: str) -> tuple[BugFixConfig, ClaudeConfig, str, str]:
    """Load and parse full pipeline configuration."""
    config = load_config_file(config_path)
    output_section = config.get("output", {})
    if not isinstance(output_section, dict):
        output_section = {}
    output_dir = expand_path(output_section.get("directory"), "./runs")

    bug_fix_config, workdir, source_path = parse_bug_fix_config(config)
    claude_config = parse_claude_config(config, output_dir)

    return bug_fix_config, claude_config, workdir, source_path
