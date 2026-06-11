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
    """Load YAML or JSON config file.

    Args:
        config_path: Path to .yaml, .yml, or .json file

    Returns:
        Parsed configuration dict

    Raises:
        ConfigError: If file doesn't exist, has wrong extension, or is invalid
    """
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


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration structure and required fields.

    Args:
        config: Configuration dict from YAML/JSON

    Raises:
        ConfigError: If required fields are missing or invalid
    """
    # Required top-level fields
    required = ["issue_id", "bug_description", "repository", "branches", "fix", "build"]
    missing = [f for f in required if f not in config]
    if missing:
        raise ConfigError(f"Missing required fields: {', '.join(missing)}")

    # Repository section
    repo = config.get("repository", {})
    if not repo.get("source_path"):
        raise ConfigError("repository.source_path is required")

    # Branches section
    branches = config.get("branches", {})
    if not branches.get("source"):
        raise ConfigError("branches.source is required")
    if not branches.get("target"):
        raise ConfigError("branches.target is required")

    # Fix section
    fix = config.get("fix", {})
    if not fix.get("commit"):
        raise ConfigError("fix.commit is required")

    # Build section
    build = config.get("build", {})
    if not build.get("command"):
        raise ConfigError("build.command is required")
    if not build.get("test_command"):
        raise ConfigError("build.test_command is required")


def expand_path(path: str) -> str:
    """Expand ~ and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))


def parse_bug_fix_config(config: dict[str, Any]) -> tuple[BugFixConfig, str, str]:
    """Parse YAML/JSON config into BugFixConfig dataclass.

    Args:
        config: Configuration dict from load_config_file()

    Returns:
        Tuple of (BugFixConfig, workdir_path, repo_path)

    Raises:
        ConfigError: If config is invalid
    """
    validate_config(config)

    # Paths
    source_path = expand_path(config["repository"]["source_path"])
    workdir = expand_path(config["repository"].get("workdir", "/tmp/bugfix-workdir"))
    repo_name = os.path.basename(source_path)
    repo_path = os.path.join(workdir, repo_name)

    build_subdir = config["repository"].get("build_subdir", "")
    if build_subdir:
        build_dir = os.path.join(repo_path, build_subdir)
    else:
        build_dir = repo_path

    # Advanced settings
    advanced = config.get("advanced", {})

    bug_fix_config = BugFixConfig(
        repo_path=repo_path,
        build_dir=build_dir,
        source_branch=config["branches"]["source"],
        target_branch=config["branches"]["target"],
        source_fix_commit=config["fix"]["commit"],
        bug_description=config["bug_description"],
        issue_id=config["issue_id"],
        build_command=config["build"]["command"],
        test_command=config["build"]["test_command"],
        max_build_test_retries=advanced.get("max_build_test_retries", 3),
        forward_patch_id_lookback=advanced.get("forward_patch_id_lookback", "12 months"),
        allowlist_rename_expansion_cap=advanced.get("allowlist_rename_expansion_cap", 3),
    )

    return bug_fix_config, workdir, source_path


def parse_claude_config(config: dict[str, Any], output_dir: str) -> ClaudeConfig:
    """Parse Claude configuration section.

    Args:
        config: Configuration dict from load_config_file()
        output_dir: Output directory for Claude's working directory

    Returns:
        ClaudeConfig dataclass
    """
    claude = config.get("claude", {})
    model = claude.get("model") or config.get("model") or "claude-sonnet-4-6"

    return ClaudeConfig(
        model=model,
        allowed_tools=claude.get("allowed_tools", ["Read", "Write", "Bash"]),
        perm_mode=claude.get("perm_mode", "acceptEdits"),
        cwd=output_dir,
    )


def load_pipeline_config(config_path: str) -> tuple[BugFixConfig, ClaudeConfig, str, str]:
    """Load and parse full pipeline configuration.

    Returns:
        Tuple of (BugFixConfig, ClaudeConfig, workdir, source_path)
    """
    config = load_config_file(config_path)
    output_dir = expand_path(config.get("output", {}).get("directory", "./runs"))

    bug_fix_config, workdir, source_path = parse_bug_fix_config(config)
    claude_config = parse_claude_config(config, output_dir)

    return bug_fix_config, claude_config, workdir, source_path
