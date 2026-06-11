"""Tests for auto_bug_fix.config_loader — YAML/JSON config parsing."""
import os
import tempfile
import pytest
from auto_bug_fix.config_loader import (
    load_config_file, validate_config, parse_bug_fix_config,
    parse_claude_config, get_prebuild_commands, get_output_dir,
    load_pipeline_config, ConfigError, expand_path,
)


def test_load_yaml_config():
    """Test loading a valid YAML config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
issue_id: CVE-2024-1234
bug_description: Test CVE
repository:
  source_path: ~/test-repo
  workdir: /tmp/workdir
  build_subdir: src
branches:
  source: main
  target: v1.0
fix:
  commit: abc123
build:
  command: make
  test_command: make test
""")
        f.flush()
        config = load_config_file(f.name)

    assert config["issue_id"] == "CVE-2024-1234"
    assert config["repository"]["source_path"] == "~/test-repo"
    assert config["branches"]["source"] == "main"
    os.unlink(f.name)


def test_load_json_config():
    """Test loading a valid JSON config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("""{
  "issue_id": "CVE-2024-5678",
  "bug_description": "Test CVE",
  "repository": {
    "source_path": "~/test-repo"
  },
  "branches": {
    "source": "main",
    "target": "v1.0"
  },
  "fix": {
    "commit": "def456"
  },
  "build": {
    "command": "make",
    "test_command": "make test"
  }
}""")
        f.flush()
        config = load_config_file(f.name)

    assert config["issue_id"] == "CVE-2024-5678"
    assert config["fix"]["commit"] == "def456"
    os.unlink(f.name)


def test_load_config_file_not_found():
    """Test loading a non-existent config file."""
    with pytest.raises(ConfigError, match="not found"):
        load_config_file("/nonexistent/config.yaml")


def test_load_config_invalid_extension():
    """Test loading a config with invalid extension."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("issue_id: test")
        f.flush()
        with pytest.raises(ConfigError, match="Unsupported config format"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_validate_config_missing_fields():
    """Test validation catches missing required fields."""
    config = {
        "issue_id": "CVE-2024-1234",
        # Missing bug_description, repository, etc.
    }
    with pytest.raises(ConfigError, match="Missing required fields"):
        validate_config(config)


def test_validate_config_missing_nested():
    """Test validation catches missing nested fields."""
    config = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test",
        "repository": {},  # Missing source_path
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc"},
        "build": {"command": "make", "test_command": "make test"},
    }
    with pytest.raises(ConfigError, match="repository.source_path is required"):
        validate_config(config)


def test_expand_path():
    """Test path expansion (~ and env vars)."""
    os.environ["TEST_VAR"] = "/test/path"
    assert expand_path("~/foo") == os.path.expanduser("~/foo")
    assert expand_path("$TEST_VAR/bar") == "/test/path/bar"


def test_parse_bug_fix_config():
    """Test parsing config dict into BugFixConfig."""
    config = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test CVE: buffer overflow",
        "repository": {
            "source_path": "/tmp/test-repo",
            "workdir": "/tmp/workdir",
            "build_subdir": "src",
        },
        "branches": {
            "source": "main",
            "target": "v1.0",
        },
        "fix": {
            "commit": "abc123def",
        },
        "build": {
            "command": "make -j4",
            "test_command": "make test",
        },
        "advanced": {
            "max_build_test_retries": 5,
            "bisect_max_commits": 100,
        },
    }

    bug_fix_config, workdir, source_path = parse_bug_fix_config(config)

    assert bug_fix_config.issue_id == "CVE-2024-1234"
    assert bug_fix_config.bug_description == "Test CVE: buffer overflow"
    assert bug_fix_config.source_branch == "main"
    assert bug_fix_config.target_branch == "v1.0"
    assert bug_fix_config.source_fix_commit == "abc123def"
    assert bug_fix_config.build_command == "make -j4"
    assert bug_fix_config.test_command == "make test"
    assert bug_fix_config.max_build_test_retries == 5
    assert bug_fix_config.bisect_max_commits == 100
    assert workdir == "/tmp/workdir"
    assert source_path == "/tmp/test-repo"
    assert bug_fix_config.repo_path == "/tmp/workdir/test-repo"
    assert bug_fix_config.build_dir == "/tmp/workdir/test-repo/src"


def test_parse_claude_config():
    """Test parsing Claude configuration."""
    config = {
        "claude": {
            "model": "claude-opus-4-7",
            "allowed_tools": ["Read", "Bash"],
            "perm_mode": "interactive",
        },
    }

    claude_config = parse_claude_config(config, "/tmp/output")

    assert claude_config.model == "claude-opus-4-7"
    assert claude_config.allowed_tools == ["Read", "Bash"]
    assert claude_config.perm_mode == "interactive"
    assert claude_config.cwd == "/tmp/output"


def test_parse_claude_config_defaults():
    """Test Claude config falls back to defaults."""
    config = {}  # No claude section

    claude_config = parse_claude_config(config, "/tmp/output")

    assert claude_config.model == "claude-sonnet-4-6"
    assert claude_config.allowed_tools == ["Read", "Write", "Bash"]
    assert claude_config.perm_mode == "acceptEdits"


def test_get_prebuild_commands():
    """Test extracting prebuild commands."""
    config = {
        "build": {
            "prebuild_commands": ["./autogen.sh", "./configure --enable-debug"],
        },
    }

    commands = get_prebuild_commands(config)
    assert commands == ["./autogen.sh", "./configure --enable-debug"]


def test_get_prebuild_commands_empty():
    """Test prebuild commands defaults to empty list."""
    config = {"build": {}}
    assert get_prebuild_commands(config) == []


def test_get_output_dir():
    """Test getting output directory."""
    config = {"output": {"directory": "/custom/output"}}
    assert get_output_dir(config) == "/custom/output"


def test_get_output_dir_default():
    """Test output directory defaults to ./runs."""
    config = {}
    assert get_output_dir(config) == "./runs"
