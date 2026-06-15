"""Tests for auto_bug_fix.config_loader — YAML/JSON config parsing."""
import os
import tempfile
import pytest
from auto_bug_fix.config_loader import (
    load_config_file, validate_config, parse_bug_fix_config,
    parse_claude_config,
    load_pipeline_config, ConfigError, expand_path,
)


def test_load_yaml_config():
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
    with pytest.raises(ConfigError, match="not found"):
        load_config_file("/nonexistent/config.yaml")


def test_load_config_invalid_extension():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("issue_id: test")
        f.flush()
        with pytest.raises(ConfigError, match="Unsupported config format"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_load_config_empty_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        f.flush()
        with pytest.raises(ConfigError, match="must contain a YAML/JSON mapping"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_load_config_non_mapping_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("- item1\n- item2\n")
        f.flush()
        with pytest.raises(ConfigError, match="must contain a YAML/JSON mapping"):
            load_config_file(f.name)
    os.unlink(f.name)


def test_validate_config_missing_fields():
    config = {
        "issue_id": "CVE-2024-1234",
    }
    with pytest.raises(ConfigError, match="Missing required fields"):
        validate_config(config)


def test_validate_config_missing_nested():
    config = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test",
        "repository": {},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc"},
        "build": {"command": "make", "test_command": "make test"},
    }
    with pytest.raises(ConfigError, match="repository.source_path is required"):
        validate_config(config)


def test_expand_path():
    os.environ["TEST_VAR"] = "/test/path"
    assert expand_path("~/foo") == os.path.expanduser("~/foo")
    assert expand_path("$TEST_VAR/bar") == "/test/path/bar"


def test_parse_bug_fix_config():
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
    assert workdir == "/tmp/workdir"
    assert source_path == "/tmp/test-repo"
    assert bug_fix_config.repo_path == "/tmp/workdir/test-repo"
    assert bug_fix_config.build_dir == "/tmp/workdir/test-repo/src"


def test_parse_claude_config():
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
    config = {}

    claude_config = parse_claude_config(config, "/tmp/output")

    assert claude_config.model == "claude-sonnet-4-6"
    assert claude_config.allowed_tools == ["Read", "Write", "Bash"]
    assert claude_config.perm_mode == "acceptEdits"


def test_parse_claude_config_top_level_model():
    config = {"model": "claude-opus-4-6"}
    claude_config = parse_claude_config(config, "/tmp/output")
    assert claude_config.model == "claude-opus-4-6"
