"""Tests for auto_bug_fix.config_loader — YAML/JSON config parsing."""
import os
import tempfile
import pytest
from auto_bug_fix.config_loader import (
    load_config_file, validate_config, parse_bug_fix_config,
    parse_claude_config,
    load_pipeline_config, ConfigError, expand_path,
    TargetSpec, _sanitize_ref, parse_targets, load_multi_target_config,
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


# --- timeout_minutes -> command_timeout (seconds) ---

def _timeout_config(advanced):
    return {
        "issue_id": "CVE-2024-0000",
        "bug_description": "Test",
        "repository": {"source_path": "/tmp/test-repo", "workdir": "/tmp/workdir"},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc123"},
        "build": {"command": "make", "test_command": "make test"},
        "advanced": advanced,
    }


def test_timeout_minutes_unset_means_no_limit():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({}))
    assert cfg.command_timeout is None


def test_timeout_minutes_zero_means_no_limit():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": 0}))
    assert cfg.command_timeout is None


def test_timeout_minutes_positive_to_seconds():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": 5}))
    assert cfg.command_timeout == 300


def test_timeout_minutes_fractional():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": 1.5}))
    assert cfg.command_timeout == 90


def test_timeout_minutes_negative_means_no_limit():
    cfg, _, _ = parse_bug_fix_config(_timeout_config({"timeout_minutes": -3}))
    assert cfg.command_timeout is None


# --- _sanitize_ref ---

def test_sanitize_ref_strips_origin():
    assert _sanitize_ref("origin/release-1.34") == "release-1.34"


def test_sanitize_ref_replaces_slashes():
    assert _sanitize_ref("feature/deep/path") == "feature_deep_path"


def test_sanitize_ref_sha_passthrough():
    assert _sanitize_ref("cc8e8609e") == "cc8e8609e"


def test_sanitize_ref_origin_and_slashes():
    assert _sanitize_ref("origin/release/1.34") == "release_1.34"


# --- parse_targets ---

def _base_config(**overrides):
    """Minimal valid config dict for parse_targets tests."""
    cfg = {
        "issue_id": "CVE-2024-1234",
        "bug_description": "Test CVE",
        "repository": {"source_path": "/tmp/test-repo"},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc123"},
        "build": {"command": "make", "test_command": "make test"},
    }
    cfg.update(overrides)
    return cfg


def test_parse_targets_list_of_strings():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = ["cc8e8609e", "b9ac8350f"]
    specs = parse_targets(config)
    assert len(specs) == 2
    assert specs[0].branch == "cc8e8609e"
    assert specs[1].branch == "b9ac8350f"
    assert specs[0].issue_id is None
    assert specs[0].build_command is None


def test_parse_targets_list_of_mappings():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = [
        {"branch": "cc8e8609e", "issue_id": "fix-to-1.34"},
        {"branch": "b9ac8350f", "issue_id": "fix-to-1.33", "build_command": "make -j4"},
    ]
    specs = parse_targets(config)
    assert len(specs) == 2
    assert specs[0].issue_id == "fix-to-1.34"
    assert specs[0].build_command is None
    assert specs[1].issue_id == "fix-to-1.33"
    assert specs[1].build_command == "make -j4"


def test_parse_targets_mixed():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = [
        "cc8e8609e",
        {"branch": "b9ac8350f", "issue_id": "fix-to-1.33"},
    ]
    specs = parse_targets(config)
    assert len(specs) == 2
    assert specs[0].branch == "cc8e8609e"
    assert specs[0].issue_id is None
    assert specs[1].branch == "b9ac8350f"
    assert specs[1].issue_id == "fix-to-1.33"


def test_parse_targets_singular_fallback():
    config = _base_config()
    specs = parse_targets(config)
    assert len(specs) == 1
    assert specs[0].branch == "v1.0"
    assert specs[0].issue_id is None


def test_parse_targets_rejects_both():
    config = _base_config()
    config["branches"]["targets"] = ["cc8e8609e"]
    with pytest.raises(ConfigError, match="not both"):
        parse_targets(config)


def test_parse_targets_requires_one():
    config = _base_config()
    del config["branches"]["target"]
    with pytest.raises(ConfigError, match="must have 'target' or 'targets'"):
        parse_targets(config)


def test_parse_targets_rejects_duplicate_branches():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = ["origin/release-1.34", "release-1.34"]
    with pytest.raises(ConfigError, match="duplicate target after sanitization"):
        parse_targets(config)


def test_parse_targets_rejects_empty_list():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = []
    with pytest.raises(ConfigError, match="non-empty list"):
        parse_targets(config)


def test_parse_targets_rejects_mapping_without_branch():
    config = _base_config()
    del config["branches"]["target"]
    config["branches"]["targets"] = [{"issue_id": "no-branch"}]
    with pytest.raises(ConfigError, match="must have a 'branch' key"):
        parse_targets(config)


# --- load_multi_target_config ---

def test_load_multi_target_config_injects_synthetic_target():
    """Verify parse_bug_fix_config doesn't KeyError on targets-only YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
issue_id: CVE-2024-1234
bug_description: Test CVE
repository:
  source_path: /tmp/test-repo
  workdir: /tmp/workdir
branches:
  source: main
  targets:
    - branch: "v1.0"
      issue_id: "fix-to-v1.0"
    - "v2.0"
fix:
  commit: abc123
build:
  command: make
  test_command: make test
""")
        f.flush()
        targets, base_config, claude_config, workdir, source_path = load_multi_target_config(f.name)

    os.unlink(f.name)
    assert len(targets) == 2
    assert targets[0].branch == "v1.0"
    assert targets[0].issue_id == "fix-to-v1.0"
    assert targets[1].branch == "v2.0"
    assert targets[1].issue_id is None
    assert base_config.target_branch == "v1.0"
    assert base_config.issue_id == "CVE-2024-1234"
    assert workdir == "/tmp/workdir"


def test_load_multi_target_config_singular_target():
    """Singular target YAML still works via load_multi_target_config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
issue_id: CVE-2024-5678
bug_description: Test CVE
repository:
  source_path: /tmp/test-repo
branches:
  source: main
  target: v1.0
fix:
  commit: def456
build:
  command: make
  test_command: make test
""")
        f.flush()
        targets, base_config, _, _, _ = load_multi_target_config(f.name)

    os.unlink(f.name)
    assert len(targets) == 1
    assert targets[0].branch == "v1.0"
    assert base_config.target_branch == "v1.0"


# ---------- advanced_float: tolerant numeric parsing ----------

def _cfg_with_advanced(advanced: dict) -> dict:
    return {
        "issue_id": "t-1",
        "bug_description": "d",
        "repository": {"source_path": "/tmp/repo"},
        "branches": {"source": "main", "target": "v1.0"},
        "fix": {"commit": "abc123"},
        "build": {},
        "advanced": advanced,
    }


def test_advanced_float_accepts_quoted_string():
    """YAML quoting must not silently disable a configured threshold."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": "0.7"}))
    assert cfg.partial_success_skip_ratio == 0.7


def test_advanced_float_accepts_number():
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": 0.8}))
    assert cfg.partial_success_skip_ratio == 0.8


def test_advanced_float_null_falls_back_without_crashing(caplog):
    """A bare `partial_success_skip_ratio:` key yields None — must not raise."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": None}))
    assert cfg.partial_success_skip_ratio == 0.5
    assert "not a number" in caplog.text


def test_advanced_float_garbage_warns_and_defaults(caplog):
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": "high"}))
    assert cfg.partial_success_skip_ratio == 0.5
    assert "not a number" in caplog.text


def test_advanced_float_absent_is_silent(caplog):
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.partial_success_skip_ratio == 0.5
    assert "not a number" not in caplog.text


# --- advanced_float bounds ---
# The ratio gates a downgrade to partial_success. A value outside [0, 1] does
# not fail loudly, it silently disables (or inverts) that gate, so it must be
# rejected at load time.

@pytest.mark.parametrize("bad", [-0.5, 2, 1.0001, float("nan"),
                                 float("inf"), float("-inf")])
def test_skip_ratio_outside_unit_interval_falls_back(bad, caplog):
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": bad}))
    assert cfg.partial_success_skip_ratio == 0.5


def test_skip_ratio_nan_does_not_silently_disable_the_gate():
    """float('nan') makes every `ratio >= threshold` comparison false."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": float("nan")}))
    assert 1.0 >= cfg.partial_success_skip_ratio


def test_skip_ratio_above_one_cannot_suppress_a_total_drop():
    """A ratio of 2 would let a 100%-dropped port still report success."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": 2}))
    assert 1.0 >= cfg.partial_success_skip_ratio


@pytest.mark.parametrize("flag", [True, False])
def test_skip_ratio_boolean_is_rejected(flag, caplog):
    """float(True) is 1.0 — a config saying `true` never meant that."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": flag}))
    assert cfg.partial_success_skip_ratio == 0.5
    assert "boolean" in caplog.text


@pytest.mark.parametrize("good,expected", [(0, 0.0), (1, 1.0), ("0.75", 0.75), (0.25, 0.25)])
def test_skip_ratio_accepts_the_whole_unit_interval(good, expected):
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"partial_success_skip_ratio": good}))
    assert cfg.partial_success_skip_ratio == expected


# --- phase_4b_model ---
# Phase 4b is the last resort before escalating to a human and deliberately
# runs a stronger model than the rest of the pipeline. Making that configurable
# must not change the default, and must not let an unusable value through.

def test_phase_4b_model_defaults_to_opus():
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({}))
    assert cfg.phase_4b_model == "claude-opus-4-6"


def test_phase_4b_model_is_independent_of_the_pipeline_model():
    """Setting `model:` must not silently move the escalation model with it."""
    config = _cfg_with_advanced({})
    config["model"] = "claude-sonnet-4-6"
    cfg, _, _ = parse_bug_fix_config(config)
    assert cfg.phase_4b_model == "claude-opus-4-6"


def test_phase_4b_model_override_is_honoured():
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"phase_4b_model": "claude-opus-5"}))
    assert cfg.phase_4b_model == "claude-opus-5"


def test_phase_4b_model_is_stripped():
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced(
        {"phase_4b_model": "  claude-opus-5  "}))
    assert cfg.phase_4b_model == "claude-opus-5"


@pytest.mark.parametrize("bad", ["", "   ", None, 42, ["claude-opus-5"], True])
def test_phase_4b_model_unusable_value_falls_back(bad, caplog):
    """An empty model name would surface as an opaque API error inside 4b."""
    cfg, _, _ = parse_bug_fix_config(_cfg_with_advanced({"phase_4b_model": bad}))
    assert cfg.phase_4b_model == "claude-opus-4-6"
    assert "non-empty string" in caplog.text
