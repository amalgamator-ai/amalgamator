import pytest

from auto_bug_fix.bug_fix_config import BugFixConfig


def test_config_has_all_required_fields(bug_cfg):
    expected_attrs = [
        "repo_path",
        "build_dir",
        "source_branch",
        "target_branch",
        "source_fix_commit",
        "bug_description",
        "issue_id",
        "allowed_modules",
        "disallowed_modules",
        "build_command",
        "test_command",
        "max_build_test_retries",
        "max_resolution_retries",
        "forward_patch_id_lookback",
        "allowlist_rename_expansion_cap",
    ]
    for attr in expected_attrs:
        assert hasattr(bug_cfg, attr), f"BugFixConfig missing attribute: {attr}"


def test_config_defaults(bug_cfg):
    assert bug_cfg.forward_patch_id_lookback == "12 months"
    assert bug_cfg.allowlist_rename_expansion_cap == 3
    assert bug_cfg.max_build_test_retries == 3
    assert bug_cfg.max_resolution_retries == 3


def test_allowed_modules_default_none(bug_cfg):
    assert bug_cfg.allowed_modules is None
