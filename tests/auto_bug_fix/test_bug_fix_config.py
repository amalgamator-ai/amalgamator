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
        "port_tests",
        "build_command",
        "test_command",
        "max_build_test_retries",
        "require_unanimity",
        "allow_abstention",
        "bisect_max_commits",
        "forward_patch_id_lookback",
        "manifest_patch_command",
        "worktree_root",
        "use_per_branch_worktrees",
        "use_per_cve_worktrees",
        "rerere_enabled",
        "allowlist_rename_expansion_cap",
    ]
    for attr in expected_attrs:
        assert hasattr(bug_cfg, attr), f"BugFixConfig missing attribute: {attr}"


def test_config_defaults(bug_cfg):
    assert bug_cfg.require_unanimity is True
    assert bug_cfg.allow_abstention is True
    assert bug_cfg.bisect_max_commits == 200
    assert bug_cfg.forward_patch_id_lookback == "12 months"
    assert bug_cfg.allowlist_rename_expansion_cap == 3


def test_rerere_requires_per_cve_worktrees():
    with pytest.raises(ValueError, match="rerere_enabled=True requires use_per_cve_worktrees=True"):
        BugFixConfig(
            repo_path="/path/to/gcc",
            build_dir="/path/to/gcc/build",
            source_branch="gcc-14-branch",
            target_branch="gcc-13-branch",
            source_fix_commit="abc1234",
            bug_description="test",
            issue_id="GH-1",
            rerere_enabled=True,
            use_per_cve_worktrees=False,
        )


def test_rerere_valid_with_per_cve_worktrees():
    config = BugFixConfig(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="test",
        issue_id="GH-1",
        rerere_enabled=True,
        use_per_cve_worktrees=True,
    )
    assert config.rerere_enabled is True
    assert config.use_per_cve_worktrees is True


def test_allowed_modules_default_none(bug_cfg):
    assert bug_cfg.allowed_modules is None
