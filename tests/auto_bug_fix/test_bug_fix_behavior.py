"""BDD behavioral tests for the bug fix porting pipeline.

Each test is named as a scenario: Given / When / Then.
Tests verify pipeline-level behavior without running Claude.
"""
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import gen_prompts
from auto_bug_fix.bug_fix_prompts import RUN_AND_FIX_FAILURE_FILE, TEST_PORT_MANIFEST_FILE


def _claude_cfg():
    return ClaudeConfig(
        model="claude-opus-4-6",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd="/tmp/test_cwd",
    )


def _config(**overrides):
    defaults = dict(
        repo_path="/path/to/gcc",
        build_dir="/path/to/gcc/build",
        source_branch="gcc-14-branch",
        target_branch="gcc-13-branch",
        source_fix_commit="abc1234",
        bug_description="CVE-2024-1234: test bug",
        issue_id="CVE-2024-1234",
        disallowed_modules=[],
        port_tests=True,
        build_command="make -j$(nproc)",
        test_command="make check -j$(nproc)",
        max_build_test_retries=3,
    )
    defaults.update(overrides)
    return BugFixConfig(**defaults)


def test_scenario_repo_is_reset_before_any_prompts():
    """
    Given a BugFixConfig,
    When the pipeline generates its prompt list,
    Then the first entry must be a clear_repo shell command.
    """
    prompts = gen_prompts(_claude_cfg(), _config())
    assert isinstance(prompts[0], dict), "First step must be a shell command"
    assert "clear_repo" in prompts[0]["cmd"]


def test_scenario_test_port_precedes_code_gen_when_enabled():
    """
    Given port_tests=True,
    When the pipeline generates its prompt list,
    Then TestPortPrompt (git show <sha>) must appear before any CodeGenPrompt.

    Note: check for the formatted SHA (e.g. "git show abc1234"), not just "git show",
    because CodeTracePrompt also contains "git show <source_fix_commit>" (angle-bracket,
    unformatted) which would give a false positive.
    """
    config = _config(port_tests=True)
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    commit_sha = config.source_fix_commit  # "abc1234"
    test_port_index = next(i for i, p in enumerate(string_prompts) if "git show {}".format(commit_sha) in p)
    code_gen_index = next(i for i, p in enumerate(string_prompts) if "generate a code patch" in p.lower())
    assert test_port_index < code_gen_index


def test_scenario_test_extraction_skipped_when_port_tests_false():
    """
    Given port_tests=False,
    When the pipeline generates its prompt list,
    Then no TestPortPrompt (git show <sha>) should appear.
    """
    config = _config(port_tests=False)
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    commit_sha = config.source_fix_commit  # "abc1234"
    assert not any("git show {}".format(commit_sha) in p for p in string_prompts)


def test_scenario_run_and_fix_is_final_step():
    """
    Given a fully configured BugFixConfig,
    When the pipeline generates its prompt list,
    Then RunAndFixPrompt must be the last string prompt.
    """
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert RUN_AND_FIX_FAILURE_FILE in string_prompts[-1]


def test_scenario_ported_tests_referenced_in_code_gen():
    """
    Given port_tests=True,
    When the pipeline generates its prompt list,
    Then CodeGenPrompt must reference test_port_manifest.txt.
    """
    prompts = gen_prompts(_claude_cfg(), _config(port_tests=True))
    string_prompts = [p for p in prompts if isinstance(p, str)]
    code_gen_prompts = [p for p in string_prompts if "generate a code patch" in p.lower()]
    assert any(TEST_PORT_MANIFEST_FILE in p for p in code_gen_prompts)


def test_scenario_config_values_flow_into_run_and_fix():
    """
    Given config with specific build and test commands,
    When the pipeline generates its prompt list,
    Then RunAndFixPrompt must embed those exact commands.
    """
    config = _config(
        build_command="cmake --build build -j$(nproc)",
        test_command="ctest --output-on-failure",
    )
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    run_and_fix = string_prompts[-1]
    assert "cmake --build build -j$(nproc)" in run_and_fix
    assert "ctest --output-on-failure" in run_and_fix
