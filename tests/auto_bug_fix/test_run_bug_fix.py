import pytest
from common.claude_utils import ClaudeConfig
from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.run_bug_fix import gen_prompts, NUM_PLAN_REVIEWS, NUM_CODE_REVIEWS
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


def test_gen_prompts_returns_list():
    prompts = gen_prompts(_claude_cfg(), _config())
    assert isinstance(prompts, list)
    assert len(prompts) > 0


def test_first_entry_is_clear_repo_cmd():
    prompts = gen_prompts(_claude_cfg(), _config())
    assert isinstance(prompts[0], dict)
    assert "clear_repo" in prompts[0]["cmd"]


def test_run_and_fix_is_last_string_prompt():
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert RUN_AND_FIX_FAILURE_FILE in string_prompts[-1]


def test_includes_git_show_when_port_tests_true():
    config = _config(port_tests=True)
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    # TestPortPrompt formats the commit SHA directly into the prompt
    assert any("git show {}".format(config.source_fix_commit) in p for p in string_prompts)


def test_excludes_git_show_when_port_tests_false():
    config = _config(port_tests=False)
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    # TestPortPrompt (which formats the commit SHA) should be absent
    assert not any("git show {}".format(config.source_fix_commit) in p for p in string_prompts)


def test_plan_review_iteration_count():
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    assert len([p for p in string_prompts if "high-level multi-step coding plan" in p]) >= NUM_PLAN_REVIEWS


def test_manifest_referenced_in_code_gen():
    prompts = gen_prompts(_claude_cfg(), _config())
    string_prompts = [p for p in prompts if isinstance(p, str)]
    code_gen_prompts = [p for p in string_prompts if "generate a code patch" in p.lower()]
    assert any(TEST_PORT_MANIFEST_FILE in p for p in code_gen_prompts)


def test_build_command_appears_in_run_and_fix():
    config = _config(
        build_command="cmake --build build -j$(nproc)",
        test_command="ctest --output-on-failure",
    )
    prompts = gen_prompts(_claude_cfg(), config)
    string_prompts = [p for p in prompts if isinstance(p, str)]
    run_and_fix = string_prompts[-1]
    assert "cmake --build build -j$(nproc)" in run_and_fix
    assert "ctest --output-on-failure" in run_and_fix
