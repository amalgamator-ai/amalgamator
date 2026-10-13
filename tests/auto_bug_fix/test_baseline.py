# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auto_bug_fix.baseline — test failure parsing and regression detection."""

from auto_bug_fix.baseline import (
    TIMEOUT_EXIT_CODE,
    check_regression,
    parse_test_failures,
    run_test_suite,
)


def test_parse_test_failures_fail_colon():
    stdout = "FAIL: test_foo\nPASS: test_bar\nFAIL: test_baz"
    failures = parse_test_failures(stdout, "")
    assert failures == {"test_foo", "test_baz"}


def test_parse_test_failures_tap_format():
    stdout = "ok 1 - test_a\nnot ok 2 - test_b\nok 3 - test_c"
    failures = parse_test_failures(stdout, "")
    assert failures == {"test_b"}


def test_parse_test_failures_empty():
    failures = parse_test_failures("", "")
    assert failures == set()


def test_check_regression_no_new():
    passed, new_failures = check_regression(
        failures={"a", "b"},
        baseline={"a", "b", "c"},
    )
    assert passed is True
    assert new_failures == set()


def test_check_regression_has_new():
    passed, new_failures = check_regression(
        failures={"a", "b", "d"},
        baseline={"a", "b"},
    )
    assert passed is False
    assert new_failures == {"d"}


def test_parse_go_per_test_fail():
    stdout = "--- FAIL: TestFoo (0.01s)\n--- FAIL: TestBar/sub (0.02s)\n--- PASS: TestOk (0.00s)"
    failures = parse_test_failures(stdout, "")
    assert "TestFoo" in failures
    assert "TestBar/sub" in failures
    assert len(failures) == 2


def test_parse_go_package_fail():
    stdout = (
        "FAIL\tgithub.com/org/repo/pkg\t0.123s\nok\tgithub.com/org/repo/other\t0.050s"
    )
    failures = parse_test_failures(stdout, "")
    assert "github.com/org/repo/pkg" in failures
    assert len(failures) == 1


def test_parse_rust_fail():
    stdout = "test utils::test_parse ... ok\ntest utils::test_broken ... FAILED\ntest utils::test_other ... ok"
    failures = parse_test_failures(stdout, "")
    assert failures == {"utils::test_broken"}


def test_parse_junit_fail():
    stdout = "Tests run: 5, Failures: 1\n<<< FAILURE! - in org.example.MyTest\n"
    failures = parse_test_failures(stdout, "")
    assert "org.example.MyTest" in failures


def test_parse_ginkgo_fail():
    stdout = "[FAIL] cri-o: ContainerStop should succeed\n[PASS] cri-o: Other test"
    failures = parse_test_failures(stdout, "")
    assert "cri-o: ContainerStop should succeed" in failures


def test_parse_pytest_fail():
    stdout = (
        "FAILED tests/test_main.py::test_login\nPASSED tests/test_main.py::test_logout"
    )
    failures = parse_test_failures(stdout, "")
    assert failures == {"tests/test_main.py::test_login"}


def test_parse_jest_fail():
    stdout = "FAIL src/components/Button.test.js\nPASS src/components/Input.test.js"
    failures = parse_test_failures(stdout, "")
    assert "src/components/Button.test.js" in failures


# --- timeout handling ---


def test_run_test_suite_timeout_returns_sentinel():
    exit_code, _, _ = run_test_suite("sleep 3", "/tmp", timeout=1)
    assert exit_code == TIMEOUT_EXIT_CODE


def test_run_test_suite_completes_within_timeout():
    exit_code, _, _ = run_test_suite("true", "/tmp", timeout=30)
    assert exit_code == 0


def test_run_test_suite_no_timeout_by_default():
    exit_code, _, _ = run_test_suite("true", "/tmp")
    assert exit_code == 0
