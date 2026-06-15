"""Tests for auto_bug_fix.baseline — test failure parsing and regression detection."""
from auto_bug_fix.baseline import parse_test_failures, check_regression


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
