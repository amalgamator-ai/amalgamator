"""Test suite baseline capture and regression detection for the verify phase."""
from __future__ import annotations

import re
import subprocess


def run_test_suite(test_command: str, cwd: str, timeout: int = 1800) -> tuple[int, str, str]:
    """Run a shell test command and return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            test_command, shell=True, cwd=cwd, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        partial_out = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
        partial_err = (e.stderr or "") if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
        return (1, partial_out, partial_err + f"\nCommand timed out after {timeout}s: {test_command}")
    return (result.returncode, result.stdout, result.stderr)


_FAILURE_PATTERNS = [
    # Go per-test: --- FAIL: TestName (0.01s)
    re.compile(r"^---\s+FAIL:\s+(\S+)"),
    # Go package: FAIL\tgithub.com/org/repo/pkg
    re.compile(r"^\s*FAIL\t(\S+)"),
    # Go package (spaces): FAIL  github.com/org/repo/pkg
    re.compile(r"^\s*FAIL\s{2,}(\S+)"),
    # Generic: FAIL: test_name
    re.compile(r"^\s*FAIL:\s+(\S+)"),
    # pytest: FAILED path/test.py::test_name
    re.compile(r"^\s*FAILED\s+(\S+)"),
    # Rust: test name ... FAILED
    re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED"),
    # JUnit/Maven: <<< FAILURE! - in org.package.TestClass
    re.compile(r"<<<\s+FAILURE!\s+-\s+in\s+(\S+)"),
    # Ginkgo: [FAIL] Description
    re.compile(r"^\s*\[FAIL\]\s+(.+)$"),
    # TAP: not ok 1 - test_name
    re.compile(r"^\s*not ok\s+\d+\s+-\s+(\S+)"),
    # Jest/Mocha: FAIL src/path/test.js
    re.compile(r"^\s*FAIL\s+((?:\S+\.(?:js|ts|jsx|tsx))\S*)"),
    # RSpec: rspec ./spec/path_spec.rb:42
    re.compile(r"^\s*rspec\s+(\S+)"),
]


def parse_test_failures(stdout: str, stderr: str) -> set[str]:
    """Extract failed test names from combined stdout/stderr using common test-output patterns."""
    failures: set[str] = set()
    combined = stdout + "\n" + stderr
    for line in combined.splitlines():
        for pattern in _FAILURE_PATTERNS:
            m = pattern.match(line)
            if m:
                failures.add(m.group(1))
                break
    return failures


def capture_baseline(test_command: str, cwd: str) -> set[str]:
    """Run tests and return the set of failing test names as the baseline."""
    _, stdout, stderr = run_test_suite(test_command, cwd)
    return parse_test_failures(stdout, stderr)


def check_regression(
    failures: set[str], baseline: set[str],
) -> tuple[bool, set[str]]:
    """Compare current failures against baseline. Returns (no_regression, new_failures)."""
    new_failures = failures - baseline
    return (len(new_failures) == 0, new_failures)
