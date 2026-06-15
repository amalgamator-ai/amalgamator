"""Tests for auto_bug_fix.signature — normalization and comparison utilities."""
from __future__ import annotations

from auto_bug_fix.signature import compare_signatures, normalize_signature


def test_normalize_strips_pointer_addresses():
    raw = "segfault at 0x7fff12345678 in libfoo.so"
    result = normalize_signature(raw)
    assert "0x7fff12345678" not in result
    assert "0xADDR" in result


def test_normalize_strips_pids():
    raw = "pid=12345 crashed"
    result = normalize_signature(raw)
    assert "12345" not in result
    assert "pid=PID" in result


def test_normalize_strips_timestamps():
    raw = "2024-01-15T12:30:45 error occurred"
    result = normalize_signature(raw)
    assert "2024-01-15T12:30:45" not in result
    assert "TIMESTAMP" in result


def test_normalize_strips_line_numbers():
    raw = "file.c:42: error: undefined reference"
    result = normalize_signature(raw)
    assert ":42:" not in result
    assert ":LINE:" in result


def test_normalize_collapses_whitespace():
    raw = "error   in\n\nmodule   foo"
    result = normalize_signature(raw)
    assert result == "error in module foo"


def test_compare_identical_signatures():
    sig = "ASAN: heap-buffer-overflow in do_stuff"
    assert compare_signatures(sig, sig) == "match"


def test_compare_same_after_normalization():
    sig_a = "crash at 0x7fff00001111 in libbar.so"
    sig_b = "crash at 0x7fff99998888 in libbar.so"
    assert compare_signatures(sig_a, sig_b) == "match"


def test_compare_completely_different():
    sig_a = "ASAN: heap-buffer-overflow in do_stuff"
    sig_b = "UBSAN: shift-out-of-range in calc_offset"
    assert compare_signatures(sig_a, sig_b) == "mismatch"


def test_compare_partial_overlap():
    # Build two strings that share ~85% of their sentence fragments but differ
    # in enough content to prevent an exact normalized match.
    common = ". ".join(f"common segment {i}" for i in range(20))
    sig_a = common + ". unique alpha tail"
    sig_b = common + ". unique beta tail"
    result = compare_signatures(sig_a, sig_b)
    assert result == "partial"
