"""Trace collection must not copy a trace that is still being written.

Kineto streams the gzip archive, so copying too early yields a truncated file
that only fails much later, inside the analysis, as
``EOFError: Compressed file ended before the end-of-stream marker``. The
collection step therefore waits for a stable size *and* a valid gzip stream.
"""

import gzip

from llm.auto_profile_server_client.benchmark_runner import BenchmarkRunner


def _wait(path, timeout=1.0):
    return BenchmarkRunner._wait_for_written_file(str(path), timeout=timeout)


def test_complete_gzip_is_accepted(tmp_path):
    p = tmp_path / "trace.json.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(b'{"traceEvents": []}' * 1000)
    assert _wait(p, timeout=10.0) is True


def test_truncated_gzip_is_rejected(tmp_path):
    p = tmp_path / "trace.json.gz"
    with gzip.open(p, "wb") as fh:
        fh.write(b'{"traceEvents": []}' * 1000)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # simulate a copy taken mid-write
    assert _wait(p, timeout=1.0) is False


def test_plain_file_stability_is_enough(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text("{}")
    assert _wait(p, timeout=10.0) is True


def test_missing_file_is_rejected(tmp_path):
    assert _wait(tmp_path / "nope.json.gz", timeout=0.1) is False
