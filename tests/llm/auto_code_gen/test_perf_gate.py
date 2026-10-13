# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

from llm.auto_code_gen.run_code_gen import (
    PERF_GATE_DILUTION_FACTOR,
    _bench_throughput_pair,
    _check_performance_meets_target,
)


def _write_bench_results(tmp_path, patched, baseline):
    (tmp_path / "bench_result.json").write_text(json.dumps(patched))
    (tmp_path / "baseline_bench_result.json").write_text(json.dumps(baseline))


def test_serve_dialect_regression_fails_gate(tmp_path):
    _write_bench_results(
        tmp_path,
        {"output_throughput": 89.84},
        {"output_throughput": 118.31},
    )
    passed, actual, threshold = _check_performance_meets_target(tmp_path, 1, 3.81)
    assert passed is False
    assert actual < -20
    assert threshold == round(3.81 * PERF_GATE_DILUTION_FACTOR, 10)


def test_serve_dialect_improvement_passes_gate(tmp_path):
    _write_bench_results(
        tmp_path,
        {"output_throughput": 121.0},
        {"output_throughput": 116.1},
    )
    passed, actual, _ = _check_performance_meets_target(tmp_path, 1, 3.81)
    assert passed is True
    assert 4.0 < actual < 5.0


def test_legacy_throughput_dialect_still_read(tmp_path):
    _write_bench_results(
        tmp_path,
        {"tokens_per_second": 121.0},
        {"tokens_per_second": 116.1},
    )
    passed, actual, _ = _check_performance_meets_target(tmp_path, 1, 3.81)
    assert passed is True
    assert actual > 0


def test_gate_fails_closed_when_metric_is_unknown(tmp_path):
    _write_bench_results(
        tmp_path,
        {"mean_tpot_ms": 11.07},
        {"mean_tpot_ms": 8.39},
    )
    passed, actual, threshold = _check_performance_meets_target(tmp_path, 1, 3.81)
    assert passed is False
    assert actual == 0.0
    assert threshold > 0


def test_gate_fails_closed_when_baseline_missing(tmp_path):
    (tmp_path / "bench_result.json").write_text(
        json.dumps({"output_throughput": 120.0})
    )
    passed, _, _ = _check_performance_meets_target(tmp_path, 1, 3.81)
    assert passed is False


def test_no_expected_improvement_is_not_gated(tmp_path):
    _write_bench_results(
        tmp_path,
        {"output_throughput": 1.0},
        {"output_throughput": 100.0},
    )
    passed, actual, threshold = _check_performance_meets_target(tmp_path, 1, None)
    assert (passed, actual, threshold) == (True, 0.0, 0.0)


def test_pair_requires_the_same_key_on_both_sides():
    patched = {"output_throughput": 120.0, "tokens_per_second": 500.0}
    baseline = {"tokens_per_second": 100.0}
    assert _bench_throughput_pair(patched, baseline) == (
        500.0,
        100.0,
        "tokens_per_second",
    )


def test_pair_rejects_non_numeric_and_zero_values():
    assert _bench_throughput_pair(
        {"output_throughput": "118"}, {"output_throughput": 118.0}
    ) == (None, None, None)
    assert _bench_throughput_pair(
        {"output_throughput": 118.0}, {"output_throughput": 0}
    ) == (None, None, None)


def _make_shared_baseline(tmp_path, baseline_type="bench_off"):
    base = tmp_path / "runtime" / "baselines" / baseline_type
    base.mkdir(parents=True)
    (base / ".done").write_text("1\n")
    return base, tmp_path


def test_shared_baseline_usable_when_files_are_real(tmp_path):
    base, out = _make_shared_baseline(tmp_path)
    (base / "baseline_bench_result.json").write_text("{}")
    from llm.auto_code_gen.run_code_gen import _has_shared_baseline

    assert _has_shared_baseline(str(out), "bench_off") is True


def test_shared_baseline_rejected_when_symlink_dangles(tmp_path):
    base, out = _make_shared_baseline(tmp_path)
    gone = tmp_path / "runtime" / "V1" / "baseline_bench_result.json"
    gone.parent.mkdir(parents=True)
    gone.write_text("{}")
    link = base / "baseline_bench_result.json"
    link.symlink_to(gone.resolve())
    gone.unlink()
    from llm.auto_code_gen.run_code_gen import _has_shared_baseline

    assert _has_shared_baseline(str(out), "bench_off") is False


def test_link_baseline_results_links_only_existing_files(tmp_path):
    from llm.auto_code_gen.run_code_gen import _link_baseline_results

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "baseline_bench_result.json").write_text("{}")
    _link_baseline_results(str(src), str(dst))
    assert (dst / "baseline_bench_result.json").is_file()
    assert not (dst / "baseline_stdout.log").exists()
