import json
import os

from llm.auto_code_gen.pr_flow import gates


def _bench(tmp_path, patched, baseline):
    (tmp_path / "bench_result.json").write_text(json.dumps({"output_throughput": patched}))
    (tmp_path / "baseline_bench_result.json").write_text(
        json.dumps({"output_throughput": baseline}))


def _lm_eval(tmp_path, text):
    (tmp_path / "lm_eval.txt").write_text(text)


def test_perf_gate_holds_most_of_the_accepted_win(tmp_path):
    _bench(tmp_path, patched=123.0, baseline=119.58)          # +2.86%
    result = gates.evaluate_perf(str(tmp_path), expected_pct=1.5, recorded_pct=2.86)
    assert result["pass"] is True
    assert result["actual_pct"] > 2.8


def test_perf_gate_fails_when_a_review_fix_eats_the_win(tmp_path):
    _bench(tmp_path, patched=120.2, baseline=119.58)          # +0.52%
    result = gates.evaluate_perf(str(tmp_path), expected_pct=1.5, recorded_pct=2.86)
    assert result["pass"] is False
    assert "retention" in result["reason"]


def test_perf_gate_fails_closed_without_measurements(tmp_path):
    result = gates.evaluate_perf(str(tmp_path), expected_pct=1.5, recorded_pct=2.86)
    assert result["pass"] is False
    assert result["measured"] is False


def test_lm_eval_gate_passes_within_tolerance(tmp_path):
    _lm_eval(tmp_path, "flexible-extract=0.945, strict-match=0.935\n"
                       "Delta flexible-extract: +0.010\nDelta strict-match: +0.005\n"
                       "Verdict: PASS\n")
    result = gates.evaluate_lm_eval(str(tmp_path))
    assert result["pass"] is True
    assert result["verdict"] == "PASS"


def test_lm_eval_gate_fails_below_absolute_floor(tmp_path):
    _lm_eval(tmp_path, "flexible-extract=0.800, strict-match=0.800\n"
                       "Delta flexible-extract: +0.000\nVerdict: PASS\n")
    result = gates.evaluate_lm_eval(str(tmp_path))
    assert result["pass"] is False
    assert "floor" in result["reason"]


def test_lm_eval_gate_fails_without_a_result(tmp_path):
    assert gates.evaluate_lm_eval(str(tmp_path))["pass"] is False


def _lm_eval_results_json(base, score, stderr):
    out = base / "nvidia__GLM-5.2-NVFP4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_test.json").write_text(json.dumps({
        "results": {"gsm8k": {
            "exact_match,flexible-extract": score,
            "exact_match_stderr,flexible-extract": stderr,
            "exact_match,strict-match": score,
            "exact_match_stderr,strict-match": stderr,
        }}}))


def test_lm_eval_gate_uses_stderr_aware_delta_tolerance(tmp_path):
    # delta -0.03 is noise at 200 questions (stderr ~0.017 each arm):
    # tolerance becomes -1.65 * ~0.024 = -0.039, so -0.03 must PASS.
    _lm_eval(tmp_path, "flexible-extract=0.94, strict-match=0.94\n"
                       "Delta flexible-extract: -0.030\nDelta strict-match: -0.025\n"
                       "Verdict: PASS\n")
    _lm_eval_results_json(tmp_path / "lm_eval_output", 0.94, 0.0168)
    _lm_eval_results_json(tmp_path / "lm_eval_output_disabled", 0.97, 0.0168)
    result = gates.evaluate_lm_eval(str(tmp_path))
    assert result["pass"] is True
    assert result["delta_stderr"] > 0.02
    assert result["delta_tolerance"] < -0.03


def test_lm_eval_gate_still_fails_a_degradation_beyond_the_noise(tmp_path):
    # delta -0.08 is far beyond even the widened tolerance: a real regression.
    _lm_eval(tmp_path, "flexible-extract=0.89, strict-match=0.89\n"
                       "Delta flexible-extract: -0.080\nDelta strict-match: -0.075\n"
                       "Verdict: PASS\n")
    _lm_eval_results_json(tmp_path / "lm_eval_output", 0.89, 0.0168)
    _lm_eval_results_json(tmp_path / "lm_eval_output_disabled", 0.97, 0.0168)
    result = gates.evaluate_lm_eval(str(tmp_path))
    assert result["pass"] is False
    assert "tolerance" in result["reason"]


def test_lm_eval_gate_falls_back_to_fixed_tolerance_without_stderr(tmp_path):
    # No results JSON on disk: the old fixed -0.01 tolerance applies.
    _lm_eval(tmp_path, "flexible-extract=0.94, strict-match=0.94\n"
                       "Delta flexible-extract: -0.030\nDelta strict-match: -0.025\n"
                       "Verdict: PASS\n")
    result = gates.evaluate_lm_eval(str(tmp_path))
    assert result["pass"] is False
    assert result["delta_stderr"] is None
    assert result["delta_tolerance"] == gates.LM_EVAL_MIN_DELTA


RESOLUTION = """
# dispositions

## Comment 11: vllm/config.py:42
STATUS: fixed
REPLY: Moved the toggle into the envs module as suggested.

## Comment 12: vllm/model_executor/x.py:88
STATUS: explained
REPLY: Keeping split-K here; the fallback path needs it.

## Comment 13: vllm/model_executor/y.py:5
STATUS: unresolved
"""


def test_resolution_closure_requires_every_comment(tmp_path):
    path = tmp_path / "comment_resolution.md"
    path.write_text(RESOLUTION)
    summary = gates.parse_comment_resolution(str(path))
    assert summary["total"] == 3
    assert (summary["fixed"], summary["explained"], summary["unresolved"]) == (1, 1, 1)
    assert summary["pass"] is False
    assert summary["unresolved_ids"] == [13]
    assert summary["entries"][0]["reply"].startswith("Moved the toggle")


def test_resolution_closure_passes_when_all_answered(tmp_path):
    path = tmp_path / "comment_resolution.md"
    path.write_text(RESOLUTION.replace("STATUS: unresolved", "STATUS: fixed"))
    summary = gates.parse_comment_resolution(str(path))
    assert summary["pass"] is True
    assert summary["unresolved"] == 0


def test_resolution_file_absent_fails_closed(tmp_path):
    summary = gates.parse_comment_resolution(str(tmp_path / "missing.md"))
    assert summary["pass"] is False and summary["total"] == 0


def _perf(passed=True):
    return {"pass": passed, "actual_pct": 2.8, "reason": ""}


def _lm(passed=True):
    return {"pass": passed, "verdict": "PASS" if passed else "FAIL", "reason": ""}


def _closure(total=2, unresolved=0):
    return {"pass": unresolved == 0 and total > 0, "total": total, "fixed": total - unresolved,
            "explained": 0, "unresolved": unresolved, "unresolved_ids": [], "reason": ""}


def test_verdict_is_green_only_when_all_three_gates_are_green():
    verdict = gates.build_verdict(1, 2, "sha", _perf(), _lm(), _closure(), nsys_run=False)
    assert verdict["overall"] == "PASS"
    assert verdict["nsys_run"] is False
    assert gates.build_verdict(1, 2, "sha", _perf(False), _lm(), _closure(),
                               nsys_run=False)["overall"] == "FAIL"
    assert gates.build_verdict(1, 2, "sha", _perf(), _lm(False), _closure(),
                               nsys_run=False)["overall"] == "FAIL"
    assert gates.build_verdict(1, 2, "sha", _perf(), _lm(), _closure(unresolved=1),
                               nsys_run=True)["overall"] == "FAIL"


def test_verdict_round_trip(tmp_path):
    verdict = gates.build_verdict(1, 2, "sha", _perf(), _lm(), _closure(), nsys_run=True)
    path = gates.write_verdict(str(tmp_path / "V2" / gates.VERDICT_FILE), verdict)
    assert os.path.dirname(path).endswith("V2")
    assert gates.read_verdict(path)["iteration"] == 2
    assert gates.read_verdict(None) is None


def test_recorded_perf_prefers_an_accepted_gate():
    state = {"gates": [
        {"overall": "PASS", "accepted": True, "perf": {"actual_pct": 2.86}},
        {"overall": "FAIL", "accepted": False, "perf": {"actual_pct": -1.0}},
    ]}
    assert gates.recorded_perf_pct(state) == 2.86


def test_recorded_perf_falls_back_to_the_original_success_note(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "success_result.txt").write_text(
        "perf gate met: output_throughput +123.01 -> +2.86% vs baseline\n")
    assert gates.recorded_perf_pct({"gates": []}, str(tmp_path)) == 2.86


def test_shared_baseline_is_reusable_only_for_the_same_base_commit():
    state = {"baselines": {"bench_off_base_commit": "aaa"}}
    assert gates.baseline_is_valid(state, "bench_off", "aaa") is True
    assert gates.baseline_is_valid(state, "bench_off", "bbb") is False
    assert gates.baseline_is_valid(state, "bench_off", "") is False
    assert gates.baseline_is_valid({"baselines": {}}, "bench_off", "aaa") is False


def test_perf_gate_reports_a_real_regression_as_a_regression(tmp_path):
    """A measured win that is too small must not be blamed on missing artifacts."""
    _bench(tmp_path, patched=140.59804, baseline=140.20688)     # +0.28%
    result = gates.evaluate_perf(str(tmp_path), expected_pct=1.5, recorded_pct=2.86)
    assert result["pass"] is False
    assert result["measured"] is True
    assert "missing" not in result["reason"]
    assert "actual +0.28% < required +2.29%" in result["reason"]
