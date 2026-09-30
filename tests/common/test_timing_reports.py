from auto_code_gen.run_code_gen import _print_detailed_table, _print_phase_summary


def test_reports_accept_unknown_and_zero_costs(capsys):
    timings = [
        {"name": "code_gen_V1", "duration": 1, "input_tokens": 10, "output_tokens": 20, "cost_usd": None},
        {"name": "code_gen_V2", "duration": 1, "input_tokens": 10, "output_tokens": 20, "cost_usd": 0.0},
    ]
    _print_detailed_table(timings, 2)
    output = capsys.readouterr().out
    assert "unknown" in output and "$0.0000" in output
    _print_phase_summary([{"name": "Code gen", "duration": 2, "max_iterations": 2,
                           "iterations": 2, "converged": True}], timings, 2)
    assert "unknown" in capsys.readouterr().out
