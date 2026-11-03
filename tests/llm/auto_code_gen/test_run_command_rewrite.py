"""The inherited serving command is rewritten for the runtime gate.

Three rewrites happen, in order: tracing-only profiler flags are dropped,
campaign-level extra server flags are appended, and the bench sample count is
raised. These tests pin the contract that makes an A/B gate measurement
comparable to the trace it was derived from.
"""

import shlex

from llm.auto_code_gen.code_gen_configs import (
    _apply_run_command_prefix,
    _append_serve_args,
    _apply_runtime_bench_num_prompts,
    _strip_trace_profiler_flags,
)

VLLM_SERVE = (
    "VLLM_ROCM_USE_AITER=1 vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --port 8123 "
    "--max-num-seqs 1 --trust-remote-code"
)


class TestStripTraceProfilerFlags:
    def test_drops_flag_with_value(self):
        command = (
            VLLM_SERVE
            + ' --profiler-config \'{"profiler": "torch", "max_iterations": 50}\''
        )
        assert "--profiler-config" not in _strip_trace_profiler_flags(command)
        assert "max_iterations" not in _strip_trace_profiler_flags(command)

    def test_drops_equals_form(self):
        command = VLLM_SERVE + ' --profiler-config={"max_iterations": 50}'
        assert "max_iterations" not in _strip_trace_profiler_flags(command)

    def test_keeps_other_flags(self):
        command = VLLM_SERVE + ' --profiler-config \'{"a": 1}\' --seed 7'
        stripped = _strip_trace_profiler_flags(command)
        assert "--max-num-seqs 1" in stripped
        assert "--seed 7" in stripped

    def test_bench_command_is_untouched(self):
        # A batch benchmark has no "serve" token: its profiler flags drive the
        # benchmark itself and must survive.
        command = "vllm bench latency --profiler-config '{\"a\": 1}'"
        assert _strip_trace_profiler_flags(command) == command

    def test_no_profiler_flags_is_a_no_op(self):
        assert _strip_trace_profiler_flags(VLLM_SERVE) == VLLM_SERVE


class TestAppendServeArgs:
    def test_appends_after_existing_tokens(self):
        command = _append_serve_args(VLLM_SERVE, ["--max-num-seqs", "64"])
        tokens = shlex.split(command)
        # Later occurrences win in argparse, so the override must come last.
        assert tokens[-2:] == ["--max-num-seqs", "64"]
        assert tokens.count("--max-num-seqs") == 2

    def test_no_extra_args_is_a_no_op(self):
        assert _append_serve_args(VLLM_SERVE, []) == VLLM_SERVE

    def test_bench_command_is_not_extended(self):
        command = "vllm bench latency --num-prompts 8"
        assert _append_serve_args(command, ["--max-num-seqs", "64"]) == command

    def test_quoted_values_survive(self):
        spec = '{"method": "mtp", "num_speculative_tokens": 3}'
        command = _append_serve_args(
            VLLM_SERVE, ["--speculative-config", spec])
        tokens = shlex.split(command)
        assert tokens[tokens.index("--speculative-config") + 1] == spec


class TestApplyRuntimeBenchNumPrompts:
    def test_rewrites_space_form(self):
        command = "vllm bench serve --num-prompts 8 --max-concurrency 1"
        assert "--num-prompts 32" in _apply_runtime_bench_num_prompts(command, 32)

    def test_rewrites_equals_form(self):
        command = "vllm bench serve --num-prompts=8"
        assert "--num-prompts=32" in _apply_runtime_bench_num_prompts(command, 32)

    def test_rewrites_num_samples_for_latency(self):
        command = "vllm bench latency --num-samples 8"
        assert "--num-samples 32" in _apply_runtime_bench_num_prompts(command, 32)

    def test_zero_keeps_the_traced_value(self):
        command = "vllm bench serve --num-prompts 8"
        assert _apply_runtime_bench_num_prompts(command, 0) == command


class TestApplyRunCommandPrefix:
    def test_prefix_is_placed_in_front(self):
        command = _apply_run_command_prefix(
            VLLM_SERVE, "docker run --rm --network host IMAGE")
        assert command.startswith("docker run --rm --network host IMAGE ")
        assert command.endswith(VLLM_SERVE)

    def test_empty_prefix_is_a_no_op(self):
        assert _apply_run_command_prefix(VLLM_SERVE, "") == VLLM_SERVE
        assert _apply_run_command_prefix(VLLM_SERVE, "   ") == VLLM_SERVE

    def test_prefix_survives_the_other_rewrites(self):
        # Order matters: profiler flags are stripped and serve args appended
        # before the container wrapper is put in front.
        command = _apply_run_command_prefix(
            _append_serve_args(
                _strip_trace_profiler_flags(VLLM_SERVE + ' --profiler-config \'{"a": 1}\''),
                ["--max-num-seqs", "64"]),
            "docker run --rm IMAGE")
        assert command.startswith("docker run --rm IMAGE ")
        assert "--profiler-config" not in command
        assert command.rstrip().endswith("64")


class TestQuotedCommandPlaceholder:
    def test_quoted_placeholder_quotes_the_serving_command(self):
        spec = '{"method": "mtp", "num_speculative_tokens": 3}'
        command = VLLM_SERVE + f" --speculative-config '{spec}'"
        wrapped = _apply_run_command_prefix(
            command,
            "docker run --rm IMAGE --entrypoint bash -c {quoted_command}")
        # The whole serving command must arrive as one shell argument.
        tokens = shlex.split(wrapped)
        assert tokens[-1] == command
        assert tokens[0:3] == ["docker", "run", "--rm"]

    def test_raw_placeholder_is_not_quoted(self):
        wrapped = _apply_run_command_prefix(
            VLLM_SERVE, "bash wrapper.sh {command}")
        assert wrapped == f"bash wrapper.sh {VLLM_SERVE}"
