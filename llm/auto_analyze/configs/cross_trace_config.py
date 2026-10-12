import json
import os
from dataclasses import dataclass

from common.iteration_utils import resolve_output_file
from llm.auto_analyze.configs.single_trace_config import (
    GPU_OPS_TO_BLOCKS_FILE,
    HIGH_LEVEL_OPS_FILE,
    MEDIAN_BLOCK_FILE,
    PERF_ANALYSIS_SINGLE_FILE,
)

CROSS_MATCHING_FILE = "cross_matching_blocks.txt"
CROSS_COMPARE_FILE = "cross_compare_blocks.txt"
CROSS_IMPROVEMENT_FILE = "cross_improvement_plan.txt"

CROSS_TRACE_OUTPUT_FILES = {
    "cross_matching": CROSS_MATCHING_FILE,
    "cross_compare": CROSS_COMPARE_FILE,
    "cross_improvement": CROSS_IMPROVEMENT_FILE,
}


@dataclass
class TraceResult:
    """A single trace's results, loaded from its result directory."""

    result_dir: str

    # Inferred from run_params.txt
    framework_name: str = ""
    framework_source_code: str = ""
    commit_id: str = ""
    model: str = ""
    gpu_type: str = ""
    batch_size_range: str = ""
    prefill_size_range: str = ""
    output_size_range: str = ""
    run_command: str = ""

    def load_from_result_dir(self):
        params_path = os.path.join(self.result_dir, "run_params.txt")
        if not os.path.exists(params_path):
            raise RuntimeError(
                f"run_params.txt not found in result directory: {self.result_dir}\n"
                f"  Expected: {params_path}\n"
                f"  This directory must contain single-trace analysis results."
            )

        params = {}
        with open(params_path) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("="):
                    key, _, value = line.partition(":")
                    params[key.strip()] = value.strip()

        self.framework_name = params.get("framework_name", "")
        self.framework_source_code = params.get("framework_source_code", "")
        self.commit_id = params.get("commit_id", "")
        self.model = params.get("model", "")
        self.gpu_type = params.get("gpu_type", "")
        self.batch_size_range = params.get("batch_size_range", "")
        self.prefill_size_range = params.get("prefill_size_range", "")
        self.output_size_range = params.get("output_size_range", "")
        self.run_command = params.get("run_command", "")

    @property
    def trace_id(self):
        return (
            f"{self.framework_name}_{self.commit_id[:6]}"
            if self.commit_id
            else self.framework_name
        )

    def get_median_block_file(self):
        return resolve_output_file(self.result_dir, MEDIAN_BLOCK_FILE)

    def get_high_level_ops_file(self):
        return resolve_output_file(self.result_dir, HIGH_LEVEL_OPS_FILE)

    def get_perf_analysis_file(self):
        return resolve_output_file(self.result_dir, PERF_ANALYSIS_SINGLE_FILE)

    def get_gpu_ops_to_blocks_file(self):
        return resolve_output_file(self.result_dir, GPU_OPS_TO_BLOCKS_FILE)

    def get_run_command_file(self):
        return os.path.join(self.result_dir, "run_originals", "run_command.txt")

    def get_run_log_file(self):
        path = os.path.join(self.result_dir, "run_originals", "run_log.txt")
        return path if os.path.exists(path) else ""


@dataclass
class CrossTraceConfig:
    traces: list[TraceResult]
    target_trace_id: int
    output_dir: str = "./cross_trace_output"
    make_improvement_plan: bool = False

    # Iteration control for generation steps
    num_matching_iterations: int = 3
    num_compare_iterations: int = 3
    num_improvement_plan_iterations: int = 3
    matching_skip_review: bool = False
    compare_skip_review: bool = False
    improvement_plan_skip_review: bool = False
    thinking_mode: str = "deep"
    agent: dict | None = None
    config_json_path: str = ""

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)

        traces = [
            TraceResult(result_dir=t["result_dir"]) for t in data.get("traces", [])
        ]

        config = cls(
            traces=traces,
            target_trace_id=data.get("target_trace_id", 0),
            output_dir=data.get("output_dir", "./cross_trace_output"),
            make_improvement_plan=data.get("make_improvement_plan", False),
            num_matching_iterations=data.get("num_matching_iterations", 3),
            num_compare_iterations=data.get("num_compare_iterations", 3),
            num_improvement_plan_iterations=data.get(
                "num_improvement_plan_iterations", 3
            ),
            matching_skip_review=data.get("matching_skip_review", False),
            compare_skip_review=data.get("compare_skip_review", False),
            improvement_plan_skip_review=data.get(
                "improvement_plan_skip_review", False
            ),
            thinking_mode=data.get("thinking-mode", data.get("thinking_mode", "deep")),
            agent=data.get("agent"),
        )
        config.config_json_path = os.path.abspath(path)
        return config

    def to_json(self, path):
        d = {
            "traces": [{"result_dir": t.result_dir} for t in self.traces],
            "target_trace_id": self.target_trace_id,
            "output_dir": self.output_dir,
            "make_improvement_plan": self.make_improvement_plan,
            "thinking-mode": self.thinking_mode,
            "agent": self.agent,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=4)

    def get_target_result(self) -> TraceResult:
        return self.traces[self.target_trace_id]

    def infer_analysis_type(self) -> str:
        frameworks = set(t.framework_name for t in self.traces)
        if len(frameworks) == 1:
            return "cross-commit"
        return "cross-framework"

    def load_all_trace_params(self):
        print("Loading trace parameters from result directories...")
        for i, tr in enumerate(self.traces):
            print(f"  [{i}] Loading: {tr.result_dir}")
            tr.load_from_result_dir()
            print(
                f"      Framework: {tr.framework_name}, "
                f"Model: {tr.model}, "
                f"Commit: {tr.commit_id[:12]}"
            )

    def validate(self):
        errors = []

        if len(self.traces) < 2:
            errors.append("at least 2 traces are required")

        if self.target_trace_id < 0 or self.target_trace_id >= len(self.traces):
            errors.append(
                f"target_trace_id {self.target_trace_id} is out of range "
                f"(0..{len(self.traces) - 1})"
            )

        for i, tr in enumerate(self.traces):
            if not tr.result_dir:
                errors.append(f"traces[{i}].result_dir is required")
            elif not os.path.isdir(tr.result_dir):
                errors.append(f"traces[{i}].result_dir not found: {tr.result_dir}")

            if not tr.framework_name:
                errors.append(
                    f"traces[{i}]: could not infer framework_name from run_params.txt"
                )
            if not tr.commit_id:
                errors.append(
                    f"traces[{i}]: commit_id is required (run single-trace analysis with commit_id set)"
                )

            median_file = tr.get_median_block_file()
            if not os.path.exists(median_file):
                errors.append(f"traces[{i}] median block file not found: {median_file}")

            high_level_file = tr.get_high_level_ops_file()
            if not os.path.exists(high_level_file):
                errors.append(
                    f"traces[{i}] high-level ops file not found: {high_level_file}"
                )

        # Check consistency of execution parameters across all traces
        def _check_consistency(field_name, display_name):
            values = set(
                getattr(t, field_name) for t in self.traces if getattr(t, field_name)
            )
            if len(values) > 1:
                errors.append(f"traces have different {display_name}: {values}")

        _check_consistency("model", "models")
        _check_consistency("gpu_type", "GPU types")
        _check_consistency("batch_size_range", "batch sizes")
        _check_consistency("prefill_size_range", "prefill sizes")
        _check_consistency("output_size_range", "output sizes")

        return errors

    def save_run_params(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "run_params_cross.txt")

        target = self.get_target_result()
        analysis_type = self.infer_analysis_type()

        with open(path, "w") as f:
            f.write("CROSS TRACE ANALYSIS PARAMETERS\n")
            f.write("=" * 50 + "\n")
            f.write(f"analysis_type: {analysis_type}\n")
            f.write(f"model: {target.model}\n")
            f.write(f"gpu_type: {target.gpu_type}\n")
            f.write(f"batch_size_range: {target.batch_size_range}\n")
            f.write(f"prefill_size_range: {target.prefill_size_range}\n")
            f.write(f"output_size_range: {target.output_size_range}\n")
            f.write(f"target_trace_id: {self.target_trace_id}\n")
            f.write(f"target_framework: {target.framework_name}\n")
            f.write(f"target_commit_id: {target.commit_id}\n")
            f.write(f"output_dir: {os.path.abspath(self.output_dir)}\n")
            f.write(f"num_traces: {len(self.traces)}\n")
            f.write("\n")
            for i, tr in enumerate(self.traces):
                marker = " (TARGET)" if i == self.target_trace_id else ""
                f.write(f"trace_{i}{marker}:\n")
                f.write(f"  framework_name: {tr.framework_name}\n")
                f.write(f"  commit_id: {tr.commit_id}\n")
                f.write(f"  framework_source_code: {tr.framework_source_code}\n")
                f.write(f"  result_dir: {tr.result_dir}\n")

        print(f"  Saved cross-trace run parameters: {path}")
