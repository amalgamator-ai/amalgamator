import json
import os
import subprocess
from dataclasses import dataclass, field

from common.claude_utils import ClaudeConfig

from auto_analyze.configs.cross_trace_config import (
    CrossTraceConfig,
    TraceResult,
    CROSS_IMPROVEMENT_FILE,
)

from auto_analyze.configs.single_trace_config import (
    MEDIAN_BLOCK_FILE,
    HIGH_LEVEL_OPS_FILE,
)


_CODE_PORT_DISALLOWED_MODULES_MAP = {
    "sgl": ["sglang", "sgl_kernel", "sgl*"],
    "sglang": ["sglang", "sgl_kernel", "sgl*"],
    "vllm": ["vllm"],
    "trt": ["tensorrt_llm"],
}


def _infer_code_port_disallowed_modules(source_framework: str) -> list[str]:
    return _CODE_PORT_DISALLOWED_MODULES_MAP.get(source_framework, [source_framework])


def _run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


@dataclass
class CodeGenConfig:
    # --- Primary inputs (from JSON) ---
    cross_trace_config_path: str = ""
    improvement_id: int = 0
    source_code_dir: str = ""
    output_dir: str = ""
    num_code_port_plan_iterations: int = 3
    num_test_plan_iterations: int = 3
    num_code_gen_iterations: int = 3
    code_port_disallowed_modules: list[str] = field(default_factory=list)

    # --- Auto-inferred fields ---
    model: str = ""
    gpu_type: str = ""
    batch_size: str = ""
    isl: str = ""
    osl: str = ""

    framework_names: list[str] = field(default_factory=list)
    framework_source_codes: list[str] = field(default_factory=list)
    framework_test_dirs: list[str] = field(default_factory=list)
    transformer_block_high_level_ops_files: list[str] = field(default_factory=list)
    median_transformer_block_files: list[str] = field(default_factory=list)

    plan_file: str = ""
    plan_step: int = 0

    target_framework: str = ""
    source_framework: str = ""
    target_commit_id: str = ""
    source_commit_id: str = ""
    source_framework_code_dir: str = ""

    @classmethod
    def from_json(cls, path: str) -> "CodeGenConfig":
        with open(path) as f:
            data = json.load(f)

        errors = []

        cross_trace_config_path = data.get("cross_trace_config", "")
        if not cross_trace_config_path:
            errors.append('"cross_trace_config" is required.')
        elif not os.path.isfile(cross_trace_config_path):
            errors.append(f'"cross_trace_config" file not found: {cross_trace_config_path}')

        improvement_id = data.get("improvement_id")
        if improvement_id is None or not isinstance(improvement_id, int) or improvement_id <= 0:
            errors.append('"improvement_id" must be a positive integer.')

        source_code_dir = data.get("source_code_dir", "")
        if not source_code_dir:
            errors.append('"source_code_dir" is required.')
        elif not os.path.isdir(source_code_dir):
            errors.append(f'"source_code_dir" directory not found: {source_code_dir}')

        output_dir = data.get("output_dir", "")
        if not output_dir:
            errors.append('"output_dir" is required.')

        num_code_port_plan_iterations = data.get("num_code_port_plan_iterations", 3)
        if not isinstance(num_code_port_plan_iterations, int) or num_code_port_plan_iterations <= 0:
            errors.append('"num_code_port_plan_iterations" must be a positive integer.')

        num_test_plan_iterations = data.get("num_test_plan_iterations", 3)
        if not isinstance(num_test_plan_iterations, int) or num_test_plan_iterations <= 0:
            errors.append('"num_test_plan_iterations" must be a positive integer.')

        num_code_gen_iterations = data.get("num_code_gen_iterations", 3)
        if not isinstance(num_code_gen_iterations, int) or num_code_gen_iterations <= 0:
            errors.append('"num_code_gen_iterations" must be a positive integer.')

        if errors:
            raise ValueError(
                "Code gen config errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        code_port_disallowed_modules = data.get("code_port_disallowed_modules", [])

        config = cls(
            cross_trace_config_path=cross_trace_config_path,
            improvement_id=improvement_id,
            source_code_dir=os.path.abspath(source_code_dir),
            output_dir=os.path.abspath(output_dir),
            num_code_port_plan_iterations=num_code_port_plan_iterations,
            num_test_plan_iterations=num_test_plan_iterations,
            num_code_gen_iterations=num_code_gen_iterations,
            plan_step=improvement_id,
            code_port_disallowed_modules=code_port_disallowed_modules,
        )
        config._load_from_cross_trace()
        return config

    def _load_from_cross_trace(self):
        cross_cfg = CrossTraceConfig.from_json(self.cross_trace_config_path)
        cross_cfg.load_all_trace_params()

        errors = cross_cfg.validate()
        if errors:
            raise ValueError(
                "Cross-trace config validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        if len(cross_cfg.traces) != 2:
            raise ValueError(
                f"Code generation requires exactly 2 traces (source and target), "
                f"got {len(cross_cfg.traces)}."
            )

        target_idx = cross_cfg.target_trace_id
        source_idx = 1 - target_idx

        target_trace = cross_cfg.traces[target_idx]
        source_trace = cross_cfg.traces[source_idx]

        self.target_framework = target_trace.framework_name
        self.source_framework = source_trace.framework_name
        self.target_commit_id = target_trace.commit_id
        self.source_commit_id = source_trace.commit_id
        self.source_framework_code_dir = source_trace.framework_source_code

        self.model = target_trace.model
        self.gpu_type = target_trace.gpu_type
        self.batch_size = target_trace.batch_size_range
        self.isl = target_trace.prefill_size_range
        self.osl = target_trace.output_size_range

        # Order: [target, source] matching trace order for target_trace_id=0,
        # but we always build lists in trace-index order for consistency.
        ordered = [(target_idx, target_trace), (source_idx, source_trace)]
        ordered.sort(key=lambda x: x[0])
        traces_ordered = [t for _, t in ordered]

        self.framework_names = [t.framework_name for t in traces_ordered]

        # Source codes: use source_code_dir for target, run_params for source
        source_codes = []
        for idx, trace in ordered:
            if idx == target_idx:
                source_codes.append(self.source_code_dir)
            else:
                source_codes.append(trace.framework_source_code)
        self.framework_source_codes = source_codes

        self.framework_test_dirs = [
            os.path.join(t.result_dir, "run_originals") for t in traces_ordered
        ]

        self.transformer_block_high_level_ops_files = [
            t.get_high_level_ops_file() for t in traces_ordered
        ]
        self.median_transformer_block_files = [
            t.get_median_block_file() for t in traces_ordered
        ]

        plan_path = os.path.join(cross_cfg.output_dir, CROSS_IMPROVEMENT_FILE)
        if not os.path.isfile(plan_path):
            raise FileNotFoundError(
                f"Improvement plan file not found: {plan_path}\n"
                f"  Run cross-trace analysis with make_improvement_plan=true first."
            )
        self.plan_file = plan_path

        if not self.code_port_disallowed_modules:
            self.code_port_disallowed_modules = (
                _infer_code_port_disallowed_modules(self.source_framework)
            )

    def _prepare_branch(self, repo_dir: str, commit: str, branch: str, label: str):
        if not commit:
            raise RuntimeError(
                f"Cannot prepare {label} branch: commit_id is empty.\n"
                f"  Ensure run_params.txt in the {label} trace has a valid commit_id."
            )

        if not os.path.isdir(repo_dir):
            raise RuntimeError(f"{label} code directory not found: {repo_dir}")

        result = _run_git(["rev-parse", "--is-inside-work-tree"], repo_dir, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"{label} code directory is not a git repository: {repo_dir}")

        result = _run_git(["cat-file", "-t", commit], repo_dir, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} commit {commit[:12]} not found in {repo_dir}.\n"
                f"  Fetch or pull to make this commit available."
            )

        result = _run_git(["branch", "--list", branch], repo_dir, check=False)
        branch_exists = branch in result.stdout

        if branch_exists:
            print(f"  [{label}] Branch '{branch}' exists, switching to it...")
            _run_git(["checkout", branch], repo_dir)

            head = _run_git(["rev-parse", "HEAD"], repo_dir).stdout.strip()
            if head == commit:
                print(f"  [{label}] Branch HEAD is at the tested commit ({commit[:12]}).")
            else:
                ancestor_check = _run_git(
                    ["merge-base", "--is-ancestor", commit, head],
                    repo_dir,
                    check=False,
                )
                if ancestor_check.returncode == 0:
                    print(
                        f"  [{label}] Branch HEAD ({head[:12]}) descends from tested "
                        f"commit ({commit[:12]}). Existing work preserved."
                    )
                else:
                    raise RuntimeError(
                        f"[{label}] Branch '{branch}' HEAD ({head[:12]}) does NOT "
                        f"descend from the tested commit ({commit[:12]}).\n"
                        f"  The branch has diverged from the commit used in analysis.\n"
                        f"  Options:\n"
                        f"    1. Delete the branch: git -C {repo_dir} branch -D {branch}\n"
                        f"    2. Reset it: git -C {repo_dir} checkout {branch} && "
                        f"git -C {repo_dir} reset --hard {commit[:12]}"
                    )
        else:
            print(f"  [{label}] Creating branch '{branch}' at commit {commit[:12]}...")
            _run_git(["checkout", "-b", branch, commit], repo_dir)
            print(f"  [{label}] Branch '{branch}' created successfully.")

    def prepare_branches(self):
        model_slug = self.model.replace("/", "_").replace("-", "_")
        target_branch = (
            f"auto_code_gen_{model_slug}"
            f"_isl_{self.isl}_osl_{self.osl}_b_{self.batch_size}"
            f"_plan_{self.improvement_id}"
            f"_commit_{self.target_commit_id[:6]}"
        )
        source_branch = (
            f"auto_analyze_{self.source_framework}"
            f"_commit_{self.source_commit_id[:6]}"
        )

        print(f"Preparing target framework ({self.target_framework}) branch...")
        self._prepare_branch(
            self.source_code_dir, self.target_commit_id, target_branch, "target"
        )

        print(f"Preparing source framework ({self.source_framework}) branch...")
        self._prepare_branch(
            self.source_framework_code_dir, self.source_commit_id, source_branch, "source"
        )

    def make_claude_config(self) -> ClaudeConfig:
        return ClaudeConfig(
            model="claude-opus-4-6[1m]",
            allowed_tools=["Read", "Write", "Bash"],
            perm_mode="acceptEdits",
            cwd=self.output_dir,
        )
