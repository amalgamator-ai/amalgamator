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


def _infer_disallowed_modules(source_framework: str) -> list[str]:
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
class PipelineConfig:
    output_dir: str = ""
    source_code_dir: str = ""
    num_code_port_plan_iterations: int = 3
    num_test_plan_iterations: int = 3
    num_code_gen_iterations: int = 3
    disallowed_modules: list[str] = field(default_factory=list)
    thinking_mode: str = "deep"
    code_port_plan_skip_review: bool = False
    test_plan_skip_review: bool = False
    code_gen_skip_review: bool = False


@dataclass
class CodeGenConfig(PipelineConfig):
    # --- Primary inputs (from JSON) ---
    cross_trace_config_path: str = ""
    improvement_id: int = 0
    num_runtime_iterations: int = 10
    gpu_wait_timeout_minutes: int = 30
    use_smaller_model_for_runtime: bool = False
    disable_new_feature_for_runtime: bool = False

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
    target_run_command: str = ""

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

        num_runtime_iterations = data.get("num_runtime_iterations", 10)
        if not isinstance(num_runtime_iterations, int) or num_runtime_iterations <= 0:
            errors.append('"num_runtime_iterations" must be a positive integer.')

        gpu_wait_timeout_minutes = data.get("gpu_wait_timeout_minutes", 30)
        if not isinstance(gpu_wait_timeout_minutes, (int, float)) or gpu_wait_timeout_minutes <= 0:
            errors.append('"gpu_wait_timeout_minutes" must be a positive number.')

        use_smaller_model_for_runtime = data.get("use_smaller_model_for_runtime", False)
        if not isinstance(use_smaller_model_for_runtime, bool):
            errors.append('"use_smaller_model_for_runtime" must be a boolean.')

        disable_new_feature_for_runtime = data.get("disable_new_feature_for_runtime", False)
        if not isinstance(disable_new_feature_for_runtime, bool):
            errors.append('"disable_new_feature_for_runtime" must be a boolean.')

        thinking_mode = data.get("thinking-mode", "deep")
        if thinking_mode not in ("normal", "deep"):
            errors.append('"thinking-mode" must be "normal" or "deep".')

        code_port_plan_skip_review = data.get("code_port_plan_skip_review", False)
        test_plan_skip_review = data.get("test_plan_skip_review", False)
        code_gen_skip_review = data.get("code_gen_skip_review", False)

        if errors:
            raise ValueError(
                "Code gen config errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        disallowed_modules = data.get("disallowed_modules", [])

        config = cls(
            cross_trace_config_path=cross_trace_config_path,
            improvement_id=improvement_id,
            source_code_dir=os.path.abspath(source_code_dir),
            output_dir=os.path.abspath(output_dir),
            num_code_port_plan_iterations=num_code_port_plan_iterations,
            num_test_plan_iterations=num_test_plan_iterations,
            num_code_gen_iterations=num_code_gen_iterations,
            num_runtime_iterations=num_runtime_iterations,
            gpu_wait_timeout_minutes=int(gpu_wait_timeout_minutes),
            use_smaller_model_for_runtime=use_smaller_model_for_runtime,
            disable_new_feature_for_runtime=disable_new_feature_for_runtime,
            plan_step=improvement_id,
            disallowed_modules=disallowed_modules,
            thinking_mode=thinking_mode,
            code_port_plan_skip_review=code_port_plan_skip_review,
            test_plan_skip_review=test_plan_skip_review,
            code_gen_skip_review=code_gen_skip_review,
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
        self.target_run_command = target_trace.run_command

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

        if not self.disallowed_modules:
            self.disallowed_modules = (
                _infer_disallowed_modules(self.source_framework)
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

    def get_target_branch_name(self) -> str:
        model_slug = self.model.replace("/", "_").replace("-", "_")
        return (
            f"auto_code_gen_{model_slug}"
            f"_isl_{self.isl}_osl_{self.osl}_b_{self.batch_size}"
            f"_plan_{self.improvement_id}"
            f"_commit_{self.target_commit_id[:6]}"
        )

    def verify_target_branch(self) -> bool:
        """Verify the target source code repo is on the correct branch.

        Attempts to switch if not already on the branch.
        Returns True on success, False if the repo needs manual cleanup.
        """
        branch = self.get_target_branch_name()
        cwd = self.source_code_dir

        result = _run_git(["branch", "--show-current"], cwd, check=False)
        current = result.stdout.strip()

        if current == branch:
            head = _run_git(["rev-parse", "HEAD"], cwd).stdout.strip()
            ancestor_check = _run_git(
                ["merge-base", "--is-ancestor", self.target_commit_id, head],
                cwd, check=False,
            )
            if ancestor_check.returncode == 0:
                print(f"Target repo on correct branch '{branch}', HEAD={head[:12]}")
                return True
            else:
                print(
                    f"WARNING: Branch '{branch}' HEAD ({head[:12]}) does NOT "
                    f"descend from tested commit ({self.target_commit_id[:12]})."
                )
                return False

        status = _run_git(["status", "--porcelain"], cwd, check=False)
        dirty = [
            l for l in status.stdout.strip().splitlines()
            if l and not l.startswith("??")
        ]
        if dirty:
            print(
                f"WARNING: Cannot switch to branch '{branch}': "
                f"repo at {cwd} has uncommitted changes.\n"
                f"  Please clean the repo before running runtime iterations."
            )
            return False

        branch_check = _run_git(["branch", "--list", branch], cwd, check=False)
        if branch not in branch_check.stdout:
            print(
                f"WARNING: Branch '{branch}' does not exist in {cwd}.\n"
                f"  Run the code generation pipeline first."
            )
            return False

        result = _run_git(["checkout", branch], cwd, check=False)
        if result.returncode != 0:
            print(
                f"WARNING: Cannot switch to branch '{branch}': "
                f"{result.stderr.strip()}\n"
                f"  Please clean the repo at {cwd} and retry."
            )
            return False

        head = _run_git(["rev-parse", "HEAD"], cwd).stdout.strip()
        print(f"Switched to branch '{branch}', HEAD={head[:12]}")
        return True

    _MODE_PARAMS = {
        "deep": {
            "model": "claude-opus-4-6[1m]",
            "thinking": {"type": "adaptive"},
            "effort": "max",
            "max_thinking_tokens": 1048576,
        },
        "normal": {
            "model": "claude-sonnet-4-6",
            "thinking": {"type": "adaptive"},
            "effort": "medium",
            "max_thinking_tokens": 65536,
        },
    }

    def make_claude_config(self) -> ClaudeConfig:
        params = self._MODE_PARAMS[self.thinking_mode]
        return ClaudeConfig(
            model=params["model"],
            allowed_tools=["Read", "Write", "Bash"],
            perm_mode="acceptEdits",
            cwd=self.output_dir,
            thinking=params["thinking"],
            effort=params["effort"],
            max_thinking_tokens=params["max_thinking_tokens"],
        )


@dataclass
class BugFixConfig(PipelineConfig):
    repo_path: str = ""
    build_dir: str = ""
    source_branch: str = ""
    target_branch: str = ""
    source_fix_commit: str = ""
    bug_description: str = ""
    issue_id: str = ""
    build_command: str = ""
    test_command: str = ""
    max_build_test_retries: int = 3
    use_combined_code_and_test_port_plan: bool = True

    @classmethod
    def from_json(cls, path: str) -> "BugFixConfig":
        with open(path) as f:
            data = json.load(f)

        errors = []

        # Validate required fields
        repo_path = data.get("repo_path", "")
        if not repo_path:
            errors.append('"repo_path" is required.')
        elif not os.path.isdir(repo_path):
            errors.append(f'"repo_path" directory not found: {repo_path}')

        source_branch = data.get("source_branch", "")
        if not source_branch:
            errors.append('"source_branch" is required.')

        target_branch = data.get("target_branch", "")
        if not target_branch:
            errors.append('"target_branch" is required.')

        source_fix_commit = data.get("source_fix_commit", "")
        if not source_fix_commit:
            errors.append('"source_fix_commit" is required.')

        output_dir = data.get("output_dir", "")
        if not output_dir:
            errors.append('"output_dir" is required.')

        build_command = data.get("build_command", "")
        if not build_command:
            errors.append('"build_command" is required.')

        test_command = data.get("test_command", "")
        if not test_command:
            errors.append('"test_command" is required.')

        # Optional fields with defaults
        build_dir = data.get("build_dir", repo_path)
        bug_description = data.get("bug_description", "")
        issue_id = data.get("issue_id", "")
        max_build_test_retries = data.get("max_build_test_retries", 3)
        use_combined_code_and_test_port_plan = data.get("use_combined_code_and_test_port_plan", True)
        disallowed_modules = data.get("disallowed_modules", [])
        thinking_mode = data.get("thinking-mode", "deep")

        # Iteration counts
        num_code_port_plan_iterations = data.get("num_code_port_plan_iterations", 3)
        num_test_plan_iterations = data.get("num_test_plan_iterations", 3)
        num_code_gen_iterations = data.get("num_code_gen_iterations", 3)

        code_port_plan_skip_review = data.get("code_port_plan_skip_review", False)
        test_plan_skip_review = data.get("test_plan_skip_review", False)
        code_gen_skip_review = data.get("code_gen_skip_review", False)

        if thinking_mode not in ("normal", "deep"):
            errors.append('"thinking-mode" must be "normal" or "deep".')

        if errors:
            raise ValueError(
                "Bug fix config errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return cls(
            output_dir=os.path.abspath(output_dir),
            source_code_dir=os.path.abspath(repo_path),
            num_code_port_plan_iterations=num_code_port_plan_iterations,
            num_test_plan_iterations=num_test_plan_iterations,
            num_code_gen_iterations=num_code_gen_iterations,
            disallowed_modules=disallowed_modules,
            thinking_mode=thinking_mode,
            repo_path=os.path.abspath(repo_path),
            build_dir=os.path.abspath(build_dir),
            source_branch=source_branch,
            target_branch=target_branch,
            source_fix_commit=source_fix_commit,
            bug_description=bug_description,
            issue_id=issue_id,
            build_command=build_command,
            test_command=test_command,
            max_build_test_retries=max_build_test_retries,
            use_combined_code_and_test_port_plan=use_combined_code_and_test_port_plan,
            code_port_plan_skip_review=code_port_plan_skip_review,
            test_plan_skip_review=test_plan_skip_review,
            code_gen_skip_review=code_gen_skip_review,
        )

    # Reuse the same _MODE_PARAMS pattern from CodeGenConfig
    def make_claude_config(self) -> ClaudeConfig:
        params = CodeGenConfig._MODE_PARAMS[self.thinking_mode]
        return ClaudeConfig(
            model=params["model"],
            allowed_tools=["Read", "Write", "Bash"],
            perm_mode="acceptEdits",
            cwd=self.output_dir,
            thinking=params["thinking"],
            effort=params["effort"],
            max_thinking_tokens=params["max_thinking_tokens"],
        )


def load_config_and_use_case(path: str):
    """Load a pipeline config and its corresponding UseCase from a JSON file.

    The JSON must have a "use_case" field ("llm_framework" or "bug_fix").
    Defaults to "llm_framework" if not specified.
    """
    with open(path) as f:
        data = json.load(f)

    use_case_name = data.get("use_case", "llm_framework")

    if use_case_name == "llm_framework":
        from auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase
        return CodeGenConfig.from_json(path), LLMFrameworkUseCase()
    elif use_case_name == "bug_fix":
        from auto_code_gen.use_cases.bug_fix import BugFixUseCase
        return BugFixConfig.from_json(path), BugFixUseCase()
    else:
        raise ValueError(f'Unknown use_case: "{use_case_name}". Must be "llm_framework" or "bug_fix".')
