import json
import os
import re
import subprocess
from dataclasses import dataclass, field

from common.claude_utils import ClaudeConfig
from llm.auto_code_gen.claude_utils import CodeGenClaudeConfig
from llm.auto_code_gen.agents.config import MODE_PARAMS, resolve_agent_config

from llm.auto_analyze.configs.cross_trace_config import (
    CrossTraceConfig,
    TraceResult,
    CROSS_IMPROVEMENT_FILE,
)

from llm.auto_analyze.configs.single_trace_config import (
    MEDIAN_BLOCK_FILE,
    HIGH_LEVEL_OPS_FILE,
)

from common.iteration_utils import resolve_output_file


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


def _ensure_worktree(source_dir: str, worktree_name: str) -> str:
    """Give this run its own git worktree.

    Concurrent runs (a new plan's codegen while review comments are being
    fixed on another PR) cannot share one working tree: the agent edits the
    tree in place and the PR flow never commits, so a second run would corrupt
    the first. A git worktree gives every run its own checkout while sharing
    one .git object store.

    Worktree path: <parent of source_code_dir>/<worktree_name>. The caller
    passes "vllm_worktree_<output_dir basename>" so the directory says both
    what it is (a vllm worktree) and which run it belongs to.

    Idempotent: returns the existing run-named worktree when present, and
    creates it from source_dir's current HEAD otherwise. source_dir itself is
    left untouched (the main clone keeps whatever branch it had checked out).
    """
    source_dir = os.path.abspath(source_dir)
    parent = os.path.dirname(source_dir)
    worktree = os.path.join(parent, worktree_name)

    if source_dir == worktree:
        return source_dir  # already this run's worktree (resumed run)

    # Refuse when source_dir is itself a linked worktree: worktrees hang off
    # the main clone, and creating one from another worktree's path nests
    # confusingly. Run from the main clone instead.
    if os.path.isfile(os.path.join(source_dir, ".git")):
        raise RuntimeError(
            f"source_code_dir {source_dir} is itself a linked worktree; "
            f"point the config at the main clone so {worktree} can be created."
        )
    if os.path.exists(os.path.join(worktree, ".git")):
        # A file (not dir) at .git is what a linked worktree looks like; a
        # plain dir means a full clone someone placed there -- refuse to touch.
        if os.path.isfile(os.path.join(worktree, ".git")):
            return worktree
        raise RuntimeError(
            f"Worktree path {worktree} exists but is not a linked worktree "
            f"(no .git file). Move it aside or use a different output_dir."
        )
    if os.path.exists(worktree):
        raise RuntimeError(
            f"Worktree path {worktree} already exists and is not a git "
            f"worktree. Move it aside or pick another run name."
        )

    head = _run_git(["rev-parse", "HEAD"], source_dir, check=False)
    if head.returncode != 0:
        raise RuntimeError(
            f"Cannot create worktree {worktree}: {source_dir} is not a git "
            f"repository ({head.stderr.strip()})"
        )
    result = _run_git(["worktree", "add", "--detach", worktree,
                       head.stdout.strip()], source_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed for {worktree}: {result.stderr.strip()}"
        )
    print(f"  [worktree] created {worktree} at {head.stdout.strip()[:12]}")
    return worktree


def _apply_runtime_bench_num_prompts(run_command: str, num_prompts: int) -> str:
    """Rewrite the bench command's sample count for gate measurements.

    The command inherited from the trace's run_params.txt was tuned for fast
    profiling, and under max-concurrency 1 its wall time is roughly
    num_prompts x output_len x step_time -- a few dozen seconds. Shared-cluster
    nodes inject host-side transients of the same order that uniformly inflate
    a whole arm (observed: +240us/step for ~60s, one arm per measurement,
    kernel-identical under NSYS), which is enough to flip an A/B verdict.
    Longer arms dilute such a transient below the typical effect size.
    """
    if not run_command or num_prompts <= 0:
        return run_command

    # vllm bench serve uses --num-prompts; bench latency uses --num-samples
    # (--num-prompts is not a flag there). Support both " " and "=" forms.
    for flag in ("--num-prompts", "--num-samples"):
        for pattern, repl in (
                (r"{} (\d+)".format(flag), "{} {}".format(flag, num_prompts)),
                (r"{}=(\d+)".format(flag), "{}={}".format(flag, num_prompts))):
            rewritten, n = re.subn(pattern, repl, run_command)
            if n:
                return rewritten
    return run_command


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
    use_knowledge_base: bool = False
    generate_pr_commands: bool = False
    pr_base_branch: str = "main"
    pr_remote: str = "origin"
    config_json_path: str = ""
    agent: dict | None = None
    agent_config_path: str = ""

    def make_agent_config(self, agent_config_path=None):
        if agent_config_path:
            self.agent_config_path = os.path.abspath(os.path.expanduser(agent_config_path))
        return resolve_agent_config(
            self.agent, cwd=self.output_dir, mode=self.thinking_mode,
            config_path=self.config_json_path,
            agent_config_path=self.agent_config_path or None,
        )


@dataclass
class CodeGenConfig(PipelineConfig):
    # --- Primary inputs (from JSON) ---
    cross_trace_config_path: str = ""
    improvement_id: int = 0
    num_runtime_iterations: int = 10
    gpu_wait_timeout_minutes: int = 30
    # "local"  — use local GPUs via nvidia-smi (default, unchanged behavior).
    # "remote" — delegate all GPU phases to OpenShift cluster Jobs via cluster_runner.py.
    runtime_backend: str = "local"
    # remote_* fields: all have working defaults for the known cluster setup.
    # Only override if your cluster differs. remote_disable_env_var is intentionally
    # absent — it is auto-detected at runtime from patch/summary files.
    remote_namespace: str = ""
    remote_pvc_name: str = "vllm-workspace"    # RWX NFS: vllm source, pip-cache, output
    remote_model_pvc_name: str = ""            # required when runtime_backend="remote":
                                               # HF model cache PVC mounted at /model-cache.
                                               # Sets HF_HOME=/model-cache. Model is downloaded
                                               # there on first run if not already cached.
    remote_image: str = "quay.io/example/vllm-sandbox-nsys:latest"
    remote_gpu_count: int = 4
    remote_gpu_product: str = "NVIDIA-B200"
    # Per-run PVC subdir under /workspace holding this run's vllm tree.
    # Empty (default) derives "pvc_<output_dir basename>" so every run gets a
    # run-named dir on the PVC -- concurrent runs never share /workspace/vllm.
    remote_vllm_subdir: str = ""
    # Command used to launch the API server before ``vllm bench serve``.
    # Keep workload-specific server options here so remote runs reproduce the
    # traced configuration instead of guessing them from the client command.
    remote_server_command: str = ""
    # Extra "KEY=VALUE" environment for remote Job containers, e.g.
    # HF_HUB_OFFLINE=1 when the model cache PVC is read-only.
    remote_extra_env: list[str] = field(default_factory=list)
    # Override the number of prompts in the inherited bench command. The trace's
    # run_params.txt command was tuned for fast profiling (~60s of decode); a
    # shared-cluster node can inject tens-of-seconds host-side transients that
    # uniformly inflate a whole arm, so A/B gate measurements need longer arms.
    # Applied by rewriting --num-prompts (vllm bench serve) / --num-samples
    # (bench latency) in target_run_command; 0 (default) keeps the trace value.
    runtime_bench_num_prompts: int = 0
    max_investigation_revisions: int = 2
    use_smaller_model_for_runtime: bool = False
    disable_new_feature_for_runtime: bool = False
    additional_benchmark_configs: list[dict] = field(default_factory=list)

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

        runtime_bench_num_prompts = data.get("runtime_bench_num_prompts", 0)
        if not isinstance(runtime_bench_num_prompts, int) or runtime_bench_num_prompts < 0:
            errors.append('"runtime_bench_num_prompts" must be a non-negative integer (0 = keep the trace value).')

        use_knowledge_base = data.get("use_knowledge_base", False)
        if not isinstance(use_knowledge_base, bool):
            errors.append('"use_knowledge_base" must be a boolean.')

        max_investigation_revisions = data.get("max_investigation_revisions", 2)
        if not isinstance(max_investigation_revisions, int) or max_investigation_revisions < 0:
            errors.append('"max_investigation_revisions" must be a non-negative integer.')

        use_smaller_model_for_runtime = data.get("use_smaller_model_for_runtime", False)
        if not isinstance(use_smaller_model_for_runtime, bool):
            errors.append('"use_smaller_model_for_runtime" must be a boolean.')

        disable_new_feature_for_runtime = data.get("disable_new_feature_for_runtime", False)
        if not isinstance(disable_new_feature_for_runtime, bool):
            errors.append('"disable_new_feature_for_runtime" must be a boolean.')

        additional_benchmark_configs = data.get("additional_benchmark_configs", [])
        if not isinstance(additional_benchmark_configs, list):
            errors.append('"additional_benchmark_configs" must be a list.')

        thinking_mode = data.get("thinking-mode", "deep")
        if thinking_mode not in ("normal", "deep"):
            errors.append('"thinking-mode" must be "normal" or "deep".')

        runtime_backend = data.get("runtime_backend", "local")
        if runtime_backend not in ("local", "remote"):
            errors.append('"runtime_backend" must be "local" or "remote".')
        _default_image = "quay.io/example/vllm-sandbox-nsys:latest"
        remote_namespace = data.get("remote_namespace", "")
        remote_pvc_name = data.get("remote_pvc_name", "vllm-workspace")
        remote_model_pvc_name = data.get("remote_model_pvc_name", "")
        remote_image = data.get("remote_image", _default_image)
        remote_gpu_count = int(data.get("remote_gpu_count", 4))
        remote_gpu_product = data.get("remote_gpu_product", "NVIDIA-B200")
        remote_server_command = data.get("remote_server_command", "")
        remote_vllm_subdir = str(data.get("remote_vllm_subdir", "")).strip("/")
        if remote_vllm_subdir:
            import re as _re
            if not _re.fullmatch(r"[A-Za-z0-9._-]+", remote_vllm_subdir):
                errors.append(
                    '"remote_vllm_subdir" must be a single path segment '
                    '(letters, digits, ".", "_", "-"): {!r}'.format(remote_vllm_subdir)
                )
        elif output_dir:
            # Default: "pvc_" + the run's own name, so the PVC tree is
            # self-identifying and concurrent runs never share /workspace/vllm.
            remote_vllm_subdir = "pvc_" + (
                os.path.basename(os.path.normpath(output_dir)) or "vllm")
        remote_extra_env = []
        for entry in data.get("remote_extra_env", []) or []:
            entry = str(entry)
            if "=" not in entry:
                errors.append('"remote_extra_env" entries must be "KEY=VALUE".')
            else:
                remote_extra_env.append(entry)

        if runtime_backend == "remote" and not remote_model_pvc_name:
            errors.append(
                '"remote_model_pvc_name" is required when runtime_backend is "remote".'
            )

        code_port_plan_skip_review = data.get("code_port_plan_skip_review", False)
        test_plan_skip_review = data.get("test_plan_skip_review", False)
        code_gen_skip_review = data.get("code_gen_skip_review", False)
        generate_pr_commands = data.get("generate_pr_commands", False)
        pr_base_branch = data.get("pr_base_branch", "main")
        pr_remote = data.get("pr_remote", "origin")

        if errors:
            raise ValueError(
                "Code gen config errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        disallowed_modules = data.get("disallowed_modules", [])

        config = cls(
            agent=data.get("agent"),
            cross_trace_config_path=cross_trace_config_path,
            improvement_id=improvement_id,
            source_code_dir=os.path.abspath(source_code_dir),
            output_dir=os.path.abspath(output_dir),
            num_code_port_plan_iterations=num_code_port_plan_iterations,
            num_test_plan_iterations=num_test_plan_iterations,
            num_code_gen_iterations=num_code_gen_iterations,
            num_runtime_iterations=num_runtime_iterations,
            gpu_wait_timeout_minutes=int(gpu_wait_timeout_minutes),
            runtime_bench_num_prompts=runtime_bench_num_prompts,
            runtime_backend=runtime_backend,
            remote_namespace=remote_namespace,
            remote_pvc_name=remote_pvc_name,
            remote_model_pvc_name=remote_model_pvc_name,
            remote_image=remote_image,
            remote_gpu_count=remote_gpu_count,
            remote_gpu_product=remote_gpu_product,
            remote_server_command=remote_server_command,
            remote_vllm_subdir=remote_vllm_subdir,
            remote_extra_env=remote_extra_env,
            use_knowledge_base=use_knowledge_base,
            max_investigation_revisions=max_investigation_revisions,
            use_smaller_model_for_runtime=use_smaller_model_for_runtime,
            disable_new_feature_for_runtime=disable_new_feature_for_runtime,
            additional_benchmark_configs=additional_benchmark_configs,
            plan_step=improvement_id,
            disallowed_modules=disallowed_modules,
            thinking_mode=thinking_mode,
            code_port_plan_skip_review=code_port_plan_skip_review,
            test_plan_skip_review=test_plan_skip_review,
            code_gen_skip_review=code_gen_skip_review,
            generate_pr_commands=generate_pr_commands,
            pr_base_branch=pr_base_branch,
            pr_remote=pr_remote,
        )
        config._load_from_cross_trace()
        config.config_json_path = os.path.abspath(path)
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
        self.target_run_command = _apply_runtime_bench_num_prompts(
            target_trace.run_command, self.runtime_bench_num_prompts)

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

        # Source codes: use source_code_dir for target, run_params for source.
        # The target gets its own run-named worktree first: the agent edits the
        # tree in place, so concurrent runs need separate working trees.
        run_name = os.path.basename(os.path.normpath(self.output_dir)) or "run"
        self.source_code_dir = _ensure_worktree(
            self.source_code_dir, "vllm_worktree_" + run_name,
        )
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

        plan_path = resolve_output_file(cross_cfg.output_dir, CROSS_IMPROVEMENT_FILE)
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

    _MODE_PARAMS = MODE_PARAMS

    def make_claude_config(self) -> ClaudeConfig:
        params = self._MODE_PARAMS[self.thinking_mode]
        # Keep the profiling persona the target's codegen pipeline has
        # always run with (see llm/auto_code_gen/claude_utils).
        return CodeGenClaudeConfig(
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

    The optional "use_case" field must be "llm_framework", which is also the
    default. Bug-fix porting lives in the ``auto_bug_fix`` package.
    """
    with open(path) as f:
        data = json.load(f)

    use_case_name = data.get("use_case", "llm_framework")

    if use_case_name == "llm_framework":
        from llm.auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase
        return CodeGenConfig.from_json(path), LLMFrameworkUseCase()
    if use_case_name == "bug_fix":
        raise ValueError(
            'use_case "bug_fix" was removed from llm.auto_code_gen. Use the '
            "auto_bug_fix pipeline instead: python -m auto_bug_fix.run_bug_fix "
            "<config.yaml>"
        )
    raise ValueError(f'Unknown use_case: "{use_case_name}". Must be "llm_framework".')
