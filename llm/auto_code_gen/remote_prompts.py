# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Remote runtime prompts: GPU phases delegated to OpenShift cluster Jobs.

Selected by runtime_backend = "remote" in CodeGenConfig. The local backend
is byte-identical when runtime_backend = "local".

Three prompt classes:

  RemoteApplyCodePrompt
      Apply the patch to the local source tree and rsync it to the PVC.
      No compilation — that happens inside the GPU pod as the first step.
      Replaces: ApplyCodeAndCompilePrompt

  RemoteRunIterationPrompt
      Call cluster_runner.py run-iteration. One GPU Job does:
        compile → bench patched → bench baseline → lm_eval (optional).
      Replaces: RunAndLogPrompt (bench path) and RunLMEvalPrompt (lm_eval path).
      The lm_eval flag is toggled by which selector function constructs it.

  RemoteRunDiagnosticPrompt
      Call cluster_runner.py run-diagnostic. One GPU Job does:
        compile → nsys profile patched → nsys profile baseline → SQLite export.
      The agent then runs Steps 2-5 of DiagnosticProfileAndAnalyzePrompt locally
      in the SAME Claude session so the full reasoning history is preserved.
      Replaces: the GPU portion of DiagnosticProfileAndAnalyzePrompt (Step 0-1).

File contract: identical to the local prompts. runtime/V{i}/ layout is unchanged
so every downstream phase (perf gate, investigate, fix, summaries) is unaffected.

PVC layout (single PVC mounted at /workspace):
    /workspace/vllm/            patched vllm source
    /model-cache/               model weights (from model PVC, read-only)
    /workspace/pip-cache/       pip cache (incremental compile)
    /workspace/output/<nonce>/  per-job results (copied back by cluster_runner.py)
"""

import os
from dataclasses import dataclass
from typing import ClassVar

from llm.auto_code_gen.code_gen_prompts import DIAGNOSTIC_PROFILE_DIR, RUNTIME_DIR

REMOTE_VLLM_DIR = "/workspace/vllm"


# ── RemoteApplyCodePrompt ─────────────────────────────────────────────────────


@dataclass
class RemoteApplyCodePrompt:
    """Apply patch locally + rsync to PVC. No compile."""

    context: str
    patch_file: str
    iteration: int
    output_status_file: str
    remote_namespace: str
    remote_pvc_name: str

    prompt_template: ClassVar[str] = """
{context}

<definitions>
<patch_file>
{patch_file}
</patch_file>
<remote_namespace>{remote_namespace}</remote_namespace>
<remote_pvc_name>{remote_pvc_name}</remote_pvc_name>
<remote_vllm_dir>{remote_vllm_dir}</remote_vllm_dir>
</definitions>

<instructions>
Apply the code patch <patch_file> to the local vllm source tree and sync the \
changed files to the cluster PVC (runtime iteration {iteration}).

Do NOT run pip install, cmake, or any build command locally — compilation \
happens as the first step inside the GPU pod, in the correct CUDA environment.

IMPORTANT: If <target_branch> is provided in the context, ensure you are on \
that branch before proceeding.

Step 1: Check if patch is already committed (post-rebase case)
- `git -C <target_source_code_dir> log --oneline -1`
- `git -C <target_source_code_dir> diff --stat HEAD~1..HEAD`
- If the diff matches the patch file, print "Patch already committed. Skipping \
apply." and skip Step 2.

Step 2: Verify clean repository state and correct branch
- If <target_branch> is provided: verify with \
`git -C <target_source_code_dir> branch --show-current`. If wrong branch, ERROR and STOP.
- `git -C <target_source_code_dir> status --porcelain`. If ANY tracked files \
are modified:
    - Write "REPO_NOT_CLEAN" to <output_dir>/{output_status_file}
    - STOP.
- Apply the patch:
    - `cd <target_source_code_dir> && git apply <output_dir>/{patch_file}`
    - On failure: `git apply --3way <output_dir>/{patch_file}`
    - On failure: `patch -p1 < <output_dir>/{patch_file}`
    - Verify with `git -C <target_source_code_dir> diff --stat`
    - ALL modifications must be inside <target_source_code_dir>.

Step 3: Sync changed files to the cluster PVC
Find a running pod in namespace <remote_namespace> that has the PVC \
<remote_pvc_name> mounted:
  `oc get pods -n {remote_namespace} -o json | python3 -c "import json,sys; \
[print(p['metadata']['name']) for p in json.load(sys.stdin)['items'] \
if p.get('status',{{}}).get('phase')=='Running' and \
any(v.get('persistentVolumeClaim',{{}}).get('claimName')=='{remote_pvc_name}' \
for v in p['spec'].get('volumes',[]))]"`

If no running pod found, create a helper pod (IMPORTANT: use ubi:latest, NOT ubi-minimal — ubi-minimal lacks tar which oc cp/rsync requires):
  `oc run pvc-helper -n {remote_namespace} \
--image=registry.access.redhat.com/ubi9/ubi:latest \
--overrides='{{"spec":{{"containers":[{{"name":"pvc-helper","image":"registry.access.redhat.com/ubi9/ubi:latest","command":["sleep","infinity"],"volumeMounts":[{{"name":"workspace","mountPath":"/workspace"}}]}}],"volumes":[{{"name":"workspace","persistentVolumeClaim":{{"claimName":"{remote_pvc_name}"}}}}]}}}}' \
--restart=Never`
  Wait for the pod to be Running: `oc wait -n {remote_namespace} pod/pvc-helper --for=condition=Ready --timeout=120s`

With HELPER_POD found, rsync:
  `oc rsync -n {remote_namespace} <target_source_code_dir>/ \
HELPER_POD:{remote_vllm_dir}/ --no-perms --delete`

If oc rsync is unavailable, copy only the changed files:
  - Get changed files: `git -C <target_source_code_dir> diff --name-only`
  - For each: `oc cp -n {remote_namespace} <target_source_code_dir>/<file> \
HELPER_POD:{remote_vllm_dir}/<file>`

Verify the sync exit code is 0. If non-zero, print the error and STOP.

Step 4: Write success status
- Write "SUCCESS" to <output_dir>/{output_status_file}
</instructions>

<output>
- <output_dir>/{output_status_file}
</output>
"""

    def prompt(self) -> str:
        return self.prompt_template.format(
            context=self.context,
            patch_file=self.patch_file,
            iteration=self.iteration,
            output_status_file=self.output_status_file,
            remote_namespace=self.remote_namespace,
            remote_pvc_name=self.remote_pvc_name,
            remote_vllm_dir=REMOTE_VLLM_DIR,
        )


def gen_RemoteApplyCodePrompt(
    context: str,
    patch_file: str,
    iteration: int,
    remote_namespace: str = "",
    remote_pvc_name: str = "vllm-cache",
    remote_model_pvc_name: str = "",
):
    return RemoteApplyCodePrompt(
        context=context,
        patch_file=patch_file,
        iteration=iteration,
        output_status_file=os.path.join(
            RUNTIME_DIR, f"V{iteration}", "apply_status.txt"
        ),
        remote_namespace=remote_namespace,
        remote_pvc_name=remote_pvc_name,
    )


# ── RemoteRunIterationPrompt ──────────────────────────────────────────────────


@dataclass
class RemoteRunIterationPrompt:
    """Compile + bench patched + bench baseline + optional lm_eval in GPU Job(s)."""

    context: str
    execution_command: str
    runtime_logs_dir: str
    iteration: int
    disable_new_feature: bool
    remote_namespace: str
    remote_pvc_name: str
    remote_model_pvc_name: str
    remote_image: str
    remote_gpu_count: int
    remote_gpu_product: str
    disable_env_var: str
    run_lm_eval: bool
    gpu_wait_timeout_minutes: int
    cluster_runner_path: str
    parallel: bool = False
    output_file: str = "lm_eval.txt"

    prompt_template: ClassVar[str] = """
{context}

<definitions>
<runtime_logs_dir>
{runtime_logs_dir}
</runtime_logs_dir>
<reference_execution_command>
{execution_command}
</reference_execution_command>
</definitions>

<instructions>
Measure the patched code (runtime iteration {iteration}) by submitting a GPU Job \
to the OpenShift cluster. You have NO local GPUs. Compilation, benchmarking, and \
lm_eval all run inside a single GPU pod. The workloads are fixed.

Step 1: Ensure output directory exists
- `mkdir -p {runtime_logs_dir}`

Step 2: Determine the feature disable env var
- The auto-detected disable env var is: `{disable_env_var}`
- If empty, search the latest summary/patch files in <output_dir> for a \
VLLM_DISABLE_* name and use that instead.

Step 3: Submit Job and wait
Run exactly:
```
python {cluster_runner_path} {subcommand} \
  --namespace {remote_namespace} \
  --pvc-name {remote_pvc_name} \
  --image {remote_image} \
  --gpu-count {remote_gpu_count} \
  --gpu-product {remote_gpu_product} \
  --bench-command {bench_command_quoted} \
  --disable-env-var {disable_env_var} \
  --out-dir {runtime_logs_dir} \
  --gpu-wait-timeout-minutes {gpu_wait_timeout_minutes}{model_pvc_flag}{run_lm_eval_flag}
```

This call BLOCKS until the Job(s) complete. Kubernetes schedules each pod when \
{remote_gpu_count} GPUs of type {remote_gpu_product} are free — do NOT \
sleep-loop while it says "waiting for pod".

{parallel_desc}

If the command exits non-zero:
- Check {runtime_logs_dir}/run_status.txt for the failure stage.
- BUILD_FAILED → read {runtime_logs_dir}/build.log
- BENCH_FAILED → read {runtime_logs_dir}/run_combined.log
- BASELINE_FAILED → read {runtime_logs_dir}/baseline_combined.log
- Write "FAILED" to {runtime_logs_dir}/run_status.txt if not already set.
- STOP — these logs are input to the investigation step.

Step 4: Verify output files
Confirm in {runtime_logs_dir}:
- run_status.txt → must contain "COMPLETED"
- bench_result.json → must contain numeric `tokens_per_second`
- baseline_bench_result.json
- perf_comparison.txt
- run_combined.log
- baseline_combined.log
{lm_eval_verify}

Step 5: Feature verification
- Read {runtime_logs_dir}/run_combined.log.
- {feature_check}
</instructions>

<output>
- {runtime_logs_dir}/run_status.txt
- {runtime_logs_dir}/build.log
- {runtime_logs_dir}/runner.log
- {runtime_logs_dir}/run_command.txt
- {runtime_logs_dir}/run_stdout.log
- {runtime_logs_dir}/run_stderr.log
- {runtime_logs_dir}/run_combined.log
- {runtime_logs_dir}/run_exit_code.txt
- {runtime_logs_dir}/bench_result.json
- {runtime_logs_dir}/baseline_command.txt
- {runtime_logs_dir}/baseline_stdout.log
- {runtime_logs_dir}/baseline_stderr.log
- {runtime_logs_dir}/baseline_combined.log
- {runtime_logs_dir}/baseline_exit_code.txt
- {runtime_logs_dir}/perf_comparison.txt
{lm_eval_output}
</output>
"""

    def prompt(self) -> str:
        import shlex

        bench_quoted = shlex.quote(self.execution_command)
        subcommand = "run-iteration-parallel" if self.parallel else "run-iteration"
        model_pvc_flag = (
            f" \\\n  --model-pvc-name {self.remote_model_pvc_name}"
            if self.remote_model_pvc_name
            else ""
        )
        run_lm_eval_flag = " \\\n  --run-lm-eval" if self.run_lm_eval else ""

        if self.parallel:
            parallel_desc = (
                "Parallel mode: submits bench, baseline"
                + (", and lm_eval" if self.run_lm_eval else "")
                + " as separate GPU pods that run concurrently. "
                "Each pod does venv setup (cmake skipped) then its phase. "
                "Results are merged into --out-dir."
            )
        else:
            parallel_desc = (
                "The pod runs: incremental compile → bench (feature ON) "
                "→ bench (feature OFF / baseline)"
                + (" → lm_eval" if self.run_lm_eval else "")
                + ". All output streams live."
            )

        if self.run_lm_eval:
            lm_eval_verify = (
                "- lm_eval.txt → must contain Verdict line\n"
                "- lm_eval_combined.log\n"
                "- lm_eval_enabled_exit_code.txt\n"
                "- lm_eval_disabled_exit_code.txt"
            )
            lm_eval_output = (
                f"- {self.runtime_logs_dir}/lm_eval.txt\n"
                f"- {self.runtime_logs_dir}/lm_eval_combined.log\n"
                f"- {self.runtime_logs_dir}/lm_eval_enabled_stdout.log\n"
                f"- {self.runtime_logs_dir}/lm_eval_enabled_stderr.log\n"
                f"- {self.runtime_logs_dir}/lm_eval_enabled_exit_code.txt\n"
                f"- {self.runtime_logs_dir}/lm_eval_disabled_stdout.log\n"
                f"- {self.runtime_logs_dir}/lm_eval_disabled_stderr.log\n"
                f"- {self.runtime_logs_dir}/lm_eval_disabled_exit_code.txt"
            )
        else:
            lm_eval_verify = lm_eval_output = ""

        if self.disable_new_feature:
            feature_check = (
                'Search for "is DISABLED (env override)". If found, feature was '
                "successfully disabled. If NOT found, print a WARNING."
            )
        else:
            feature_check = (
                'Search for "is ENABLED". If found, feature is active. If NOT '
                'found, print a WARNING and write "FEATURE_NOT_VERIFIED" to '
                f"{self.runtime_logs_dir}/run_status.txt with details."
            )

        return self.prompt_template.format(
            context=self.context,
            execution_command=self.execution_command,
            runtime_logs_dir=self.runtime_logs_dir,
            iteration=self.iteration,
            cluster_runner_path=self.cluster_runner_path,
            subcommand=subcommand,
            remote_namespace=self.remote_namespace,
            remote_pvc_name=self.remote_pvc_name,
            remote_image=self.remote_image,
            remote_gpu_count=self.remote_gpu_count,
            remote_gpu_product=self.remote_gpu_product,
            bench_command_quoted=bench_quoted,
            disable_env_var=self.disable_env_var,
            gpu_wait_timeout_minutes=self.gpu_wait_timeout_minutes,
            model_pvc_flag=model_pvc_flag,
            run_lm_eval_flag=run_lm_eval_flag,
            parallel_desc=parallel_desc,
            lm_eval_verify=lm_eval_verify,
            lm_eval_output=lm_eval_output,
            feature_check=feature_check,
        )


def gen_RemoteRunIterationPrompt(
    context: str,
    execution_command: str,
    runtime_logs_dir: str,
    iteration: int,
    remote_namespace: str = "",
    remote_pvc_name: str = "vllm-workspace",
    remote_model_pvc_name: str = "",
    remote_image: str = "",
    remote_gpu_count: int = 4,
    remote_gpu_product: str = "NVIDIA-B200",
    disable_env_var: str = "",
    run_lm_eval: bool = False,
    disable_new_feature: bool = False,
    gpu_wait_timeout_minutes: int = 60,
    cluster_runner_path: str = "",
    parallel: bool = False,
):
    if not cluster_runner_path:
        cluster_runner_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cluster_runner.py"
        )
    if not remote_image:
        remote_image = "quay.io/example/vllm-sandbox-nsys:latest"
    return RemoteRunIterationPrompt(
        context=context,
        execution_command=execution_command,
        runtime_logs_dir=runtime_logs_dir,
        iteration=iteration,
        disable_new_feature=disable_new_feature,
        remote_namespace=remote_namespace,
        remote_pvc_name=remote_pvc_name,
        remote_model_pvc_name=remote_model_pvc_name,
        remote_image=remote_image,
        remote_gpu_count=remote_gpu_count,
        remote_gpu_product=remote_gpu_product,
        disable_env_var=disable_env_var,
        run_lm_eval=run_lm_eval,
        gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
        cluster_runner_path=cluster_runner_path,
        parallel=parallel,
    )


# ── RemoteRunDiagnosticPrompt ─────────────────────────────────────────────────


@dataclass
class RemoteRunDiagnosticPrompt:
    """Run nsys profiling on the cluster then do all analysis locally.

    The GPU Job runs Steps 0-1 of DiagnosticProfileAndAnalyzePrompt:
      compile → nsys profile patched → nsys profile baseline → SQLite export

    After cluster_runner.py returns, this prompt instructs the agent to run
    Steps 2-5 locally in the same session:
      extract GPU ops → correlate to transformer blocks → compare → diagnostic_report.txt

    The agent reads original_analysis_dir files (extract script, reference block
    files) directly from the local filesystem — they are never uploaded to the
    cluster. The SQLite traces are copied back from the PVC and placed in
    diagnostic_dir so the rest of the prompt is identical to the local variant.
    """

    context: str
    execution_command: str
    delay_iterations: int
    feature_disable_env_var: str
    original_analysis_dir: str
    diagnostic_dir: str
    iteration: int
    trace_gpu_focus: int
    gpu_wait_timeout_minutes: int
    remote_namespace: str
    remote_pvc_name: str
    remote_model_pvc_name: str
    remote_image: str
    remote_gpu_count: int
    remote_gpu_product: str
    cluster_runner_path: str

    prompt_template: ClassVar[str] = """
{context}

<definitions>
<execution_command>
{execution_command}
</execution_command>
<delay_iterations>
{delay_iterations}
</delay_iterations>
<feature_disable_env_var>
{feature_disable_env_var}
</feature_disable_env_var>
<original_analysis_dir>
{original_analysis_dir}
</original_analysis_dir>
<diagnostic_dir>
{diagnostic_dir}
</diagnostic_dir>
<trace_gpu_focus>
{trace_gpu_focus}
</trace_gpu_focus>
</definitions>

<instructions>
Run kernel-level NSYS diagnostic profiling for runtime iteration {iteration}, \
then extract transformer block timings and identify the KEY REASONS for the \
performance result. Think hard and be extremely thorough.

All output files MUST be written to <diagnostic_dir>. Create it first: \
`mkdir -p <diagnostic_dir>`.

The NSYS profiling runs on the cluster GPU pod (Step 1). Steps 2-5 run locally \
in this same session so the full reasoning history is preserved.

Step 1: Run NSYS profiles on the cluster (output: <diagnostic_dir>/step1_nsys_profiles.txt)
- `mkdir -p {diagnostic_dir}`
- Run the following command exactly:
```
python {cluster_runner_path} run-diagnostic \
  --namespace {remote_namespace} \
  --pvc-name {remote_pvc_name} \
  --image {remote_image} \
  --gpu-count {remote_gpu_count} \
  --gpu-product {remote_gpu_product} \
  --bench-command {bench_command_quoted} \
  --disable-env-var {feature_disable_env_var} \
  --delay-iterations {delay_iterations} \
  --out-dir {diagnostic_dir} \
  --gpu-wait-timeout-minutes {gpu_wait_timeout_minutes}{model_pvc_flag}
```
This call BLOCKS until the Job completes. The pod runs:
  - incremental compile
  - nsys profile patched (feature ON)
  - nsys profile baseline (feature OFF)
  - nsys export both traces to SQLite

On completion, these files will be in <diagnostic_dir>:
  - trace_patched.nsys-rep, trace_patched.sqlite
  - trace_baseline.nsys-rep, trace_baseline.sqlite
  - step1_nsys_profiles.txt (profiling log)
  - build.log, runner.log, diagnostic_status.txt

If the command exits non-zero:
  - Read diagnostic_status.txt for the failure stage.
  - DIAGNOSTIC_BUILD_FAILED → read build.log
  - DIAGNOSTIC_NSYS_PATCHED_FAILED / DIAGNOSTIC_NSYS_BASELINE_FAILED → \
read step1_nsys_profiles.txt
  - Write "DIAGNOSTIC_FAILED" to <diagnostic_dir>/step1_nsys_profiles.txt and STOP.

Verify trace_patched.sqlite and trace_baseline.sqlite both exist and are non-empty \
before proceeding.

Step 2: Extract GPU Ops (output: <diagnostic_dir>/step2_extract_gpu_ops.txt)
- Look for the original analysis's GPU ops extraction script in <original_analysis_dir> \
(e.g., extract_gpu_ops.py or similar). Read it to understand the extraction approach.
- Run it (or adapt it) against both new SQLite traces to produce:
    <diagnostic_dir>/gpu_ops_patched.txt
    <diagnostic_dir>/gpu_ops_baseline.txt
- Focus on GPU <trace_gpu_focus> only.
- The output format must match the original gpu_ops.txt format (pipe-separated table \
with columns: Idx, Category, Short Name, Start, End, Duration, Stream, Grid, Block, \
Regs, ShMem, CommAlgo, CommSize, XferBytes, XferKind, Full Name).
- Log the number of ops extracted from each trace, any errors or adaptations needed.
- If the extraction script fails, diagnose the error, fix the script, and retry.

Step 3: Correlate to Transformer Blocks (output: <diagnostic_dir>/step3_correlation.txt)
- Read the original analysis results from <original_analysis_dir> as REFERENCE CONTEXT \
(not as ground truth — the patch may have changed kernels, added/removed ops, or \
altered execution patterns):
    - transformer_block_high_level_ops_V*.txt (latest version)
    - gpu_ops_to_blocks_V*.txt (latest version)
    - median_block_V*.txt (latest version)
- Correlate the new GPU ops (from Step 2) to transformer blocks:
    - Detect block boundaries using the same kernel signatures as the original analysis
    - Map each kernel to its high_level_op name using (kernel_short_name, stream) identity
    - If the patch added/removed kernels or changed names, adapt and document
- Select median-representative blocks (by total wall time) for both patched and baseline
- Write:
    <diagnostic_dir>/median_block_patched.txt
    <diagnostic_dir>/median_block_baseline.txt
- Log block count, block types, any differences from the original analysis.

Step 4: Compare and Identify Key Reasons (output: <diagnostic_dir>/step4_comparison.txt)
- Produce a per-operation comparison table:
    | # | Operation | Baseline (ns) | Patched (ns) | Delta (ns) | Delta (%) |
  Each row is one high_level_op. Include a TOTAL row.
- Identify the KEY REASONS for the performance result:
    - If target kernel(s) time UNCHANGED: optimization not taking effect — why?
    - If target kernel(s) SLOWER: assumption wrong for this workload — why?
    - If target kernel(s) FASTER but total unchanged: what is offsetting the gain?
    - New or removed kernels vs original analysis?
    - Does the improvement plan's predicted mechanism match observed reality?

Step 5: Write Final Diagnostic Report (output: <diagnostic_dir>/diagnostic_report.txt)
This is the single file the investigate prompt reads. It MUST contain:
    1. SUMMARY: one-paragraph assessment of root cause
    2. PER-OPERATION COMPARISON TABLE: full table from Step 4
    3. KEY REASONS: specific reasons with kernel-level evidence
    4. BLOCK STRUCTURE CHANGES: new/removed kernels, correlation adaptations
    5. RAW TIMING DATA: patched and baseline block wall times, absolute and percentage delta
    6. IMPROVEMENT PLAN VALIDATION: whether predicted mechanism matches observed reality
    7. RECOMMENDED DIRECTION: what the next fix attempt should target, justified by key reasons
</instructions>

<output>
All files in <diagnostic_dir>/:
- diagnostic_status.txt
- step1_nsys_profiles.txt
- step2_extract_gpu_ops.txt
- step3_correlation.txt
- step4_comparison.txt
- diagnostic_report.txt
- gpu_ops_patched.txt, gpu_ops_baseline.txt
- median_block_patched.txt, median_block_baseline.txt
- trace_patched.sqlite, trace_baseline.sqlite
- trace_patched.nsys-rep, trace_baseline.nsys-rep
- build.log, runner.log
</output>
"""

    def prompt(self) -> str:
        import shlex

        model_pvc_flag = (
            f" \\\n  --model-pvc-name {self.remote_model_pvc_name}"
            if self.remote_model_pvc_name
            else ""
        )
        return self.prompt_template.format(
            context=self.context,
            execution_command=self.execution_command,
            delay_iterations=self.delay_iterations,
            feature_disable_env_var=self.feature_disable_env_var,
            original_analysis_dir=self.original_analysis_dir,
            diagnostic_dir=self.diagnostic_dir,
            iteration=self.iteration,
            trace_gpu_focus=self.trace_gpu_focus,
            gpu_wait_timeout_minutes=self.gpu_wait_timeout_minutes,
            cluster_runner_path=self.cluster_runner_path,
            remote_namespace=self.remote_namespace,
            remote_pvc_name=self.remote_pvc_name,
            remote_image=self.remote_image,
            remote_gpu_count=self.remote_gpu_count,
            remote_gpu_product=self.remote_gpu_product,
            bench_command_quoted=shlex.quote(self.execution_command),
            model_pvc_flag=model_pvc_flag,
        )


def gen_RemoteRunDiagnosticPrompt(
    context: str,
    execution_command: str,
    delay_iterations: int,
    feature_disable_env_var: str,
    original_analysis_dir: str,
    runtime_logs_dir: str,
    iteration: int,
    gpu_wait_timeout_minutes: int = 60,
    trace_gpu_focus: int = 0,
    remote_namespace: str = "",
    remote_pvc_name: str = "vllm-workspace",
    remote_model_pvc_name: str = "",
    remote_image: str = "",
    remote_gpu_count: int = 4,
    remote_gpu_product: str = "NVIDIA-B200",
    cluster_runner_path: str = "",
):
    if not cluster_runner_path:
        cluster_runner_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cluster_runner.py"
        )
    if not remote_image:
        remote_image = "quay.io/example/vllm-sandbox-nsys:latest"
    diagnostic_dir = os.path.join(runtime_logs_dir, DIAGNOSTIC_PROFILE_DIR)
    return RemoteRunDiagnosticPrompt(
        context=context,
        execution_command=execution_command,
        delay_iterations=delay_iterations,
        feature_disable_env_var=feature_disable_env_var,
        original_analysis_dir=original_analysis_dir,
        diagnostic_dir=diagnostic_dir,
        iteration=iteration,
        trace_gpu_focus=trace_gpu_focus,
        gpu_wait_timeout_minutes=gpu_wait_timeout_minutes,
        remote_namespace=remote_namespace,
        remote_pvc_name=remote_pvc_name,
        remote_model_pvc_name=remote_model_pvc_name,
        remote_image=remote_image,
        remote_gpu_count=remote_gpu_count,
        remote_gpu_product=remote_gpu_product,
        cluster_runner_path=cluster_runner_path,
    )
