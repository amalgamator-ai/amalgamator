# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Create Single-Trace Analysis Config
=====================================

Helper script that generates a single-trace analysis config JSON file
from the provided parameters. Automatically infers framework name,
commit ID, and run command from the run log and source code directory
when not explicitly provided.

Usage:
    python -m llm.auto_analyze.create_single_trace_config \
        --model nvidia/Kimi-K2.5-NVFP4 \
        --gpu-type B200 \
        --batch-size-range 1 \
        --prefill-size-range 4 \
        --output-size-range 1024 \
        --trace-file /path/to/trace.json.gz \
        --run-log-file /path/to/run_log.txt \
        --clean-source-code-path /path/to/vllm \
        --analyze-output-dir /path/to/analysis_output \
        --output-config-file /path/to/config

    The output file path should NOT include the .json extension — it is
    added automatically.

Required parameters:
    --model               HuggingFace model ID (e.g. nvidia/Kimi-K2.5-NVFP4)
    --gpu-type            GPU type used for the run (e.g. B200, H200)
    --batch-size-range    Batch size(s) used (e.g. 1)
    --prefill-size-range  Prefill / input sequence length(s) (e.g. 4)
    --output-size-range   Output / decode sequence length(s) (e.g. 1024)
    --trace-file          Path to trace file (.nsys-rep, .sqlite, .json, .json.gz)
    --run-log-file        Path to the run log (run command is extracted from it)
    --clean-source-code-path    Path to the framework source code directory (must be
                           a clean git repo with no modified files; a separate
                           branch is created automatically for the commit ID)
    --commit-id           Git commit ID or "HEAD" to use current source code state
    --analyze-output-dir          Directory where analysis outputs will be stored
    --output-config-file         Output path for the config JSON (without .json suffix)

Optional parameters:
    --framework-name      Framework name (default: inferred from run log)
    --run-command-file    File containing the run command (default: extracted from run log)

Advanced parameters:
    --enable-single-trace-perf-analysis
                           Enable single-trace performance analysis (default: disabled)
    --trace-gpu-focus      GPU focus for the trace (e.g. "0", "ALL"). For NSYS
                           multi-GPU traces. Not needed for PyTorch single-GPU traces.
    --high-level-focus     Focus guidance for source code analysis
                           (e.g. "Focus on pure decode execution and low-latency batch sizes")
    --perf-analysis-focus  Focus guidance for performance analysis (e.g.
                           "Pay special attention to torch.compile fusion opportunities")
    --max-gpu-ops          Max GPU ops to extract from the trace (default: 1000)
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import json
import os
import re
import subprocess
import sys

from llm.auto_analyze.configs.single_trace_config import VALID_TRACE_EXTENSIONS

KNOWN_FRAMEWORKS = {
    "vllm": ["vllm"],
    "sglang": ["sglang", "sgl"],
    "trt": ["trtllm", "tensorrt-llm", "tensorrt_llm", "trt-llm", "trt_llm"],
}


def infer_framework_name(run_log_path):
    """Infer the framework name from the run log content.

    Scans the first 50 lines for known framework keywords in the run
    command or log output.
    """
    with open(run_log_path) as f:
        head = "".join(f.readline() for _ in range(50)).lower()

    for fw_name, keywords in KNOWN_FRAMEWORKS.items():
        for kw in keywords:
            if kw in head:
                return fw_name
    return None


def extract_run_command(path):
    """Read a file and return its content as a single-line run command."""
    with open(path) as f:
        lines = f.readlines()
    parts = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts.append(stripped.rstrip("\\").rstrip())
        if not line.rstrip().endswith("\\"):
            break
    return " ".join(parts) if parts else None


def extract_run_command_from_log(run_log_path):
    """Extract the run command from a run log file.

    Looks for the first line that looks like a shell command, handling
    shell prompts, environment variable prefixes, and backslash
    continuations.
    """
    command_prefixes = (
        "python",
        "python3",
        "torchrun",
        "nsys ",
        "docker ",
        "mpirun",
        "deepspeed",
        "accelerate",
        "vllm ",
        "sglang",
        "trtllm",
        "mpiexec",
        "bash ",
        "sh ",
        "./",
    )

    def _strip_shell_prompt(text):
        idx = text.find("$ ")
        if idx >= 0:
            return text[idx + 2 :]
        return text

    def _strip_env_vars(text):
        return re.sub(r"^(\s*[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", "", text)

    def _looks_like_command(text):
        lower = text.lower()
        return any(lower.startswith(p) for p in command_prefixes)

    with open(run_log_path) as f:
        lines = f.readlines()

    for i, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        after_prompt = _strip_shell_prompt(stripped)
        after_env = _strip_env_vars(after_prompt).strip()

        if _looks_like_command(after_prompt) or _looks_like_command(after_env):
            cmd_parts = [after_prompt.rstrip("\\").rstrip()]
            j = i + 1
            while after_prompt.rstrip().endswith("\\") or (
                j < len(lines) and j == i + 1 and cmd_parts[-1].endswith("\\")
            ):
                if j >= len(lines):
                    break
                cont = lines[j].strip()
                if not cont:
                    break
                cmd_parts.append(cont.rstrip("\\").rstrip())
                if not lines[j].rstrip().endswith("\\"):
                    break
                j += 1
            return " ".join(cmd_parts)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single-trace analysis config JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    required = parser.add_argument_group("required parameters")
    required.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. nvidia/Kimi-K2.5-NVFP4)",
    )
    required.add_argument(
        "--gpu-type", required=True, help="GPU type (e.g. B200, H200)"
    )
    required.add_argument(
        "--batch-size-range", required=True, help="Batch size(s) used for the run"
    )
    required.add_argument(
        "--prefill-size-range", required=True, help="Prefill / input sequence length(s)"
    )
    required.add_argument(
        "--output-size-range", required=True, help="Output / decode sequence length(s)"
    )
    required.add_argument(
        "--trace-file",
        required=True,
        help="Path to trace file (.nsys-rep, .sqlite, .json, .json.gz)",
    )
    required.add_argument(
        "--run-log-file",
        required=True,
        help="Path to the run log file (run command is extracted from it)",
    )
    required.add_argument(
        "--clean-source-code-path",
        required=True,
        help="Path to the framework source code directory. "
        "Must be a clean git repo with no modified or "
        "uncommitted files, and should be up to date "
        "(git pull). The analysis will automatically "
        "create a separate branch for the specific "
        "commit ID used for the run.",
    )
    required.add_argument(
        "--analyze-output-dir",
        required=True,
        help="Directory where analysis outputs will be stored",
    )
    required.add_argument(
        "--commit-id",
        required=True,
        help='Git commit ID or "HEAD" to use current source code state',
    )
    required.add_argument(
        "--output-config-file",
        required=True,
        help="Output path for the config JSON (without .json suffix)",
    )

    optional = parser.add_argument_group("optional parameters")
    optional.add_argument(
        "--framework-name",
        default=None,
        help="Framework name (default: inferred from run log)",
    )
    optional.add_argument(
        "--run-command-file",
        default=None,
        help="File containing the run command (default: extracted from run log)",
    )

    advanced = parser.add_argument_group("advanced parameters")
    advanced.add_argument(
        "--enable-single-trace-perf-analysis",
        action="store_true",
        default=False,
        help="Enable single-trace performance analysis (default: disabled)",
    )
    advanced.add_argument(
        "--trace-gpu-focus",
        default=None,
        help="GPU focus for the trace (e.g. '0', 'ALL')",
    )
    advanced.add_argument(
        "--high-level-focus",
        default=None,
        help="Focus guidance for source code analysis",
    )
    advanced.add_argument(
        "--perf-analysis-focus",
        default=None,
        help="Focus guidance for performance analysis "
        "(e.g. 'Pay special attention to torch.compile "
        "fusion opportunities')",
    )
    advanced.add_argument(
        "--max-gpu-ops",
        default=1000,
        type=int,
        help="Max GPU ops to extract from the trace (default: 1000)",
    )

    args = parser.parse_args()

    errors = []

    # Validate required files/dirs exist
    if not os.path.exists(args.trace_file):
        errors.append(f"trace file not found: {args.trace_file}")
    elif not any(args.trace_file.endswith(ext) for ext in VALID_TRACE_EXTENSIONS):
        errors.append(
            f"trace file must end with one of {VALID_TRACE_EXTENSIONS}, "
            f"got: {args.trace_file}"
        )

    if not os.path.exists(args.run_log_file):
        errors.append(f"run log file not found: {args.run_log_file}")

    if not os.path.isdir(args.clean_source_code_path):
        errors.append(
            f"source code path is not a directory: {args.clean_source_code_path}"
        )

    if args.run_command_file and not os.path.exists(args.run_command_file):
        errors.append(f"run command file not found: {args.run_command_file}")

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # --- Infer framework name ---
    if args.framework_name:
        framework_name = args.framework_name
        print(f"  Framework name: {framework_name} (provided)")
    else:
        framework_name = infer_framework_name(args.run_log_file)
        if framework_name:
            print(f"  Framework name: {framework_name} (inferred from run log)")
        else:
            print("ERROR: Could not infer framework name from run log.")
            print("  Please provide --framework-name explicitly.")
            sys.exit(1)

    # --- Verify commit ID ---
    commit_id = args.commit_id
    src = args.clean_source_code_path
    result = subprocess.run(
        ["git", "-C", src, "log", "-1", "--format=%H %ai", commit_id],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Commit '{commit_id}' not found in {src}")
        print(f"  git error: {result.stderr.strip()}")
        print(
            "  Please verify the commit ID and ensure the repo is up to date (git fetch --all)."
        )
        sys.exit(1)
    full_hash, commit_date = result.stdout.strip().split(" ", 1)
    print(f"  Commit ID: {commit_id} (verified: {full_hash[:12]}, {commit_date})")

    # --- Infer run command ---
    if args.run_command_file:
        run_command = extract_run_command(args.run_command_file)
        if run_command:
            print(f"  Run command: {run_command[:80]}... (from run command file)")
        else:
            run_command = None
            print("  WARNING: Run command file is empty, setting to null")
    else:
        run_command = extract_run_command_from_log(args.run_log_file)
        if run_command:
            print(f"  Run command: {run_command[:80]}... (extracted from run log)")
        else:
            run_command = None
            print(
                "  WARNING: Could not extract run command from run log, setting to null"
            )

    # --- Resolve paths ---
    output_json = args.output_config_file
    if not output_json.endswith(".json"):
        output_json += ".json"
    output_dir = args.analyze_output_dir

    # --- Build config (advanced params last) ---
    config = {
        "framework_name": framework_name,
        "model": args.model,
        "gpu_type": args.gpu_type,
        "batch_size_range": args.batch_size_range,
        "prefill_size_range": args.prefill_size_range,
        "output_size_range": args.output_size_range,
        "trace_file": os.path.abspath(args.trace_file),
        "framework_source_code": os.path.abspath(args.clean_source_code_path),
        "commit_id": commit_id,
        "run_command": run_command,
        "run_log": os.path.abspath(args.run_log_file),
        "output_dir": os.path.abspath(output_dir),
        "skip_perf_analysis": not args.enable_single_trace_perf_analysis,
        "trace_gpu_focus": args.trace_gpu_focus,
        "high_level_focus": args.high_level_focus,
        "perf_analysis_focus": args.perf_analysis_focus,
        "max_gpu_ops": args.max_gpu_ops,
    }

    # --- Write config ---
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(config, f, indent=4)

    print()
    print(f"Config written to: {output_json}")
    print()
    print("=" * 70)
    print("  SINGLE-TRACE CONFIG")
    print("=" * 70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == "__main__":
    main()
