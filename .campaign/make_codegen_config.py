#!/usr/bin/env python3
"""Generate a per-plan-step codegen config for the isolation ladder.

One config = one plan step = one branch = one patch = one PR candidate. Every
knob that decides *what environment the measurement runs in* is pinned here:

  run_command_prefix      serves the mounted tree inside the pinned ROCm 10
                          image, so both arms of the A/B see the same build and
                          the patched tree, not the image's own vLLM
  benchmark_client_command runs the client in the same pinned image
  source_code_dir         the worktree that carries P1 as the baseline

usage:
  make_codegen_config.py --improvement-id 2 --out configs/qwen35_p1_plan2.json
"""

import argparse
import json
import pathlib

VLLM_IMAGE = ("vllm/vllm-openai-rocm@sha256:"
              "b1374c3814b6e7706c49968ba17ca3b35484382f7b9fc7115dd3e66068d53f2f")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--improvement-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cross-trace-config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source-code-dir", default="/home/amatveev/work/vllm_e0d")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--num-runtime-iterations", type=int, default=2)
    ap.add_argument("--num-code-gen-iterations", type=int, default=2)
    args = ap.parse_args()

    client = (
        f"docker run --rm --network host --entrypoint vllm {VLLM_IMAGE} "
        f"bench serve --model {args.model} --port {args.port} --backend openai "
        f"--dataset-name random --random-input-len {args.input_len} "
        f"--random-output-len {args.output_len} "
        f"--num-prompts {args.num_prompts} "
        f"--max-concurrency {args.max_concurrency} --ignore-eos "
        f"--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99 "
        f"--trust-remote-code --save-result --result-dir /tmp/vllm_bench_results "
        f"--result-filename bench_result.json"
    )

    cfg = {
        "cross_trace_config": args.cross_trace_config,
        "improvement_id": args.improvement_id,
        "source_code_dir": args.source_code_dir,
        "output_dir": args.output_dir,
        "use_source_worktree": False,
        "skip_runtime_diagnostics": True,
        "benchmark_client_command": client,
        "run_command_prefix":
            "bash /home/amatveev/qwen38/scripts/serve_in_container.sh "
            "{quoted_command}",
        "num_code_port_plan_iterations": 2,
        "num_test_plan_iterations": 1,
        "num_code_gen_iterations": args.num_code_gen_iterations,
        "num_runtime_iterations": args.num_runtime_iterations,
        "runtime_continue_after_success": False,
        "max_investigation_revisions": 1,
        "gpu_wait_timeout_minutes": 30,
        "gpu_ids": "0",
        "disallowed_modules": ["sglang", "sgl_kernel", "sgl*"],
        "use_smaller_model_for_runtime": False,
        "disable_new_feature_for_runtime": False,
        "additional_benchmark_configs": [],
        "code_port_plan_skip_review": False,
        "test_plan_skip_review": False,
        "code_gen_skip_review": False,
        "generate_pr_commands": False,
        "enable_parallel_research_iteration": False,
        "thinking-mode": "deep",
        "agent": {
            "backend": "claude_sdk",
            "model": "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "context_length": 1000000,
        },
    }

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"wrote {out} (plan step {args.improvement_id})")


if __name__ == "__main__":
    main()
