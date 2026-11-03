#!/usr/bin/env python3
"""Write a single-trace analysis config for a profile produced by
auto_profile_server_client.

usage: gen_single_trace_config.py <profile_results_dir> <source_tree> <commit> <out_json>
"""
import glob
import json
import os
import sys

res_dir, tree, commit, out_json = sys.argv[1:5]

traces = sorted(glob.glob(os.path.join(res_dir, "**", "*.trace.json.gz"), recursive=True))
if not traces:
    raise SystemExit(f"no trace under {res_dir}")
trace = traces[0]
run_dir = os.path.dirname(os.path.dirname(trace))
server_log = os.path.join(run_dir, "profiling", "server.log")
if not os.path.exists(server_log):
    server_log = os.path.join(run_dir, "server.log")

run_command = None
if os.path.exists(server_log):
    for line in open(server_log, errors="replace"):
        if "vllm serve" in line:
            run_command = line.strip()
            break

cfg = {
    "framework_name": "vllm",
    "model": "Qwen/Qwen3.8-27B-FP8",
    "gpu_type": "AMD Instinct MI300X",
    "batch_size_range": "1",
    "prefill_size_range": "8192",
    "output_size_range": "1024",
    "trace_file": trace,
    "framework_source_code": tree,
    "commit_id": commit,
    "run_command": run_command,
    "run_log": server_log if os.path.exists(server_log) else None,
    "output_dir": os.path.join(os.path.dirname(out_json), "vllm"),
    "skip_perf_analysis": False,
    "trace_gpu_focus": None,
    "high_level_focus": None,
    "perf_analysis_focus": None,
    "max_gpu_ops": 1000,
}
with open(out_json, "w") as f:
    json.dump(cfg, f, indent=4)
print(out_json)
print("trace:", trace)
