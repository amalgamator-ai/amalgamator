"""Programmatic parsing of benchmark results — replaces LLM-driven parse step.

Restructures results into test_results/<test_id>/ directories,
generates comparison tables, and creates analysis config JSONs.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from llm.auto_profile_server_client.frameworks import get_adapter
from llm.auto_profile_server_client.models import (
    BenchmarkResult,
    save_json,
)

log = logging.getLogger(__name__)


def restructure_results(results_dir: str | Path, output_dir: str | Path) -> dict:
    """Reorganize per-framework results into test_results/<test_id_with_batch>/ dirs.

    Returns a mapping: test_id_with_batch -> {framework -> BenchmarkResult}.
    """
    results_dir = Path(results_dir)
    test_results_dir = Path(output_dir) / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, BenchmarkResult]] = {}

    # Scan the pipeline results
    for result_file in sorted(results_dir.rglob("results_*.json")):
        try:
            data = json.loads(result_file.read_text())
        except Exception:
            log.warning("Skipping unreadable result: %s", result_file)
            continue

        fw = data.get("framework", "")
        test_name = data.get("test_name", "")
        bench = data.get("benchmark", {})

        if not fw or not test_name:
            continue

        # Extract test_id_with_batch from test_name by removing the framework prefix
        parts = test_name.split("-", 1)
        if len(parts) < 2:
            continue
        test_id_batch = parts[1]

        if test_id_batch not in all_results:
            all_results[test_id_batch] = {}

        get_adapter(fw)
        result = BenchmarkResult(
            **{
                k: bench.get(k, 0)
                for k in [
                    "ttft_ms",
                    "tpot_ms",
                    "throughput_tok_per_sec",
                    "ttft_p50_ms",
                    "ttft_p90_ms",
                    "ttft_p99_ms",
                    "tpot_p50_ms",
                    "tpot_p90_ms",
                    "tpot_p99_ms",
                    "e2e_latency_ms",
                    "num_prompts",
                    "total_time_sec",
                ]
            }
        )
        all_results[test_id_batch][fw] = result

        # Copy files into restructured directory
        test_dir = test_results_dir / test_id_batch / fw
        test_dir.mkdir(parents=True, exist_ok=True)

        src_dir = result_file.parent
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, test_dir / f.name)

    log.info("Restructured %d test cases", len(all_results))
    return all_results


def generate_comparison_tables(
    all_results: dict[str, dict[str, BenchmarkResult]],
    output_dir: str | Path,
) -> list[str]:
    """Generate cross-framework comparison tables.

    Returns list of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Group by test_id (without batch)
    by_test_id: dict[str, dict[int, dict[str, BenchmarkResult]]] = {}
    for test_id_batch, fw_results in all_results.items():
        parts = test_id_batch.rsplit("-b_", 1)
        if len(parts) != 2:
            continue
        test_id = parts[0]
        batch = int(parts[1])
        if test_id not in by_test_id:
            by_test_id[test_id] = {}
        by_test_id[test_id][batch] = fw_results

    all_tables = []
    for test_id, batch_results in sorted(by_test_id.items()):
        table = _format_comparison_table(test_id, batch_results)
        all_tables.append(table)

        table_path = output_dir / f"comparison_{test_id}.txt"
        table_path.write_text(table)
        generated.append(str(table_path))
        log.info("Generated comparison table: %s", table_path)

    # Combined file
    if all_tables:
        combined_path = output_dir / "comparison_all.txt"
        combined_path.write_text("\n\n".join(all_tables))
        generated.append(str(combined_path))
        log.info("Generated combined comparison: %s", combined_path)

    return generated


def _format_comparison_table(
    test_id: str,
    batch_results: dict[int, dict[str, BenchmarkResult]],
) -> str:
    frameworks = sorted({fw for br in batch_results.values() for fw in br})
    batches = sorted(batch_results.keys())

    lines = [
        f"Test: {test_id}",
        "=" * 80,
        "",
    ]

    # Header
    header = f"{'Batch':>8}"
    for _fw in frameworks:
        header += f" | {'TTFT (ms)':>12} {'TPOT (ms)':>12} {'Tput (t/s)':>12}"
    lines.append(header)

    fw_header = f"{'':>8}"
    for fw in frameworks:
        fw_header += f" | {fw:>12} {'':>12} {'':>12}"
    lines.append(fw_header)
    lines.append("-" * len(header))

    for batch in batches:
        row = f"{batch:>8}"
        for fw in frameworks:
            r = batch_results[batch].get(fw)
            if r:
                row += f" | {r.ttft_ms:>12.2f} {r.tpot_ms:>12.2f} {r.throughput_tok_per_sec:>12.1f}"
            else:
                row += f" | {'N/A':>12} {'N/A':>12} {'N/A':>12}"
        lines.append(row)

    return "\n".join(lines)


def generate_analysis_configs(
    all_results: dict[str, dict[str, BenchmarkResult]],
    results_dir: str,
    output_dir: str,
    source_codes: dict[str, str] | None = None,
) -> list[str]:
    """Generate single-trace and cross-trace analysis config JSONs.

    Returns list of generated config file paths.
    """
    test_results_dir = Path(output_dir) / "test_results"
    generated = []

    for test_id_batch, fw_results in all_results.items():
        if "vllm" not in fw_results:
            continue

        vllm_result = fw_results["vllm"]
        is_vllm_slower = any(
            fw_results[fw].tpot_ms < vllm_result.tpot_ms
            for fw in fw_results
            if fw != "vllm"
        )
        if not is_vllm_slower:
            continue

        test_dir = test_results_dir / test_id_batch

        # Generate single-trace configs
        single_configs = []
        for fw, _result in fw_results.items():
            fw_dir = test_dir / fw
            trace_files = sorted(fw_dir.glob("trace-*.sqlite"))
            trace_file = str(trace_files[0]) if trace_files else ""

            log_files = sorted(fw_dir.glob("bench_run_*.log"))
            log_file = str(log_files[0]) if log_files else ""

            source_path = (source_codes or {}).get(fw, "")

            single_config = {
                "framework": fw,
                "trace_file": trace_file,
                "trace_type": "nsys",
                "trace_gpu_focus": 0,
                "run_log": log_file,
                "source_code_path": source_path,
                "commit_id": "HEAD",
                "output_dir": str(test_dir / f"analyze_{fw}"),
                "skip_perf_analysis": True,
            }

            config_path = test_dir / f"single_trace_config_{fw}.json"
            save_json(single_config, str(config_path))
            single_configs.append(str(config_path))
            generated.append(str(config_path))

        # Generate cross-trace config
        cross_config = {
            "target_framework": "vllm",
            "single_trace_configs": single_configs,
            "output_dir": str(test_dir / "analyze_cross"),
            "make_improvement_plan": True,
        }

        cross_path = test_dir / "cross_trace_config.json"
        save_json(cross_config, str(cross_path))
        generated.append(str(cross_path))

    log.info("Generated %d analysis configs", len(generated))
    return generated
