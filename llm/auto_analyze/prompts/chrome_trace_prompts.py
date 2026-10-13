# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import ClassVar

from common.iteration_utils import resolve_output_file
from llm.auto_analyze.configs.cross_trace_config import (
    CROSS_COMPARE_FILE,
    CROSS_MATCHING_FILE,
    CrossTraceConfig,
)
from llm.auto_analyze.configs.single_trace_config import (
    GPU_OPS_TXT_FILE,
    HIGH_LEVEL_OPS_FILE,
    MEDIAN_BLOCK_FILE,
    PERF_ANALYSIS_SINGLE_FILE,
    SingleTraceConfig,
)

SINGLE_TRACE_JSON_FILE = "single_trace_transformer_block.json"
SINGLE_TRACE_TXT_FILE = "single_trace_transformer_block.txt"
CROSS_TRACE_JSON_FILE = "cross_trace_transformer_blocks.json"
CROSS_TRACE_TXT_FILE = "cross_trace_transformer_blocks.txt"

SCRIPT_ERROR_HANDLING = (
    "If the script fails with an error, read the error message, "
    "diagnose the root cause, fix the script, and re-execute. "
    "Repeat until the script runs successfully."
)


def _format_trace_entry(
    idx,
    framework_name,
    framework_source_code,
    model,
    gpu_type,
    batch_size,
    prefill_size,
    output_size,
    result_dir,
    marker="",
    include_perf_analysis=True,
):
    lines = [
        f"Trace [{idx}]{marker}:",
        f"  framework_name: {framework_name}",
        f"  framework_source_code: {framework_source_code}",
        f"  model: {model}",
        f"  gpu_type: {gpu_type}",
        f"  execution_params: BS={batch_size} ISL={prefill_size} OSL={output_size}",
        f"  median_block_file: {resolve_output_file(result_dir, MEDIAN_BLOCK_FILE)}",
        f"  high_level_ops_file: {resolve_output_file(result_dir, HIGH_LEVEL_OPS_FILE)}",
        f"  gpu_ops_txt_file: {os.path.join(result_dir, GPU_OPS_TXT_FILE)}",
    ]
    if include_perf_analysis:
        lines.append(
            f"  perf_analysis_file: {resolve_output_file(result_dir, PERF_ANALYSIS_SINGLE_FILE)}"
        )
    lines.append("")
    return lines


def _build_trace_context_single(
    config: SingleTraceConfig, include_perf_analysis: bool = True
) -> str:
    output_dir = os.path.abspath(config.output_dir)
    lines = [
        "MODE: Single-trace visualization (one median transformer block).",
        "",
    ]
    lines.extend(
        _format_trace_entry(
            idx=0,
            framework_name=config.framework_name,
            framework_source_code=config.framework_source_code,
            model=config.model,
            gpu_type=config.gpu_type,
            batch_size=config.batch_size_range,
            prefill_size=config.prefill_size_range,
            output_size=config.output_size_range,
            result_dir=output_dir,
            include_perf_analysis=include_perf_analysis,
        )
    )
    return "\n".join(lines)


def _build_trace_context_cross(config: CrossTraceConfig) -> str:
    target = config.get_target_result()
    output_dir = os.path.abspath(config.output_dir)

    lines = [
        "MODE: Cross-trace visualization (multiple median transformer blocks side by side).",
        f"[target_framework] = {target.framework_name}",
        f"[matching_file] = {resolve_output_file(output_dir, CROSS_MATCHING_FILE)}",
        f"[cross_analysis_file] = {resolve_output_file(output_dir, CROSS_COMPARE_FILE)}",
        "",
    ]

    for i, tr in enumerate(config.traces):
        marker = " (TARGET)" if i == config.target_trace_id else ""
        lines.extend(
            _format_trace_entry(
                idx=i,
                framework_name=tr.framework_name,
                framework_source_code=tr.framework_source_code,
                model=tr.model,
                gpu_type=tr.gpu_type,
                batch_size=tr.batch_size_range,
                prefill_size=tr.prefill_size_range,
                output_size=tr.output_size_range,
                result_dir=tr.result_dir,
                marker=marker,
            )
        )

    return "\n".join(lines)


@dataclass
class ChromeTraceJsonPrompt:
    """Unified prompt for generating Chrome trace JSON for Perfetto visualization.

    Works for both single-trace (one block) and cross-trace (multiple blocks).
    """

    context: str
    output_json_file: str
    output_txt_file: str
    output_dir: str
    mode: str  # "single" or "cross"
    has_perf_analysis: bool = True

    prompt_template: ClassVar[str] = """
{context}
[output_json_file] = {output_json_file}
[output_txt_file] = {output_txt_file}
IMPORTANT: Write ALL output files to {output_dir} only. Do not create files anywhere else.

{mode_instructions}

Generate a Chrome trace JSON and human-readable summary for the median transformer block(s) listed above. In Perfetto, a user should:
- See execution context in the info bar
- See ALL operations on their CUDA streams — NO operation may be dropped or hidden
- Click any operation for source code and improvement details

Write and execute a single self-contained Python script:

1. Read all input files listed above. For each trace, read its median_block_file, high_level_ops_file, and gpu_ops_txt_file. Match median block ops to gpu_ops_txt_file by start_ns/end_ns to get exact kernel timestamps and launch parameters.

2. Build Chrome trace JSON:

INFO BAR (pid=0, tid=0): full block duration, name with model, framework(s) + short version of commit id, GPU, and execution parameters. Include execution_config and median_block info in args.

{trace_layout}

CUDA STREAM LANES:
- Each CUDA stream gets one pid (starting from pid=2; pid=0 is reserved for the info bar).
- HANDLING OVERLAPS (PDL): Operations on the same CUDA stream can overlap in time (common with Programmatic Dependent Launch). Use multiple tids within the same pid to separate overlapping operations into lanes:
    - tid=0 is the main lane. tid=1 is the first overflow lane, tid=2 the second, etc.
    - Use a greedy lane assignment: for each operation sorted by start time, find the first tid for this stream where the operation does not overlap with any existing event on that tid. If no tid works, allocate a new one.
    - Use the MINIMUM number of tids needed to avoid all within-tid overlaps.
    - Overlap check: event_A.ts + event_A.dur > event_B.ts means they overlap.
- LANE NAMING:
    - Use process_name metadata for the pid: "Stream 23".
    - Use thread_name metadata for each tid: tid=0 -> "Stream 23", tid=1 -> "Stream 23 (PDL lane 2)", etc.
    - Use process_sort_index metadata to order streams logically in Perfetto.
- Every single operation from each median block MUST appear in the trace.

OPERATION NAMING: Prepend a brief high-level prefix per kernel (e.g., "Attn: ...", "MoE: ...", "Comm: ..."). Keep prefix short.

OPERATION ARGS: For each kernel event, add "args" with:
- "high_level_op": correlated operation name
- "source_code": file:line references with explanations
- "call_chain": The source code call-chain that invokes this kernel. Make sure to provide file/line references here as well
{perf_args_instruction}

TIMING: All blocks start at time 0.

3. Write outputs:
[output_json_file]: Chrome trace JSON, displayTimeUnit: "ns", all ts/dur in nanoseconds.
[output_txt_file]: header, properly aligned pipe-separated op table with all operations (ensure all "|" column separators are aligned across every row), per-stream summary.

4. VALIDATE the output (in the script, before finishing):
- Load the generated JSON back and count kernel events (ph="X", excluding info bar).
- Compare against the total number of operations across all median blocks. They must be equal.
- For every (pid, tid) pair, collect all events (ph="X"), sort by ts, and verify ZERO overlaps: event_A.ts + event_A.dur <= event_B.ts for all consecutive pairs.
- If any validation fails, print the error, fix the lane assignment, and re-generate.

{script_error_handling}

VERIFICATION CHECKLIST:
- [ ] Script executes, both files generated, JSON valid
- [ ] Info bar present
- [ ] ALL operations from ALL median blocks are present in the trace (count matches)
- [ ] One pid per CUDA stream (starting from pid=2), multiple tids for PDL overflow lanes
- [ ] ZERO overlaps within any (pid, tid) pair — validated programmatically in the script
- [ ] Every kernel has high_level_op and source_code in args{perf_checklist_item}
- [ ] All blocks start at t=0, all ts/dur in nanoseconds
"""

    SINGLE_MODE_INSTRUCTIONS: ClassVar[str] = (
        "This is a SINGLE-TRACE visualization. There is one median transformer block to visualize."
    )

    CROSS_MODE_INSTRUCTIONS: ClassVar[str] = (
        "This is a CROSS-TRACE visualization. There are multiple median transformer blocks to visualize side by side.\n"
        "- Color-code similar type operations with the same color across all traces.\n"
        "- For each operation in the [target_framework] trace, include the relevant improvement plan from [cross_analysis_file] in its description.\n"
        "- Follow all guidelines strictly, DO NOT SKIP ANYTHING.\n"
        "- Critically review the work: ensure all operations of all traces are shown — no operation is missed or skipped."
    )

    SINGLE_TRACE_LAYOUT: ClassVar[str] = (
        "TRACE LAYOUT: Visualize the single median transformer block. "
        "Show all CUDA streams used by this block."
    )

    CROSS_TRACE_LAYOUT: ClassVar[str] = (
        "TRACE LAYOUT: Visualize all median transformer blocks, one per trace, all starting at time 0.\n"
        "- Group each trace's CUDA streams together (all streams for trace 0, then all for trace 1, etc.).\n"
        "- Use process_name metadata to clearly label which trace/framework each stream belongs to.\n"
        "- Ensure all operations are shown across all CUDA streams of each trace."
    )

    def prompt(self):
        if self.mode == "single":
            mode_instructions = self.SINGLE_MODE_INSTRUCTIONS
            trace_layout = self.SINGLE_TRACE_LAYOUT
        else:
            mode_instructions = self.CROSS_MODE_INSTRUCTIONS
            trace_layout = self.CROSS_TRACE_LAYOUT

        if self.has_perf_analysis:
            perf_args = '- "proposed_improvements": relevant proposals from the perf analysis file for this trace, or "No improvements proposed for this operation."'
            perf_check = ", and proposed_improvements"
        else:
            perf_args = ""
            perf_check = ""

        return self.prompt_template.format(
            context=self.context,
            output_json_file=self.output_json_file,
            output_txt_file=self.output_txt_file,
            output_dir=self.output_dir,
            mode_instructions=mode_instructions,
            trace_layout=trace_layout,
            perf_args_instruction=perf_args,
            perf_checklist_item=perf_check,
            script_error_handling=SCRIPT_ERROR_HANDLING,
        )


def build_single_trace_json_prompt(config: SingleTraceConfig) -> ChromeTraceJsonPrompt:
    has_perf = not config.skip_perf_analysis
    output_dir = os.path.abspath(config.output_dir)
    return ChromeTraceJsonPrompt(
        context=_build_trace_context_single(config, include_perf_analysis=has_perf),
        output_json_file=os.path.join(output_dir, SINGLE_TRACE_JSON_FILE),
        output_txt_file=os.path.join(output_dir, SINGLE_TRACE_TXT_FILE),
        output_dir=output_dir,
        mode="single",
        has_perf_analysis=has_perf,
    )


def build_cross_trace_json_prompt(config: CrossTraceConfig) -> ChromeTraceJsonPrompt:
    output_dir = os.path.abspath(config.output_dir)
    return ChromeTraceJsonPrompt(
        context=_build_trace_context_cross(config),
        output_json_file=os.path.join(output_dir, CROSS_TRACE_JSON_FILE),
        output_txt_file=os.path.join(output_dir, CROSS_TRACE_TXT_FILE),
        output_dir=output_dir,
        mode="cross",
    )
