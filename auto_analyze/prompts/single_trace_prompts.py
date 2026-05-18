from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.configs.single_trace_config import (
    SingleTraceParams,
    SingleTraceConfig,
    TRACE_FILE_TYPE_NSYS,
    TRACE_FILE_TYPE_PYTORCH,
    HIGH_LEVEL_OPS_FILE,
    GPU_OPS_TXT_FILE,
    GPU_OPS_TO_BLOCKS_FILE,
    MEDIAN_BLOCK_FILE,
    PERF_ANALYSIS_SINGLE_FILE,
)
from auto_analyze.prompts.chrome_trace_prompts import build_single_trace_json_prompt


TRACE_DESCRIPTIONS = {
    TRACE_FILE_TYPE_NSYS: (
        "- The file <trace_file> is an NSYS profile result file in SQLite"
        " format of <framework_name> running <model> on <gpu_type> GPU."
    ),
    TRACE_FILE_TYPE_PYTORCH: (
        "- The file <trace_file> is a PyTorch Profiler trace in Chrome trace"
        " JSON format of <framework_name> running <model> on <gpu_type> GPU."
    ),
}


# ---------------------------------------------------------------------------
# Context header fields (emitted by SingleTraceParams.context_header())
# These use <angle_brackets> and are defined in the context block.
#
# Always present:
#   <model>                 HuggingFace model tag (e.g. "nvidia/DeepSeek-R1-NVFP4")
#   <model_hf_url>          Derived: https://huggingface.co/<model>
#   <gpu_type>              GPU hardware type (e.g. "B200")
#   <framework_name>        Framework being analyzed (e.g. "vllm", "sglang")
#   <framework_source_code> Path to the framework source tree
#   <trace_file>            Path to the trace file (.sqlite or .json)
#   <trace_file_type>       Auto-detected: "NSYS" (.sqlite) or "PYTORCH" (.json)
#   <batch_size_range>      Batch size(s) used in the run
#   <prefill_size_range>    Prefill / input sequence length(s)
#   <output_size_range>     Output / decode sequence length(s)
#   <max_gpu_ops>           Max GPU operations to extract (default 2000)
#
# Conditional (included when set):
#   <run_command>           Command line used to launch the framework
#   <trace_gpu_focus>       "ALL" or a GPU ID (e.g. "0"); omitted for single-GPU traces
#   <run_log>               Path to the framework run log
#   <high_level_focus>      Free-text guidance on what to focus the analysis on
#   <output_dir>            Absolute path to the output directory for all results
#
# Step-specific file parameters use [square_brackets] and are resolved
# via Python .format() substitution per prompt.
# ---------------------------------------------------------------------------


SCRIPT_ERROR_HANDLING = (
    "If the script fails with an error, read the error message, "
    "diagnose the root cause, fix the script, and re-execute. "
    "Repeat until the script runs successfully."
)


@dataclass
class TransformerBlockHighLevelPrompt:
    params: SingleTraceParams
    output_file: str
    prompt_template: ClassVar[str] = """
{context}
[output_file] = {output_file}

TASK — IDENTIFY TRANSFORMER BLOCK HIGH-LEVEL OPERATIONS:

The model <model> is a sequence of transformer blocks. Your main task is to investigate the source code at <framework_source_code>, find the files that implement <model>, and determine the sequence of high-level operations in each transformer block type.

This output will serve as a GUIDE for a subsequent step that correlates actual GPU trace operations to these high-level operations using both this summary and independent source code analysis. Your goal is to provide an accurate logical map of the transformer block structure — not to predict exact GPU kernel names or counts.

First, inspect <model_hf_url> (the HuggingFace model page for <model>) to learn about this model's architecture, precision/quantization, and key config values (num_layers, num_experts, hidden_size, etc.).

To guide your source code investigation, use the run command and run logs to understand which code paths are actually taken:
{run_command_section}
{run_log_section}

{focus_section}

Read and analyze in-detail the source code to detect the pieces that implement the transformer block types for the currently executed model.

For each transformer block type:
- Find the sequence of operations by reading the decoder layer's forward() method and all sub-modules it calls.
- Each operation should be a distinct logical compute step (e.g. layernorm, projection GEMM, attention, quantization, allreduce, activation function).
- Note fusions, stream parallelization, and cross-layer boundary effects.
- For each operation provide: source code reference (file:line), what it computes, and relevant details.

EXECUTION MODES:
Different execution modes, (1) prefill-only, (2) decode-only or (3) mixed, often take different code paths and produce different kernel sequences. Produce a different transformer block table for each mode.

OUTPUT FORMAT:
Dump to [output_file] a well-structured human-readable text file with:

For each combination of (block type x execution mode), a separate section with block type name, layer range, execution mode, and a properly aligned pipe-separated table where each row is one operation with columns: index, operation name, source code reference, execution details, and possible kernel names. Ensure all "|" column separators are aligned across every row. Each table must be self-contained — do not combine multiple block types or execution modes into a single table.

"""

    def prompt(self):
        p = self.params

        run_command_section = ""
        if p.run_command:
            run_command_section = (
                "- The command used to run the framework is: <run_command>"
            )

        run_log_section = ""
        if p.run_log:
            run_log_section = (
                "- The run log of executing the framework is at: <run_log>\n"
                "- Read the run log to identify which modules, classes, backends, and code paths "
                "were actually activated during execution. The log is ground truth — use it to "
                "confirm or override the code path analysis from flags."
            )

        focus_section = ""
        if p.high_level_focus:
            focus_section = (
                "Focus area for this analysis: <high_level_focus>"
            )

        return self.prompt_template.format(
            context=p.context_header(),
            output_file=self.output_file,
            run_command_section=run_command_section,
            run_log_section=run_log_section,
            focus_section=focus_section,
        )


@dataclass
class GpuOpsPrompt:
    params: SingleTraceParams
    output_txt_file: str
    prompt_template: ClassVar[str] = """
{context}
[output_txt_file] = {output_txt_file}
{trace_description}

Write and execute a single self-contained Python script that does the following:

1. Read the trace file at <trace_file>.
2. Extract all GPU operations with the fields listed below.
3. Write the output file.

The script must handle the trace format (<trace_file_type>):
- For NSYS SQLite traces:
    - Query CUPTI_ACTIVITY_KIND_KERNEL (with grid/block/registers/shared memory columns).
    - Check if CUPTI_ACTIVITY_KIND_MEMCPY exists; if so, query and merge memory copy operations.
    - Query StringIds to resolve kernel name IDs.
- For PyTorch JSON traces: parse the Chrome trace JSON and extract GPU kernel events.

{gpu_focus_instruction}

For each GPU operation, extract:
    a. short_name (max 50 chars), start_ns, end_ns, duration_ns, stream, full_name (max 200 chars)
    b. category: "compute_kernel", "memory_transfer", "communication", or "synchronization"
    c. Kernel launch parameters (NSYS only): grid, block, registers_per_thread, shared_memory_bytes
    d. For communication ops: comm_algorithm, comm_size_hint
    e. For memory transfers: transfer_bytes, transfer_kind

Keep max <max_gpu_ops> operations. If more exist, keep the LAST <max_gpu_ops> (steady-state). Sort by start_ns.

OUTPUT FILE:
[output_txt_file]: A properly aligned table with a header row and one row per GPU operation. Include all extracted fields as columns. Use consistent column widths and pipe-separated format so the table is easy to read.

{script_error_handling}

VERIFICATION CHECKLIST:
- [ ] Script executes successfully
- [ ] Output file generated with properly aligned table
- [ ] Sorted by start_ns, count <= <max_gpu_ops>, every op has non-null category
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.params.context_header(),
            output_txt_file=self.output_txt_file,
            gpu_focus_instruction=self.params.gpu_focus_instruction(),
            trace_description=TRACE_DESCRIPTIONS[self.params.trace_file_type],
            script_error_handling=SCRIPT_ERROR_HANDLING,
        )


@dataclass
class GpuOpsToTransformerBlocksPrompt:
    params: SingleTraceParams
    high_level_ops_file: str
    gpu_ops_txt_file: str
    output_file: str
    median_block_file: str
    prompt_template: ClassVar[str] = """
{context}
[high_level_ops_file] = {high_level_ops_file}
[gpu_ops_txt_file] = {gpu_ops_txt_file}
[output_file] = {output_file}
[median_block_file] = {median_block_file}
- [high_level_ops_file] describes the transformer block types with per-kernel operations and expected kernels.
- [gpu_ops_txt_file] contains a table of all GPU operations sorted by start time, with kernel names, timestamps, stream IDs, and launch parameters.
- <framework_source_code> is the source code of <framework_name>.

Your task is to analyze the sequence of GPU low-level operations in [gpu_ops_txt_file] and detect the transformer blocks and their types. Use the high-level block types from [high_level_ops_file] as a guide, but do NOT rely solely on them — you must perform your own independent source code deep-dive analysis for each and every low-level GPU operation.

THIS IS THE MOST CRITICAL STEP IN THE ENTIRE ANALYSIS. The goal is to find the 100% precise correlation between every low-level GPU operation and its corresponding high-level transformer block operation. No mistakes are acceptable.

STEP 1 — SOURCE CODE DEEP-DIVE FOR EACH GPU OPERATION:
For each distinct GPU kernel name that appears in [gpu_ops_txt_file]:
- Search the source code at <framework_source_code> to find where this kernel is launched.
- Trace the call chain from the kernel launch point UP to the decoder layer's forward() method.
- Determine which high-level transformer block operation this kernel implements.
- Record the kernel's identifying signature: kernel name, CUDA stream ID, and grid dimensions. This triplet (kernel_name, stream, grid) is what uniquely identifies a kernel's role — NOT its time-sorted position within the block.
- Cross-reference with [high_level_ops_file] to validate your finding, but trust your source code analysis if there is a conflict.

IMPORTANT: The [high_level_ops_file] from the previous step provides a SUMMARY of the expected high-level operations, but it may have missed operations, mis-identified kernel names, or made errors. YOUR analysis here must be MORE thorough because:
1. You can see the ACTUAL GPU operations and their actual kernel names (not just expected names).
2. You can see the sequence and context of surrounding operations.
3. You have the high-level summary as a starting reference point.

CRITICAL — MULTI-STREAM AWARENESS:
Operations on different CUDA streams run concurrently and their relative time-ordering varies between blocks due to GPU scheduling jitter. For example, a kernel on stream 3649 may appear before or after a kernel on stream 23 in the time-sorted list, even though both are always present. You MUST assign high_level_op names based on the kernel's identity (kernel_name, stream, grid), NOT based on its positional index in the time-sorted sequence. The same kernel on the same stream always gets the same high_level_op name, regardless of where it falls relative to operations on other streams.

STEP 2 — BLOCK BOUNDARY DETECTION:
- Using your kernel-to-operation mappings from Step 1, identify where each transformer block starts and ends in the GPU ops sequence.
- The block boundaries must be consistent with BOTH [high_level_ops_file] AND the decoder layer's forward() method in the source code. Cross-check both to confirm.
- When the same kernel name appears multiple times within a block serving different roles, disambiguate by examining the surrounding operations in the sequence to determine which occurrence corresponds to which role in the forward() method.
- After detecting boundaries, validate each block: verify that the first and last operations match what [high_level_ops_file] and the source code say the block should start and end with. If they don't match, the boundary is wrong — shift it and re-detect.

STEP 3 — CLASSIFICATION AND WARMUP:
- Classify each block by type. Flag first 2 and last 2 as WARMUP/COOLDOWN, rest as REPRESENTATIVE.

STEP 4 — PER-BLOCK CORRELATION:
- For each block, list every GPU operation with its correlated high-level operation name.
- Assign high_level_op names by matching the kernel's identity (kernel_name, stream, grid) to the mapping built in Step 1 — NOT by positional index. The same kernel on the same stream always gets the same high_level_op name across all blocks.
- Each kernel must have a UNIQUE specific high_level_op name within its block.
- The high_level_op name must be descriptive enough to understand what the operation does (e.g. "q_b_proj_gemm", "moe_gate_router_gemm", "fused_allreduce_rms_norm_fp4_quant"), but concise — aim for 3-6 words in snake_case, max ~50 characters.
- VALIDATE: within each stream, the operation sequence must follow a logical order consistent with the decoder layer's forward() method. Operations on different streams may interleave in any time order.

STEP 5 — PER-BLOCK TIMING:
- Calculate wall_time (from first op start to last op end), compute_time (sum of all op durations), idle_time, and idle_pct per block.

STEP 6 — FULL CONSISTENCY VALIDATION:
Perform a thorough validation across ALL blocks and ALL operations:

a. BLOCK-LEVEL CONSISTENCY:
   - Verify that all blocks of the same type have the SAME number of operations.
   - Verify that all blocks of the same type have the SAME SET of (high_level_op, kernel_name, stream) tuples. The time-ordering of operations across different streams may vary between blocks — this is normal. But operations within the same stream must appear in the same relative order.
   - If any block has a different kernel at a given (high_level_op, stream) than other blocks, it is a misclassification — the correlation used positional matching instead of kernel identity matching. Fix it.

b. ZERO UNCORRELATED OPERATIONS:
   - Every single GPU operation in [gpu_ops_txt_file] must be accounted for — either assigned to a transformer block or explicitly classified as an inter-block operation (e.g. embedding, sampling, LM head, scheduling overhead).
   - There must be NO operations left without a high-level correlation. If any operation lacks a correlation, perform a source code deep-dive for that specific kernel to determine what it does.
   - Report the total number of GPU operations, number assigned to blocks, and number classified as inter-block, and verify they sum correctly.

c. SOURCE CODE GROUNDING:
   - Every high_level_op name used in the correlation must be traceable back to a specific location in the source code at <framework_source_code>.
   - No operation should be labeled with a generic name like "unknown" or "misc" — if you cannot identify what a kernel does, search harder in the source code.

STEP 7 — DOMINANT BLOCK SELECTION (maybe a couple of them, based on implementation of the forward-pass):
- Based on high-level model implementation, identify the transformer block types that dominate the forward inference pass. Ensure the block types cover the full forward pass, and state for each block type, what's the relative part of the forward pass it consumes.
- For each such transformer block type, select a single REPRESENTATIVE example of such a block from the found blocks. This example should have a "median" wall time across all blocks of such a type.

STEP 8 — WRITE OUTPUTS:

FILE 1 — [output_file]: Structured text with:
    - Header (model, framework, GPU, block counts, warmup/cooldown, consistency)
    - A FULL properly aligned pipe-separated table for EACH AND EVERY detected block (not just the median — ALL blocks including warmup and cooldown). Each table has columns for:
        - index
        - high_level_op 
        - kernel name
        - cuda stream
        - duration
        - source_code_ref: the file:line references where this kernel is launched
        - source_code_explanation: a detailed explanation of what this kernel does and its context in the call chain
      CRITICAL: Never truncate any cell value with "..." or ellipsis. Write the FULL text for every cell, even if it makes the line long. This applies to all columns, especially source_code_ref and source_code_explanation. Long lines are acceptable — truncated information is not.
      Ensure all "|" column separators are aligned across every row.
    - Median selection section.

FILE 2 — [median_block_file]: Just the selected dominant block types content, that cover the forward pass.

STEP 9 — CRITICAL REVIEW — before writing the output files, perform a thorough review of the entire correlation:

a. BLOCK TYPE ASSIGNMENT REVIEW:
   - For each detected block, re-verify that its type assignment is correct by comparing its operation sequence against the high-level block type definitions in [high_level_ops_file] AND the decoder layer's forward() method in the source code.
   - Confirm that every block boundary is placed correctly — the first and last operations of each block must correspond to the entry and exit points of the decoder layer's forward() method for that block type.
   - If any block's operation count or composition differs from other blocks of the same type, investigate and fix the boundary or type assignment before proceeding.

b. PER-OPERATION HIGH-LEVEL CORRELATION REVIEW:
   - For EACH low-level GPU operation in EVERY block, re-verify that its assigned high_level_op name correctly describes what the kernel does according to the source code.
   - Re-verify that each operation's source_code_ref (file:line) is accurate — open the referenced location in <framework_source_code> and confirm it corresponds to this kernel's launch site.
   - Re-verify that each operation's source_code_explanation accurately describes the kernel's role in the forward pass, its call chain from the decoder layer, and its relationship to the high-level operation.
   - If any high_level_op name is vague, generic, or does not precisely describe the kernel's role, refine it with a more specific name derived from your source code analysis.

c. CROSS-BLOCK CONSISTENCY REVIEW:
   - Verify that the same kernel (identified by kernel_name, stream, grid) always receives the same high_level_op name across ALL blocks — no inconsistencies.
   - Verify that within each stream, operations appear in the same relative order across all blocks of the same type.
   - Fix any inconsistencies found before proceeding to the next review.

STEP 10 — SECOND REVIEW — after fixing issues from the first review, perform a deeper pass:

a. COMPLETENESS CHECK:
   - Re-scan [gpu_ops_txt_file] from start to end and confirm that every single GPU operation is accounted for — either assigned to a block or classified as inter-block. No gaps allowed.
   - For each inter-block operation, verify that it genuinely belongs outside block boundaries (e.g., embedding, sampling, LM head) by checking the source code.
   - Verify the arithmetic: total GPU ops = ops in blocks + inter-block ops. If the sum is wrong, find the missing or double-counted operations.

b. SOURCE CODE GROUNDING CHECK:
   - For each DISTINCT kernel type that appears in the correlation, confirm you have a valid source_code_ref that traces back to the actual kernel launch in <framework_source_code>.
   - Ensure no source_code_explanation is speculative or generic — each must describe the specific code path from the decoder layer's forward() method down to the kernel launch.
   - If any source code reference cannot be verified, re-do the source code search for that kernel and correct the reference.

c. HIGH-LEVEL OP NAMING QUALITY CHECK:
   - Verify that each high_level_op name is unique within its block and descriptive enough to distinguish it from all other operations.
   - Verify that the naming is consistent with the terminology used in the framework's source code (function names, class names, variable names).
   - Fix any naming issues found before proceeding.

STEP 11 — THIRD REVIEW — final quality and precision check:

a. MEDIAN BLOCK VALIDATION:
   - Re-verify that the selected median block(s) are correctly chosen — they must be REPRESENTATIVE blocks (not warmup/cooldown) with wall_time closest to the median of their block type.
   - Verify that the median block's operation table is complete and matches the format of all other blocks.

b. OUTPUT FORMAT AND CONTENT VALIDATION:
   - Verify that every table row has ALL columns filled with complete, non-truncated values — no "..." or ellipsis anywhere.
   - Verify that all "|" column separators are properly aligned across every row in every table.
   - Verify that source_code_ref columns contain specific file:line references, not just file names.
   - Verify that source_code_explanation columns contain meaningful explanations, not just kernel name repetitions.

c. END-TO-END COHERENCE:
   - Read through the complete output and verify it tells a coherent story: the block types, their boundaries, and the per-operation correlations must all be mutually consistent.
   - Verify that the information in [median_block_file] is an exact subset of [output_file] — no discrepancies between the two files.
   - Fix any issues found. Do NOT proceed to writing the output files until all three reviews pass cleanly.

{script_error_handling}

VERIFICATION CHECKLIST:
- [ ] Both output files generated
- [ ] Every GPU op accounted for
- [ ] Block boundaries are correct: verified by source code analysis, not just pattern matching
- [ ] Operation order within each block matches the decoder layer's forward() method. Make sure this for EVERY block (based on its type).
- [ ] Every kernel has UNIQUE high_level_op name within its block, determined by source code deep-dive
- [ ] Warmup/cooldown flagged, median selected correctly
- [ ] Source code was consulted for EACH distinct kernel type, not just the ones listed in [high_level_ops_file]
- [ ] All three review phases completed and all issues fixed
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.params.context_header(),
            high_level_ops_file=self.high_level_ops_file,
            gpu_ops_txt_file=self.gpu_ops_txt_file,
            output_file=self.output_file,
            median_block_file=self.median_block_file,
            script_error_handling=SCRIPT_ERROR_HANDLING,
        )


@dataclass
class PerfAnalysisPrompt:
    params: SingleTraceParams
    median_block_file: str
    high_level_ops_file: str
    gpu_ops_to_blocks_file: str
    gpu_ops_txt_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
{context}
[median_block_file] = {median_block_file}
[high_level_ops_file] = {high_level_ops_file}
[gpu_ops_to_blocks_file] = {gpu_ops_to_blocks_file}
[gpu_ops_txt_file] = {gpu_ops_txt_file}
[output_file] = {output_file}
- The model is <model> running via <framework_name> on <gpu_type> GPU.
- Execution parameters: batch size <batch_size_range>, prefill/input size <prefill_size_range>, output size <output_size_range>.
- IMPORTANT: The execution parameters above are FIXED constraints. All analysis and improvements must target the framework implementation under EXACTLY these parameters. Every observation, classification, and proposal must be specific to this batch size, prefill size, and output size — not generic advice. For example, at BS=1 decode, most GEMMs are memory-bound because the matrices are too small to saturate compute; at BS=64 the same GEMMs would be compute-bound. Your analysis must reflect the actual regime.
- [median_block_file] is the median transformer block with correlated high-level operations.
- [high_level_ops_file] describes high-level operations with source code references.
- [gpu_ops_to_blocks_file] has all detected blocks with timing statistics and median selection.
- [gpu_ops_txt_file] is the full GPU operations table with kernel launch parameters.
- <framework_source_code> is the source code of <framework_name>.

{perf_analysis_focus_section}

BEFORE WRITING THE OUTPUT, perform the following analysis steps:

PHASE 1 — SOURCE CODE DEEP-DIVE:
For each operation in the median block:
- Read the source code at <framework_source_code> and trace the full call chain from the decoder layer's forward() method down to the kernel launch point.
- Understand not just this operation in isolation, but its surrounding context: what runs before it, what runs after it, what data it consumes and produces, and how it interacts with adjacent operations.
- Look up the kernel launch parameters from [gpu_ops_txt_file] (grid, block, registers, shared memory).
- Classify as compute-bound, memory-bound, or communication-bound for the specific execution parameters (BS=<batch_size_range>, ISL=<prefill_size_range>, OSL=<output_size_range>).
- Read surrounding source code for TODO/FIXME comments, alternative code paths, optimization flags, and fused/optimized variants that exist but may not be activated.

Additionally, determine the execution context from the run command, run log, and source code:
- Is CUDA graph capture enabled? If so, what mode (full, piecewise, or other)? Which batch sizes are captured?
- Is torch.compile enabled? What compilation passes and fusions are active?
- What other framework-level optimizations are active (e.g., async scheduling, custom allreduce backends)?
This context is essential — proposals must not suggest improvements that are already handled by the active execution mode. For example, if CUDA graphs are used in full-capture mode for decode, kernel launch overhead is already eliminated and should not be proposed as an improvement.

PHASE 2 — IMPROVEMENT ANALYSIS:
Using the understanding from Phase 1, analyze improvement opportunities:
- For EACH individual kernel: can it be replaced with a faster variant, better configured, or fused with an adjacent kernel?
- For SEQUENCES of kernels: are there fusion opportunities, unnecessary dtype conversions, reducible inter-kernel gaps, or stream parallelization possibilities?
- For the BLOCK as a whole: are there scheduling improvements, stream overlap opportunities, or algorithmic changes?
- Every kernel must have at least one improvement proposal, even if the improvement is small — small gains compound across all operations.
- Derive all impact estimates from actual trace timing data, not speculation.
- IMPORTANT: Do not propose improvements that are already handled by the active execution context. For example, do not propose reducing kernel launch gaps if CUDA graphs are already capturing the execution path, and do not propose kernel fusions that are already performed by active torch.compile passes.

PHASE 3 — REVIEW AND VALIDATE:
Before writing the output, review your analysis:
- Verify that every source code reference (file:line) is accurate.
- Verify that every improvement proposal is grounded in what you actually saw in the source code, not generic advice.
- Verify that impact estimates are consistent with the trace data (durations, gaps, percentages sum correctly).
- Ensure no operation was skipped — every kernel in the median block must have been analyzed and have at least one proposal.
- Verify that no proposal conflicts with the active execution context (CUDA graphs, torch.compile, etc.).

Now, write the output document.

FORMATTING RULES — apply throughout the entire output:
- Use clear section headers and consistent indentation.
- Bullet points and numbered lists must have each item on its own line — never concatenate multiple bullets into a single continuous paragraph.
- Use blank lines between sections, between list items when they contain multi-line content, and before/after tables.
- Tables must be properly aligned with consistent column widths.
- The output should be easy to scan and read — a reader should be able to jump to any section and immediately understand the structure.

The document is structured from high-level overview down to deep details. Produce the following sections in order:

1. EXECUTIVE SUMMARY
   - Model, framework, GPU, execution config
   - Median block: index, layer, wall time, variance
   - One paragraph: top 3 findings and total estimated improvement potential

2. MEDIAN BLOCK TIMING OVERVIEW
   - Wall time, compute vs idle, timing distribution (min/p25/median/p75/max/CoV)
   - Per-stream breakdown if multiple streams are used

3. KEY OBSERVATIONS AND ACTION PLAN
   - Ranked list of ALL bottlenecks and improvement opportunities for BS=<batch_size_range> / ISL=<prefill_size_range> / OSL=<output_size_range>, from largest to smallest impact. Include every opportunity, even small ones — improvements compound.
   - For each: one-line description, estimated savings (ns and % of block wall time), difficulty, and proposal ID reference (P1, P2, ...).
   - Projected median block wall time if all low-difficulty proposals were implemented: current → projected (% improvement).
   - Projected median block wall time if ALL proposals were implemented: current → projected (% improvement).

4. MEDIAN BLOCK OPERATIONS TABLE WITH IMPROVEMENTS
   THIS IS THE MOST IMPORTANT SECTION — it provides the at-a-glance overview that readers will focus on.

   Produce a table with ALL operations from [median_block_file] in execution order.

   ALIGNMENT IS CRITICAL: First compute the maximum width of each column across all rows, then pad every cell to that width. Every "|" separator must line up vertically across all rows including the header. Use fixed-width formatting (e.g., printf-style or f-string padding) — do not rely on tab characters.

   Columns:
   | Idx | High-Level Op | Dur (us) | % Blk | Stream | Proposal | Impact % | Improvement Summary | Low-Level Kernel | Details |

   Column descriptions:
   - Idx: operation index in the block
   - High-Level Op: the correlated high_level_op name
   - Dur (us): duration in microseconds
   - % Blk: percentage of total block wall time
   - Stream: CUDA stream ID
   - Proposal: short reference (P1, P2, ...) linking to section 5; comma-separated if multiple (e.g., "P3, P7")
   - Impact %: estimated savings as percentage of total block wall time
   - Improvement Summary: one-line explanation of what to do
   - Low-Level Kernel: the actual GPU kernel name (short_name from trace)
   - Details: additional context that helps understand the improvement — do not truncate

   Rules:
   - Every row must have a proposal. Even if the kernel is already near-optimal, propose something (e.g., "Could save ~0.1us via fusion with adjacent op" or "Near-optimal; consider stream overlap"). Small gains compound.

   After the table, include a TOTALS row:
   ```
   TOTAL ESTIMATED SAVINGS: ~<ns> ns (~<pct>% of block wall time)
   Key ideas: <2-3 sentence summary of the biggest improvement themes>
   ```

5. IMPROVEMENT PROPOSALS
   Each proposal is referenced by its ID (P1, P2, ...) from the table in section 4.
   Include the impact % in the title so it is immediately visible:
   ```
   P<N>: <Title> [<pct>% of block wall time]
   ```

   For each proposal:
   a. AFFECTED OPERATIONS — Which operations from the median block this targets.
   b. EVIDENCE — Specific timing data from the trace. Why this matters at BS=<batch_size_range>/ISL=<prefill_size_range>/OSL=<output_size_range>.
   c. ROOT CAUSE — Why it is slow, grounded in source code. Reference specific files and lines.
   d. PROPOSED FIX — What to change.
      - Type: "kernel_fusion", "kernel_replacement", "scheduling", "config_tuning", or "algorithmic".
      - Detailed step-by-step implementation guide. This must be professional and thorough enough for a developer to execute on. For each step:
        1. State what to change and where (file:line).
        2. Explain the technical rationale: why the current code is suboptimal and what the fix achieves.
        3. Show a BEFORE snippet: the actual current code from the source (10-20 lines, enough to show the full context of the change including surrounding logic, function signature, and relevant variables).
        4. Show an AFTER snippet: the proposed modified code with the change applied, including any new imports, helper functions, or API calls needed.
        5. Explain the expected effect: what changes in the kernel launch pattern, data flow, or scheduling.
   e. IMPACT ESTIMATE — Derived from actual trace data:
      "<operation>: <current_ns> ns -> <projected_ns> ns (savings: <saved_ns> ns, <pct>% of block wall time)"
   f. Difficulty: low / medium / high
   g. Risks/trade-offs

   Rank proposals by (impact * ease), largest first.

   End with:
   ```
   TOTAL ESTIMATED SAVINGS
   =======================
   All proposals: ~<ns> ns (~<pct>%)
   Low-difficulty only: ~<ns> ns (~<pct>%)
   ```

6. SOURCE CODE ANALYSIS AND UTILIZATION
   For each operation in the median block (ordered by duration, largest first):
   - Trace the call chain from forward() to the kernel launch.
   - Source code references (file:line).
   - Kernel launch parameters from [gpu_ops_txt_file] (grid, block, registers, shared memory).
   - Classify as compute-bound, memory-bound, or communication-bound for the specific execution parameters.
   - Utilization estimate vs <gpu_type> specs.
   - Gap to next op with cause attribution.
   - Performance observations grounded in source code or trace data.

   Format per operation:
   ```
   <operation_name> (<duration_ns> ns, <pct>% of block)
     Call chain: <class.method> -> ... -> <kernel>
     Source: <file1:line>, <file2:line>
     Kernel params: grid=<>, block=<>, regs=<>, smem=<>
     Classification: <type> — <rationale for these execution params>
     Utilization: <estimate vs GPU specs>
     Gap to next op: <gap_ns> ns on stream <S> (<cause>)
     Performance observations:
       - <observation>
   ```

7. SUMMARY
   - Concise recap: total block wall time, total estimated savings, top 3 proposals by impact.
   - Regime analysis: would bottleneck priorities shift at a different batch size?
   - Any unusual patterns or anomalies in the trace.

OUTPUT FORMAT: Structured text, sections 1-7.
Write the output to [output_file].
"""

    def prompt(self):
        focus = self.params.perf_analysis_focus
        focus_section = ""
        if focus:
            focus_section = (
                f"PERFORMANCE ANALYSIS FOCUS:\n{focus}\n"
                "Pay special attention to the above focus areas, "
                "especially in sections 3-5."
            )

        return self.prompt_template.format(
            context=self.params.context_header(),
            median_block_file=self.median_block_file,
            high_level_ops_file=self.high_level_ops_file,
            gpu_ops_to_blocks_file=self.gpu_ops_to_blocks_file,
            gpu_ops_txt_file=self.gpu_ops_txt_file,
            output_file=self.output_file,
            perf_analysis_focus_section=focus_section,
        )


def gen_single_trace_prompts(config: SingleTraceConfig, file_prefix=""):
    import os
    output_dir = os.path.abspath(config.output_dir)

    def _path(rel):
        return os.path.join(output_dir, f"{file_prefix}{rel}")

    high_level_ops_file = _path(HIGH_LEVEL_OPS_FILE)
    gpu_ops_txt_file = _path(GPU_OPS_TXT_FILE)
    gpu_ops_to_blocks_file = _path(GPU_OPS_TO_BLOCKS_FILE)
    median_block_file = _path(MEDIAN_BLOCK_FILE)
    perf_analysis_file = _path(PERF_ANALYSIS_SINGLE_FILE)

    params: SingleTraceParams = config

    prompt_objects = [
        TransformerBlockHighLevelPrompt(
            params=params,
            output_file=high_level_ops_file,
        ),
        GpuOpsPrompt(
            params=params,
            output_txt_file=gpu_ops_txt_file,
        ),
        GpuOpsToTransformerBlocksPrompt(
            params=params,
            high_level_ops_file=high_level_ops_file,
            gpu_ops_txt_file=gpu_ops_txt_file,
            output_file=gpu_ops_to_blocks_file,
            median_block_file=median_block_file,
        ),
    ]

    step_names = [
        "Extracting high-level transformer block operations from source code",
        "Extracting GPU operations from trace",
        "Correlating GPU operations to transformer blocks and selecting median",
    ]

    output_files = {
        "high_level_ops": high_level_ops_file,
        "gpu_ops_txt": gpu_ops_txt_file,
        "gpu_ops_to_blocks": gpu_ops_to_blocks_file,
        "median_block": median_block_file,
    }

    if not config.skip_perf_analysis:
        prompt_objects.append(
            PerfAnalysisPrompt(
                params=params,
                median_block_file=median_block_file,
                high_level_ops_file=high_level_ops_file,
                gpu_ops_to_blocks_file=gpu_ops_to_blocks_file,
                gpu_ops_txt_file=gpu_ops_txt_file,
                output_file=perf_analysis_file,
            )
        )
        step_names.append("Generating performance analysis")
        output_files["perf_analysis"] = perf_analysis_file

    trace_prompt = build_single_trace_json_prompt(config)
    prompt_objects.append(trace_prompt)
    step_names.append("Generating transformer block trace")
    output_files.update({
        "transformer_block_trace_json": trace_prompt.output_json_file,
        "transformer_block_trace_txt": trace_prompt.output_txt_file,
    })

    total_steps = len(step_names)
    prompts = []
    for i, (step_name, prompt_obj) in enumerate(
        zip(step_names, prompt_objects), 1
    ):
        prompts.append(
            {
                "cmd": (
                    f"print('\\n=== [Step {i}/{total_steps}]"
                    f" {step_name}... ===')"
                )
            }
        )
        prompts.append(prompt_obj.prompt())

    return prompts, output_files
