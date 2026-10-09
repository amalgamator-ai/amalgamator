import os
from dataclasses import dataclass
from typing import ClassVar, Optional

from llm.auto_analyze.configs.cross_trace_config import (
    CrossTraceConfig,
    TraceResult,
    CROSS_MATCHING_FILE,
    CROSS_COMPARE_FILE,
    CROSS_IMPROVEMENT_FILE,
)
from common.iteration_utils import prev_iteration_section as _prev_iteration_section
from common.claude_utils import PipelineStep


# ---------------------------------------------------------------------------
# File prefix constants for versioned iteration outputs
# ---------------------------------------------------------------------------
MATCHING_PREFIX = "cross_matching_blocks"
COMPARE_PREFIX = "cross_compare_blocks"
IMPROVEMENT_PREFIX = "cross_improvement_plan"
REDO_IMPROVEMENT_PREFIX = "redo_improvement_plan"
REDO_CONTEXT_PREFIX = "redo_analysis_context"
REDO_IMPROVEMENT_FILE = "redo_improvement_plan.txt"


def _build_traces_context(
    traces: list[TraceResult], target_idx: int, output_dir: str
) -> str:
    lines = []
    for i, tr in enumerate(traces):
        marker = " (TARGET)" if i == target_idx else ""
        lines.append(f"Trace [{i}]{marker}:")
        lines.append(f"  trace_id: {tr.trace_id}")
        lines.append(f"  framework_name: {tr.framework_name}")
        lines.append(f"  framework_source_code: {tr.framework_source_code}")
        lines.append(f"  commit_id: {tr.commit_id}")
        lines.append(f"  model: {tr.model}")
        lines.append(f"  gpu_type: {tr.gpu_type}")
        lines.append(
            f"  execution_params: BS={tr.batch_size_range} ISL={tr.prefill_size_range} OSL={tr.output_size_range}"
        )
        lines.append(f"  median_block_file: {tr.get_median_block_file()}")
        lines.append(f"  high_level_ops_file: {tr.get_high_level_ops_file()}")
        perf_file = tr.get_perf_analysis_file()
        if os.path.exists(perf_file):
            lines.append(f"  perf_analysis_file: {perf_file}")
        lines.append(f"  gpu_ops_to_blocks_file: {tr.get_gpu_ops_to_blocks_file()}")
        lines.append("")
    abs_output = os.path.abspath(output_dir)
    lines.append(
        f"IMPORTANT: Write ALL output files to {abs_output} only. Do not create files anywhere else."
    )
    return "\n".join(lines)


# ============================================================================
# Generation prompt classes
# ============================================================================


@dataclass
class MatchMedianTransformerBlocksPrompt:
    config: CrossTraceConfig
    output_file: str
    prev_output_file: Optional[str] = None
    prev_output_summary_file: Optional[str] = None
    iteration: int = 1
    output_summary_file: str = ""
    prompt_template: ClassVar[str] = """
{traces_context}
{prev_iteration_section}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}

This is a {analysis_type} matching.
{analysis_description}

The goal of this step is to MATCH the operations between the median transformer blocks of all traces.

Each trace has its own low-level GPU operations correlated to high-level operations, and since high-level operations as similar between traces (they all implement the same logical LLM transformer block), exploit this to help match the low-level operations between the transformer block traces.

Do the following:
- Read and analyze in-detail the high-level transformer block file of each framework
- Read and analyze in-detail the median transformer block file of each framework, with their low-level => high-level operation matchings.
- For each block, understand its operation sequence: the high level operation names, the low-level kernel names, and what each operation does.
- Match operations across blocks, one-by-one, with the help of:
    - High-level logical purpose
    - Order in the logical sequence of a transformer block
    - Source code inspections and deep dives
    - It may be that each framework has a couple block types, then match each type individually.
- Take into account separate GPU streams and their synchronization points.
- When operations do not match cleanly, perform a deeper analysis and source code deep-dive to fully understand it and fix it.
- Read the RUNTIME CONFIGURATION section from each trace's high_level_ops_file. Note which backends, modes, and code paths are active in each framework — including the dispatch points and active branches. This is essential context for understanding why operations differ between frameworks and for ensuring downstream improvement proposals target the correct code paths. If a trace's high_level_ops_file does not contain a RUNTIME CONFIGURATION section, read the run log directly and extract the backend/mode selections by looking for lines where the framework logs which implementation was selected.

Produce a structured output with:
1. A matched operations table showing, for each logical operation, the corresponding high_level_op name, low-level kernel name, duration, and stream for each block — side by side.
2. Operations that exist in one trace but not the other (added/removed/fused differently).
3. Overall wall time for each trace.

Dump results to [output_file].
Dump a summary to [output_summary_file] that includes:
- An explanation of the work done to generate the matching in this iteration
- A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
"""

    def prompt(self):
        analysis_type = self.config.infer_analysis_type()
        if analysis_type == "cross-framework":
            desc = "The traces come from different frameworks running the same model."
        else:
            desc = "The traces come from the SAME framework but different versions/commits."

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file
            ),
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            analysis_type=analysis_type,
            analysis_description=desc,
            iteration=self.iteration,
        )


@dataclass
class CompareMedianTransformerBlocksPrompt:
    config: CrossTraceConfig
    matching_file: str
    output_file: str
    prev_output_file: Optional[str] = None
    prev_output_summary_file: Optional[str] = None
    iteration: int = 1
    output_summary_file: str = ""
    prompt_template: ClassVar[str] = """
{traces_context}
{prev_iteration_section}
[matching_file] = {matching_file}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}
- {analysis_description}
- [matching_file] contains a detailed per-operation matching of median transformer blocks across traces.
- The TARGET trace is [{target_idx}] ({target_label}) — the trace we want to optimize/fix.

Do the following:
- Read, analyze in-detail, and understand the matching of blocks in [matching_file].
- Detect all of the performance differences (both positive and negative), from the pespective of the target trace. For each such difference:
    - Analyze in-detail the performance difference in the context of the high-level logical operation involved and the low-level operations produced.
    - Perform source code analysis to get all of the details necessary to understand the performance difference.
    - Understand in-detail the root causes for the difference, grounded in the source code and concrete precise reasons.
    - Understand in-detail why one trace is slower and the other is faster.

- Generate a comparison summary of the performance differences found (both positive and negative), from the pespective of the target trace:
    - Executive summary with gaps, key findings/ideas, and brief explanations.
    - A detailed sequence of the performance differences, sorted from negative to positive, where the negative are sorted from worst negative to less worst, and positive, from most positive to less positive. For each difference, provide:
        - An explanation of root causes, key findings/ideas that make this difference, and how it affects the comparison.
        - Provide code snippet examples to better explain why the difference exist.
        - Provide source code, pull-requests (PRs), or specific commit references and explanations that are responsible for this difference.
    - Make sure the summary is clear, concise and professional and it fully explains the differences between the blocks.

CRITICAL REVIEW — before writing the output, perform a thorough review of every performance difference found:
- For each difference, re-verify the root cause by re-reading the relevant source code sections in both traces. Ensure the explanation is grounded in what the code actually does, not assumptions.
- Cross-check timing deltas against the matched operations in [matching_file] — verify that the numbers are consistent and sum correctly.
- For each code snippet or source reference cited, confirm it exists and is relevant to the claimed root cause.
- Verify that no performance difference was missed — re-scan the matched operations for any delta that was overlooked.
- If any difference cannot be fully explained with concrete source code evidence, flag it explicitly rather than speculating.
- Fix any inaccuracies found during the review before proceeding.

SECOND REVIEW — after fixing issues from the first review, perform one more pass:
- Re-read the complete comparison end-to-end and verify it tells a coherent, consistent story.
- Ensure the executive summary accurately reflects the detailed findings — no contradictions.
- Verify that all timing percentages and impact numbers are arithmetically correct and consistent with each other.
- Confirm the sort order is correct: negatives from worst to least, positives from most to least.
- Fix any issues found before proceeding.

THIRD REVIEW — final quality check:
- For each performance difference, verify that the explanation would be clear and actionable to a developer who has not seen the traces before.
- Ensure every claim about "why" a kernel is faster or slower has a specific source code reference — not just a file name, but the relevant line range and what the code does.
- Verify that the total net delta (sum of all individual deltas) is consistent with the overall wall-time difference between the two blocks.

Dump the comparison summary to [output_file].
Dump a summary to [output_summary_file] that includes:
- An explanation of the work done to generate the comparison in this iteration
- A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
"""

    def prompt(self):
        target = self.config.get_target_result()
        analysis_type = self.config.infer_analysis_type()

        if analysis_type == "cross-framework":
            desc = "This is a CROSS-FRAMEWORK comparison of different frameworks running the same model."
            label = target.framework_name
        else:
            desc = f"This is a CROSS-COMMIT comparison of {target.framework_name} across different commits."
            label = f"{target.framework_name} commit {target.commit_id[:12]}"

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file
            ),
            matching_file=self.matching_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            analysis_description=desc,
            target_idx=self.config.target_trace_id,
            target_label=label,
            target_framework=target.framework_name,
            iteration=self.iteration,
        )


@dataclass
class ImprovementPlanPrompt:
    config: CrossTraceConfig
    compare_file: str
    output_file: str
    prev_output_file: Optional[str] = None
    prev_output_summary_file: Optional[str] = None
    iteration: int = 1
    output_summary_file: str = ""
    knowledge_base_file: Optional[str] = None
    prompt_template: ClassVar[str] = """
{traces_context}
{prev_iteration_section}
{knowledge_base_section}
[compare_file] = {compare_file}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}
- {analysis_description}
- [compare_file] contains a detailed comparison of median transformer blocks across traces, covering all performance differences (both positive and negative).
- The TARGET trace is [{target_idx}] ({target_label}) — the trace we want to improve.

Your task is to generate an improvement plan that fully recovers performance for ALL negative differences identified in [compare_file] from the perspective of the target trace.

Do the following:
- Read, analyze in-detail, and understand the comparison in [compare_file].
- Identify ALL cases where the target trace is slower than the other trace(s).
- For each such negative difference:
    - Understand in-detail the performance difference in the context of the high-level logical operation involved and the low-level operations produced.
    - Perform source code analysis to get all of the details necessary to understand the performance difference:
        - Do source code analysis for the faster framework, including call-chains and execution details.
        - Do source code analysis for the slower framework, including call-chains and execution details.
    - Understand in-detail the root cause, grounded in the source code and concrete precise reasons.
    - Understand in-detail why the other framework is faster and see how to fix this performance difference by porting the code from the other framework. I.e we want to COPY-PASTE as much as possible from the other framework while doing MINIMAL changes to the target framework.

- CRITICAL — ARCHITECTURAL PORTABILITY VALIDATION: For each performance difference, before proposing a fix:
    - Verify that the optimization mechanism is architecturally equivalent between frameworks. Specifically:
        - Do both frameworks use the same module structure for this operation? (e.g., same expert count, same weight dimensions, same config lookup keys)
        - If framework A uses E=257 experts (fused shared expert) and framework B uses E=256 (separate shared expert), they load DIFFERENT config files — copying config values across won't work.
        - If framework A uses separate configs per GEMM direction and framework B uses a shared config, the parameter space is different — matching individual values won't replicate the behavior.
        - If the performance gap stems from an architectural difference (different module design, different dispatch mechanism, different weight layout), explicitly state this and rate the proposal as requiring ARCHITECTURAL CHANGES, not config/parameter changes.
    - For config-level proposals: verify that the exact config file, key, and lookup mechanism used by the source framework also exists and is reachable in the target framework. If the config lookup keys differ (e.g., different E or N values due to different architecture), the config-level approach is INVALID.
    - Rate each proposal's difficulty accurately: TRIVIAL (config-only, verified reachable), MEDIUM (code change within existing architecture), HIGH (architectural change needed).

- Generate a ranked improvement plan for the target trace that fully recovers its performance as follows:
    - Executive summary with gaps, key findings, key ideas, and ranked plan summary (that also has impact per each improvement in % of total block time).
    - A detailed ranked sequence of improvement proposals (from most impactful to less), where each proposal has:
        - A summary of root causes, key findings, key ideas, and estimated improvement/impact in % of total block time.
        - A step-by-step guide that describes how to port code from the other framework with minimal changes to the target framework, in order to fix the performance difference fully. For each step provide:
            - An explanation of the step
            - A code snippet example that shows what needs to be changed. Ensure the code snippet is properly formatted and detailed enough.
    - Make sure the plan is clear, concise and professional so that it can be executed on by a professional LLM programmer.

Dump the improvement plan to [output_file].
Dump a summary to [output_summary_file] that includes:
- An explanation of the work done to generate the improvement plan in this iteration
- A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
"""

    def prompt(self):
        target = self.config.get_target_result()
        analysis_type = self.config.infer_analysis_type()

        if analysis_type == "cross-framework":
            desc = "This is a CROSS-FRAMEWORK comparison of different frameworks running the same model."
            label = target.framework_name
        else:
            desc = f"This is a CROSS-COMMIT comparison of {target.framework_name} across different commits."
            label = f"{target.framework_name} commit {target.commit_id[:12]}"

        kb_section = ""
        if self.knowledge_base_file and os.path.isfile(self.knowledge_base_file):
            kb_section = (
                "IMPORTANT — Read the knowledge base before generating proposals. "
                "It contains learnings from previous code generation runs that MUST "
                "inform your analysis (architectural pitfalls, invalid assumptions, "
                "proven constraints). Apply these learnings to avoid repeating mistakes.\n"
                f"[knowledge_base_file] = {self.knowledge_base_file}\n"
            )

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file
            ),
            knowledge_base_section=kb_section,
            compare_file=self.compare_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            analysis_description=desc,
            target_idx=self.config.target_trace_id,
            target_label=label,
            target_framework=target.framework_name,
            iteration=self.iteration,
        )


# ============================================================================
# Review prompt classes
# ============================================================================


@dataclass
class ReviewMatchingPrompt:
    config: CrossTraceConfig
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int = 1
    prompt_template: ClassVar[str] = """
{traces_context}
[input_file] = {input_file}
[input_summary_file] = {input_summary_file}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}

This is a review of the matched operations produced in iteration {iteration}.

Read and analyze in-detail [input_file] and [input_summary_file]. Then perform the following domain-specific verification checks:

1. RUNTIME CONFIGURATION CONTEXT:
   - Read the RUNTIME CONFIGURATION section from each trace's high_level_ops_file. If not available, read the run log directly and find lines where the framework logs which implementation was selected.
   - Note which backends, modes, and code paths are active in each framework.
   - When verifying matches below, use this context to understand WHY operations differ between frameworks — different backends produce different kernel sequences for the same logical operation.

2. MATCHED PAIR VERIFICATION — for each matched operation pair:
   - Verify that both operations serve the same logical role in the transformer block (e.g., both are QKV projections, both are attention softmax, etc.).
   - If the logical roles differ, flag the mismatch and provide the correct matching with evidence from the high-level ops files and source code.
   - When operations use different kernel names, verify via source code that both kernels implement the same logical step under their respective active backends.

3. COVERAGE VERIFICATION — check for missing operations:
   - Re-read each trace's median block file and high-level ops file.
   - Verify that every operation in each trace is accounted for in the matching — either as a matched pair, or explicitly listed as added/removed/fused.
   - If any operations were missed, add them to the appropriate category with an explanation.

4. CLASSIFICATION VERIFICATION — for operations marked as added/removed/fused:
   - Verify that the classification is correct by checking the source code of both frameworks.
   - Confirm that "added" operations truly have no counterpart in the other trace. Consider whether the operation might exist under a different name due to a different active backend.
   - Confirm that "removed" operations truly have no counterpart.
   - Confirm that "fused" operations correctly identify which operations were combined and in which trace.

5. WALL TIME CONSISTENCY — verify that the overall wall time totals for each trace:
   - Are consistent with the sum of individual matched and unmatched operation durations.
   - Match the wall times reported in the median block files.
   - Flag any arithmetic discrepancies.

For each issue found, document:
- What is wrong
- Why it matters
- The correction applied

Produce a corrected matching with all issues fixed and dump to [output_file].

Dump a summary to [output_summary_file] that includes:
- Documentation of the issues found and fixed in this review
- A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed to the matching, write CONVERGED as the very first line of [output_summary_file]. Otherwise, do NOT write CONVERGED.
"""

    def prompt(self):
        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class ReviewComparePrompt:
    config: CrossTraceConfig
    matching_file: str
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int = 1
    prompt_template: ClassVar[str] = """
{traces_context}
[matching_file] = {matching_file}
[input_file] = {input_file}
[input_summary_file] = {input_summary_file}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}
- The TARGET trace is [{target_idx}] ({target_label}) — the trace we want to optimize/fix.

This is a review of the performance comparison produced in iteration {iteration}.

Read and analyze in-detail [input_file] and [input_summary_file]. Then perform the following domain-specific verification checks:

1. ACTIVE CODE PATH VERIFICATION — for each performance difference:
   - Read the RUNTIME CONFIGURATION section from each trace's high_level_ops_file (or the run log directly if the section is not available).
   - Verify that the source code analysis for each difference follows the active code path — the branch actually taken at runtime given the logged backend/mode selections.
   - If a root cause explanation references a code path that is NOT active (e.g., analyzing the mega_moe path when the runtime uses flashinfer_trtllm), flag it immediately and redirect the analysis to the correct active path.

2. NET DELTA CONSISTENCY — verify that the net delta (sum of all individual performance deltas) equals the overall wall-time difference between the two blocks as reported in [matching_file]. If they do not match, identify the missing or incorrect deltas and fix them.

3. ROOT CAUSE VERIFICATION — for each performance difference:
   - Verify that the root cause has a concrete source code reference — not just a file name, but a specific line range and an explanation of what the code does.
   - Re-read the cited source code sections in both traces and confirm the explanation is accurate.
   - Verify the cited code is on the active code path (not a dead branch).
   - If a root cause is vague, speculative, or lacks a concrete source reference, flag it and provide the correct grounded explanation.

4. SORT ORDER VERIFICATION — verify the sort ordering:
   - Negative differences must be sorted from worst (largest negative) to least (smallest negative).
   - Positive differences must be sorted from most positive (largest positive) to least positive (smallest positive).
   - Fix any ordering violations.

5. MISSING DELTA CHECK — re-scan the matched operations in [matching_file]:
   - Identify any operation pairs with a non-trivial timing delta that are not covered in the comparison.
   - For any missing deltas found, add them with proper root cause analysis.

For each issue found, document:
- What is wrong
- Why it matters
- The correction applied

Produce a corrected comparison with all issues fixed and dump to [output_file].

Dump a summary to [output_summary_file] that includes:
- Documentation of the issues found and fixed in this review
- A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed to the comparison, write CONVERGED as the very first line of [output_summary_file]. Otherwise, do NOT write CONVERGED.
"""

    def prompt(self):
        target = self.config.get_target_result()
        analysis_type = self.config.infer_analysis_type()

        if analysis_type == "cross-framework":
            label = target.framework_name
        else:
            label = f"{target.framework_name} commit {target.commit_id[:12]}"

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            matching_file=self.matching_file,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            target_idx=self.config.target_trace_id,
            target_label=label,
            iteration=self.iteration,
        )


@dataclass
class ReviewImprovementPlanPrompt:
    config: CrossTraceConfig
    compare_file: str
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int = 1
    knowledge_base_file: Optional[str] = None
    prompt_template: ClassVar[str] = """
{traces_context}
{knowledge_base_section}
[compare_file] = {compare_file}
[input_file] = {input_file}
[input_summary_file] = {input_summary_file}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}
- The TARGET trace is [{target_idx}] ({target_label}) — the trace we want to improve.

This is a review of the improvement plan produced in iteration {iteration}. This is the LAST QUALITY GATE before the improvement plan is used for code generation — thoroughness is critical.

Read and analyze in-detail [input_file] and [input_summary_file]. Then perform the following domain-specific verification checks:

1. ROOT CAUSE GROUNDING — for each improvement proposal:
   - Verify that the root cause is grounded in the source code of BOTH frameworks (the target and the reference).
   - Re-read the cited source code sections and confirm the explanation is accurate.
   - If a root cause is vague, speculative, or references code that does not exist, flag it and provide the correct grounded explanation.

2. CRITICAL — ACTIVE CODE PATH VERIFICATION — for each improvement proposal:
   - Read the RUNTIME CONFIGURATION section from the target trace's high_level_ops_file.
   - For the subsystem being changed, find the logged runtime selection and the dispatch point in the source code.
   - Evaluate the dispatch condition with the runtime value to identify which branch is taken.
   - Verify that the proposed code changes target ONLY the active code path — the branch that is actually executed at runtime.
   - If the proposal targets a dead code path (a branch that is NOT taken at runtime), flag it immediately and redirect the proposal to the correct active code path.
   - This is the single most important check — changes to inactive code paths have ZERO effect at runtime.

3. STEP-BY-STEP GUIDE VERIFICATION — for each proposal's implementation guide:
   - Verify that all referenced files and line ranges actually exist in the target framework source code.
   - Verify that code snippet examples are syntactically correct and match the actual code structure.
   - Verify that the steps are in the correct logical order (dependencies before dependents).
   - Flag any steps that are ambiguous or missing critical details.

4. CRITICAL — ARCHITECTURAL PORTABILITY VERIFICATION — for each proposal in a cross-framework comparison:
   - Verify that the optimization mechanism is architecturally equivalent between the source and target frameworks.
   - For config/parameter-level proposals: confirm that the target framework uses the SAME config lookup keys (expert count, weight dimensions, config file naming) as the source. If the source uses E=257 but the target uses E=256, they load different config files — the proposal is INVALID as a config-only change.
   - For proposals that reference how the source framework structures a module: verify that the target framework uses the same structure. If the source fuses a shared expert into the MoE kernel (changing expert count and weight layout) but the target runs the shared expert separately, the performance gap is architectural — not fixable by parameter tuning.
   - Flag any proposal where the root cause of the performance gap is an architectural difference between frameworks, and reclassify it from TRIVIAL/config-only to requiring ARCHITECTURAL CHANGES.

5. IMPACT ESTIMATE CONSISTENCY — verify that:
   - The estimated improvement percentages are consistent with the timing data in [compare_file].
   - The sum of all proposal impacts is consistent with the total performance gap.
   - No improvement is double-counted across proposals.

5. COVERAGE CHECK — verify that ALL negative performance differences from [compare_file] are addressed:
   - Re-read [compare_file] and list all negative differences.
   - Verify each one has a corresponding proposal in the improvement plan.
   - If any negative difference is unaddressed, add a proposal for it or explain why it cannot be addressed.

For each issue found, document:
- What is wrong
- Why it matters (especially for active code path violations)
- The correction applied

Produce a corrected improvement plan with all issues fixed and dump to [output_file].

Dump a summary to [output_summary_file] that includes:
- Documentation of the issues found and fixed in this review
- A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed to the improvement plan, write CONVERGED as the very first line of [output_summary_file]. Otherwise, do NOT write CONVERGED.
"""

    def prompt(self):
        target = self.config.get_target_result()
        analysis_type = self.config.infer_analysis_type()

        if analysis_type == "cross-framework":
            label = target.framework_name
        else:
            label = f"{target.framework_name} commit {target.commit_id[:12]}"

        kb_section = ""
        if self.knowledge_base_file and os.path.isfile(self.knowledge_base_file):
            kb_section = (
                "IMPORTANT — Read the knowledge base before reviewing proposals. "
                "It contains learnings from previous code generation runs. Verify "
                "that no proposal violates a known constraint or repeats a proven mistake.\n"
                f"[knowledge_base_file] = {self.knowledge_base_file}\n"
            )

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            knowledge_base_section=kb_section,
            compare_file=self.compare_file,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            target_idx=self.config.target_trace_id,
            target_label=label,
            iteration=self.iteration,
        )


# ============================================================================
# Iteration step generator functions
# ============================================================================


def gen_matching_iter_steps(
    config: CrossTraceConfig,
    traces_context: str,
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
    iteration: int,
):
    """Generate generation + review steps for one matching iteration.

    Returns (steps, review_prompt) where steps is a list of PipelineStep
    and review_prompt is the ReviewMatchingPrompt instance (for accessing
    output_file / output_summary_file for convergence checks).
    """
    output_dir = os.path.abspath(config.output_dir)

    gen_prompt = MatchMedianTransformerBlocksPrompt(
        config=config,
        output_file=os.path.join(output_dir, "{}_V{}.txt".format(MATCHING_PREFIX, iteration)),
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
        output_summary_file=os.path.join(
            output_dir, "{}_summary_V{}.txt".format(MATCHING_PREFIX, iteration)
        ),
    )

    review_prompt = ReviewMatchingPrompt(
        config=config,
        input_file=gen_prompt.output_file,
        input_summary_file=gen_prompt.output_summary_file,
        output_file=os.path.join(
            output_dir, "{}_review_V{}.txt".format(MATCHING_PREFIX, iteration)
        ),
        output_summary_file=os.path.join(
            output_dir, "{}_review_summary_V{}.txt".format(MATCHING_PREFIX, iteration)
        ),
        iteration=iteration,
    )

    steps = [
        PipelineStep(
            name="matching_V{}".format(iteration),
            prompt=gen_prompt.prompt(),
            output_files=[gen_prompt.output_file, gen_prompt.output_summary_file],
        ),
        PipelineStep(
            name="matching_review_V{}".format(iteration),
            prompt=review_prompt.prompt(),
            output_files=[review_prompt.output_file, review_prompt.output_summary_file],
        ),
    ]

    return steps, review_prompt


def gen_compare_iter_steps(
    config: CrossTraceConfig,
    traces_context: str,
    matching_file: str,
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
    iteration: int,
):
    """Generate generation + review steps for one compare iteration.

    Returns (steps, review_prompt).
    """
    output_dir = os.path.abspath(config.output_dir)

    gen_prompt = CompareMedianTransformerBlocksPrompt(
        config=config,
        matching_file=matching_file,
        output_file=os.path.join(output_dir, "{}_V{}.txt".format(COMPARE_PREFIX, iteration)),
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
        output_summary_file=os.path.join(
            output_dir, "{}_summary_V{}.txt".format(COMPARE_PREFIX, iteration)
        ),
    )

    review_prompt = ReviewComparePrompt(
        config=config,
        matching_file=matching_file,
        input_file=gen_prompt.output_file,
        input_summary_file=gen_prompt.output_summary_file,
        output_file=os.path.join(
            output_dir, "{}_review_V{}.txt".format(COMPARE_PREFIX, iteration)
        ),
        output_summary_file=os.path.join(
            output_dir, "{}_review_summary_V{}.txt".format(COMPARE_PREFIX, iteration)
        ),
        iteration=iteration,
    )

    steps = [
        PipelineStep(
            name="compare_V{}".format(iteration),
            prompt=gen_prompt.prompt(),
            output_files=[gen_prompt.output_file, gen_prompt.output_summary_file],
        ),
        PipelineStep(
            name="compare_review_V{}".format(iteration),
            prompt=review_prompt.prompt(),
            output_files=[review_prompt.output_file, review_prompt.output_summary_file],
        ),
    ]

    return steps, review_prompt


def _resolve_knowledge_base_file():
    """Resolve the global knowledge base file path."""
    from llm.auto_code_gen.code_gen_prompts import KNOWLEDGE_BASE_FILE
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        KNOWLEDGE_BASE_FILE,
    )
    return kb_path if os.path.isfile(kb_path) else None


def gen_improvement_plan_iter_steps(
    config: CrossTraceConfig,
    traces_context: str,
    compare_file: str,
    prev_output_file: Optional[str],
    prev_output_summary_file: Optional[str],
    iteration: int,
):
    """Generate generation + review steps for one improvement plan iteration.

    Returns (steps, review_prompt).
    """
    output_dir = os.path.abspath(config.output_dir)
    kb_file = _resolve_knowledge_base_file()

    gen_prompt = ImprovementPlanPrompt(
        config=config,
        compare_file=compare_file,
        output_file=os.path.join(output_dir, "{}_V{}.txt".format(IMPROVEMENT_PREFIX, iteration)),
        prev_output_file=prev_output_file,
        prev_output_summary_file=prev_output_summary_file,
        iteration=iteration,
        output_summary_file=os.path.join(
            output_dir, "{}_summary_V{}.txt".format(IMPROVEMENT_PREFIX, iteration)
        ),
        knowledge_base_file=kb_file,
    )

    review_prompt = ReviewImprovementPlanPrompt(
        config=config,
        compare_file=compare_file,
        input_file=gen_prompt.output_file,
        input_summary_file=gen_prompt.output_summary_file,
        output_file=os.path.join(
            output_dir, "{}_review_V{}.txt".format(IMPROVEMENT_PREFIX, iteration)
        ),
        output_summary_file=os.path.join(
            output_dir, "{}_review_summary_V{}.txt".format(IMPROVEMENT_PREFIX, iteration)
        ),
        iteration=iteration,
        knowledge_base_file=kb_file,
    )

    steps = [
        PipelineStep(
            name="improvement_plan_V{}".format(iteration),
            prompt=gen_prompt.prompt(),
            output_files=[gen_prompt.output_file, gen_prompt.output_summary_file],
        ),
        PipelineStep(
            name="improvement_plan_review_V{}".format(iteration),
            prompt=review_prompt.prompt(),
            output_files=[review_prompt.output_file, review_prompt.output_summary_file],
        ),
    ]

    return steps, review_prompt


# ============================================================================
# Legacy flat prompt generation (backward compatibility)
# ============================================================================


def gen_cross_trace_prompts(config: CrossTraceConfig, file_prefix=""):
    import os

    output_dir = os.path.abspath(config.output_dir)

    def _path(rel):
        return os.path.join(output_dir, f"{file_prefix}{rel}")

    matching_file = _path(CROSS_MATCHING_FILE)
    compare_file = _path(CROSS_COMPARE_FILE)

    prompt_objects = [
        MatchMedianTransformerBlocksPrompt(
            config=config,
            output_file=matching_file,
        ),
        CompareMedianTransformerBlocksPrompt(
            config=config,
            matching_file=matching_file,
            output_file=compare_file,
        ),
    ]

    step_names = [
        "Matching median transformer blocks across traces",
        "Comparing performance differences across block traces",
    ]

    output_files = {
        "cross_matching": matching_file,
        "cross_compare": compare_file,
    }

    if config.make_improvement_plan:
        improvement_file = _path(CROSS_IMPROVEMENT_FILE)
        prompt_objects.append(
            ImprovementPlanPrompt(
                config=config,
                compare_file=compare_file,
                output_file=improvement_file,
            )
        )
        step_names.append("Generating improvement plan for target trace")
        output_files["cross_improvement"] = improvement_file

    total_steps = len(step_names)
    prompts = []
    for i, (step_name, prompt_obj) in enumerate(zip(step_names, prompt_objects), 1):
        prompts.append(
            {
                "cmd": (
                    f"print('\\n=== [Step {i}/{total_steps}] {step_name}... ===')"
                )
            }
        )
        prompts.append(prompt_obj.prompt())

    return prompts, output_files


# ============================================================================
# Redo cross-trace analysis prompt
# ============================================================================


@dataclass
class RedoCrossTraceAnalysisPrompt:
    config: CrossTraceConfig
    code_gen_dirs: list[str]
    knowledge_base_file: str
    original_improvement_plan_file: str
    original_compare_file: str
    output_file: str
    output_summary_file: str
    output_context_file: str
    iteration: int = 1
    full_decode_step_analysis: bool = False
    prompt_template: ClassVar[str] = """
{traces_context}

[original_improvement_plan_file] = {original_improvement_plan_file}
[original_compare_file] = {original_compare_file}
[knowledge_base_file] = {knowledge_base_file}
[code_gen_dirs] = {code_gen_dirs}
[output_file] = {output_file}
[output_summary_file] = {output_summary_file}
[output_context_file] = {output_context_file}
- {analysis_description}
- The TARGET trace is [{target_idx}] ({target_label}) — the trace we want to improve.

This is a REDO of the cross-trace improvement plan (iteration {iteration}). The original analysis has already been used to generate code — some proposals succeeded, some failed, and significant learnings were accumulated. This redo incorporates all of that feedback to produce a corrected and updated improvement plan.

Think hard and be extremely thorough.

Do the following:

Step 1: Read the original analysis.
- Read [original_improvement_plan_file] — the original improvement plan with its proposals and estimated impacts.
- Read [original_compare_file] — the original per-operation performance comparison between traces.

Step 2: Read the knowledge base.
- Read [knowledge_base_file] — accumulated learnings from all code generation runs. Each entry documents an architectural constraint, performance insight, methodology lesson, or comparison pitfall. These are GROUND TRUTH from actual implementation attempts and MUST inform every proposal.

Step 3: Read code generation results.
For each directory in [code_gen_dirs]:
- Read iteration_history_summary.txt (or the latest version) — comprehensive history of all phases
- Read all code_gen_runtime_summary_V*.txt files — what was tried, what worked, what failed
- Read runtime_results_manifest.json — final throughput results
- Read diagnostic profile reports (runtime_*/runtime_logs/diagnostic_profile_iteration_V*/diagnostic_report.txt) — kernel-level per-op comparisons showing ground truth
- Understand: which proposal was attempted, what the predicted improvement was, what the actual improvement was, and WHY the prediction was wrong (if it was)

Step 4: Re-assess each original proposal.
For each proposal in the original improvement plan:
- Was it attempted? If so, what happened?
- Was the predicted improvement accurate? If not, by how much and why?
- Is it fully implemented (done), partially implemented (residual opportunity), or not attempted (still open)?
- Does the knowledge base contain entries that invalidate or modify this proposal?
- What is the CORRECTED estimate based on actual data?

Step 5: Identify the CURRENT performance gap.
- Use the latest NSYS diagnostic profile data (from code-gen results) to see where the gap stands NOW, after applied optimizations
- The original traces show the gap BEFORE any code changes; the diagnostic profiles show the gap AFTER
- Identify: what bottlenecks remain, what new bottlenecks appeared, what was successfully fixed

{full_decode_step_section}

Step 6: Generate the UPDATED improvement plan.
Produce a new ranked improvement plan that:
- Marks implemented proposals as DONE with actual results
- Corrects estimates for open proposals using learnings from the knowledge base and actual diagnostic data
- Adds NEW proposals for bottlenecks that emerged from previous optimizations
- Removes proposals that the knowledge base proves are invalid (e.g., architectural mismatches)
- For each proposal, include:
  - Corrected impact estimate with justification from kernel-level data
  - Difficulty rating informed by actual implementation experience
  - Risk assessment informed by issues encountered in previous code-gen runs
  - Reference to knowledge base entries that apply
- Apply the ARCHITECTURAL PORTABILITY VALIDATION from the knowledge base — verify that each proposal's mechanism is architecturally equivalent between frameworks

Step 7: Write outputs.
- Dump the updated improvement plan to [output_file]
- Dump a summary to [output_summary_file] that includes:
  - What changed vs the original plan and why
  - Which proposals are done, corrected, new, or removed
  - The current performance gap after implemented optimizations
  - Recommended next proposal to implement
- Dump the analysis context to [output_context_file] that documents:
  - Which knowledge base entries influenced which decisions
  - Which diagnostic profile data was used for which estimates
  - The evidence chain for each corrected estimate
"""

    def prompt(self):
        target = self.config.get_target_result()
        analysis_type = self.config.infer_analysis_type()

        if analysis_type == "cross-framework":
            desc = "This is a CROSS-FRAMEWORK comparison of different frameworks running the same model."
            label = target.framework_name
        else:
            desc = f"This is a CROSS-COMMIT comparison of {target.framework_name} across different commits."
            label = f"{target.framework_name} commit {target.commit_id[:12]}"

        full_decode_section = ""
        if self.full_decode_step_analysis:
            full_decode_section = """Step 5b: FULL DECODE STEP ANALYSIS
The per-block analysis (Step 5) only covers ONE transformer block. But TPOT (Time Per Output Token)
includes the ENTIRE decode step — all N blocks plus all non-block work. To fully explain the TPOT
difference between frameworks, analyze the complete decode step:

- Read the NSYS SQLite traces for BOTH frameworks (from the single-trace analysis directories).
  The trace files are referenced in each trace's config.
- For each framework, measure the FULL decode step duration — from the first kernel of one decode
  iteration to the first kernel of the next decode iteration. This captures everything:
    1. ALL N transformer blocks (not just the median one) — sum their wall times
    2. Pre-forward work: embedding lookup, position encoding
    3. Post-forward work: final layer norm, logits projection, sampling
    4. Inter-block gaps: CPU scheduling time between consecutive blocks
    5. Framework overhead: scheduler decisions, memory management, request handling
    6. NCCL synchronization: allreduce timing variance at full-model level
    7. Any other work between consecutive token outputs

- Write a Python extraction script that:
    1. Reads the NSYS SQLite trace for each framework
    2. Identifies decode iteration boundaries (e.g., by finding repeating kernel patterns)
    3. Measures the full decode step duration (wall time from first kernel to first kernel of next step)
    4. Breaks down the time into: total block time, pre-forward, post-forward, inter-block gaps, other
    5. Compares both frameworks side by side

- Produce a FULL DECODE STEP COMPARISON TABLE:
    | Component           | Framework A (ns) | Framework B (ns) | Delta (ns) | Delta (%) |
    |---------------------|------------------|------------------|------------|-----------|
    | All blocks (sum)    | ...              | ...              | ...        | ...       |
    | Pre-forward         | ...              | ...              | ...        | ...       |
    | Post-forward        | ...              | ...              | ...        | ...       |
    | Inter-block gaps    | ...              | ...              | ...        | ...       |
    | Other overhead      | ...              | ...              | ...        | ...       |
    | TOTAL (decode step) | ...              | ...              | ...        | ...       |

- Identify which components contribute most to the TPOT gap. If the per-block analysis shows
  framework A is faster per-block but framework B has lower TPOT, the gap must be in the non-block
  components — identify exactly which ones and by how much.
- Include this full decode step comparison in the output improvement plan as a separate section.
"""

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            original_improvement_plan_file=self.original_improvement_plan_file,
            original_compare_file=self.original_compare_file,
            knowledge_base_file=self.knowledge_base_file,
            code_gen_dirs="\n".join(self.code_gen_dirs),
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            output_context_file=self.output_context_file,
            full_decode_step_section=full_decode_section,
            analysis_description=desc,
            target_idx=self.config.target_trace_id,
            target_label=label,
            iteration=self.iteration,
        )


def gen_redo_cross_trace_steps(
    config: CrossTraceConfig,
    code_gen_dirs: list[str],
    iteration: int = 1,
    full_decode_step_analysis: bool = False,
):
    """Generate steps for a redo cross-trace analysis.

    Returns (steps, output_files).
    """
    from common.iteration_utils import resolve_output_file
    from llm.auto_code_gen.code_gen_prompts import KNOWLEDGE_BASE_FILE

    output_dir = os.path.abspath(config.output_dir)

    original_plan = resolve_output_file(output_dir, CROSS_IMPROVEMENT_FILE)
    original_compare = resolve_output_file(output_dir, CROSS_COMPARE_FILE)

    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        KNOWLEDGE_BASE_FILE,
    )

    output_file = os.path.join(
        output_dir, "{}_V{}.txt".format(REDO_IMPROVEMENT_PREFIX, iteration))
    output_summary = os.path.join(
        output_dir, "{}_summary_V{}.txt".format(REDO_IMPROVEMENT_PREFIX, iteration))
    output_context = os.path.join(
        output_dir, "{}_V{}.txt".format(REDO_CONTEXT_PREFIX, iteration))

    prompt = RedoCrossTraceAnalysisPrompt(
        config=config,
        code_gen_dirs=[os.path.abspath(d) for d in code_gen_dirs],
        knowledge_base_file=kb_path if os.path.isfile(kb_path) else "",
        original_improvement_plan_file=original_plan,
        original_compare_file=original_compare,
        output_file=output_file,
        output_summary_file=output_summary,
        output_context_file=output_context,
        iteration=iteration,
        full_decode_step_analysis=full_decode_step_analysis,
    )

    steps = [
        PipelineStep(
            name="redo_improvement_plan_V{}".format(iteration),
            prompt=prompt.prompt(),
            output_files=[output_file, output_summary, output_context],
        ),
    ]

    return steps, [output_file, output_summary, output_context]
