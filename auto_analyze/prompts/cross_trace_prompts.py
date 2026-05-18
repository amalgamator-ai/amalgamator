import os
from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.configs.cross_trace_config import (
    CrossTraceConfig,
    TraceResult,
    CROSS_MATCHING_FILE,
    CROSS_COMPARE_FILE,
    CROSS_IMPROVEMENT_FILE,
)


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


@dataclass
class MatchMedianTransformerBlocksPrompt:
    config: CrossTraceConfig
    output_file: str
    prompt_template: ClassVar[str] = """
{traces_context}
[output_file] = {output_file}

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

Produce a structured output with:
1. A matched operations table showing, for each logical operation, the corresponding high_level_op name, low-level kernel name, duration, and stream for each block — side by side.
2. Operations that exist in one trace but not the other (added/removed/fused differently).
3. Overall wall time for each trace.

Dump results to [output_file].
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
            output_file=self.output_file,
            analysis_type=analysis_type,
            analysis_description=desc,
        )


@dataclass
class CompareMedianTransformerBlocksPrompt:
    config: CrossTraceConfig
    matching_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
{traces_context}
[matching_file] = {matching_file}
[output_file] = {output_file}
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
            matching_file=self.matching_file,
            output_file=self.output_file,
            analysis_description=desc,
            target_idx=self.config.target_trace_id,
            target_label=label,
            target_framework=target.framework_name,
        )


@dataclass
class ImprovementPlanPrompt:
    config: CrossTraceConfig
    compare_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
{traces_context}
[compare_file] = {compare_file}
[output_file] = {output_file}
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

- Generate a ranked improvement plan for the target trace that fully recovers its performance as follows:
    - Executive summary with gaps, key findings, key ideas, and ranked plan summary (that also has impact per each improvement in % of total block time).
    - A detailed ranked sequence of improvement proposals (from most impactful to less), where each proposal has:
        - A summary of root causes, key findings, key ideas, and estimated improvement/impact in % of total block time.
        - A step-by-step guide that describes how to port code from the other framework with minimal changes to the target framework, in order to fix the performance difference fully. For each step provide:
            - An explanation of the step
            - A code snippet example that shows what needs to be changed. Ensure the code snippet is properly formatted and detailed enough.
    - Make sure the plan is clear, concise and professional so that it can be executed on by a professional LLM programmer.

Dump the improvement plan to [output_file].
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
            compare_file=self.compare_file,
            output_file=self.output_file,
            analysis_description=desc,
            target_idx=self.config.target_trace_id,
            target_label=label,
            target_framework=target.framework_name,
        )


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
