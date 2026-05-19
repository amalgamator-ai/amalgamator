import os
from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.configs.single_trace_config import PERF_ANALYSIS_SINGLE_FILE
from auto_analyze.configs.cross_trace_config import CROSS_COMPARE_FILE

SUMMARY_PDF_FILE = "summary_report.pdf"


@dataclass
class SummaryPDFPrompt:
    """Generate a PDF summary report from analysis results.

    Supports two modes:
      (A) Single-trace: results from one framework's trace analysis
      (B) Cross-trace: results from comparing multiple frameworks/commits
    """

    results_dir: str
    mode: str  # "single" or "cross"
    output_file: str

    # The visual design and document structure template — kept unchanged
    prompt_template: ClassVar[str] = """
[results_dir] = {results_dir}
[run_params_file] = {run_params_file}
[output_file] = {output_file}

{mode_context}

Read [run_params_file] to understand the analysis that was performed.
Read ALL result files in [results_dir] to gather the data for the report.

=== VISUAL DESIGN ===

Color palette (use these exact hex values):
  primary:       #1B2A4A  (dark navy — titles, headers)
  secondary:     #2E86AB  (steel blue — section accents, rule lines)
  accent:        #E8630A  (burnt orange — impact badges)
  success:       #1A7F37  (forest green — positive deltas)
  danger:        #CF222E  (signal red — regressions)
  bg_light:      #F6F8FA  (ghost white — backgrounds)
  bg_alt_row:    #EEF2F7  (pale blue-gray — alternating table rows)
  border:        #D0D7DE  (silver — gridlines)
  text_primary:  #1F2328  (near-black — body text)
  text_secondary:#656D76  (slate gray — captions)

Page: LETTER portrait, margins 0.75in left/right, 0.6in top/bottom.

Typography:
  - Title: Helvetica-Bold 22pt, primary color.
  - Section headers: Helvetica-Bold 15pt, primary color, with a secondary-colored
    rule line underneath.
  - Subsection headers: Helvetica-Bold 12pt, secondary color.
  - Body: Helvetica 10pt, text_primary, 14pt leading.
  - Captions: Helvetica-Oblique 8.5pt, text_secondary, centered.

Tables:
  - Header row: primary background, white bold text, 9pt.
  - Body: Helvetica 9pt, alternating row backgrounds (white / bg_alt_row).
  - Grid: 0.5pt border color. Cell padding: 5pt vertical, 6pt horizontal.
  - Performance deltas: green for improvements, red for regressions.
  - Appendix tables: use smaller font (7.5pt) if needed to fit columns.
  - Bold rows where timing delta exceeds 10%.
  - Always add spacing before and after every table to prevent overlap.

Code blocks — LIGHT THEME:
  - Background: #F6F8FA (ghost white). Border: 1pt #D0D7DE (silver), rounded feel.
  - Default text: Courier 8.5pt, color #1F2328 (near-black).
  - Use pygments tokenization with XPreformatted (NOT Preformatted) for syntax coloring.
  - Token colors (light-theme friendly, high contrast on white):
      Keywords:          #CF222E  (red)
      Strings/chars:     #0A3069  (deep blue)
      Comments:          #656D76  (gray, italic)
      Functions/builtins:#8250DF  (purple)
      Numbers:           #0550AE  (blue)
      Decorators:        #953800  (brown)
      Operators/punct:   #1F2328  (near-black)
      Class names/types: #953800  (brown)
      Default:           #1F2328  (inherits from style)
  - Wrap the XPreformatted in a Table cell for reliable background rendering.
  - Hard-wrap lines at 88 chars; use KeepTogether to prevent page splits.
  - Add spacing above and below each code block.
  - Inline code refs: Courier 9pt, #1F2328 text on #F6F8FA background.
  - CRITICAL: Escape XML entities in all text going into Paragraph/XPreformatted.

Impact badges (small colored table cell before issue title):
  - HIGH IMPACT: danger background, white text.
  - MEDIUM IMPACT: accent background, white text.
  - LOW IMPACT: secondary background, white text.

{document_structure}

=== GENERATION RULES ===

- One self-contained Python script, deterministic, no network calls.
- Use Paragraph for wrapping text; XPreformatted for code (never Preformatted).
- Test that the script runs without errors before finishing.
- Verify: no overlapping elements, no clipped tables, code blocks have multi-color
  syntax highlighting on the light background, appendix is complete.

Write and execute the Python script. Save the PDF to [output_file].
"""

    SINGLE_TRACE_CONTEXT: ClassVar[str] = f"""This is a SINGLE-TRACE analysis report.
[results_dir] contains the results of analyzing one framework's GPU trace, including:
  - transformer_block_high_level_ops.txt: high-level transformer block operations
  - gpu_ops_to_blocks.txt: GPU operations correlated to transformer blocks
  - median_block.txt: the median transformer block with operation details
  - {PERF_ANALYSIS_SINGLE_FILE}: detailed performance analysis with improvement proposals
  - transformer_block_trace.json: Perfetto trace visualization

The report should present the performance analysis findings and improvement proposals
for this single framework execution."""

    CROSS_TRACE_CONTEXT: ClassVar[str] = f"""This is a CROSS-TRACE comparison report.
[results_dir] contains the results of comparing multiple framework traces, including:
  - cross_matching_blocks.txt: operation-by-operation matching across traces
  - {CROSS_COMPARE_FILE}: detailed performance difference analysis

The report should present the cross-framework comparison findings, performance gaps,
and the improvement plan for the target framework."""

    SINGLE_TRACE_STRUCTURE: ClassVar[str] = f"""=== DOCUMENT STRUCTURE ===

1. COVER BLOCK
   - Title: "Performance Analysis Report"
   - Subtitle: model, GPU, execution parameters (from [run_params_file]). 
        - For each framework's commit_id provide also the concrete date it relates to.
        - For the "execution mode" (ISL/OSL and batch size) provide a comment that briefly describes its goal.
   - Date line: "Generated: [today's date]"
   - Horizontal rule separator.

2. EXECUTIVE SUMMARY (max 1 page)
   - Key findings from {PERF_ANALYSIS_SINGLE_FILE}: top bottlenecks, total estimated savings.
   - Summary table: [Proposal #, Title, Impact (ns), Impact (%), Difficulty].
   - Page break.

3. MEDIAN BLOCK OVERVIEW
   - Median block timing, operation count, variance.
   - Operations table with improvement proposals (from {PERF_ANALYSIS_SINGLE_FILE} section 4).

4. DETAILED IMPROVEMENT PROPOSALS (one sub-section per proposal, ranked)
   Per proposal:
   - Impact badge + title.
   - "Problem": 2-3 executive-friendly sentences.
   - "Root Cause": technical explanation with inline code refs.
   - "Implementation Guide": steps with code snippets.
   - Spacing between proposals.

5. APPENDIX: SOURCE CODE ANALYSIS
   - Per-operation source code analysis details from {PERF_ANALYSIS_SINGLE_FILE}."""

    CROSS_TRACE_STRUCTURE: ClassVar[str] = """=== DOCUMENT STRUCTURE ===

1. COVER BLOCK
   - Title: "Cross-Framework Performance Comparison Report"
   - Subtitle: model, GPU, execution parameters, frameworks compared (from [run_params_file]; make sure to include commit id date as well)
   - Date line: "Generated: [today's date]"
   - Horizontal rule separator.

2. EXECUTIVE SUMMARY (max 1 page)
   - What was compared, total performance gap, top 3 issues, estimated total gain.
   - Summary table: [Issue #, Title, Impact (ns), Difficulty, Priority].
   - Page break.

3. DETAILED IMPROVEMENT PLANS (one sub-section per issue, ranked)
   Per issue:
   - Impact badge + title.
   - "Problem": 2-3 executive-friendly sentences.
   - "Root Cause": technical explanation with inline code refs.
   - "Performance Impact" table: [Metric, Target Framework, Other, Delta].
   - "Implementation Guide": numbered steps, each with description,
     source file ref, syntax-highlighted code snippet, and brief explanation.
   - Spacing between issues.

4. APPENDIX A: OP-BY-OP MATCHING
   - Render the COMPLETE contents of the matching file as formatted tables.
   - Do NOT omit any operations.

5. APPENDIX B: SOURCE REFERENCES
   - All referenced source files, grouped by framework."""

    def prompt(self):
        if self.mode == "single":
            mode_context = self.SINGLE_TRACE_CONTEXT
            document_structure = self.SINGLE_TRACE_STRUCTURE
            run_params_file = os.path.join(self.results_dir, "run_params.txt")
        else:
            mode_context = self.CROSS_TRACE_CONTEXT
            document_structure = self.CROSS_TRACE_STRUCTURE
            run_params_file = os.path.join(self.results_dir, "run_params_cross.txt")

        return self.prompt_template.format(
            results_dir=self.results_dir,
            run_params_file=run_params_file,
            output_file=self.output_file,
            mode_context=mode_context,
            document_structure=document_structure,
        )
