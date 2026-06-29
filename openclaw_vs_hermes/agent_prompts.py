from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# File prefixes and constants
# ---------------------------------------------------------------------------

FEATURE_DETECTION_PREFIX = "feature_detection"
FEATURE_HERMES_FILE = "features_hermes.txt"
FEATURE_OPENCLAW_FILE = "features_openclaw.txt"
DETECTED_FEATURES_FILE = "detected_features.txt"
DETECTED_FEATURES_JSON = "detected_features.json"

CODE_TRACE_FILE = "code_trace.txt"
CODE_PORT_PLAN_FILE_PREFIX = "code_port_plan"
TEST_PLAN_PREFIX = "test_plan"
CODE_GEN_FILE_PREFIX = "code_gen"
RUNTIME_FILE_PREFIX = "code_gen_runtime"
RUNTIME_SUCCESS_FILE = "runtime_success_result.txt"
RUNTIME_LOGS_DIR = "runtime_logs"


def code_trace_filename(project: str) -> str:
    return "{}_{}".format(project, CODE_TRACE_FILE)


# ---------------------------------------------------------------------------
# Helper: previous iteration section
# ---------------------------------------------------------------------------

def _prev_iteration_section(prev_output_file, prev_output_summary_file):
    if prev_output_file is None:
        return ""
    return """
<prev_output_file>
{prev_output_file}
</prev_output_file>
<prev_output_summary_file>
{prev_output_summary_file}
</prev_output_summary_file>

Read and analyze in-detail <prev_output_file> and <prev_output_summary_file> which contain the results from the previous iterations. Understand what was done, what issues were found and fixed, and what the iteration evolution looks like. The current attempt is done from scratch, but must incorporate all learnings from the previous iterations to avoid repeating mistakes and to build on what worked.
""".format(prev_output_file=prev_output_file, prev_output_summary_file=prev_output_summary_file)


# ---------------------------------------------------------------------------
# Context string for all prompts
# ---------------------------------------------------------------------------

def create_context_str(config):
    return """
<context>

<output_dir>
{output_dir}
</output_dir>

<source_project>hermes</source_project>
<source_project_full_name>Hermes Agent (NousResearch)</source_project_full_name>
<source_project_language>Python</source_project_language>
<source_project_dir>
{hermes_source_dir}
</source_project_dir>
<source_project_github>
{hermes_github_url}
</source_project_github>

<target_project>openclaw</target_project>
<target_project_full_name>OpenClaw</target_project_full_name>
<target_project_language>TypeScript</target_project_language>
<target_project_dir>
{openclaw_source_dir}
</target_project_dir>
<target_project_github>
{openclaw_github_url}
</target_project_github>

</context>

<definitions>
<code_trace>
code-paths, code-pieces, and their associated call-chains, including all relevant classes, functions, modules, imports, and data flow
</code_trace>
<feature>
A distinct, self-contained capability or subsystem that can be identified, analyzed, and potentially ported as a unit
</feature>
</definitions>

<context_explanations>
- <output_dir> is the output directory where all generated artifacts must be saved. ALL output files MUST be written to this directory.
- <source_project> is the project from which features are ported (Hermes Agent, written in Python)
- <target_project> is the project to which features are ported (OpenClaw, written in TypeScript)
- <source_project_dir> is the local path to the Hermes Agent source code
- <target_project_dir> is the local path to the OpenClaw source code
- <source_project_github> is the GitHub URL for the Hermes Agent repository
- <target_project_github> is the GitHub URL for the OpenClaw repository
- Porting involves translating Python code to TypeScript while adapting to OpenClaw's architecture and conventions
</context_explanations>

""".format(
        output_dir=config.output_dir,
        hermes_source_dir=config.hermes_source_dir,
        openclaw_source_dir=config.openclaw_source_dir,
        hermes_github_url=config.hermes_github_url,
        openclaw_github_url=config.openclaw_github_url,
    )


# ===================================================================
# PHASE 0: FEATURE DETECTION PROMPTS
# ===================================================================

ANALYZE_HERMES_FEATURES_TEMPLATE = """

{context}

<instructions>
The goal of this task is to perform an exhaustive analysis of the Hermes Agent project to identify ALL major features and capabilities it provides. Think hard and be very thorough.

Do the following:
1. Explore the Hermes Agent source code at <source_project_dir> comprehensively:
   - Read the README.md, AGENTS.md, and any documentation files
   - Examine the top-level directory structure
   - Analyze each major module/directory to understand its purpose
   - Look at key source files in: agent/, tools/, gateway/, skills/, plugins/, cron/, hermes_cli/

2. For each identified feature/capability, document:
   - Feature name (clear, concise identifier)
   - Category (e.g., memory, communication, automation, UI, security, etc.)
   - Description (2-3 sentences explaining what the feature does)
   - Key implementation files (list the primary source files)
   - Architecture notes (how the feature is structured, key classes/functions)
   - Dependencies (what other features or external systems it depends on)
   - Complexity estimate (LOW / MEDIUM / HIGH / VERY HIGH) for porting to another project
   - Language and frameworks used

3. Be especially thorough about these capability areas:
   - Memory and persistence systems
   - Conversation handling (compression, caching, context management)
   - Multi-platform messaging/gateway
   - Error handling and fallback chains
   - Search and retrieval capabilities
   - Skill/plugin lifecycle management
   - Multi-agent orchestration (kanban, delegation, subagents)
   - Scheduling and automation (cron)
   - Security and approval systems
   - UI theming and customization
   - Batch processing and trajectory generation

4. For each feature, note whether it is:
   - Core (fundamental to the agent's operation)
   - Extension (adds capability but not required)
   - Integration (connects to external systems)
</instructions>

<output>
- Dump the comprehensive feature analysis to <output_dir>/{output_file}
- Structure the output as a numbered list of features, each with the fields described above
- Be exhaustive -- do not miss any significant feature
</output>

"""


ANALYZE_OPENCLAW_FEATURES_TEMPLATE = """

{context}

<instructions>
The goal of this task is to perform an exhaustive analysis of the OpenClaw project to identify ALL major features and capabilities it currently provides. Think hard and be very thorough.

Do the following:
1. Explore the OpenClaw source code at <target_project_dir> comprehensively:
   - Read the AGENTS.md, VISION.md, CHANGELOG.md (recent entries), and documentation files in docs/
   - Examine the top-level directory structure
   - Analyze each major module/directory under src/ to understand its purpose
   - Look at extensions/, skills/, packages/, ui/, apps/

2. For each identified feature/capability, document:
   - Feature name (clear, concise identifier)
   - Category (e.g., memory, communication, automation, UI, security, etc.)
   - Description (2-3 sentences explaining what the feature does)
   - Key implementation files (list the primary source files or directories)
   - Maturity level (MATURE / DEVELOPING / BASIC / PLANNED)
   - Known limitations or issues (if apparent from code comments, TODOs, or documentation)

3. Be especially thorough about these capability areas (matching the Hermes analysis):
   - Memory and persistence systems
   - Conversation handling (compression, compaction, context management)
   - Multi-platform messaging/channels
   - Error handling and fallback chains
   - Search and retrieval capabilities
   - Skill/plugin lifecycle management
   - Multi-agent orchestration
   - Scheduling and automation (cron)
   - Security and approval systems
   - UI theming and customization
   - Batch processing

4. For each feature, note its current quality:
   - Working well
   - Has known issues
   - Incomplete/basic implementation
   - Missing entirely
</instructions>

<output>
- Dump the comprehensive feature analysis to <output_dir>/{output_file}
- Structure the output as a numbered list of features, each with the fields described above
- Be exhaustive -- do not miss any significant feature
</output>

"""


DETECT_PORTABILITY_CANDIDATES_TEMPLATE = """

{context}

<definitions>
<hermes_features_file>
{hermes_features_file}
</hermes_features_file>
<openclaw_features_file>
{openclaw_features_file}
</openclaw_features_file>
</definitions>

<instructions>
The goal of this task is to identify features that exist in Hermes Agent but are MISSING or INADEQUATE in OpenClaw, AND that have demonstrated community demand. This is the critical step that determines what is worth porting. Think very hard and be precise.

Do the following:

1. Read and fully understand the Hermes feature analysis in <hermes_features_file>
2. Read and fully understand the OpenClaw feature analysis in <openclaw_features_file>

3. For EACH Hermes feature, determine its status in OpenClaw:
   - MISSING: Feature does not exist at all in OpenClaw
   - INADEQUATE: Feature exists but has significant gaps, bugs, or limitations compared to Hermes
   - EQUIVALENT: Feature exists and is roughly equivalent
   - SUPERIOR: OpenClaw's implementation is better than Hermes's

4. For features that are MISSING or INADEQUATE in OpenClaw, research community demand:
   - Use the `gh` CLI tool to search OpenClaw GitHub issues:
     Run: gh api "search/issues?q=repo:openclaw/openclaw+is:issue+<relevant_search_terms>&per_page=20" --jq '.items[] | {{number, title, state, reactions: .reactions.total_count, comments, created_at}}'
   - Search for feature requests, enhancement requests, bug reports related to the feature
   - Look for issues with high reaction counts or many comments (indicating community demand)
   - Also search Hermes issues to verify the feature works well there:
     Run: gh api "search/issues?q=repo:nousresearch/hermes-agent+<relevant_search_terms>&per_page=10" --jq '.items[] | {{number, title, state}}'

5. Score each portability candidate on these dimensions (1-10):
   - Community demand: How many people want this? How strongly? (based on issue counts, reactions, comments)
   - Gap severity: How much worse is OpenClaw's current solution? (MISSING=10, INADEQUATE=5-8, etc.)
   - Porting feasibility: How practical is it to port from Python to TypeScript? Consider architectural differences.
   - Impact: How much would this improve OpenClaw for its users?
   - Independence: Can this feature be ported independently, or does it require other features first?

6. Rank the candidates by overall priority score (weighted average: demand=30%, gap=25%, impact=25%, feasibility=15%, independence=5%)

7. For each top candidate, provide:
   - Feature name
   - Hermes implementation summary (key files, architecture)
   - OpenClaw gap description (what's missing/broken, with specific issue references)
   - Community evidence (list of GitHub issues with numbers, titles, reaction counts)
   - Porting approach (high-level strategy for adapting Python to TypeScript)
   - Risk assessment
   - Estimated complexity (LOW / MEDIUM / HIGH / VERY HIGH)
   - Score breakdown (each dimension + overall)

IMPORTANT RULES:
- Only include features where you found CONCRETE evidence of community demand (actual GitHub issues, not speculation)
- Only include features where there is a CLEAR gap between Hermes and OpenClaw
- Do NOT include features that OpenClaw already handles well
- Be honest about porting complexity -- Python to TypeScript is non-trivial
</instructions>

<output>
- Dump the full analysis to <output_dir>/{report_file}
- Also dump a structured JSON summary to <output_dir>/{json_file} with the following format:
  {{
    "detected_features": [
      {{
        "rank": 1,
        "feature_name": "...",
        "category": "...",
        "hermes_status": "IMPLEMENTED",
        "openclaw_status": "MISSING" or "INADEQUATE",
        "community_demand_score": 8,
        "gap_severity_score": 9,
        "impact_score": 8,
        "feasibility_score": 7,
        "independence_score": 9,
        "overall_score": 8.2,
        "key_hermes_files": ["..."],
        "openclaw_issues": [{{  "number": 123, "title": "...", "reactions": 5 }}],
        "porting_complexity": "HIGH",
        "summary": "..."
      }}
    ],
    "total_candidates_analyzed": 25,
    "total_candidates_with_demand": 8
  }}
- Limit the detected_features list to the top {max_features} candidates
</output>

"""


# ===================================================================
# PHASE 1: CODE TRACE PROMPTS
# ===================================================================

CODE_TRACE_TEMPLATE = """

{context}

<definitions>
<project>
{project}
</project>
<project_dir>
{project_dir}
</project_dir>
<feature_to_trace>
{feature_description}
</feature_to_trace>
</definitions>

<instructions>
The goal of this task is to detect the <code_trace> inside <project> that implements <feature_to_trace>. Think hard for this task and follow these guidelines:

- Analyze and understand in-detail the feature described in <feature_to_trace>
- Detect the <code_trace> inside <project> (located at <project_dir>) that implements this feature:
    - Find all of the trace from higher levels to lower levels of the code
    - Go deep into all implementation details including utility functions, configuration, data models, and storage
    - Include all relevant files: source code, tests, configuration, types/interfaces
    - Do not miss anything critical to understanding or porting this feature

- For the detected <code_trace>, perform a detailed analysis:
    - Document the entry points (how the feature is triggered/activated)
    - Document the data flow (what data enters, how it's transformed, where it goes)
    - Document the state management (what persistent state is maintained, where/how)
    - Document external dependencies (libraries, APIs, services)
    - Document configuration options and defaults
    - Document error handling patterns

- Document each major component with:
    - Its purpose and responsibility
    - Key classes/functions with their signatures, inputs, and outputs
    - How it connects to other components
    - Any important invariants or assumptions

- Be professional, clear, and concise while documenting the trace gradually, in an incremental way from entry points through to implementation details.
</instructions>

<output>
- Dump results to <output_dir>/{output_file}
</output>

"""


# ===================================================================
# PHASE 2: CODE PORT PLAN PROMPTS
# ===================================================================

CODE_PORT_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for Hermes (source) and OpenClaw (target) respectively. Each code trace file describes the <code_trace> of the specific project relevant to the feature being ported.
</definition_explanations>

<instructions>
The goal of this task is to provide a high-level multi-step coding plan that ports the feature described in <feature_description> from Hermes Agent (Python) to OpenClaw (TypeScript). Think hard for this task and follow these guidelines:

- Analyze and understand in-detail the <code_trace> of Hermes from <code_trace_files>.
- Analyze and understand in-detail the <code_trace> of OpenClaw from <code_trace_files> (the corresponding area where this feature should be integrated).
- Compare in-detail how Hermes implements this feature vs what OpenClaw currently has (or lacks).

- Design a porting plan that:
    - Translates Python code to TypeScript idiomatically (use TypeScript patterns, not direct Python translation)
    - Follows OpenClaw's existing code conventions, module structure, and architectural patterns
    - Integrates naturally with OpenClaw's existing systems (config, plugins, gateway, agent core)
    - Maximizes reuse of existing OpenClaw infrastructure where possible
    - Preserves the core logic and behavior of the Hermes implementation
    - Handles the language differences properly:
        - Python classes -> TypeScript classes/interfaces
        - Python asyncio -> TypeScript async/await with Node.js patterns
        - Python dataclasses -> TypeScript interfaces/types
        - Python imports -> TypeScript ESM imports
        - Python SQLite3 -> TypeScript better-sqlite3 or similar
        - Python dict -> TypeScript Map/Record/object as appropriate

- For each step in the plan:
    - Describe what code to create or modify
    - Specify the target file path within OpenClaw's directory structure
    - Describe the TypeScript implementation approach
    - Note any OpenClaw APIs or patterns to reuse
    - Note any new dependencies needed

- Ensure the plan:
    - Can be implemented incrementally (each step builds on the previous)
    - Has clear integration points with existing OpenClaw code
    - Handles configuration (add to openclaw.json schema if needed)
    - Includes proper TypeScript types and interfaces
    - Follows OpenClaw's error handling patterns

- Provide risk analysis and potential breaking points
</instructions>

<output>
- Dump the high-level multi-step coding plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the work done to generate the coding plan in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}, describing the progression of fixes and improvements
</output>

"""


REVIEW_CODE_PORT_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for Hermes (source) and OpenClaw (target) respectively.
- <input_file> is a file that describes the high-level multi-step coding plan for porting the feature from Hermes to OpenClaw.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the porting plan in <input_file> (iteration {iteration}). Do the following and think hard:

- Understand in-detail the plan in <input_file>, and restate the plan step-by-step.
- Review the plan for:
    - Correctness of Python-to-TypeScript translation approach
    - Proper use of OpenClaw's existing APIs and patterns (check the actual source code to verify)
    - Proper TypeScript typing and interfaces
    - Integration completeness (does it hook into all necessary OpenClaw systems?)
    - Configuration handling (is it added to the config schema properly?)
    - Error handling patterns (matches OpenClaw's approach?)
    - Missing steps or gaps in the plan
    - Incorrect assumptions about either codebase
    - Dependencies that might not exist or might need to be added

- Review for architectural issues:
    - Does the plan follow OpenClaw's module organization?
    - Are new files placed in the correct directories?
    - Does it respect OpenClaw's plugin/extension architecture?
    - Are there circular dependency risks?

- For each issue found, document:
    - The affected part of the plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected plan with all issues fixed.
</instructions>

<output>
- Dump the corrected plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed to the code port plan, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""


# ===================================================================
# PHASE 3: TEST PLAN PROMPTS
# ===================================================================

TEST_PLAN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<definition_explanations>
- <code_trace_files> is a list of code trace files for Hermes (source) and OpenClaw (target) respectively.
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan for porting the feature.
</definition_explanations>

<instructions>
The goal of this task is to generate a comprehensive testing plan for the ported feature described in <code_port_plan_file> (iteration {iteration}). Do the following and think hard:

- Analyze and understand in-detail the coding plan in <code_port_plan_file>
- Analyze how Hermes tests this feature (look at the test files in the Hermes repo at <source_project_dir>)
- Analyze how OpenClaw structures its tests (look at existing tests in <target_project_dir>)

- Plan tests following OpenClaw's testing conventions:
    - Unit tests (*.test.ts colocated with source files)
    - Integration tests if needed
    - E2E tests (*.e2e.test.ts) if the feature has user-facing behavior

- For each test:
    - Describe what it verifies
    - Specify the test file path
    - Describe key test cases (happy path, edge cases, error cases)
    - Note any test fixtures or mocks needed

- Ensure test coverage for:
    - Core feature functionality
    - Configuration handling
    - Error cases and fallbacks
    - Integration with existing OpenClaw systems
    - Edge cases identified in the code trace
    - Regression scenarios

- Plan tests that can actually run in OpenClaw's test infrastructure (using vitest or the project's test runner)
</instructions>

<output>
- Dump the planned tests to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the test plan generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
</output>

"""


REVIEW_TEST_PLAN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<input_file>
{input_file}
</input_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<instructions>
The goal of this task is to perform a critical, in-depth review of the testing plan in <input_file> (iteration {iteration}). Do the following and think hard:

- Understand in-detail the test plan in <input_file>, and restate the plan step-by-step.
- Verify the test plan:
    - Covers all components from the coding plan
    - Uses OpenClaw's testing patterns correctly (check existing tests for conventions)
    - Has proper unit tests for each new module/function
    - Has integration tests for interaction with existing OpenClaw systems
    - Covers error handling and edge cases
    - Tests are actually implementable (correct APIs, proper mocking)

- Review for issues:
    - Missing test coverage
    - Incorrect test assumptions
    - Tests that would not actually verify correctness
    - Tests that would not detect regressions
    - Missing test fixtures or mocks

- For each issue found, document and fix.
- Produce a corrected test plan.
</instructions>

<output>
- Dump the corrected test plan to <output_dir>/{output_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""


# ===================================================================
# PHASE 4: CODE GEN PROMPTS
# ===================================================================

CODE_GEN_TEMPLATE = """

{context}

{prev_iteration_section}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<instructions>
The goal of this task is to generate the actual TypeScript code that ports the feature from Hermes to OpenClaw, implementing the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file> (iteration {iteration}). Do the following and think hard:

- Analyze and understand in-detail the coding plan in <code_port_plan_file>
- Analyze and understand in-detail the testing plan in <test_plan_file>
- Implement the plans EXACTLY as described:
    - Follow strictly the code and test plans, DO NOT DIVERGE
    - Write idiomatic TypeScript (not a literal Python translation)
    - Follow OpenClaw's coding conventions (check existing files for patterns)
    - Use proper TypeScript types, interfaces, and generics
    - Use OpenClaw's existing utilities and patterns where specified in the plan

- For each file to create or modify:
    - Write the complete file content
    - Include proper imports
    - Include proper TypeScript types
    - Add exports as needed

- IMPORTANT: ALL code modifications MUST be made exclusively inside <target_project_dir>. Do NOT modify any files outside of this directory.

- After writing the code:
    - Run the build command to verify compilation: cd <target_project_dir> && {build_command}
    - Fix any TypeScript compilation errors
    - Run the tests: cd <target_project_dir> && {test_command}
    - Fix any test failures

- Create a unified diff/patch of all changes made to <target_project_dir>:
    - Run: cd <target_project_dir> && git diff HEAD > <output_dir>/{{output_patch_file}}

- Ensure the code is written professionally, clearly, well-typed, and follows OpenClaw conventions.
- Review your work critically and fix issues.
</instructions>

<output>
- Dump the code patch to <output_dir>/{output_patch_file} (it must have both code and its tests inside)
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - An explanation of the code patch and tests generated in this iteration
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
    - List of all files created or modified
    - Build and test results
</output>

"""


REVIEW_CODE_GEN_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<input_patch_file>
{input_patch_file}
</input_patch_file>
<input_summary_file>
{input_summary_file}
</input_summary_file>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<instructions>
The goal of this task is to perform a critical, in-depth review of the code patch in <input_patch_file> (iteration {iteration}). Do the following and think hard:

- Understand in-detail the code in <input_patch_file>, and restate the implementation step-by-step.
- Review the code for:
    - Correctness of the TypeScript implementation vs the Hermes Python original
    - Proper TypeScript typing (no `any` types unless truly necessary)
    - Proper use of OpenClaw's APIs and patterns
    - Proper error handling
    - Proper async/await usage
    - Memory leaks or resource management issues
    - Security issues (injection, XSS, etc.)
    - Test quality and coverage

- Review for architectural issues:
    - Proper module organization
    - Clean interfaces and abstractions
    - No circular dependencies
    - Proper configuration integration

- For each issue found, document:
    - The affected code
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected code patch with all issues fixed.
- Re-run build and tests after fixes.
- IMPORTANT: ALL code modifications MUST be made exclusively inside <target_project_dir>. Do NOT modify files outside of this directory.
</instructions>

<output>
- Dump the corrected code patch to <output_dir>/{output_patch_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - Documentation of the issues found and fixed in this review
    - A summary of the iteration evolution across all iterations up to and including iteration {iteration}
- IMPORTANT: If no changes were needed, write CONVERGED as the very first line of <output_dir>/{output_summary_file}. Otherwise, do NOT write CONVERGED.
</output>

"""


# ===================================================================
# PHASE 5: RUNTIME PROMPTS
# ===================================================================

APPLY_AND_BUILD_TEMPLATE = """

{context}

<definitions>
<patch_file>
{patch_file}
</patch_file>
</definitions>

<instructions>
Apply the code patch in <patch_file> to the OpenClaw project and build it.

Steps:
1. Read the patch from <output_dir>/<patch_file>
2. Apply it to <target_project_dir>:
   cd <target_project_dir> && git apply <output_dir>/<patch_file>
   (If git apply fails, try applying manually by reading the patch and writing files)
3. Build the project:
   cd <target_project_dir> && {build_command}
4. Report whether the build succeeded or failed.
   - If it succeeded, write "SUCCESS" to <output_dir>/{status_file}
   - If it failed, write "FAILED" followed by the error output to <output_dir>/{status_file}
</instructions>

<output>
- Write build status to <output_dir>/{status_file}
</output>

"""


RUN_AND_TEST_TEMPLATE = """

{context}

<instructions>
Run the OpenClaw test suite to verify the ported feature works correctly.

Steps:
1. Run the tests:
   cd <target_project_dir> && {test_command}
2. Capture and analyze the test output
3. Report the results:
   - Write test results to <output_dir>/{results_file}
   - If all tests pass, also write "COMPLETED" to <output_dir>/{status_file}
   - If tests fail, write the failure details to <output_dir>/{results_file}
</instructions>

<output>
- Write test results to <output_dir>/{results_file}
- Write status to <output_dir>/{status_file}
</output>

"""


INVESTIGATE_AND_FIX_TEMPLATE = """

{context}

<definitions>
<code_trace_files>
{code_trace_files}
</code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
<test_plan_file>
{test_plan_file}
</test_plan_file>
<prev_patch_file>
{prev_patch_file}
</prev_patch_file>
<runtime_logs_dir>
{runtime_logs_dir}
</runtime_logs_dir>
<feature_description>
{feature_description}
</feature_description>
</definitions>

<instructions>
Investigate the build/test failures from runtime iteration {iteration} and fix the code.

Steps:
1. Read the build/test logs from <runtime_logs_dir>
2. Read the current patch from <output_dir>/<prev_patch_file>
3. Analyze the failures:
   - Identify the root cause of each failure
   - Determine whether it's a build error, test error, or runtime error
   - Check if the error is in the ported code or in test code

4. Fix the issues:
   - Apply the previous patch to <target_project_dir> if not already applied
   - Make the necessary code fixes
   - Re-run build: cd <target_project_dir> && {build_command}
   - Re-run tests: cd <target_project_dir> && {test_command}
   - Iterate until build and tests pass

5. Generate a new patch:
   cd <target_project_dir> && git diff HEAD > <output_dir>/{{output_patch_file}}

6. If all tests pass, create the success marker:
   Write results to <output_dir>/{success_file}

- IMPORTANT: ALL code modifications MUST be made exclusively inside <target_project_dir>.
</instructions>

<output>
- Dump the fixed code patch to <output_dir>/{output_patch_file}
- Dump a summary to <output_dir>/{output_summary_file} that includes:
    - What failures were found
    - What fixes were applied
    - Current build/test status
- If fully successful, write results to <output_dir>/{success_file}
</output>

"""


# ===================================================================
# PROMPT DATACLASSES AND GENERATORS
# ===================================================================

@dataclass
class AnalyzeHermesFeaturesPrompt:
    context: str
    output_file: str

    def prompt(self):
        return ANALYZE_HERMES_FEATURES_TEMPLATE.format(
            context=self.context,
            output_file=self.output_file,
        )


@dataclass
class AnalyzeOpenClawFeaturesPrompt:
    context: str
    output_file: str

    def prompt(self):
        return ANALYZE_OPENCLAW_FEATURES_TEMPLATE.format(
            context=self.context,
            output_file=self.output_file,
        )


@dataclass
class DetectPortabilityCandidatesPrompt:
    context: str
    hermes_features_file: str
    openclaw_features_file: str
    report_file: str
    json_file: str
    max_features: int

    def prompt(self):
        return DETECT_PORTABILITY_CANDIDATES_TEMPLATE.format(
            context=self.context,
            hermes_features_file=self.hermes_features_file,
            openclaw_features_file=self.openclaw_features_file,
            report_file=self.report_file,
            json_file=self.json_file,
            max_features=self.max_features,
        )


@dataclass
class CodeTracePrompt:
    context: str
    project: str
    project_dir: str
    feature_description: str
    output_file: str

    def prompt(self):
        return CODE_TRACE_TEMPLATE.format(
            context=self.context,
            project=self.project,
            project_dir=self.project_dir,
            feature_description=self.feature_description,
            output_file=self.output_file,
        )


@dataclass
class CodePortPlanPrompt:
    context: str
    code_trace_files: list
    feature_description: str
    output_file: str
    output_summary_file: str
    prev_output_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int

    def prompt(self):
        return CODE_PORT_PLAN_TEMPLATE.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file,
            ),
            code_trace_files=self.code_trace_files,
            feature_description=self.feature_description,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class ReviewCodePortPlanPrompt:
    context: str
    code_trace_files: list
    feature_description: str
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int

    def prompt(self):
        return REVIEW_CODE_PORT_PLAN_TEMPLATE.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            feature_description=self.feature_description,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class TestPlanPrompt:
    context: str
    code_trace_files: list
    code_port_plan_file: str
    feature_description: str
    output_file: str
    output_summary_file: str
    prev_output_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int

    def prompt(self):
        return TEST_PLAN_TEMPLATE.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_file, self.prev_output_summary_file,
            ),
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            feature_description=self.feature_description,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class ReviewTestPlanPrompt:
    context: str
    code_trace_files: list
    code_port_plan_file: str
    feature_description: str
    input_file: str
    input_summary_file: str
    output_file: str
    output_summary_file: str
    iteration: int

    def prompt(self):
        return REVIEW_TEST_PLAN_TEMPLATE.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            feature_description=self.feature_description,
            input_file=self.input_file,
            input_summary_file=self.input_summary_file,
            output_file=self.output_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class CodeGenPrompt:
    context: str
    code_trace_files: list
    code_port_plan_file: str
    test_plan_file: str
    feature_description: str
    build_command: str
    test_command: str
    output_patch_file: str
    output_summary_file: str
    prev_output_patch_file: Optional[str]
    prev_output_summary_file: Optional[str]
    iteration: int

    def prompt(self):
        return CODE_GEN_TEMPLATE.format(
            context=self.context,
            prev_iteration_section=_prev_iteration_section(
                self.prev_output_patch_file, self.prev_output_summary_file,
            ),
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            feature_description=self.feature_description,
            build_command=self.build_command,
            test_command=self.test_command,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class ReviewCodeGenPrompt:
    context: str
    code_trace_files: list
    code_port_plan_file: str
    test_plan_file: str
    feature_description: str
    input_patch_file: str
    input_summary_file: str
    output_patch_file: str
    output_summary_file: str
    iteration: int

    def prompt(self):
        return REVIEW_CODE_GEN_TEMPLATE.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            feature_description=self.feature_description,
            input_patch_file=self.input_patch_file,
            input_summary_file=self.input_summary_file,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            iteration=self.iteration,
        )


@dataclass
class ApplyAndBuildPrompt:
    context: str
    patch_file: str
    build_command: str
    status_file: str
    iteration: int

    def prompt(self):
        return APPLY_AND_BUILD_TEMPLATE.format(
            context=self.context,
            patch_file=self.patch_file,
            build_command=self.build_command,
            status_file=self.status_file,
        )


@dataclass
class RunAndTestPrompt:
    context: str
    test_command: str
    results_file: str
    status_file: str
    iteration: int

    def prompt(self):
        return RUN_AND_TEST_TEMPLATE.format(
            context=self.context,
            test_command=self.test_command,
            results_file=self.results_file,
            status_file=self.status_file,
        )


@dataclass
class InvestigateAndFixPrompt:
    context: str
    code_trace_files: list
    code_port_plan_file: str
    test_plan_file: str
    feature_description: str
    prev_patch_file: str
    runtime_logs_dir: str
    build_command: str
    test_command: str
    output_patch_file: str
    output_summary_file: str
    success_file: str
    iteration: int

    def prompt(self):
        return INVESTIGATE_AND_FIX_TEMPLATE.format(
            context=self.context,
            code_trace_files=self.code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            test_plan_file=self.test_plan_file,
            feature_description=self.feature_description,
            prev_patch_file=self.prev_patch_file,
            runtime_logs_dir=self.runtime_logs_dir,
            build_command=self.build_command,
            test_command=self.test_command,
            output_patch_file=self.output_patch_file,
            output_summary_file=self.output_summary_file,
            success_file=self.success_file,
            iteration=self.iteration,
        )
