from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# File prefixes and constants
# ---------------------------------------------------------------------------

FEATURE_DETECTION_PREFIX = "feature_detection"
FEATURE_KAGENTI_FILE = "features_kagenti.txt"
FEATURE_OPENSHELL_FILE = "features_openshell.txt"
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

<source_project>kagenti</source_project>
<source_project_full_name>Kagenti (Cloud-Native Agentic Platform)</source_project_full_name>
<source_project_languages>Python (51%), TypeScript (31%), Go (5%), Shell (10%)</source_project_languages>
<source_project_dir>
{kagenti_source_dir}
</source_project_dir>
<source_project_github>
{kagenti_github_url}
</source_project_github>

<target_project>openshell</target_project>
<target_project_full_name>OpenShell (NVIDIA)</target_project_full_name>
<target_project_languages>Rust (89%), Shell (6%), Python (4%)</target_project_languages>
<target_project_dir>
{openshell_source_dir}
</target_project_dir>
<target_project_github>
{openshell_github_url}
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
- <source_project> is the project from which features are ported (Kagenti, a cloud-native Kubernetes middleware for deploying and orchestrating AI agents)
- <target_project> is the project to which features are ported (OpenShell, NVIDIA's sandboxed execution environment for AI agents, written primarily in Rust)
- <source_project_dir> is the local path to the Kagenti source code
- <target_project_dir> is the local path to the OpenShell source code
- <source_project_github> is the GitHub URL for the Kagenti repository
- <target_project_github> is the GitHub URL for the OpenShell repository
- Porting involves translating Python/TypeScript/Go code to Rust while adapting to OpenShell's architecture and conventions
- Kagenti is a Kubernetes middleware platform for AI agent orchestration (deployment, security, networking, observability)
- OpenShell is a sandboxed execution environment for AI agents (isolation, policy enforcement, privacy routing)
- These are complementary projects: Kagenti orchestrates agents at the platform level, OpenShell secures agent execution at the runtime level
- Features being ported are infrastructure/platform capabilities that exist in Kagenti but are missing from OpenShell
</context_explanations>

""".format(
        output_dir=config.output_dir,
        kagenti_source_dir=config.kagenti_source_dir,
        openshell_source_dir=config.openshell_source_dir,
        kagenti_github_url=config.kagenti_github_url,
        openshell_github_url=config.openshell_github_url,
    )


# ===================================================================
# PHASE 0: FEATURE DETECTION PROMPTS
# ===================================================================

ANALYZE_KAGENTI_FEATURES_TEMPLATE = """

{context}

<instructions>
The goal of this task is to perform an exhaustive analysis of the Kagenti project to identify ALL major features and capabilities it provides. Think hard and be very thorough.

Do the following:
1. Explore the Kagenti source code at <source_project_dir> comprehensively:
   - Read the README.md, CLAUDE.md, CONTRIBUTING.md, and any documentation files in docs/
   - Examine the top-level directory structure
   - Analyze each major module/directory to understand its purpose
   - Look at key source files in: kagenti/ (backend, ui-v2, tui, auth, keycloak, llm-budget-proxy, common, installer, tools, examples)
   - Look at charts/ for Helm chart configurations
   - Look at deployments/ for Kubernetes manifests
   - Look at docs/ for architecture, components, identity, access control, security, observability docs

2. For each identified feature/capability, document:
   - Feature name (clear, concise identifier)
   - Category (e.g., security/identity, networking/gateway, orchestration, observability, deployment, discovery, policy, UI, etc.)
   - Description (2-3 sentences explaining what the feature does)
   - Key implementation files (list the primary source files or directories)
   - Architecture notes (how the feature is structured, key classes/functions/modules)
   - Dependencies (what other features or external systems it depends on)
   - Complexity estimate (LOW / MEDIUM / HIGH / VERY HIGH) for porting to a Rust project
   - Languages and frameworks used (Python, TypeScript, Go, etc.)

3. Be especially thorough about these capability areas:
   - SPIFFE/SPIRE workload identity and zero-trust security
   - OAuth2/OIDC integration (Keycloak) and token exchange (RFC 8693)
   - Identity & Auth Bridge (AuthBridge sidecar pattern)
   - MCP Gateway (Envoy-based MCP tool routing and discovery)
   - Agent lifecycle management (Kubernetes operator, AgentCard CRDs)
   - Agent/tool discovery via Kubernetes labels and CRDs
   - A2A (Agent-to-Agent) protocol support
   - Observability (MLflow, Phoenix/Langfuse tracing, OpenTelemetry, Kiali)
   - LLM Budget Proxy (token tracking and cost control)
   - Skills system (ConfigMap-based skill registry, Skillberry store)
   - Plugins adapter for Envoy (ext_proc)
   - Fine-grained access control (Cortex, OPA)
   - Multi-tenant Kubernetes deployment
   - Dynamic agent materialization

4. For each feature, note whether it is:
   - Core (fundamental to the platform's operation)
   - Extension (adds capability but not required)
   - Integration (connects to external systems like Keycloak, Envoy, MLflow)
</instructions>

<output>
- Dump the comprehensive feature analysis to <output_dir>/{output_file}
- Structure the output as a numbered list of features, each with the fields described above
- Be exhaustive -- do not miss any significant feature
</output>

"""


ANALYZE_OPENSHELL_FEATURES_TEMPLATE = """

{context}

<instructions>
The goal of this task is to perform an exhaustive analysis of the OpenShell project to identify ALL major features and capabilities it currently provides. Think hard and be very thorough.

Do the following:
1. Explore the OpenShell source code at <target_project_dir> comprehensively:
   - Read the README.md, AGENTS.md, CHANGELOG.md (recent entries), and documentation files in docs/ and architecture/
   - Examine the top-level directory structure
   - Analyze each major Rust crate under crates/ to understand its purpose
   - Look at proto/ for gRPC service definitions
   - Look at providers/ for credential provider implementations
   - Look at deploy/ for Helm charts and deployment configs
   - Look at e2e/ for end-to-end tests
   - Look at python/ for Python SDK
   - Look at .agents/skills/ for agent skill definitions
   - Look at rfc/ for design proposals

2. For each identified feature/capability, document:
   - Feature name (clear, concise identifier)
   - Category (e.g., security/isolation, networking/proxy, policy, sandbox lifecycle, compute drivers, credentials, observability, SDK, etc.)
   - Description (2-3 sentences explaining what the feature does)
   - Key implementation files (list the primary Rust crates, source files, or directories)
   - Maturity level (MATURE / DEVELOPING / BASIC / PLANNED)
   - Known limitations or issues (if apparent from code comments, TODOs, RFCs, or documentation)

3. Be especially thorough about these capability areas (matching the Kagenti analysis):
   - Security and isolation (Landlock, seccomp, credential injection)
   - Networking (L7 proxy, privacy router, inference routing)
   - Authentication and identity (providers, OAuth, workload identity)
   - Agent/tool discovery and registration
   - Multi-agent orchestration
   - Kubernetes deployment (Helm, operator, multi-tenant)
   - Observability (metrics, tracing, logging)
   - MCP support
   - Policy engine (OPA, filesystem, network, process policies)
   - SDK and API (gRPC, Python SDK, CLI)
   - Compute drivers (Docker, Podman, MicroVM, Kubernetes)
   - Gateway management

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
<kagenti_features_file>
{kagenti_features_file}
</kagenti_features_file>
<openshell_features_file>
{openshell_features_file}
</openshell_features_file>
</definitions>

<instructions>
The goal of this task is to identify features that exist in Kagenti but are MISSING or INADEQUATE in OpenShell, AND that have demonstrated community demand. This is the critical step that determines what is worth porting. Think very hard and be precise.

IMPORTANT CONTEXT: Kagenti and OpenShell are COMPLEMENTARY projects, not direct competitors:
- Kagenti is a Kubernetes middleware platform for AI agent orchestration (deployment, security, networking, observability)
- OpenShell is a sandboxed execution environment for AI agents (isolation, policy enforcement, privacy routing)
- Features being ported are infrastructure/platform capabilities from Kagenti that would enhance OpenShell's platform layer

Do the following:

1. Read and fully understand the Kagenti feature analysis in <kagenti_features_file>
2. Read and fully understand the OpenShell feature analysis in <openshell_features_file>

3. For EACH Kagenti feature, determine its status in OpenShell:
   - MISSING: Feature does not exist at all in OpenShell
   - INADEQUATE: Feature exists but has significant gaps, bugs, or limitations compared to Kagenti
   - EQUIVALENT: Feature exists and is roughly equivalent
   - SUPERIOR: OpenShell's implementation is better than Kagenti's

4. For features that are MISSING or INADEQUATE in OpenShell, research community demand:
   - Use the `gh` CLI tool to search OpenShell GitHub issues:
     Run: gh api "search/issues?q=repo:NVIDIA/openshell+is:issue+<relevant_search_terms>&per_page=20" --jq '.items[] | {{number, title, state, reactions: .reactions.total_count, comments, created_at}}'
   - Search for feature requests, enhancement requests, bug reports related to the feature
   - Look for issues with high reaction counts or many comments (indicating community demand)
   - Also search Kagenti issues to verify the feature works well there:
     Run: gh api "search/issues?q=repo:kagenti/kagenti+<relevant_search_terms>&per_page=10" --jq '.items[] | {{number, title, state}}'
   - Search for related terms: MCP gateway, SPIFFE, workload identity, OAuth, OIDC, Keycloak, agent discovery, lifecycle operator, A2A protocol, observability, tracing, OpenTelemetry, budget proxy, skills registry, multi-tenant, Kubernetes operator, agent orchestration

5. Score each portability candidate on these dimensions (1-10):
   - Community demand: How many people want this? How strongly? (based on issue counts, reactions, comments)
   - Gap severity: How much worse is OpenShell's current solution? (MISSING=10, INADEQUATE=5-8, etc.)
   - Porting feasibility: How practical is it to port from Python/Go/TypeScript to Rust? Consider architectural differences.
   - Impact: How much would this improve OpenShell for its users?
   - Independence: Can this feature be ported independently, or does it require other features first?

6. Rank the candidates by overall priority score (weighted average: demand=30%, gap=25%, impact=25%, feasibility=15%, independence=5%)

7. For each top candidate, provide:
   - Feature name
   - Kagenti implementation summary (key files, architecture, language)
   - OpenShell gap description (what's missing/broken, with specific issue references)
   - Community evidence (list of GitHub issues with numbers, titles, reaction counts)
   - Porting approach (high-level strategy for adapting Python/Go/TypeScript to Rust)
   - Risk assessment
   - Estimated complexity (LOW / MEDIUM / HIGH / VERY HIGH)
   - Score breakdown (each dimension + overall)

IMPORTANT RULES:
- Only include features where you found CONCRETE evidence of community demand (actual GitHub issues, not speculation)
- Only include features where there is a CLEAR gap between Kagenti and OpenShell
- Do NOT include features that OpenShell already handles well
- Be honest about porting complexity -- Python/Go/TypeScript to Rust is non-trivial and requires understanding Rust ownership, lifetimes, and async patterns
- Consider that OpenShell is Rust-first: ported features should feel native to Rust, not like translated code
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
        "kagenti_status": "IMPLEMENTED",
        "openshell_status": "MISSING" or "INADEQUATE",
        "community_demand_score": 8,
        "gap_severity_score": 9,
        "impact_score": 8,
        "feasibility_score": 7,
        "independence_score": 9,
        "overall_score": 8.2,
        "key_kagenti_files": ["..."],
        "kagenti_language": "Python",
        "openshell_issues": [{{  "number": 123, "title": "...", "reactions": 5 }}],
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
    - Include all relevant files: source code, tests, configuration, types/interfaces/traits
    - Do not miss anything critical to understanding or porting this feature

- For the detected <code_trace>, perform a detailed analysis:
    - Document the entry points (how the feature is triggered/activated)
    - Document the data flow (what data enters, how it's transformed, where it goes)
    - Document the state management (what persistent state is maintained, where/how)
    - Document external dependencies (libraries, APIs, services, Kubernetes resources)
    - Document configuration options and defaults
    - Document error handling patterns
    - Document concurrency/async patterns used

- Document each major component with:
    - Its purpose and responsibility
    - Key classes/functions/structs/traits with their signatures, inputs, and outputs
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
- <code_trace_files> is a list of code trace files for Kagenti (source) and OpenShell (target) respectively. Each code trace file describes the <code_trace> of the specific project relevant to the feature being ported.
</definition_explanations>

<instructions>
The goal of this task is to provide a high-level multi-step coding plan that ports the feature described in <feature_description> from Kagenti to OpenShell. Think hard for this task and follow these guidelines:

- Analyze and understand in-detail the <code_trace> of Kagenti from <code_trace_files>.
- Analyze and understand in-detail the <code_trace> of OpenShell from <code_trace_files> (the corresponding area where this feature should be integrated).
- Compare in-detail how Kagenti implements this feature vs what OpenShell currently has (or lacks).

- Design a porting plan that:
    - Translates Python/Go/TypeScript code to idiomatic Rust (use Rust patterns, not direct translation)
    - Follows OpenShell's existing code conventions, crate structure, and architectural patterns
    - Integrates naturally with OpenShell's existing systems (gateway, sandbox, policy engine, providers, CLI)
    - Maximizes reuse of existing OpenShell infrastructure where possible
    - Preserves the core logic and behavior of the Kagenti implementation
    - Handles the language differences properly:
        - Python classes -> Rust structs + impl blocks + traits
        - Python asyncio -> Rust async/await with tokio
        - Python dataclasses -> Rust structs with derive macros (Debug, Clone, Serialize, Deserialize)
        - Python imports -> Rust mod/use statements
        - Python dict -> Rust HashMap/BTreeMap or typed structs
        - Python exceptions -> Rust Result<T, E> with custom error types (thiserror)
        - Go interfaces -> Rust traits
        - Go goroutines -> Rust tokio tasks
        - TypeScript interfaces -> Rust traits or structs
        - gRPC/protobuf definitions -> tonic-based Rust gRPC services

- For each step in the plan:
    - Describe what code to create or modify
    - Specify the target file path within OpenShell's crate structure
    - Describe the Rust implementation approach
    - Note any OpenShell APIs or patterns to reuse
    - Note any new Cargo dependencies needed

- Ensure the plan:
    - Can be implemented incrementally (each step builds on the previous)
    - Has clear integration points with existing OpenShell code
    - Handles configuration (add to OpenShell's config structures if needed)
    - Includes proper Rust types, traits, and error handling
    - Follows OpenShell's error handling patterns (using thiserror, anyhow, or the project's error types)
    - Follows OpenShell's async patterns (tokio-based)

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
- <code_trace_files> is a list of code trace files for Kagenti (source) and OpenShell (target) respectively.
- <input_file> is a file that describes the high-level multi-step coding plan for porting the feature from Kagenti to OpenShell.
- <input_summary_file> is a file that summarizes the work done to generate <input_file>.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical, in-depth review of the porting plan in <input_file> (iteration {iteration}). Do the following and think hard:

- Understand in-detail the plan in <input_file>, and restate the plan step-by-step.
- Review the plan for:
    - Correctness of Python/Go/TypeScript-to-Rust translation approach
    - Proper use of OpenShell's existing APIs and patterns (check the actual source code to verify)
    - Proper Rust typing, traits, and lifetime annotations
    - Proper ownership and borrowing semantics
    - Integration completeness (does it hook into all necessary OpenShell systems?)
    - Configuration handling (is it added to the config structures properly?)
    - Error handling patterns (matches OpenShell's approach with Result types?)
    - Missing steps or gaps in the plan
    - Incorrect assumptions about either codebase
    - Dependencies that might not exist or might need to be added to Cargo.toml
    - Proper async/await usage with tokio

- Review for architectural issues:
    - Does the plan follow OpenShell's crate organization?
    - Are new files placed in the correct crates/modules?
    - Does it respect OpenShell's gateway/sandbox/supervisor architecture?
    - Are there circular dependency risks between crates?

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
- <code_trace_files> is a list of code trace files for Kagenti (source) and OpenShell (target) respectively.
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan for porting the feature.
</definition_explanations>

<instructions>
The goal of this task is to generate a comprehensive testing plan for the ported feature described in <code_port_plan_file> (iteration {iteration}). Do the following and think hard:

- Analyze and understand in-detail the coding plan in <code_port_plan_file>
- Analyze how Kagenti tests this feature (look at the test files in the Kagenti repo at <source_project_dir>)
- Analyze how OpenShell structures its tests (look at existing tests in <target_project_dir>)

- Plan tests following OpenShell's testing conventions:
    - Rust unit tests (#[cfg(test)] mod tests in source files)
    - Integration tests (tests/ directory in each crate)
    - E2E tests (e2e/ directory) if the feature has user-facing behavior

- For each test:
    - Describe what it verifies
    - Specify the test file path (crate and module)
    - Describe key test cases (happy path, edge cases, error cases)
    - Note any test fixtures or mocks needed
    - Specify whether it requires #[tokio::test] for async tests

- Ensure test coverage for:
    - Core feature functionality
    - Configuration handling
    - Error cases and fallbacks
    - Integration with existing OpenShell systems
    - Edge cases identified in the code trace
    - Regression scenarios
    - Concurrency safety (if applicable)

- Plan tests that can actually run in OpenShell's test infrastructure (using cargo test, with proper feature flags)
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
    - Uses OpenShell's testing patterns correctly (check existing tests for conventions)
    - Has proper unit tests for each new module/function
    - Has integration tests for interaction with existing OpenShell systems
    - Covers error handling and edge cases
    - Tests are actually implementable (correct APIs, proper mocking with mockall or similar)
    - Uses #[tokio::test] for async tests
    - Properly handles Rust ownership in test assertions

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
The goal of this task is to generate the actual Rust code that ports the feature from Kagenti to OpenShell, implementing the coding plan in <code_port_plan_file> and the testing plan in <test_plan_file> (iteration {iteration}). Do the following and think hard:

- Analyze and understand in-detail the coding plan in <code_port_plan_file>
- Analyze and understand in-detail the testing plan in <test_plan_file>
- Implement the plans EXACTLY as described:
    - Follow strictly the code and test plans, DO NOT DIVERGE
    - Write idiomatic Rust (not a literal Python/Go/TypeScript translation)
    - Follow OpenShell's coding conventions (check existing files for patterns)
    - Use proper Rust types, traits, generics, and derive macros
    - Use OpenShell's existing utilities and patterns where specified in the plan
    - Handle ownership and borrowing correctly
    - Use proper async/await with tokio

- For each file to create or modify:
    - Write the complete file content
    - Include proper use/mod statements
    - Include proper Rust types and trait implementations
    - Add pub exports as needed
    - Update Cargo.toml if new dependencies are needed

- IMPORTANT: ALL code modifications MUST be made exclusively inside <target_project_dir>. Do NOT modify any files outside of this directory.

- After writing the code:
    - Run the build command to verify compilation: cd <target_project_dir> && {build_command}
    - Fix any Rust compilation errors (type errors, lifetime errors, borrow checker errors)
    - Run the tests: cd <target_project_dir> && {test_command}
    - Fix any test failures

- Create a unified diff/patch of all changes made to <target_project_dir>:
    - Run: cd <target_project_dir> && git diff HEAD > <output_dir>/{{output_patch_file}}

- Ensure the code is written professionally, clearly, well-typed, and follows OpenShell conventions.
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
    - Correctness of the Rust implementation vs the Kagenti Python/Go/TypeScript original
    - Proper Rust typing (no unnecessary clones, proper references and lifetimes)
    - Proper use of OpenShell's APIs and patterns
    - Proper error handling with Result types
    - Proper async/await usage with tokio
    - Memory safety (no unsafe blocks unless truly necessary)
    - Thread safety (Send + Sync bounds where needed)
    - Security issues (injection, path traversal, etc.)
    - Test quality and coverage

- Review for architectural issues:
    - Proper crate organization
    - Clean trait abstractions
    - No circular dependencies between crates
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
Apply the code patch in <patch_file> to the OpenShell project and build it.

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
Run the OpenShell test suite to verify the ported feature works correctly.

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
   - Determine whether it's a build error (Rust compilation), test error, or runtime error
   - Check if the error is in the ported code or in test code
   - Pay special attention to: borrow checker errors, lifetime issues, type mismatches, missing trait implementations

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
class AnalyzeKagentiFeaturesPrompt:
    context: str
    output_file: str

    def prompt(self):
        return ANALYZE_KAGENTI_FEATURES_TEMPLATE.format(
            context=self.context,
            output_file=self.output_file,
        )


@dataclass
class AnalyzeOpenShellFeaturesPrompt:
    context: str
    output_file: str

    def prompt(self):
        return ANALYZE_OPENSHELL_FEATURES_TEMPLATE.format(
            context=self.context,
            output_file=self.output_file,
        )


@dataclass
class DetectPortabilityCandidatesPrompt:
    context: str
    kagenti_features_file: str
    openshell_features_file: str
    report_file: str
    json_file: str
    max_features: int

    def prompt(self):
        return DETECT_PORTABILITY_CANDIDATES_TEMPLATE.format(
            context=self.context,
            kagenti_features_file=self.kagenti_features_file,
            openshell_features_file=self.openshell_features_file,
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
