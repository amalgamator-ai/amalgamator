# Amalgamator documentation

User guides for installing Amalgamator and running its pipelines. For design
contracts aimed at coding agents and contributors changing the code, see
[AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).

The published docs site ([amalgamator.ai](https://amalgamator.ai/docs/)) pulls
this tree. Sidebar titles and order live in [`docs_nav.yaml`](docs_nav.yaml) —
update that file in the same PR when you add or rename a guide (no website PR
needed for product docs).

## Start here

| Doc | When to read it |
|-----|-----------------|
| [Getting started](getting-started.md) | Install, prerequisites, and choosing a workflow |
| [Concepts](concepts.md) | What Amalgamator does, evidence-first design, and scope |
| [Authentication and models](authentication.md) | API keys, `models.yaml`, and which pipelines need which providers |

## Workflows

### GPU performance analysis

| Doc | Description |
|-----|-------------|
| [Overview](gpu/overview.md) | Profile → analyze → codegen → PR flow |
| [Profiling](gpu/profiling.md) | Capture benchmarks and GPU traces |
| [Trace analysis](gpu/analysis.md) | Single-trace and cross-trace analysis |
| [Code generation and PRs](gpu/codegen.md) | Port optimizations and open reviewable PRs |

### Bug-fix backporting

| Doc | Description |
|-----|-------------|
| [Overview](backporting/overview.md) | Scan for missing fixes, then backport them |
| [Scanner](backporting/scanning.md) | Find branches/tags that still need a fix |
| [Backport pipeline](backporting/backporting.md) | Git-first port of a fix to another branch |

## Reference

| Doc | Description |
|-----|-------------|
| [CLI entry points](reference/cli.md) | Module commands and orchestrator scripts |
| [Artifacts and outputs](reference/artifacts.md) | Where runs write results |
| [FAQ and troubleshooting](reference/faq.md) | Common failures and how to recover |

## Package deep dives

These READMEs are the authoritative detail for configuration fields, phases,
and examples. The guides above summarize; use these when you need the full
contract:

| Package | README |
|---------|--------|
| GPU subtree | [llm/README.md](../llm/README.md) |
| Trace analysis | [llm/auto_analyze/README.md](../llm/auto_analyze/README.md) |
| Code generation | [llm/auto_code_gen/README.md](../llm/auto_code_gen/README.md) |
| Agent backends | [llm/auto_code_gen/agents/README.md](../llm/auto_code_gen/agents/README.md) |
| Upstream PR flow | [llm/auto_code_gen/pr_flow/README.md](../llm/auto_code_gen/pr_flow/README.md) |
| Bug-fix backport | [auto_bug_fix/README.md](../auto_bug_fix/README.md) |
| Backport scanner | [backport_scanner/README.md](../backport_scanner/README.md) |
