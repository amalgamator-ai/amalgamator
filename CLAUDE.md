# Claude Code — Amalgamator

Shared repository instructions:

@AGENTS.md

## Skills (Slash Commands)

Pipeline runbooks live in `.claude/skills/`. Use the relevant skill when running
that workflow; development setup and verification commands are in `AGENTS.md`.

| Skill | Pipeline | Description |
|-------|----------|-------------|
| `/profile` | Perf | Run GPU profiling benchmarks |
| `/parse` | Perf | Parse profiling results, generate analysis configs |
| `/analyze` | Perf | Run trace analysis (single-trace or cross-trace) |
| `/perf-pipeline` | Perf | Full end-to-end performance pipeline |
| `/codegen` | CodeGen | AI code generation pipeline |
| `/runtime` | CodeGen | Standalone runtime iterations |
| `/investigate` | CodeGen | Post-pipeline issue investigation |
| `/fix` | CodeGen | Post-pipeline issue fixing |
| `/work-items` | CodeGen | Execute arbitrary work items |
| `/bugfix` | BugFix | Git-first bug fix backport pipeline |
| `/chrome-trace` | Util | Generate Chrome trace visualization |
| `/summary` | Util | Generate summary reports (PDF/PPTX) |
| `/jiras` | Util | Create JIRA tasks from analysis |
| `/create-config` | Util | Generate config files interactively |
