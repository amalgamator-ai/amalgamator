# AI-Based Automatic Performance Analysis

End-to-end automation for profiling, analyzing, and comparing GPU traces of LLM inference frameworks (vLLM, SGLang, TensorRT-LLM), and for porting bug fixes between branches of C/systems projects. Uses Claude as an AI agent.

## Overview

This repo contains two independent AI pipelines:

### GPU Performance Analysis

Automate profiling, analyzing, and comparing GPU traces of LLM inference frameworks (vLLM, SGLang, TensorRT-LLM) using Claude as an AI agent.

See **[GPU_Performance_Analysis_README.md](GPU_Performance_Analysis_README.md)** for the full setup and usage guide, including single-trace analysis, cross-trace comparison, profiling orchestration, and annotated Perfetto trace generation.



### Bug Fix Porting

Port a bug fix from one branch of a C/systems project (gcc, openssl, glibc, curl, CRI-O, Kubernetes, etc.) to another using Claude as the AI engine. The pipeline tries `git cherry-pick` first, resolves conflicts with an LLM, and falls back to a full LLM-powered autonomous build-test-fix loop when needed.

See **[BUG_FIX_README.md](BUG_FIX_README.md)** for the full setup and usage guide, including the gitted pipeline, auto_code_gen details, and config examples.

**Quick start:**

```bash
# 1. Create a YAML config (see auto_bug_fix/configs/)
# 2. Run
python -m auto_bug_fix.run_bug_fix auto_bug_fix/configs/your_config.yaml
```


## Project Structure

```
ai_auto_analyze/
├── common/                             # Shared utilities
│   ├── claude_utils.py                 #   Claude Agent SDK wrapper
│   ├── utils.py                        #   Logging, directory cleanup, output dir helpers
│   └── convert_nsys_to_sqlite.sh       #   NSYS trace format converter
├── auto_analyze/                       # Analysis pipeline
│   ├── run_single_trace.py             #   Single-trace analysis entry point
│   ├── run_cross_trace.py              #   Cross-trace analysis entry point
│   ├── run_chrome_trace.py             #   Chrome trace JSON generation
│   ├── run_summary_pdf.py              #   PDF report generation
│   ├── run_jiras.py                    #   JIRA task creation
│   ├── create_single_trace_config.py   #   Helper: generate single-trace config JSON
│   ├── create_cross_trace_config.py    #   Helper: generate cross-trace config JSON
│   ├── configs/                        #   Config dataclasses and examples
│   │   ├── single_trace_config.py
│   │   ├── single_trace_config_example.json
│   │   ├── cross_trace_config.py
│   │   ├── cross_trace_config_example.json
│   │   └── claude_config.json
│   ├── prompts/                        #   Prompt templates for all analysis steps
│   │   ├── single_trace_prompts.py
│   │   ├── cross_trace_prompts.py
│   │   ├── chrome_trace_prompts.py
│   │   ├── jira_prompts.py
│   │   └── summary_pdf_prompts.py
│   └── examples/                       #   Guides and example runs
├── auto_profile/                       # Profiling orchestration
│   ├── run_profile_core.sh             #   Main profiling script
│   ├── run_profile_summary.py          #   Parse results into analysis configs
│   ├── parse_run_config.py             #   Config parser and validator
│   └── test_configs/                   #   JSON configs (infra, run, docker, GPUs)
├── auto_code_gen/                      # AI-based code generation pipeline
│   ├── run_code_gen.py                #   Pipeline orchestrator (all use cases)
│   ├── code_gen_prompts.py            #   Shared prompt classes
│   ├── code_gen_configs.py            #   PipelineConfig, CodeGenConfig, BugFixConfig
│   ├── use_cases/                     #   Pluggable use case definitions
│   │   ├── base.py                    #     UseCase ABC
│   │   ├── llm_framework.py           #     LLM framework optimization porting
│   │   └── bug_fix.py                 #     Bug fix branch porting
│   ├── configs/                       #   Example JSON configs
│   ├── run_runtime_iters.py           #   Standalone runtime iteration runner
│   ├── run_investigate_issue.py
│   ├── run_fix_issue.py
│   ├── run_work_items.py
│   └── run_summary.py
├── env.sh                              # Environment variables
├── run_all.sh                          # Full pipeline orchestrator
└── run_all_scheduled.sh                # Scheduled pipeline execution
```
