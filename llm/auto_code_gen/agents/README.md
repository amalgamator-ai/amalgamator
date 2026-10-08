# Agent backends

`auto_code_gen` and standalone `auto_analyze` support the existing Claude Agent
SDK and LiteLLM tool-calling endpoints through this shared layer. Framework
porting, codegen bug-fix, and GPU trace analysis use the same provider-neutral
session, tool loop, and event logging.

## Configuration

Existing pipeline configs work unchanged: `thinking-mode: deep|normal` retains
the current Claude model and thinking settings. To select another agent, add
an `agent` object to the existing codegen pipeline JSON:

```json
{
  "agent": {
    "backend": "litellm",
    "model": "openai/<served-model-id>",
    "api_base": "http://localhost:8000/v1",
    "api_key_file": "~/.keys/codegen-endpoint",
    "model_params": {"temperature": 0.2, "max_tokens": 8192},
    "context_length": 32768,
    "command_timeout": 1800,
    "request_timeout": 300,
    "max_agent_iterations": null,
    "billing": "self_hosted",
    "allowed_tools": ["Read", "Write", "Bash"]
  }
}
```

Use the actual served model ID and context/output limits. The endpoint must
support tool calling. `openai/` identifies an OpenAI-compatible endpoint; it
does not mean the benchmark workload is an OpenAI model. `model_params` contains
provider parameters, while transport fields such as endpoint, key, tools and
timeouts belong directly in `agent`.

The workload's existing `model` and GPU `runtime_backend: local|remote` fields
keep their current meanings. They are independent of `agent.model/backend`.

Every codegen command that invokes an agent accepts `--agent-config FILE`:
`run_code_gen`, `run_runtime_iters`, `run_work_items`, `run_summary`,
`run_rebase_pr`, `run_generate_pr_commands`, and
`pr_review.step3_fix_review_comments`. The file can contain the agent object
directly or under an `agent` key. It replaces the pipeline's agent section.

Standalone `auto_analyze` commands accept the same option. Single- and
cross-trace JSON configs may also embed the same `agent` object. The legacy
`--claude-config` spelling remains an alias for backward compatibility.

```sh
python -m llm.auto_code_gen.run_code_gen --config pipeline.json --agent-config agent.json
python -m llm.auto_code_gen.run_runtime_iters --config pipeline.json --agent-config agent.json --resume
python -m llm.auto_code_gen.pr_review.step3_fix_review_comments --output-dir review --repo-dir repo --agent-config agent.json
```

Generated PR-description and rebase follow-up commands include the explicit
agent-file override. Keep that file at a stable location outside directories
cleaned between runs. With nested agent settings, follow-ups read the same
pipeline config. Existing command wrappers that only pass the pipeline config
can use its nested `agent` section.

Credentials: an explicit `api_key` takes priority over `api_key_file`, followed
by the provider's usual environment lookup. Relative key-file paths resolve
against the file declaring them. Missing/empty selected key files fail at
startup. The runner does not print configuration credentials. For an endpoint
without authentication, supply a dummy key if its client requires one.

For Claude, use `backend: claude_sdk` and a bare `claude-*` model name. The
adapter retains the existing SDK environment authentication and profiling
persona; transport parameters are rejected rather than ignored. To route a
provider-qualified Claude model through LiteLLM, select `backend: litellm`
explicitly and use that provider's model syntax. SDK `[1m]` suffixes are not
accepted by the generic backend. Claude `thinking-mode` settings are not
forwarded to generic providers.

## Runtime contract

- `agent_session` retains history across steps and calls; a call without a
  supplied session uses a temporary one. Existing deliberate fresh sessions
  remain fresh. A failed generic session must be replaced before retrying.
- Resume remains artifact/convergence based. It does not reload a conversation
  from a prior process. Existing step-level resume reruns the last completed
  step; outer iteration skipping remains in the pipeline.
- The generic tools are Read (with line offset/limit), Write and Bash. Explicit
  empty tool lists grant no tools; unsupported required tools fail validation.
  Generic `acceptEdits` executes allowed tools with the process's filesystem and
  shell permissions. It does not implement the SDK approval UI or a filesystem
  sandbox. Source/build directories may legitimately be outside the output cwd.
- Tool output is capped. Context compaction reserves response tokens and a
  margin, preserves the original/current task and whole recent tool groups, and
  summarizes older codegen progress. Failed summaries fall back to dropping
  old groups. An oversized protected task fails clearly. For unrecognized
  endpoint models, explicitly set context limits; the fallback is 32,768 tokens.
- `command_timeout` defaults to 1,800 seconds; explicit `null` means unbounded.
  `max_agent_iterations` defaults to 100; explicit `null` lets a step continue
  until it completes or another progress/error guard stops it.
  Model request timeout defaults to 300 seconds. Transient transport failures
  have at most `max_retries` (default 2) retries; partial streams and auth
  failures are not silently replayed. Unsupported streaming can fall back to
  non-streaming. `stream: false` disables streaming explicitly.
- Generic runs stop with an error on provider failure, incomplete model
  output, absent required artifacts, repeated calls or iteration exhaustion.
  `AgentError.timings` retains completed/partial step usage. The codegen bug-fix
  runtime accepts a current SUCCESS marker as success and treats a failure
  marker or absent outcome as failure; stale markers are removed before retry.
- Token/price information comes from LiteLLM responses and metadata. Unknown
  prices and incomplete/estimated usage are flagged in logs and reports;
  numeric USD totals contain known charges only. `billing: self_hosted`
  explicitly records zero per-token API charges, excluding infrastructure
  costs. A custom endpoint alone does not imply free billing.

Each generic run writes two event logs in the agent working directory:
`agent_events.jsonl` is the structured machine-readable record, and
`agent_events.log` is a chronological human-readable transcript containing
session metadata, prompts, assistant replies, tool calls, tool results,
provider reasoning (`reasoning` or `reasoning_content`), context compaction,
and step completion/failure events. Both logs use mode
`0600` and redact the configured API key.

`summary_model_params` can override provider parameters only for context
summarization requests. For reasoning models, this can disable thinking during
summarization without changing the main agent requests.

To render an older JSONL file produced before the text log was added, run:

```sh
python -m llm.auto_code_gen.agents.render_events path/to/agent_events.jsonl
```

## Implementation and tests

Transport, tool calling and context-management ideas are adapted from
`amalgamator` commits `c0b487d`, `de20339` and `3fc802e`. Provider settings and
conversation state are per session; no source global configuration or backport
pipeline code is imported. The canonical `PipelineStep` and the native Claude
runner remain in `common.claude_utils` unchanged. The SDK remains a dependency.

Validated with Python 3.14, LiteLLM 1.100.1 and Claude Agent SDK 0.2.152. Focused
test dependencies are in `tests/auto_code_gen/requirements-agent-tests.txt`.

```sh
python -m pip install -r tests/auto_code_gen/requirements-agent-tests.txt
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest tests/auto_code_gen
```

Tests include disposable local HTTP endpoints for real LiteLLM streaming and
tool-call conversion, plus mocked sessions and artifact fixtures. They require
no provider credentials, GPU jobs or changes to another repository. Real model
quality and GPU/cluster execution should be validated against the intended
endpoint and workload before relying on generated patches.
