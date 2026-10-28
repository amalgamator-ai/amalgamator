# Authentication and models

Amalgamator does not ship with cloud credentials. You configure access per
pipeline.

## Quick matrix

| Pipeline | Typical provider | How credentials are supplied |
|----------|------------------|------------------------------|
| Bug-fix backport | Claude, OpenAI, Ollama, OpenAI-compatible | Named profiles in `models.yaml`, or legacy inline case fields |
| Backport scanner | Same registry as backport | `llm.model` profile ID in the scan config |
| GPU analyze / codegen | Claude Agent SDK | `ANTHROPIC_API_KEY`, SDK login, or Vertex/backend setup; optional agent JSON for LiteLLM endpoints |
| GPU profiling (benchmarks) | N/A for the shell/Docker runners | Framework/Docker/GPU env; analysis stage still needs Claude |

Tests in this repository do **not** require live API keys.

## Model registry (`models.yaml`)

Backporting and scanning share one registry format. Copy the template:

```bash
cp auto_bug_fix/configs/example_models.yaml ~/.config/amalgamator/models.yaml
# or keep models.yaml beside each case/scan file
```

Minimal example:

```yaml
models:
  m1:
    provider: anthropic
    model: claude-sonnet-4-6
    # api_key_file: ~/.keys/anthropic.key

  m2:
    provider: litellm
    model: openai/deepseek-r1
    api_base: http://localhost:8080/v1
    params:
      temperature: 0.2
      max_tokens: 16384

default: m1
# strict: true   # reject unknown profile IDs
```

### Discovery order

Registries are **not** merged. The first match wins:

1. `--models PATH` on the CLI
2. `models.yaml` in the same directory as the case/scan file
3. `~/.config/amalgamator/models.yaml`

An explicit `--models` path that is missing or invalid is an error (no fallback).

### Credentials tips

- Prefer `api_key_file` over embedding secrets in YAML. Relative key-file paths
  resolve against the **command working directory**, not the registry directory.
- To use provider environment credentials (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  Claude SDK login), omit `api_key` / `api_key_file` on that profile.
- Do not write `api_key: "${ANTHROPIC_API_KEY}"` — values are not env-interpolated.
- Each profile carries its own key; Phase 4b overrides (`advanced.phase_4b_model`)
  do not inherit another profile's credentials.

Full field reference:
[auto_bug_fix README — Model registry](../auto_bug_fix/README.md#model-registry-reference).

## Selecting a model in a case or scan

```yaml
# case.yaml
model: m1

# optional recovery profile for Phase 4b
advanced:
  phase_4b_model: m3
```

```yaml
# scan.yaml
llm:
  model: m1
```

Without a registry, bare model names and legacy inline fields still work; with
`strict: true`, only registered profile IDs are accepted.

## GPU analysis and codegen (Claude / agents)

Trace analysis and the default codegen path use the **Claude Agent SDK**, not the
backport `models.yaml` schema.

Common setup:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or rely on an existing Claude / Vertex SDK login
```

To run codegen against a LiteLLM tool-calling endpoint (including local
OpenAI-compatible servers), add an `agent` section to the codegen JSON or pass
`--agent-config`. See:

- [Agent runtime README](../llm/auto_code_gen/agents/README.md)
- Example: [agent_vllm_example.json](../llm/auto_code_gen/configs/agent_vllm_example.json)

Important distinction:

- **Backport registry** `model` / `provider` — selects the backport/scanner runner.
- **GPU `agent.model` / `agent.backend`** — selects the coding agent for codegen.
- Workload fields like `model` / `runtime_backend` in GPU configs describe the
  **inference job under test**, not the coding agent.

## Keeping secrets out of git

Never commit API keys, token files, or case/scan files that embed credentials.
Keep local registries and keys outside the repo or in ignored paths. See
[CONTRIBUTING.md](../CONTRIBUTING.md#credentials-and-local-files) and
[SECURITY.md](../SECURITY.md).
