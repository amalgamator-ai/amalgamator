<!--
Thanks for contributing to Amalgamator. Keep this PR to one logical change and
fill in the sections below; write `None` where a section does not apply. Commit
subjects use a `feat:` / `fix:` / `docs:` / `test:` / `chore:` prefix (see
CONTRIBUTING.md), so matching the PR title keeps the history readable. This
template governs PRs to Amalgamator — PRs opened against an upstream framework
(vLLM, SGLang, TensorRT-LLM) follow that repo's template.
-->

## Summary

<!-- What changes, and why? One short paragraph. Link the issue this closes. -->

Closes #

## Area

<!-- Tick the parts this change touches. -->

- [ ] `auto_bug_fix`
- [ ] `backport_scanner`
- [ ] `llm/auto_config`
- [ ] `llm/auto_profile`
- [ ] `llm/auto_profile_server_client`
- [ ] `llm/auto_analyze`
- [ ] `llm/auto_code_gen` (including `pr_flow` / `pr_review`)
- [ ] `common/`
- [ ] `tests/`, docs, tooling, or CI

## Evidence

<!--
Verdicts must rest on measured artifacts, not assertions. Point at the files or
paste the numbers, and say how to reproduce them. Omit bullets that do not apply.
-->

- Artifact or verdict: <!-- e.g. runtime/V3/verify_verdict.json overall=PASS,
  a dossier path, report.json -->
- Performance: <!-- before/after, framework and version, hardware, target trace id -->
- Reproduced with: <!-- exact command(s) -->

## Tests

<!-- Which tests were added or updated, and why they cover the change. -->

## Interface changes

<!--
Prompt fields, artifact names, config schema, CLI flags, and the registry /
AgentConfig shapes are interfaces. List every one you changed, or write "None".
See AGENTS.md for the trace-target, model-profile, and artifact contracts.
-->

None.

## Compatibility

<!--
Breaking change? Describe the config/CLI/Python API impact and the migration
path. While pre-1.0 a breaking change lands in a minor release. Otherwise
write "None".
-->

None.

## Checklist

- [ ] `uv run pre-commit run --all-files --show-diff-on-failure` passes
- [ ] `uv run pyrefly check` passes
- [ ] `uv run pytest` passes — or list the focused suites run under "Tests"
- [ ] Added or updated tests for the changed behavior
- [ ] Updated docs, READMEs, or example configs for user-visible changes
- [ ] Regenerated `uv.lock` if dependencies or `version` changed
- [ ] Commits carry a DCO `Signed-off-by` trailer (`git commit -s`)
- [ ] No credentials, local model registries, secret-bearing case files, or
  run artifacts included
- [ ] No AI assistance was used
- [ ] AI assistance was used, and a human reviewed the result
  <!-- Tick one of the two AI-assistance items. -->
