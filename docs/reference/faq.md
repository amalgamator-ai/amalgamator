# FAQ and troubleshooting

## Install and environment

**Which Python version?**
3.12+. Use `uv sync --python 3.12` to match CI's floor. Syntax must stay 3.12-
compatible even if you develop on newer interpreters.

**`uv sync` vs `pip install -e .`?**
`uv sync --locked` uses the committed `uv.lock` (what CI does). Editable pip
installs resolve from `pyproject.toml` without the lock. Prefer uv for
reproducible runs.

**Do tests need API keys?**
No. Live pipeline runs do.

## Models and auth

**Backport fails with auth / 401.**
Check the selected profile: key file path (cwd-relative), `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY`, and that `api_base` includes `/v1` for OpenAI-compatible
servers. See [Authentication](../authentication.md).

**GPU analysis ignores my `models.yaml`.**
Expected. Analyze/codegen use the Claude Agent SDK (or an `agent` JSON), not the
backport registry schema.

**`--models` path errors instead of falling back.**
By design. Explicit invalid registries fail closed.

**Scanner vs backport Claude defaults differ.**
Without a registry, scanner historically defaults to Opus; backport cases
default to Sonnet. Prefer an explicit profile ID.

## Backporting

**Cherry-pick conflicts every time.**
Normal for drifted trees. The pipeline tries mechanical apply before LLM
resolution. Read `dossier.md` for which files were LLM-touched and which audits
fired.

**RED passed but I expected failure.**
Phase 1.5 may conclude the target is not vulnerable, mitigated, or the ported
test is wrong. The dossier explains the stop/continue decision.

**Build/test skipped.**
Empty `build.command` / `build.test_command` skip those gates. Fill them in for
real verification.

**Where is the branch with my backport?**
In the worktree repository under a name like `amalgamator/<issue-id>`.

## GPU analysis / codegen

**Perfetto shows an empty or useless trace.**
Confirm you opened `single_trace_transformer_block.json` from a finished
single-trace run, and that the input trace format was supported. For
`.nsys-rep`, ensure `nsys` conversion succeeded.

**Cross-trace improvement targets the wrong side.**
Set `target_trace_id` explicitly; do not assume order means faster/slower.

**Codegen resume redoes finished work / skips needed work.**
Resume is artifact-based. Check `output_dir` for empty or partial versioned
files; delete or complete them intentionally before `--resume`.

**Runtime iterations never converge.**
Check GPU availability, baseline OFF identity, and whether correctness/perf
gates are failing for environmental reasons. Use investigate/fix entry points
after the main pipeline stops.

**PR flow won't push.**
Correct — network git/`gh` steps are human-run scripts with confirmation. The
agent is intentionally blocked from remotes.

## General

**Can Amalgamator merge my PR?**
No. Human review and merge only.

**Windows?**
Not supported.

**Security issues?**
Follow [SECURITY.md](../../SECURITY.md) — use GitHub security advisories, not
public issues, for vulnerabilities.

## Still stuck

1. Re-run with verbose logging where available (`--verbose` on the scanner;
   full `run_log.txt` for backport).
2. Confirm the matching package README for your pipeline.
3. Open a GitHub issue with the command, sanitized config (no keys), and
   expected vs actual behavior.
