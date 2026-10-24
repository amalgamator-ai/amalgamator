# Backport scanner

Finds which **branches, tags, and releases** are missing a given fix commit, and
can generate Amalgamator case files for those refs.

Full docs: [backport_scanner/README.md](../../backport_scanner/README.md).

## Prerequisites

- Local clone of the repository to scan
- Amalgamator install
- Model credentials for ambiguous/LLM steps (and for config generation) —
  [Authentication](../authentication.md)

```bash
git clone https://github.com/<org>/<repo>.git /path/to/repo
cd /path/to/amalgamator
uv sync
cp auto_bug_fix/configs/example_models.yaml models.yaml
```

## Write a scan config

Examples live in `backport_scanner/configs/examples/`
(`profile_scan.yaml`, `kubernetes_scan.yaml`, `openssl_scan.yaml`).

Minimal branch scan:

```yaml
repository:
  local_path: "/path/to/repo"

fix:
  commit: "abc1234"
  description: "fix: short description of the bug fix"
  # vuln_commit: "def5678"   # optional: older refs → NOT AFFECTED

scan:
  branch_pattern: "release-1.*"
  # tag_pattern: "v1.*"
  # branches: ["release-1.33"]
  # tags: ["v1.33.13"]

llm:
  model: m1
  discover_commands: false
```

## Detection cascade

For each ref, the scanner stops at the first conclusive result:

1. Exact ancestry of the fix commit
2. Equivalent patch-id already present
3. Bug-introducing commit absent → not affected (when `vuln_commit` given)
4. File-diff heuristic (≥85% of added fix lines present → likely fixed)
5. LLM semantic check for remaining ambiguous cases

Statuses you will see: already fixed (exact/equivalent), not affected, likely
fixed, needs fix, ambiguous. A **heuristic match is not proof of equivalence** —
verify before mass-backporting.

## Run

```bash
uv run python -m backport_scanner.run_scanner --config my_scan.yaml
uv run python -m backport_scanner.run_scanner --config my_scan.yaml --output report.json
uv run python -m backport_scanner.run_scanner --config my_scan.yaml --models /path/to/models.yaml
uv run python -m backport_scanner.run_scanner --config my_scan.yaml --verbose
```

## Generate backport cases

Bridge scanner results into `auto_bug_fix` configs:

```bash
uv run python -m backport_scanner.generate_config \
  --config my_scan.yaml \
  --mode generate \
  --output-dir ./generated/
```

Other modes (`add`, `verify`) are documented in the package README. Then run
cases with [the backport pipeline](backporting.md).
