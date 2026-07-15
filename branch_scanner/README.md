# Branch Scanner

Scans a git repository to determine which branches are **missing a given bug fix** and need it backported.

Designed to feed directly into the [Amalgamator](../auto_bug_fix/) pipeline — the scanner tells you *where* the fix is needed, and Amalgamator applies it automatically.

---

## How it works

For each branch, checks run in order — the first conclusive result wins:

| Level | Method | What it detects |
|---|---|---|
| 1 | `git_is_ancestor()` | Exact fix commit already on branch |
| 2 | `forward_patch_id_check()` | Same patch content, different SHA (e.g. manual cherry-pick) |
| 3 | `backward_patch_id_check()` | Branch predates the bug — never affected |
| 4 | File diff heuristic | 85%+ of the fix's added lines present in branch files |
| 5 *(optional)* | LLM semantic check | Claude resolves remaining `AMBIGUOUS` cases |

**Possible results per branch:**

| Status | Meaning |
|---|---|
| `✓ ALREADY FIXED (exact commit)` | Same SHA is in this branch's history |
| `✓ ALREADY FIXED (equivalent patch)` | Different SHA but identical patch content |
| `—  NOT AFFECTED` | Branch predates the bug — no fix needed |
| `~  LIKELY FIXED` | Fix lines found via heuristic — verify manually |
| `✗  NEEDS FIX` | Fix is absent — backport needed |
| `?  AMBIGUOUS` | Inconclusive — enable LLM check |

---

## Prerequisites

```bash
# Clone the target repository locally (only needed once)
git clone https://github.com/<org>/<repo>.git /path/to/repo

# Install amalgamator dependencies
cd amalgamator/
pip install -e .
```

---

## Quickstart

### Step 1 — Write a scan config

```yaml
# my_scan.yaml
repository:
  local_path: "/path/to/repo"     # local clone of the repository

fix:
  commit: "abc1234"               # SHA of the fix on the source branch (main)
  description: "fix: short description of the bug fix"

  # Optional: SHA of the commit that introduced the bug.
  # Branches that predate this commit are marked NOT_AFFECTED.
  # vuln_commit: "def5678"

scan:
  # Auto-discover branches matching a pattern:
  branch_pattern: "release-1.*"

  # OR specify branches explicitly:
  # branches:
  #   - "release-1.33"
  #   - "release-1.34"

llm:
  enabled: false          # set true to resolve AMBIGUOUS cases with Claude
  model: "claude-opus-4-6"
  discover_commands: false  # set true to auto-discover build/test commands
```

### Step 2 — Run the scanner

```bash
# From the amalgamator/ root directory:
python -m branch_scanner.run_scanner --config my_scan.yaml

# Save a JSON report:
python -m branch_scanner.run_scanner --config my_scan.yaml --output report.json

# Verbose logging:
python -m branch_scanner.run_scanner --config my_scan.yaml --verbose
```

### Step 3 — Generate Amalgamator configs and run the pipeline

```bash
# Auto-generate one YAML config per NEEDS_FIX branch
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode generate \
  --output-dir ./generated/

# Run the Amalgamator pipeline for one branch
python -m auto_bug_fix.run_bug_fix --config ./generated/release-1-33.yaml

# Or run all generated configs in parallel
for f in ./generated/*.yaml; do
  python -m auto_bug_fix.run_bug_fix --config "$f" &
done
wait
```

> Each generated YAML targets one branch. The scanner produces one file per `NEEDS_FIX` branch automatically.

---

## Example output

```
======================================================================
  BRANCH SCANNER REPORT
======================================================================
  Repo:        /path/to/repo
  Fix commit:  abc1234
  Description: fix: short description of the bug fix
  Branches scanned: 8
======================================================================

  BRANCH          STATUS
  --------------  ----------------------------------------
  release-1.30    ✗  NEEDS FIX
                  → Fix lines not found in branch files
  release-1.31    ✗  NEEDS FIX
                  → Fix lines not found in branch files
  release-1.37    ✓  ALREADY FIXED (exact commit)
                  → Commit abc1234 is a direct ancestor of origin/release-1.37
  release-1.0     —  NOT AFFECTED  (branch predates the bug)

----------------------------------------------------------------------
  NEEDS FIX    : 6 branch(es)
  ALREADY FIXED: 1 branch(es)
  NOT AFFECTED : 1 branch(es) — predates the bug
======================================================================
```

---

## Auto-generate Amalgamator YAML (`generate_config.py`)

Three modes to bridge scanner output → `auto_bug_fix` pipeline configs:

**`generate` — create YAMLs from scratch**

```bash
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode generate \
  --output-dir ./generated/
```

**Auto-discovering build and test commands**

By default, `build.command` and `build.test_command` are left empty for you to fill in. Enable `discover_commands` to have the LLM read the repo's `README`, `Makefile`, CI workflows, and build files to suggest the right commands:

```yaml
llm:
  enabled: false
  model: "claude-opus-4-6"
  discover_commands: true   # LLM fills in build/test commands
```

The LLM reads: `README.md`, `Makefile`, `go.mod`, `pom.xml`, `pyproject.toml`, `setup.py`, `.github/workflows/*.yml`

> Always review discovered commands before running the pipeline.

**`add` — append newly-found branches to an existing YAML**

```bash
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode add \
  --existing-yaml ./configs/my_fix.yaml
```

**`verify` — confirm branches in an existing YAML still need the fix**

```bash
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode verify \
  --existing-yaml ./configs/my_fix.yaml
```

---

## JSON report structure

```json
{
  "repo": "/path/to/repo",
  "fix_commit": "abc1234",
  "fix_description": "fix: short description",
  "summary": {
    "total": 8,
    "needs_fix": 6,
    "already_fixed": 1,
    "ambiguous": 0,
    "errors": 0
  },
  "branches": [
    {
      "branch": "release-1.33",
      "status": "NEEDS_FIX",
      "method": "file diff heuristic",
      "detail": "Fix lines not found in branch files",
      "changed_files": ["path/to/changed_file.go"]
    },
    {
      "branch": "release-1.37",
      "status": "ALREADY_FIXED",
      "method": "git_is_ancestor",
      "detail": "Commit abc1234 is a direct ancestor of origin/release-1.37",
      "changed_files": ["path/to/changed_file.go"]
    }
  ]
}
```

---

## Example configs

See [`configs/examples/openssl_scan.yaml`](configs/examples/openssl_scan.yaml)
for a ready-to-use example with OpenSSL release branches.
