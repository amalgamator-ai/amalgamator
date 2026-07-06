# Branch Scanner

Scans a git repository to determine which branches are **missing a given bug fix** and need it backported.

Designed to feed directly into the [Amalgamator](../auto_bug_fix/) pipeline — the scanner tells you *where* the fix is needed, and Amalgamator applies it automatically.

---

## How it works

For each branch, checks run in order — the first conclusive result wins:

| Level | Function | What it detects |
|---|---|---|
| 1 | `git_is_ancestor()` | Exact fix commit already on branch |
| 2 | `forward_patch_id_check()` | Same patch content, different SHA (e.g. manual cherry-pick) |
| 3 | `backward_patch_id_check()` | Branch predates the bug — never affected, skip |
| 4 | File diff heuristic | 85%+ of the fix's added lines present in branch files |
| 5 *(optional)* | LLM semantic check | Claude resolves remaining `AMBIGUOUS` cases |

Levels 1–3 reuse existing functions from `auto_bug_fix` — no duplication.
Level 4 (file diff heuristic) is new and also fills a gap in Phase 0.

**Possible results per branch:**

| Status | Meaning |
|---|---|
| `✓ ALREADY FIXED (exact commit)` | Same SHA is in this branch's history |
| `✓ ALREADY FIXED (equivalent patch)` | Different SHA but identical patch content |
| `—  NOT AFFECTED` | Branch predates the bug — no fix needed |
| `~  LIKELY FIXED (file diff)` | Fix lines found via heuristic — verify manually |
| `✗  NEEDS FIX` | Fix is absent — backport needed |
| `?  AMBIGUOUS` | Inconclusive — enable LLM check |

---

## Prerequisites

```bash
# Clone the target repository locally (only needed once)
git clone https://github.com/cri-o/cri-o.git /path/to/cri-o

# Install amalgamator dependencies
cd amalgamator/
pip install -e .
```

---

## Quickstart

### Step 1 — Write a config file

```yaml
# my_scan.yaml
repository:
  local_path: "/path/to/cri-o"   # local clone of the repo

fix:
  commit: "d3b1036cc"            # SHA of the fix on the source branch (main)
  description: "oci: use shared lock for read-only stats collection"

  # Optional: SHA of the commit that INTRODUCED the bug.
  # Branches that predate this commit are marked NOT_AFFECTED (no false positives).
  # vuln_commit: "abc1234"

scan:
  # Auto-discover branches matching a pattern:
  branch_pattern: "release-1.3*"

  # OR specify branches explicitly:
  # branches:
  #   - "release-1.33"
  #   - "release-1.34"

llm:
  enabled: false   # set true to resolve AMBIGUOUS cases with Claude
  model: "claude-opus-4-6"
```

### Step 2 — Run the scanner

```bash
# From the amalgamator/ root directory:
python -m branch_scanner.run_scanner --config my_scan.yaml

# Save a JSON report as well:
python -m branch_scanner.run_scanner --config my_scan.yaml --output report.json

# Verbose logging:
python -m branch_scanner.run_scanner --config my_scan.yaml --verbose
```

### Step 3 — Use the output to drive backporting

Branches listed under **"NEEDS FIX"** are your backport targets. Feed them into
the Amalgamator `auto_bug_fix` YAML:

```yaml
# auto_bug_fix config generated from scan output
issue_id: "crio-shared-lock-d3b1036cc"
repository:
  source_path: "/path/to/cri-o"
  workdir: "/tmp/bugfix-workdir"
branches:
  source: "origin/main"
  target: "origin/release-1.33"   # ← one entry per NEEDS_FIX branch
fix:
  commit: "d3b1036cc"
build:
  command: "go build ./..."
  test_command: "go test ./..."
model: "claude-opus-4-6"
```

Or use `generate_config.py` to generate the YAML automatically — see below.

---

## Example output

Running against `cri-o` with commit `d3b1036cc` (fix merged to main + release-1.37):

```
======================================================================
  BRANCH SCANNER REPORT
======================================================================
  Repo:        /path/to/cri-o
  Fix commit:  d3b1036cc
  Description: oci: use shared lock for read-only stats collection
  Branches scanned: 8
======================================================================

  BRANCH          STATUS
  --------------  ----------------------------------------
  release-1.30    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.30
  release-1.31    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.31
  release-1.32    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.32
  release-1.33    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.33
  release-1.34    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.34
  release-1.35    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.35
  release-1.36    ✗  NEEDS FIX
                  → Fix not found on origin/release-1.36
  release-1.37    ✓  ALREADY FIXED (exact commit)
                  → Commit d3b1036cc is a direct ancestor of origin/release-1.37

----------------------------------------------------------------------
  NEEDS FIX    : 7 branch(es)
  ALREADY FIXED: 1 branch(es)
======================================================================

  Branches that NEED the fix:
    - release-1.30
    - release-1.31
    - release-1.32
    - release-1.33
    - release-1.34
    - release-1.35
    - release-1.36
```

---

## Auto-generate Amalgamator YAML (`generate_config.py`)

Instead of manually writing the `auto_bug_fix` YAML for each branch, use the
generator to do it automatically from scan results.

### Three modes

**`generate` — create YAML from scratch**

Scans all branches and generates one YAML per `NEEDS_FIX` branch:

```bash
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode generate \
  --output-dir ./configs/generated/
```

**`add` — append branches to an existing YAML**

Scans and adds any newly-discovered `NEEDS_FIX` branches to an existing config:

```bash
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode add \
  --existing-yaml ./configs/crio_fix.yaml
```

**`verify` — confirm existing YAML branches still need the fix**

Checks each target branch in an existing YAML and reports any that are already fixed:

```bash
python -m branch_scanner.generate_config \
  --config my_scan.yaml \
  --mode verify \
  --existing-yaml ./configs/crio_fix.yaml
```

---

## JSON report structure

```json
{
  "repo": "/path/to/cri-o",
  "fix_commit": "d3b1036cc",
  "fix_description": "oci: use shared lock for read-only stats collection",
  "summary": {
    "total": 8,
    "needs_fix": 7,
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
      "changed_files": ["server/container_server.go"]
    },
    {
      "branch": "release-1.37",
      "status": "ALREADY_FIXED",
      "method": "git_is_ancestor",
      "detail": "Commit d3b1036cc is a direct ancestor of origin/release-1.37",
      "changed_files": ["server/container_server.go"]
    }
  ]
}
```

---

## Example configs

See [`configs/examples/openssl_scan.yaml`](configs/examples/openssl_scan.yaml)
for a ready-to-use example with OpenSSL release branches.
