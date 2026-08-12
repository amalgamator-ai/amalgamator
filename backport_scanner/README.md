# Backport Scanner

Scans a git repository to determine which **branches, tags, and releases** are missing a given bug fix and need it backported.

Given a fix commit SHA, the scanner checks every target ref — whether it's a release branch (`release-1.33`), a stable release tag (`v6.12.25`), or a versioned distribution branch (`rhel-9.8.z`) — and reports whether the fix is already present, not applicable, or needs to be backported.

Designed to feed directly into the [Amalgamator](../auto_bug_fix/) pipeline — the scanner tells you *where* the fix is needed, and Amalgamator applies it automatically.

### What you can scan

| Ref type | Example | How to configure |
|---|---|---|
| Release branches | `release-1.33`, `release-1.34` | `branch_pattern: "release-1.*"` or `branches: [...]` |
| Stable release tags | `v6.1.120`, `v6.12.25` | `tag_pattern: "v6.*"` or `tags: [...]` |
| Distribution branches | `rhel-9.8.z`, `centos-stream-10/main` | `branches: [...]` |
| Any git ref | Mix of branches and tags | Combine `branch_pattern` + `tag_pattern` in one config |

---

## How it works

### Detection flowchart

For each ref, the scanner runs a 5-level cascade. It stops at the first conclusive result:

```mermaid
flowchart TD
    START([Start: fix commit + target ref]) --> FILES[Identify changed files in fix commit]
    FILES --> L1{Level 1: Is fix commit<br/>an ancestor of this ref?}

    L1 -- Yes --> EXACT[✓ ALREADY FIXED<br/>exact commit]
    L1 -- No --> L2{Level 2: Does any commit on<br/>this ref have the same patch-id?}

    L2 -- Yes --> EQUIV[✓ ALREADY FIXED<br/>equivalent patch]
    L2 -- No --> L3{Level 3: Does the bug-introducing<br/>commit exist on this ref?}

    L3 -- "No (ref predates bug)" --> NOTAFF[— NOT AFFECTED<br/>predates the bug]
    L3 -- "Yes / no vuln_commit given" --> L4{Level 4: Are 85%+ of the fix's<br/>added lines present in ref's files?}

    L4 -- "≥85% present" --> LIKELY[~ LIKELY FIXED<br/>file diff heuristic]
    L4 -- "0% present" --> NEEDS[✗ NEEDS FIX]
    L4 -- "Partial / inconclusive" --> L5{Level 5: LLM semantic check<br/>enabled?}

    L5 -- Yes --> LLM[Claude analyzes fix diff<br/>vs. ref file contents]
    LLM --> LLM_FIXED[✓ FIXED]
    LLM --> LLM_NEEDS[✗ NEEDS FIX]
    LLM --> LLM_UNC[? UNCERTAIN]
    L5 -- No --> AMB[? AMBIGUOUS<br/>recommend LLM check]

    style EXACT fill:#2d6a4f,color:#fff
    style EQUIV fill:#2d6a4f,color:#fff
    style LIKELY fill:#52796f,color:#fff
    style NOTAFF fill:#6c757d,color:#fff
    style NEEDS fill:#d32f2f,color:#fff
    style AMB fill:#f9a825,color:#000
    style LLM_FIXED fill:#2d6a4f,color:#fff
    style LLM_NEEDS fill:#d32f2f,color:#fff
    style LLM_UNC fill:#f9a825,color:#000
```

### Detection levels

For each ref (branch or tag), a 5-level detection cascade runs in order — the first conclusive result wins:

| Level | Method | What it detects |
|---|---|---|
| 1 | `git_is_ancestor()` | Exact fix commit already reachable from ref |
| 2 | `forward_patch_id_check()` | Same patch content, different SHA (e.g. manual cherry-pick) |
| 3 | `backward_patch_id_check()` | Ref predates the bug — never affected |
| 4 | File diff heuristic | 85%+ of the fix's added lines present in ref's files |
| 5 *(optional)* | LLM semantic check | Claude resolves remaining `AMBIGUOUS` cases |

**Possible results per ref:**

| Status | Meaning |
|---|---|
| `✓ ALREADY FIXED (exact commit)` | Same SHA is reachable from this ref |
| `✓ ALREADY FIXED (equivalent patch)` | Different SHA but identical patch content |
| `—  NOT AFFECTED` | Ref predates the bug — no fix needed |
| `~  LIKELY FIXED` | Fix lines found via heuristic — verify manually |
| `✗  NEEDS FIX` | Fix is absent — backport needed |
| `?  AMBIGUOUS` | Inconclusive — LLM semantic check will resolve |

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

**Scan branches only:**

```yaml
# scan_branches.yaml
repository:
  local_path: "/path/to/repo"

fix:
  commit: "abc1234"
  description: "fix: short description of the bug fix"

scan:
  branch_pattern: "release-1.*"    # auto-discover matching branches

llm:
  model: "claude-opus-4-6"
```

**Scan release tags only:**

```yaml
# scan_tags.yaml
repository:
  local_path: "/path/to/linux"

fix:
  commit: "8a8a9fac9efa"
  description: "fix(net/core): CVE-2026-43208"

scan:
  tag_pattern: "v6.*"              # auto-discover matching tags

llm:
  model: "claude-opus-4-6"
```

**Scan both branches and tags together:**

```yaml
# scan_all.yaml
repository:
  local_path: "/path/to/repo"

fix:
  commit: "abc1234"
  description: "fix: short description of the bug fix"
  # vuln_commit: "def5678"         # optional: refs predating this are marked NOT_AFFECTED

scan:
  branch_pattern: "release-1.*"    # branches matching this glob
  tag_pattern: "v1.*"              # tags matching this glob
  # OR list explicitly:
  # branches: ["release-1.33", "release-1.34"]
  # tags: ["v1.33.13", "v1.34.9"]

llm:
  model: "claude-opus-4-6"
  discover_commands: false          # set true to auto-discover build/test commands
```

> You can scan **branches only**, **tags only**, or **both** in a single config. The scanner runs the same 5-level cascade on each ref.

### Step 2 — Run the scanner

```bash
# From the amalgamator/ root directory:
python -m backport_scanner.run_scanner --config my_scan.yaml

# Save a JSON report:
python -m backport_scanner.run_scanner --config my_scan.yaml --output report.json

# Verbose logging:
python -m backport_scanner.run_scanner --config my_scan.yaml --verbose
```

### Step 3 — Generate Amalgamator configs and run the pipeline

See the [Auto-generate Amalgamator YAML](#auto-generate-amalgamator-yaml-generate_configpy) section below for `generate`, `add`, and `verify` modes.

---

## Example output

```
======================================================================
  BACKPORT SCANNER REPORT
======================================================================
  Repo:        /path/to/repo
  Fix commit:  abc1234
  Description: fix: short description of the bug fix
  Refs scanned: 4 branch(es) + 3 tag(s)
======================================================================

  REF                STATUS
  -----------------  ----------------------------------------
  release-1.30       ✗  NEEDS FIX
                     → Fix lines not found in origin/release-1.30 files
  release-1.31       ✗  NEEDS FIX
                     → Fix lines not found in origin/release-1.31 files
  release-1.37       ✓  ALREADY FIXED (exact commit)
                     → Commit abc1234 is reachable from origin/release-1.37
  release-1.0        —  NOT AFFECTED  (predates the bug)
  [tag] v6.1.120     ✗  NEEDS FIX
                     → Fix lines not found in v6.1.120 files
  [tag] v6.6.87      ✓  ALREADY FIXED (exact commit)
                     → Commit abc1234 is reachable from v6.6.87
  [tag] v6.12.25     ✓  ALREADY FIXED (equivalent patch)

----------------------------------------------------------------------
  NEEDS FIX    : 3 ref(s)
  ALREADY FIXED: 3 ref(s)
  NOT AFFECTED : 1 ref(s) — predates the bug
======================================================================
```

---

## Auto-generate Amalgamator YAML (`generate_config.py`)

Three modes to bridge scanner output → `auto_bug_fix` pipeline configs:

**`generate` — create YAMLs from scratch**

```bash
python -m backport_scanner.generate_config \
  --config my_scan.yaml \
  --mode generate \
  --output-dir ./generated/
```

**Auto-discovering build and test commands**

By default, `build.command` and `build.test_command` are left empty for you to fill in. Enable `discover_commands` to have the LLM read the repo's `README`, `Makefile`, CI workflows, and build files to suggest the right commands:

```yaml
llm:
  # LLM is always used for ambiguous cases (Level 5)
  model: "claude-opus-4-6"
  discover_commands: true   # LLM fills in build/test commands
```

The LLM reads: `README.md`, `Makefile`, `go.mod`, `pom.xml`, `pyproject.toml`, `setup.py`, `.github/workflows/*.yml`

> Always review discovered commands before running the pipeline.

**`add` — append newly-found branches to an existing YAML**

```bash
python -m backport_scanner.generate_config \
  --config my_scan.yaml \
  --mode add \
  --existing-yaml ./configs/my_fix.yaml
```

**`verify` — confirm branches in an existing YAML still need the fix**

```bash
python -m backport_scanner.generate_config \
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
    "total": 7,
    "needs_fix": 3,
    "already_fixed": 3,
    "ambiguous": 0,
    "errors": 0
  },
  "refs": [
    {
      "name": "release-1.33",
      "ref_type": "branch",
      "status": "NEEDS_FIX",
      "method": "file diff heuristic",
      "detail": "Fix lines not found in origin/release-1.33 files",
      "changed_files": ["path/to/changed_file.go"]
    },
    {
      "name": "v6.1.120",
      "ref_type": "tag",
      "status": "NEEDS_FIX",
      "method": "file diff heuristic",
      "detail": "Fix lines not found in v6.1.120 files",
      "changed_files": ["path/to/changed_file.go"]
    },
    {
      "name": "release-1.37",
      "ref_type": "branch",
      "status": "ALREADY_FIXED",
      "method": "git_is_ancestor",
      "detail": "Commit abc1234 is reachable from origin/release-1.37",
      "changed_files": ["path/to/changed_file.go"]
    }
  ]
}
```

---

## Example configs

| Config | What it scans | Ref types |
|---|---|---|
| [`openssl_scan.yaml`](configs/examples/openssl_scan.yaml) | OpenSSL release branches | Branches |
| [`kubernetes_scan.yaml`](configs/examples/kubernetes_scan.yaml) | Kubernetes release branches | Branches |
