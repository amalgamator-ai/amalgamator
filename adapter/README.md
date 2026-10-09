# adapter/ — the hybrid backport adapter (vendored)

This subdirectory vendors the **hybrid backport adapter** so a single
amalgamator checkout can run the full hybrid flow with no other repos:

```
git cherry-pick -x <upstream>
  ├── conflicts ──► conflict resolver (PLUGGABLE; tested: synthmerge,
  │                  https://gitlab.com/aarcange/synthmerge)
  │                     │
  │                     ▼
  │            check-backport.py          deterministic gate (zero LLM):
  │             dead symbols · used-but-undefined · Makefile objs
  │                     │
  │          clean ──► ship the resolver's commit      (fast path)
  │                     │
  │          findings ──► external-context.md + case yaml ──► amalgamator
  │                     │                                (seeded escalation)
  └── clean ───────────┘          │
                                  ▼
                    B1 arbiter (driver-side, deterministic): re-run the
                    same scans + diff-shape rule — keep the amalgamator
                    result or reject it and fall back to the resolver's
```

| File | Role |
|---|---|
| `hybrid-backport.sh` | the driver: worktree + cherry-pick + resolver contract, gate invocation, escalation (case-yaml generation), stray-file stripping, B1 arbiter |
| `check-backport.py` | the deterministic checker / gate / arbiter scan core; emits the external-context Markdown the amalgamator consumes |

## Usage (from an amalgamator checkout)

`--amalgamator-root` defaults to the parent of this directory, so the
shortest working invocation is:

```bash
./adapter/hybrid-backport.sh \
    --repo /path/to/downstream/repo --upstream <fix-sha> \
    --workdir /var/hybrid/run --gitscratch /var/hybrid/scratch \
    --synthmerge-bin synthmerge --synthmerge-config endpoints.yaml \
    --synthmerge-cache /var/hybrid/lmdb \
    --python <python with amalgamator deps> \
    --model openai/<model-id> --api-base https://... --api-key-file key
```

Outcome branches in the source repo: `hybrid/<issue-id>` (fast path or
rejected escalation), `amalgamator/<issue-id>` (kept escalation),
`amalgamator-rejected/<issue-id>` (study only). Full CLI reference,
resolver-pluggability contract, and validation numbers: the adapter's
own README.

## Provenance

Canonical home: **adapter-synthmerge-amalgamator**
(gitlab.cee.redhat.com/asangior/adapter-amalgamator-synthmerge). This is
a vendored copy — re-sync it when the adapter changes:

```bash
cp <adapter-repo>/hybrid-backport.sh <adapter-repo>/check-backport.py .
```

Design documentation: `../docs/external-context.md` (the contract the
checker's Markdown satisfies and the amalgamator parses).
