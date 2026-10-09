# External Resolver Context — template

Fill in and reference from the case YAML:

```yaml
external_context_file: <this-file>.md
```

Only `## Metadata`, `## Suspect findings` and `## Resolved files` are
machine-parsed; everything else is commentary passed through verbatim.

Rules (see `docs/external-context.md` for the full contract):

- Findings are copied **verbatim** from the deterministic checker
  (`check-backport.py`) — no paraphrase, no softening. Agents treat them
  as mechanical ground truth.
- A clean run emits **no** bullet under `## Suspect findings` — a
  `- (none)` placeholder would parse as a phantom finding. (In the hybrid
  flow the driver would not escalate a clean run in the first place.)
- `## Resolved files` carries only files NOT named in a finding; flagged
  files are excluded (their prior content is noise for re-derivation).
- Findings may carry inline upstream-tree ground truth and advisory
  discipline ("predates the patch — do not remove…") — obey the
  instruction inside the finding, not just its existence.

---

# External Resolver Context

## Metadata
- resolver: synthmerge 0.3.15
- endpoint: Turnstile DeepSeek V4 Flash
- command: synthmerge -c /path/to/config.yaml --cache /path/to/cache --vibe --continue
- conflicts: 26 across 11 files
- retries: 7 relocation retries, 1 extra-conflict-lines escalation
- duration_s: 61.5
- checker: check-backport.py (dead-symbol, undefined-symbol, makefile-obj)

## Suspect findings
Findings from the deterministic checks that triggered this escalation.
One bullet per finding, `path:line: reason` format preferred.

- include/acpi/actbl1.h: symbol 'ACPI_HEST_CORRECTABLE' introduced by the patch but has zero references in tree; upstream tree still references it in: include/acpi/actbl1.h — the port likely dropped those uses
- drivers/xfs/scrub/listxattr.c:129: symbol 'xchk_xattr_find_leftmost_leaf' used in added line but not defined anywhere in the tree or base (possible hallucination or un-ported upstream helper); upstream tree defines/references it in: fs/xfs/scrub/attr.c — an un-ported upstream helper: port its definition from there
- mm/memory-failure.c:1224: symbol 'num_poisoned_pages_dec' used in added line but not defined anywhere in the tree or base (possible hallucination or un-ported upstream helper); not present in the upstream tree either — likely hallucinated: remove or replace the call
- arch/s390/kernel/Makefile: 'uv.o' listed but no uv.c or uv.S exists in arch/s390/kernel
- include/net/ipv6.h: 19 pre-existing declarations with zero references in tree (dead at base and after the patch: IP6_AUTO_FLOW_LABEL_FORCED, …) — structurally diverged area, downstream removed this header's users; verify the port keeps only what the fix needs; these symbols predate the patch — do not remove or replace them on this check's account

## Resolved files
Final per-file content/patches from the external resolver for files that
passed all checks. One `### <repo-relative-path>` header per file, each
followed by a fenced block with the resolved content or the per-file
unified diff.

### fs/xfs/libxfs/xfs_attr.c
```diff
diff --git a/fs/xfs/libxfs/xfs_attr.c b/fs/xfs/libxfs/xfs_attr.c
...
```

### fs/xfs/libxfs/xfs_defer.c
```diff
diff --git a/fs/xfs/libxfs/xfs_defer.c b/fs/xfs/libxfs/xfs_defer.c
...
```
