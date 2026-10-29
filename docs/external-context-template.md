# External Resolver Context — template

Fill in and reference from the case YAML:

```yaml
external_context_file: <this-file>.md
```

Only `## Metadata`, `## Suspect findings` and `## Resolved files` are
machine-parsed; everything else is commentary passed through verbatim.

Rules (see `docs/external-context.md` for the full contract):

- Findings are copied **verbatim** from the deterministic checker
  (`check-backport.py`) — no paraphrase, no softening — and carry a
  `[severity][class][evidence/confidence]` prefix. `strong` marks
  objective breakage classes, `advisory` correlation signals,
  `observed` facts, `inferred` heuristic conclusions. Untagged bullets
  are treated as advisory/inferred. The scans are text heuristics:
  agents verify them before acting, and a demonstrated false positive
  may be kept.
- A clean run emits **no** bullet under `## Suspect findings` — a
  `- (none)` placeholder would parse as a phantom finding. (In the hybrid
  flow the driver would not escalate a clean run in the first place.)
- `## Resolved files` carries only files NOT named in a finding; flagged
  files are excluded (their prior content is noise for re-derivation).
- Findings may carry inline upstream-tree observations and advisory
  discipline ("predates the patch — do not remove…") — treat these as
  evidence to verify, not as unconditional orders.
- `resolver:` in the Metadata names the prior resolver — `synthmerge`
  when its log format is recognized, `unknown` for anything else. The
  front-end is pluggable (tested with
  [synthmerge](https://gitlab.com/aarcange/synthmerge)); absent or
  unrecognized stdout degrades every Metadata field to `unknown`.

---

# External Resolver Context

## Metadata
- resolver: synthmerge
- endpoint: Turnstile DeepSeek V4 Flash
- command: synthmerge -c /path/to/config.yaml --cache /path/to/cache --vibe --continue
- conflicts: 26 across 11 files
- retries: 7 relocation retries, 1 extra-conflict-lines escalation
- duration_s: 61.5
- checker: check-backport.py (dead-symbol, undefined-symbol, makefile-obj)

## Suspect findings
Findings from the deterministic checks that triggered this escalation.
One bullet per finding, `path:line: reason` format preferred.

- [advisory][dead_symbol_introduced][inferred/low] include/acpi/actbl1.h: symbol 'ACPI_HEST_CORRECTABLE' introduced by the patch but has zero references in tree; upstream tree also contains it in: drivers/acpi/apei/hest.c — verify whether those are uses before assuming the port dropped them (limitations: text scan cannot see macro expansion, function pointers, or references built from tokens)
- [strong][undefined_symbol][observed/medium] drivers/xfs/scrub/listxattr.c:129: symbol 'xchk_xattr_find_leftmost_leaf' used in added line but not defined anywhere in the tree or base (possible hallucination or un-ported upstream helper); the upstream tree defines it in: fs/xfs/scrub/attr.c — port that definition (limitations: the call site is observed, but 'undefined' is inferred from a text search)
- [strong][undefined_symbol][observed/medium] mm/memory-failure.c:1224: symbol 'num_poisoned_pages_dec' used in added line but not defined anywhere in the tree or base (possible hallucination or un-ported upstream helper); not present in the upstream tree either — likely hallucinated: remove or replace the call
- [strong][makefile_missing_source][observed/high] arch/s390/kernel/Makefile: 'uv.o' listed but no uv.c or uv.S exists in arch/s390/kernel
- [advisory][dead_area][inferred/low] include/net/ipv6.h: 19 pre-existing declarations with zero references in tree (dead at base and after the patch: IP6_AUTO_FLOW_LABEL_FORCED, …) — structurally diverged area, downstream removed this header's users; verify the port keeps only what the fix needs; these symbols predate the patch — do not remove or replace them on this check's account

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
