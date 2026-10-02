# External Resolver Context — template

Fill in and reference from the case YAML:

```yaml
external_context_file: <this-file>.md
```

Only `## Metadata`, `## Suspect findings` and `## Resolved files` are
machine-parsed; everything else is commentary passed through verbatim.
An empty `## Suspect findings` bullet list means "all checks passed"
(the hybrid would normally not escalate in that case).

---

# External Resolver Context

## Metadata
- resolver: synthmerge 0.3.15
- endpoint: Turnstile DeepSeek V4 Flash
- command: synthmerge -c /path/to/config.yaml --cache /path/to/cache --vibe --continue
- conflicts: 26 across 11 files
- retries: 7 relocation retries, 1 extra-conflict-lines escalation
- duration_s: 61.5
- checker: check-backport.sh (dead-symbol, undefined-symbol, makefile-obj)

## Suspect findings
Findings from the deterministic checks that triggered this escalation.
One bullet per finding, `path:line: reason` format preferred.

- fs/xfs/libxfs/xfs_defer.h:68: symbol 'dop_inodes' declared but has zero references in tree
- mm/memory-failure.c:1224: symbol 'num_poisoned_pages_dec' used but never defined
- arch/s390/kernel/Makefile:12: 'uv.o' in obj-y but no uv.c in directory

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
