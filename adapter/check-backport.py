#!/usr/bin/env python3
"""Deterministic backport checker + external-context generator.

Resolver-agnostic: scans a worktree where ANY conflict resolver — plain
git on a clean pick, a human, a mechanical applier, or an LLM — has
finished a cherry-pick. The only contract with the resolver is the git
commit it leaves behind; everything is derived from `git diff
base..HEAD` plus the worktree state. The resolver this was built and
tested with is synthmerge (https://gitlab.com/aarcange/synthmerge); its
captured stdout additionally feeds the Metadata provenance section
(optional — unrecognized or absent stdout degrades every field to
"unknown" rather than claiming a specific tool).

Scans (git/grep only, zero LLM cost):

1. Extract the resolver's diff vs the cherry-pick base (git diff).
2. Run deterministic scans:
   - dead symbols in touched headers (declared/defined, zero tree-wide refs)
   - used-but-undefined symbols (called in added lines, no definition in tree)
   - Makefile obj-* entries with no source file
3. Emit findings (exit 1 + findings on stdout) and, with --emit-external-context,
   write the external-context Markdown that seeds amalgamator
   (auto_bug_fix/external_context.py parses it).

Usage:
  ./check-backport.py --worktree <wt> --base <sha> [--upstream <sha>] \
      [--synthmerge-stdout <file>] [--ai-diff-out <file>] \
      [--emit-external-context <out.md>]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

KEYWORDS = {
    "if", "for", "while", "return", "sizeof", "struct", "enum", "union",
    "static", "const", "void", "int", "char", "bool", "long", "unsigned",
    "extern", "inline", "define", "include", "typedef", "volatile",
    "else", "switch", "case", "break", "continue", "goto", "do",
}


def sh(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------




def header_symbols(path_text):
    """Heuristic symbol extraction from a header (PoC quality)."""
    syms = set()
    for m in re.finditer(r"^\s*#\s*define\s+([A-Za-z_]\w+)", path_text, re.M):
        sym = m.group(1)
        # include guards (_LINUX_XATTR_H, _ASM_X86_XEN_HYPERCALL_H, __CXL_CORE_H__)
        if re.fullmatch(r"_+[A-Z0-9_]+_H_*", sym):
            continue
        syms.add(sym)
    for m in re.finditer(r"^[ \t]*[A-Za-z_][\w \t\*]*?[ \t\*]([a-z_][a-z0-9_]*)[ \t]*(?:\[[^\]]*\])?;", path_text, re.M):
        syms.add(m.group(1))
    for m in re.finditer(r"^[A-Za-z_][\w \t\*]*?\b([a-z_][a-z0-9_]*)\s*\(", path_text, re.M):
        syms.add(m.group(1))
    return syms


def _batched_refs(wt, patterns, treeish=None):
    """Symbol lookups; returns {(sym, path)}.

    treeish=None searches the working tree; otherwise that tree.

    Engine choice (measured on a kernel tree, case 1015's 711-symbol set):
      git grep -F -f N-patterns  ~44s per 50-pattern chunk — multi-pattern
        fixed search compiles a giant regex alternation; unusable.
      GNU grep -r -F -f          ~5s for ALL patterns (Aho-Corasick).
    So: GNU grep on the worktree; for a treeish, extract it once with
    `git archive | tar` and GNU grep that (small dead-sets use per-symbol
    git grep instead — no extraction cost).
    """
    if not patterns:
        return set()
    pats = sorted(set(patterns))
    pats_set = set(pats)
    refs = set()
    import tempfile, shutil

    def parse_gnu(output):
        for line in output.splitlines():
            if not line:
                continue
            path, _, sym = line.rpartition(":")
            if path.startswith("./"):
                path = path[2:]
            if sym in pats_set:
                refs.add((sym, path))

    with tempfile.NamedTemporaryFile("w", suffix=".pats", delete=False) as tf:
        tf.write("\n".join(pats) + "\n")
        patfile = tf.name
    try:
        if treeish is None:
            r = subprocess.run(
                ["grep", "-r", "-I", "-w", "-o", "-F", "-f", patfile,
                 "--exclude=.git", "--exclude=*.rej", "--exclude=*.orig",
                 "--exclude-dir=.git", "."],
                cwd=wt, capture_output=True, text=True)
            parse_gnu(r.stdout)
        elif len(pats) <= 15:
            # small set: per-symbol git grep (single pattern = fast kwset)
            for sym in pats:
                r = sh(["git", "grep", "-I", "-w", "-o", "-F", "-e", sym,
                        treeish, "--", "."], cwd=wt, check=False)
                for line in r.stdout.splitlines():
                    if not line:
                        continue
                    # "<tree>:<path>:<sym>"
                    rest = line.split(":", 1)[1] if ":" in line else line
                    path, _, s = rest.rpartition(":")
                    if s == sym:
                        refs.add((sym, path))
        else:
            # big set: extract the tree once, GNU grep it
            base_dir = tempfile.mkdtemp(prefix="basetree-", dir="/tmp")
            try:
                arch = subprocess.run(["git", "-C", wt, "archive", treeish],
                                      capture_output=True)
                if arch.returncode == 0:
                    subprocess.run(["tar", "-x", "-C", base_dir],
                                   input=arch.stdout, capture_output=True, check=True)
                    r = subprocess.run(
                        ["grep", "-r", "-I", "-w", "-o", "-F", "-f", patfile, "."],
                        cwd=base_dir, capture_output=True, text=True)
                    parse_gnu(r.stdout)
            finally:
                shutil.rmtree(base_dir, ignore_errors=True)
    finally:
        os.unlink(patfile)
    return refs


def scan_dead_symbols(wt, touched, findings, added=None, base=None, upstream=None):
    """Declarations in touched headers with zero references in the tree.

    Two finding classes:
      * STRONG (per symbol): the patch/resolution made the symbol dead —
        introduced by the patch's added lines; file absent at base
        (resurrected); or referenced outside the file at base and dead
        now (the patch removed its users — case 896's dop_inodes).
      * AREA SIGNAL (one per header, capped): N >= AREA_SIGNAL_MIN
        pre-existing declarations that were already dead at base and stay
        dead. Not an error by itself, but a patch touching a header whose
        symbols lost their users means downstream structurally diverged
        here — over-application risk (cases 6155/10024 were caught this
        way). Without the cap, firmware-interface headers fire 1000+
        findings (case 1015's cxgb4.h).

    Without *base* the attribution degrades: every dead symbol is a
    strong finding (legacy behavior).
    """
    AREA_SIGNAL_MIN = 3
    decls = {}  # sym -> declaring file
    for f in touched:
        if not f.endswith(".h"):
            continue
        path = os.path.join(wt, f)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        for sym in header_symbols(text):
            if len(sym) >= 4 and sym not in KEYWORDS:
                decls[sym] = f
    if not decls:
        return

    tree_refs = _batched_refs(wt, list(decls))
    dead = {sym: f for sym, f in decls.items()
            if not any(p != f for s, p in tree_refs if s == sym)}
    if not dead:
        return

    base_refs = _batched_refs(wt, list(dead), treeish=base) if base else set()
    # A2: upstream-tree ground truth for strong dead-symbol findings —
    # tells the resolver whether the symbol's users were dropped by the
    # port (upstream still references it) or the declaration is spurious.
    up_refs = _batched_refs(wt, list(dead), treeish=upstream) if upstream else set()
    added_lines = {f: [t for _ln, t in lines] for f, lines in (added or {}).items()}

    def _up_fact(sym):
        if not upstream:
            return ""
        hits = sorted({p for s, p in up_refs if s == sym})
        if hits:
            shown = ", ".join(hits[:3]) + ("…" if len(hits) > 3 else "")
            return f"; upstream tree still references it in: {shown} — the port likely dropped those uses"
        return "; upstream tree does not reference it either — the declaration is spurious, remove it"

    pre_existing_dead = {}
    for sym, f in sorted(dead.items()):
        if added_lines.get(f) and any(re.search(rf"\b{re.escape(sym)}\b", t) for t in added_lines[f]):
            findings.append(
                f"{f}: symbol '{sym}' introduced by the patch but has zero references in tree"
                f"{_up_fact(sym)}"
            )
            continue
        if not base:
            findings.append(
                f"{f}: symbol '{sym}' declared but has zero references in tree"
            )
            continue
        if sh(["git", "cat-file", "-e", f"{base}:{f}"], cwd=wt, check=False).returncode != 0:
            findings.append(
                f"{f}: symbol '{sym}' declared but has zero references in tree "
                f"(file absent at base — resurrected or introduced wholesale)"
                f"{_up_fact(sym)}"
            )
            continue
        if any(p != f for s, p in base_refs if s == sym):
            base_hits = sorted({p for s, p in base_refs if s == sym and p != f})[:3]
            findings.append(
                f"{f}: symbol '{sym}' declared but has zero references in tree "
                f"(referenced at base by: {', '.join(base_hits)} — the patch removed its users)"
                f"{_up_fact(sym)}"
            )
        else:
            pre_existing_dead.setdefault(f, []).append(sym)

    for f, syms in sorted(pre_existing_dead.items()):
        if len(syms) >= AREA_SIGNAL_MIN:
            shown = ", ".join(sorted(syms)[:10]) + ("…" if len(syms) > 10 else "")
            findings.append(
                f"{f}: {len(syms)} pre-existing declarations with zero references in tree "
                f"(dead at base and after the patch: {shown}) — structurally diverged "
                f"area, downstream removed this header's users; verify the port keeps "
                f"only what the fix needs; these symbols predate the patch — do not "
                f"remove or replace them on this check's account"
            )


def scan_undefined_symbols(wt, added, findings, base=None, upstream=None):
    """Symbols referenced in added lines that have no definition in tree.

    A symbol called in an added line is suspect when ALL of:
    (a) it has zero occurrences in the BASE tree (pre-patch) — an existing
        downstream function is fine, whatever the call looks like;
    (b) it is not defined by the patch itself — a definition/prototype in
        the ADDED lines (column-0 `type sym(...)` or `#define sym`) makes a
        new helper legitimate;
    (c) it has no references outside the calling file in the CURRENT tree
        (the AI's own added call sites live in that file — case 437:
        num_poisoned_pages_dec called 6x passed the scan before this).

    Lookups are batched (one git grep per phase): per-symbol greps made
    large driver patches (case 1015: ~200 called symbols) take minutes.
    """
    # index definition-patterns found in ADDED lines
    defined_in_patch = set()
    for f, lines in added.items():
        for lineno, text in lines:
            m = re.match(r"^[A-Za-z_][\w \t\*]*?\b([a-z_][a-z0-9_]*)\s*\(", text)
            if m:
                defined_in_patch.add(m.group(1))
            m = re.match(r"^\s*#\s*define\s+([A-Za-z_]\w+)", text)
            if m:
                defined_in_patch.add(m.group(1))

    called = {}  # sym -> (file, first call line)
    for f, lines in added.items():
        if not (f.endswith(".c") or f.endswith(".h")):
            continue
        for lineno, text in lines:
            for m in re.finditer(r"\b([a-z_][a-z0-9_]{5,})\s*\(", text):
                sym = m.group(1)
                if sym in KEYWORDS:
                    continue
                # Skip obvious control flow / common helpers
                if sym in ("printk", "pr_info", "pr_err", "pr_warn", "pr_debug", "pr_alert",
                           "BUG_ON", "WARN_ON", "ASSERT", "kmalloc", "kfree", "kzalloc",
                           "memcpy", "memset", "strlen", "strcmp", "strncmp", "container_of",
                           "list_for_each_entry", "list_add", "list_del", "spin_lock", "spin_unlock",
                           "mutex_lock", "mutex_unlock", "cpu_relax", "barrier", "return"):
                    continue
                called.setdefault(sym, (f, lineno))
    if not called:
        return

    suspects = [s for s in called if s not in defined_in_patch]
    if not suspects:
        return
    tree_refs = _batched_refs(wt, suspects)
    base_refs = _batched_refs(wt, suspects, treeish=base) if base else set()
    # A2: upstream-tree ground truth — distinguishes "un-ported upstream
    # helper" (definition exists upstream, port it from there) from
    # "hallucination" (not in upstream either — remove the call).
    up_refs = _batched_refs(wt, suspects, treeish=upstream) if upstream else set()

    for sym in sorted(suspects):
        f, lineno = called[sym]
        if any(s == sym for s, _ in base_refs):
            continue  # exists at base
        if any(s == sym and p != f for s, p in tree_refs):
            continue  # defined elsewhere in the current tree
        if upstream:
            hits = sorted({p for s, p in up_refs if s == sym})
            if hits:
                shown = ", ".join(hits[:3]) + ("…" if len(hits) > 3 else "")
                up_fact = (f"; upstream tree defines/references it in: {shown} — "
                           f"an un-ported upstream helper: port its definition from there")
            else:
                up_fact = ("; not present in the upstream tree either — likely "
                           "hallucinated: remove or replace the call")
        else:
            up_fact = ""
        findings.append(
            f"{f}:{lineno}: symbol '{sym}' used in added line but not defined "
            f"anywhere in the tree or base (possible hallucination or "
            f"un-ported upstream helper){up_fact}"
        )


def scan_makefile_objs(wt, touched, findings):
    for f in touched:
        if not os.path.basename(f) in ("Makefile", "Kbuild"):
            continue
        try:
            text = open(os.path.join(wt, f), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        d = os.path.dirname(f)
        for m in re.finditer(r"^\s*(?:obj-\S+\s*[:+]?=|[a-z0-9_-]+-objs\s*[:+]?=)\s*(.*)$", text, re.M):
            for token in m.group(1).split():
                if token.endswith(".o"):
                    base = token[:-2]
                    src = None
                    for ext in (".c", ".S"):
                        if os.path.exists(os.path.join(wt, d, base + ext)):
                            src = base + ext
                            break
                    if src is None and not token.startswith("$("):
                        findings.append(
                            f"{f}: '{token}' listed but no {base}.c or {base}.S exists in {d or '.'}"
                        )


# ---------------------------------------------------------------------------
# External-context emission
# ---------------------------------------------------------------------------

def emit_external_context(path, findings, resolved_patches, meta):
    lines = [
        "# External Resolver Context",
        "",
        "## Metadata",
        f"- resolver: {meta['resolver']}",
        f"- endpoint: {meta['endpoint']}",
        f"- command: {meta['command']}",
        f"- conflicts: {meta['conflicts']}",
        f"- retries: {meta['retries']}",
        f"- duration_s: {meta['duration_s']}",
        "- checker: check-backport.py (dead-symbol, undefined-symbol, makefile-obj)",
        "",
        "## Suspect findings",
        "Findings from the deterministic checks that triggered this escalation.",
        "",
    ]
    if findings:
        lines += [f"- {f}" for f in findings]
    # clean run: NO placeholder bullet — a bullet here would be parsed as a
    # finding by amalgamator's external_context parser
    lines += ["", "## Resolved files", ""]
    for f, patch in sorted(resolved_patches.items()):
        lines += [f"### {f}", "```diff", patch.rstrip("\n"), "```", ""]
    with open(path, "w") as out:
        out.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Deterministic backport checker + external-context generator")
    # Production mode: a live worktree where synthmerge just finished
    ap.add_argument("--worktree", required=True,
                    help="live worktree with the completed cherry-pick")
    ap.add_argument("--base", help="base commit the cherry-pick started from (prod mode)")
    ap.add_argument("--upstream", default=None,
                    help="upstream fix commit — enables upstream-tree ground-truth "
                         "facts in findings (where a suspect symbol lives upstream)")
    ap.add_argument("--synthmerge-stdout", metavar="FILE",
                    help="captured resolver stdout for the Metadata provenance "
                         "section (optional; unrecognized stdout degrades to "
                         "'unknown')")
    ap.add_argument("--ai-diff-out", metavar="FILE",
                    help="where to save the extracted ai.diff (prod mode)")
    ap.add_argument("--emit-external-context", metavar="OUT.md", default=None)
    ap.add_argument("--include-flagged-diffs", action="store_true",
                    help="include per-file diffs even for flagged files (default: exclude)")
    args = ap.parse_args()

    return run_production_mode(args)


def run_production_mode(args):
    """Scan the live worktree where synthmerge just finished.

    The caller (hybrid-backport.sh) owns the worktree, knows the base,
    and captured synthmerge's stdout.
    """
    wt = os.path.abspath(args.worktree)
    if not args.base:
        sys.exit("ERROR: --base is required with --worktree")
    base = args.base

    r = sh(["git", "-C", wt, "rev-parse", "--verify", "HEAD"], check=False)
    if r.returncode:
        sys.exit(f"ERROR: {wt} has no HEAD (cherry-pick incomplete?)")
    r = sh(["git", "-C", wt, "rev-parse", "--verify", base], check=False)
    if r.returncode:
        sys.exit(f"ERROR: base {base} not found in {wt}")

    # cherry-pick must be finished (synthmerge committed)
    r = sh(["git", "-C", wt, "status", "--porcelain"], check=False)
    unresolved = [l for l in r.stdout.splitlines() if l.startswith(("UU", "AA", "DU", "UD", "AU", "UA", "DD"))]
    if unresolved:
        sys.exit(f"ERROR: unresolved conflicts in {wt} — synthmerge did not finish")

    # No -C (copy detection reads/hashes the ENTIRE tree as copy sources —
    # minutes over a container bind mount for large patches, case 1015).
    # The -c overrides also beat the driver's inherited diff.renames=copies
    # + renameLimit=100000 env config. Plain rename detection is cheap and
    # keeps the MD's per-file diffs readable.
    ai_diff = sh(["git", "-C", wt, "-c", "diff.renames=true",
                  "-c", "diff.renameLimit=2000", "diff", "-D",
                  f"{base}..HEAD"]).stdout
    if not ai_diff.strip():
        print("cherry-pick produced no changes relative to base")
        findings = []
    else:
        if args.ai_diff_out:
            with open(args.ai_diff_out, "w") as f:
                f.write(ai_diff)
        findings = scan_worktree(wt, diff_text=ai_diff, base=base,
                                 upstream=args.upstream)
    for f in findings:
        print(f"  FINDING: {f}")

    exit_code = 0
    if args.emit_external_context:
        meta = extract_metadata(args.synthmerge_stdout, ai_diff)
        exit_code = write_context_and_report(
            args.emit_external_context, findings, ai_diff, meta,
            include_flagged=args.include_flagged_diffs)
    sys.exit(exit_code if args.emit_external_context else (1 if findings else 0))


def scan_worktree(wt, diff_text=None, diff_path=None, base=None, upstream=None):
    """Run all deterministic scans over a worktree. Returns findings list."""
    findings = []
    if diff_path:
        with open(diff_path) as f:
            diff_text = f.read()
    touched = parse_files_text(diff_text)
    added = added_lines_by_file_text(diff_text)
    scan_dead_symbols(wt, touched, findings, added=added, base=base, upstream=upstream)
    scan_undefined_symbols(wt, added, findings, base=base, upstream=upstream)
    scan_makefile_objs(wt, touched, findings)
    return findings


def extract_metadata(stdout_path, ai_diff_text):
    """Provenance for the external-context Metadata section.

    The resolver front-end is pluggable: with no recognizable resolver
    stdout, every field degrades to "unknown" rather than claiming a
    specific tool. The tested resolver is synthmerge
    (https://gitlab.com/aarcange/synthmerge); its stdout is recognized
    via the Assisted-by trailer and its log phrasing.
    """
    meta = {
        "resolver": "unknown",
        "endpoint": "unknown",
        "command": "unknown",
        "conflicts": "unknown",
        "retries": "unknown",
        "duration_s": "unknown",
    }
    if stdout_path and os.path.isfile(stdout_path):
        text = open(stdout_path, errors="replace").read()
        m = re.search(r"Found (\d+) conflicts to resolve", text)
        if m:
            meta["conflicts"] = f"{m.group(1)} across {len(set(parse_files_text(ai_diff_text)))} files"
        retries = len(re.findall(r"Will retry file", text))
        escal = re.findall(r"increased --extra-conflict-lines \((\d+ -> \d+)\)", text)
        meta["retries"] = f"{retries} relocation retries" + (
            f", extra-conflict-lines escalation ({escal[0]})" if escal else "")
        m = re.search(r"Total duration: ([\d.]+)s", text) or re.search(r"(\d+\.\d+)s wall", text)
        if m:
            meta["duration_s"] = m.group(1)
        m = re.search(r"Assisted-by: synthmerge \[([^\]]+)\]", text)
        if m:
            meta["resolver"] = "synthmerge"
            meta["command"] = "synthmerge -c <config> --cache <cache> --vibe --continue"
            meta["endpoint"] = m.group(1)
    return meta


def write_context_and_report(out_path, findings, ai_diff_text, meta, include_flagged=False):
    flagged_files = set()
    for f in findings:
        m = re.match(r"([^:]+):\d*", f)
        if m:
            flagged_files.add(m.group(1))
    resolved = {}
    for f, patch in split_diff_per_file_text(ai_diff_text).items():
        if f in flagged_files and not include_flagged:
            continue
        resolved[f] = patch
    emit_external_context(out_path, findings, resolved, meta)
    print(f"external context written: {out_path} "
          f"({len(findings)} findings, {len(resolved)} resolved files)")
    return 1 if findings else 0


def parse_files_text(diff_text):
    files = []
    for line in diff_text.splitlines():
        m = re.match(r"diff --git a/\S+ b/(.+)$", line)
        if m:
            files.append(m.group(1))
    return files


def added_lines_by_file_text(diff_text):
    out, current, newline = {}, None, 0
    for line in diff_text.splitlines():
        m = re.match(r"diff --git a/\S+ b/(.+)$", line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
        if m:
            newline = int(m.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.setdefault(current, []).append((newline, line[1:]))
            newline += 1
        elif not (line.startswith("-") and not line.startswith("---")):
            newline += 1
    return out


def split_diff_per_file_text(diff_text):
    out, cur_name, cur_lines = {}, None, []
    for line in diff_text.splitlines():
        m = re.match(r"diff --git a/\S+ b/(.+)$", line)
        if m:
            if cur_name:
                out[cur_name] = "\n".join(cur_lines)
            cur_name, cur_lines = m.group(1), [line]
        elif cur_name is not None:
            cur_lines.append(line)
    if cur_name:
        out[cur_name] = "\n".join(cur_lines)
    return out




if __name__ == "__main__":
    main()
