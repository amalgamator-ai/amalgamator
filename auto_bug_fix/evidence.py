# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic adaptation evidence for resolved backports.

Prompt instructions asking the LLM to "git grep every symbol" are followed
inconsistently. This module computes the evidence mechanically so the
post-resolution check receives facts instead of homework:

- **Token audit**: every identifier the backport adds is checked against the
  target branch. A token that exists upstream (before the fix) but not on the
  target is an upstream-only symbol that must be adapted; a token that exists
  in neither is likely hallucinated.
- **Substitution mining**: word-level pairing of the upstream-base vs target
  versions of the touched files yields downstream rename patterns
  (e.g. an API wrapper the downstream renamed). Unambiguous, repeated pairs whose
  upstream form does not exist on the target at all are applied mechanically
  to the lines the backport added; the rest are reported as evidence.
- **Follow-up scan**: upstream commits whose ``Fixes:`` tag points at the
  ported commit — downstream backports sometimes fold these in.
- **File checks**: an optional per-file command (e.g. a syntax-only compile)
  run over the touched files.

Everything here reads only the upstream history and the target branch — never
the eval ground truth or unrelated refs.
"""

from __future__ import annotations

import difflib
import os
import re
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass, field

from auto_bug_fix.git_tools import _run_git
from auto_bug_fix.snapback import _git_blob, normalize_ws

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Language keywords and ubiquitous identifiers — not worth auditing, they
# exist in every tree and only cost grep time.
STOPWORDS = frozenset(
    [
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "bool",
        "true",
        "false",
        "NULL",
        "nullptr",
        "u8",
        "u16",
        "u32",
        "u64",
        "s8",
        "s16",
        "s32",
        "s64",
        "__u8",
        "__u16",
        "__u32",
        "__u64",
        "__s8",
        "__s16",
        "__s32",
        "__s64",
        "size_t",
        "ssize_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uintptr_t",
        "intptr_t",
        "ptrdiff_t",
        "func",
        "var",
        "type",
        "range",
        "map",
        "chan",
        "interface",
        "package",
        "import",
        "defer",
        "select",
        "fallthrough",
        "nil",
        "err",
        "error",
        "string",
        "byte",
        "rune",
        "uint",
        "len",
        "cap",
        "new",
        "make",
        "append",
        "copy",
        "delete",
        "panic",
        "recover",
        "iota",
        "def",
        "class",
        "lambda",
        "None",
        "True",
        "False",
        "not",
        "and",
        "or",
        "elif",
        "except",
        "finally",
        "raise",
        "with",
        "yield",
        "assert",
        "async",
        "await",
        "pass",
        "del",
        "global",
        "nonlocal",
        "from",
        "try",
        "self",
        "this",
        "let",
        "function",
        "undefined",
        "null",
        "out",
        "ret",
        "val",
    ]
)

MAX_AUDIT_TOKENS = 200
MAX_FIX_DIFF_BYTES = 2_000_000


@dataclass
class TokenFinding:
    token: str
    status: str  # "upstream_only" | "unknown"
    files: list[str] = field(default_factory=list)
    # Set when every added occurrence sits inside a conditional-compilation
    # region whose gating config symbol is disabled on the target: the code
    # is dead there, and backporters keep upstream bytes verbatim in dead
    # regions instead of adapting them.
    dead_config: str | None = None
    # Set when the token may be generated on the target via macro
    # token-pasting (``##``).  The hint is the grep match line.
    generated_hint: str | None = None


@dataclass
class LineFinding:
    """A port line whose provenance violates three-way discipline."""

    file: str
    kind: str  # "removed_target_line" | "adapted_line" | "novel_line" | "ws_drift"
    line: str
    detail: str = ""
    dead_config: str | None = None


def _word_re(token: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")


def added_lines_by_file(diff_text: str) -> dict[str, list[str]]:
    """Extract the added lines of a unified diff, grouped by target path.

    A ``+++`` line is only a file header when it directly follows a ``---``
    header; anywhere else it is added content that happens to start with
    ``++`` (e.g. a committed patch file).
    """
    files: dict[str, list[str]] = {}
    current: str | None = None
    prev = ""
    for line in diff_text.split("\n"):
        if line.startswith("+++ ") and prev.startswith("--- "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = None if path == "/dev/null" else path
        elif line.startswith("+") and current:
            files.setdefault(current, []).append(line[1:])
        prev = line
    return files


def extract_tokens(lines: list[str]) -> set[str]:
    """Identifier-like tokens from a list of lines, minus stopwords."""
    tokens: set[str] = set()
    for line in lines:
        tokens.update(TOKEN_RE.findall(line))
    return tokens - STOPWORDS


def token_exists_on_ref(repo_path: str, ref: str, token: str) -> bool:
    """True if the token appears (word-bounded) anywhere in the ref's tree."""
    result = subprocess.run(
        ["git", "grep", "-q", "-w", "-F", "-e", token, ref],
        cwd=repo_path,
        capture_output=True,
    )
    return result.returncode == 0


def _ref_exists(repo_path: str, ref: str) -> bool:
    return _run_git(repo_path, ["rev-parse", "--verify", f"{ref}^{{commit}}"]).success


_CLASS_WITH_PARENTS_RE = re.compile(
    r"class\s+(\w+)\s*\(([^)]+)\)",
)

_CLASS_HEADER_RE = re.compile(
    r"^class\s+(\w+)\s*[\(:]",
)

_MAX_HIERARCHY_DEPTH = 5


def _find_parent_classes(
    repo_path: str,
    ref: str,
    class_name: str,
) -> list[str]:
    """Return the direct parent class names of *class_name* on *ref*."""
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            rf"class\s+{re.escape(class_name)}\s*\(",
            ref,
            "--",
            "*.py",
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    parents: list[str] = []
    for line in result.stdout.splitlines():
        m = _CLASS_WITH_PARENTS_RE.search(line)
        if m and m.group(1) == class_name:
            raw = m.group(2)
            for p in raw.split(","):
                p = p.strip().rsplit(".", 1)[-1]  # strip module prefix
                if p and p not in ("object",):
                    parents.append(p)
    return parents


def _class_defines_method(
    repo_path: str,
    ref: str,
    class_name: str,
    method: str,
) -> bool:
    """True if *class_name* directly defines *method* (``def method``) on *ref*."""
    result = subprocess.run(
        [
            "git",
            "grep",
            "-q",
            "-E",
            rf"^\s+def\s+{re.escape(method)}\s*\(",
            ref,
            "--",
            "*.py",
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return False
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-E",
            rf"^\s+def\s+{re.escape(method)}\s*\(",
            ref,
            "--",
            "*.py",
        ],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for fpath in result.stdout.strip().splitlines():
        fpath = fpath.split(":", 1)[-1] if ":" in fpath else fpath
        blob = _git_blob(repo_path, ref, fpath)
        if not blob:
            continue
        in_class = False
        for bline in blob.splitlines():
            cm = _CLASS_HEADER_RE.match(bline)
            if cm:
                in_class = cm.group(1) == class_name
            elif in_class and re.match(rf"^\s+def\s+{re.escape(method)}\s*\(", bline):
                return True
            elif in_class and re.match(r"^class\s", bline):
                in_class = False
    return False


def method_exists_on_class_hierarchy(
    repo_path: str,
    ref: str,
    class_name: str,
    method: str,
) -> bool:
    """True if *method* is available on *class_name* via direct definition or inheritance."""
    visited: set[str] = set()
    queue = [class_name]
    for _ in range(_MAX_HIERARCHY_DEPTH):
        if not queue:
            break
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if _class_defines_method(repo_path, ref, current, method):
            return True
        queue.extend(_find_parent_classes(repo_path, ref, current))
    return False


@dataclass
class TriageCorrection:
    """A factual correction to a triage prerequisite claim."""

    symbol: str
    original_reason: str
    correction: str


_NONEXISTENCE_RE = re.compile(
    r"does\s+not\s+exist|doesn'?t\s+exist|not\s+present|missing|absent"
    r"|was\s+removed|renamed\s+away|has\s+no\s|lacks?\s",
    re.IGNORECASE,
)


def _parse_dotted_symbol(symbol: str) -> tuple[str, str] | None:
    """Parse ``'ClassName.method_name'`` into (class, method).

    Returns None for degenerate inputs (empty parts, whitespace-only, etc.).
    """
    parts = symbol.split(".", 1)
    if len(parts) != 2:
        return None
    lhs_words = parts[0].strip().split()
    rhs_stripped = parts[1].strip().split("(")[0].strip()
    rhs_words = rhs_stripped.split()
    if not lhs_words or not rhs_words:
        return None
    return lhs_words[-1], rhs_words[0]


def _parse_bare_symbol(symbol: str) -> str | None:
    """Extract a single identifier from a free-form symbol string.

    Returns None for degenerate inputs.
    """
    stripped = symbol.strip().split("(")[0].strip()
    words = stripped.split()
    if not words:
        return None
    candidate = words[-1]
    return candidate if candidate else None


def validate_triage_prerequisites(
    repo_path: str,
    target_ref: str,
    prerequisites: list[dict],
) -> list[TriageCorrection]:
    """Verify each triage prerequisite claim against the target branch.

    For ``portable: false`` prerequisites whose reason claims a symbol doesn't
    exist on the target, check whether the symbol actually exists — including
    via inheritance for dotted ``Class.method`` symbols.

    Returns a list of corrections for claims that are factually wrong.
    """
    corrections: list[TriageCorrection] = []
    if not _ref_exists(repo_path, target_ref):
        return corrections

    for prereq in prerequisites:
        if not isinstance(prereq, dict):
            continue
        if prereq.get("portable", True):
            continue

        symbol = prereq.get("symbol", "")
        reason = prereq.get("reason", "")
        claims_nonexistence = bool(_NONEXISTENCE_RE.search(reason))

        if "." in symbol:
            parsed = _parse_dotted_symbol(symbol)
            if parsed is None:
                continue
            class_name, method = parsed

            if method_exists_on_class_hierarchy(
                repo_path,
                target_ref,
                class_name,
                method,
            ):
                if claims_nonexistence:
                    corrections.append(
                        TriageCorrection(
                            symbol=symbol,
                            original_reason=reason,
                            correction=(
                                f"{class_name}.{method} DOES exist on the target "
                                f"branch (via inheritance). The resolution agent "
                                f"should keep {class_name} in any isinstance/type "
                                f"checks."
                            ),
                        )
                    )
                else:
                    corrections.append(
                        TriageCorrection(
                            symbol=symbol,
                            original_reason=reason,
                            correction=(
                                f"{reason} [pipeline note: `{class_name}.{method}`"
                                f" does word-grep on the target tree — verify this"
                                f" claim before adapting]"
                            ),
                        )
                    )
        else:
            bare = _parse_bare_symbol(symbol)
            if not bare:
                continue
            if token_exists_on_ref(repo_path, target_ref, bare):
                if claims_nonexistence:
                    corrections.append(
                        TriageCorrection(
                            symbol=symbol,
                            original_reason=reason,
                            correction=(
                                f"{bare} exists on the target branch. The triage "
                                f"claim that it is missing is incorrect."
                            ),
                        )
                    )
                else:
                    corrections.append(
                        TriageCorrection(
                            symbol=symbol,
                            original_reason=reason,
                            correction=(
                                f"{reason} [pipeline note: `{bare}` does word-grep"
                                f" on the target tree — verify this claim before"
                                f" adapting]"
                            ),
                        )
                    )

    return corrections


def _fix_added_tokens(repo_path: str, fix_commit: str) -> set[str]:
    """Tokens introduced by the upstream fix commit's own added lines."""
    result = _run_git(repo_path, ["diff", f"{fix_commit}^1..{fix_commit}"])
    diff_text = result.stdout[:MAX_FIX_DIFF_BYTES]
    added = [
        line for lines in added_lines_by_file(diff_text).values() for line in lines
    ]
    tokens: set[str] = set()
    for line in added:
        tokens.update(TOKEN_RE.findall(line))
    return tokens


def _check_token_paste_hint(
    repo_path: str,
    ref: str,
    token: str,
) -> str | None:
    """Check if *token* could be generated on *ref* via ``##`` token-pasting.

    Strategy: look for a macro whose prefix matches the token's prefix and
    that uses ``##`` to paste a parameter as the suffix.  Also check for
    generator-style invocations (``TRACE_EVENT(tail)``, ``DEFINE_*(tail)``).
    Returns the first matching grep line, or None.
    """
    parts = token.split("_")
    for i in range(1, len(parts)):
        prefix = "_".join(parts[:i])
        tail = "_".join(parts[i:])
        if len(tail) < 2 or len(prefix) < 2:
            continue
        # Pattern 1: a #define that builds the prefix via ##
        # e.g.  #define MAKE(name) void helper_##name(void)
        # matches: prefix_## (the macro builds prefix_ then pastes)
        result = subprocess.run(
            [
                "git",
                "grep",
                "-n",
                "-E",
                rf"{re.escape(prefix)}_\s*##"
                rf"|##\s*{re.escape(tail)}"
                rf"|TRACE_EVENT\(\s*{re.escape(tail)}"
                rf"|DEFINE_\w+\(\s*{re.escape(tail)}",
                ref,
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    return None


def audit_added_tokens(
    repo_path: str,
    target_ref: str,
    fix_commit: str,
    added_by_file: dict[str, list[str]],
    max_tokens: int = MAX_AUDIT_TOKENS,
) -> list[TokenFinding]:
    """Audit every identifier the backport added against the target branch.

    Classification for a token missing from the target tree:
    - exists upstream before the fix        -> "upstream_only" (must adapt)
    - introduced by the fix commit itself   -> fine (its definition is ported)
    - in neither                            -> "unknown" (possible hallucination)
    """
    parent_ref = f"{fix_commit}^1"
    if not _ref_exists(repo_path, target_ref) or not _ref_exists(repo_path, parent_ref):
        return []

    token_files: dict[str, list[str]] = {}
    for path, lines in added_by_file.items():
        for token in extract_tokens(lines):
            token_files.setdefault(token, []).append(path)

    # In-memory tier: the target/parent versions of the touched files resolve
    # the overwhelming majority of tokens without a tree-wide grep.
    target_texts = "\n".join(
        _git_blob(repo_path, target_ref, f) or "" for f in added_by_file
    )
    parent_texts = "\n".join(
        _git_blob(repo_path, parent_ref, f) or "" for f in added_by_file
    )
    fix_added = _fix_added_tokens(repo_path, fix_commit)

    candidates = sorted(token_files, key=len, reverse=True)[:max_tokens]
    findings: list[TokenFinding] = []
    for token in candidates:
        word = _word_re(token)
        if word.search(target_texts):
            continue
        if token_exists_on_ref(repo_path, target_ref, token):
            continue
        in_parent = bool(word.search(parent_texts)) or token_exists_on_ref(
            repo_path, parent_ref, token
        )
        if in_parent:
            hint = _check_token_paste_hint(repo_path, target_ref, token)
            findings.append(
                TokenFinding(
                    token,
                    "upstream_only",
                    token_files[token],
                    generated_hint=hint,
                )
            )
        elif token not in fix_added:
            findings.append(TokenFinding(token, "unknown", token_files[token]))
    return findings


def mine_substitutions(
    repo_path: str,
    upstream_base_ref: str,
    target_ref: str,
    paths: list[str],
    min_count: int = 3,
) -> dict[str, str]:
    """Mine downstream rename patterns from the upstream-base vs target delta.

    Pairs lines that differ by exactly one token and returns
    {upstream_token: target_token} for pairs seen at least ``min_count`` times
    with a single consistent mapping.
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    for path in paths:
        result = _run_git(
            repo_path,
            ["diff", "-U0", f"{upstream_base_ref}:{path}", f"{target_ref}:{path}"],
        )
        if not result.success:
            continue
        removed: list[str] = []
        added: list[str] = []

        def flush(removed=removed, added=added) -> None:  # noqa: B023
            if removed and len(removed) == len(added):
                for old_line, new_line in zip(removed, added, strict=False):
                    pair = _single_token_diff(old_line, new_line)
                    if pair:
                        pair_counts[pair] += 1
            removed.clear()
            added.clear()

        for line in result.stdout.split("\n"):
            if line.startswith("@@"):
                flush()
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
        flush()

    by_old: dict[str, Counter[str]] = {}
    for (old, new), count in pair_counts.items():
        by_old.setdefault(old, Counter())[new] += count

    substitutions: dict[str, str] = {}
    for old, news in by_old.items():
        if len(news) == 1:
            new, count = next(iter(news.items()))
            if count >= min_count:
                substitutions[old] = new
    return substitutions


def _single_token_diff(old_line: str, new_line: str) -> tuple[str, str] | None:
    """Return (old_token, new_token) if the lines differ by exactly one token."""
    old_tokens = TOKEN_RE.findall(old_line)
    new_tokens = TOKEN_RE.findall(new_line)
    if len(old_tokens) != len(new_tokens) or not old_tokens:
        return None
    diffs = [(a, b) for a, b in zip(old_tokens, new_tokens, strict=False) if a != b]
    if len(diffs) != 1:
        return None
    old, new = diffs[0]
    if old in STOPWORDS or new in STOPWORDS:
        return None
    return old, new


def apply_substitutions(
    current: str,
    preimage: str,
    substitutions: dict[str, str],
) -> tuple[str, Counter[str]]:
    """Apply substitutions to the lines of ``current`` not in ``preimage``.

    Only lines the backport added or replaced are touched — pre-existing
    target lines are never rewritten. Returns (new_text, {old_token: hits}).
    """
    if not substitutions:
        return current, Counter()
    pre_lines = preimage.split("\n")
    cur_lines = current.split("\n")
    matcher = difflib.SequenceMatcher(None, pre_lines, cur_lines, autojunk=False)
    changed: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))

    applied: Counter[str] = Counter()
    for j in changed:
        line = cur_lines[j]
        for old, new in substitutions.items():
            line, hits = _word_re(old).subn(new, line)
            if hits:
                applied[old] += hits
        cur_lines[j] = line
    return "\n".join(cur_lines), applied


def port_line_changes(diff_text: str) -> dict[str, dict[str, list]]:
    """Per file: added lines with new-file line numbers, and removed lines.

    Returns {path: {"added": [(lineno, text)], "removed": [text]}}.
    """
    files: dict[str, dict[str, list]] = {}
    current: str | None = None
    new_lineno = 0
    prev = ""
    for line in diff_text.split("\n"):
        if line.startswith("+++ ") and prev.startswith("--- "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path == "/dev/null":
                # Whole-file deletion: track removals under the old path so
                # deleting a file the fix doesn't delete is still auditable.
                old = prev[4:].strip()
                if old.startswith("a/"):
                    old = old[2:]
                path = old
            current = None if path == "/dev/null" else path
            if current:
                files.setdefault(current, {"added": [], "removed": []})
        elif line.startswith("@@") and current:
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            new_lineno = int(m.group(1)) if m else 0
        elif current and line.startswith("+"):
            files[current]["added"].append((new_lineno, line[1:]))
            new_lineno += 1
        elif current and line.startswith("-") and not line.startswith("---"):
            files[current]["removed"].append(line[1:])
        elif current and not line.startswith("\\"):
            new_lineno += 1
        prev = line
    return files


# --- Conditional-compilation awareness -------------------------------------
#
# Backporters keep upstream bytes verbatim inside regions that are compiled
# out on the target; adapting dead code is wasted divergence. The scanner
# models only simple, single-symbol guards — anything more complex is
# conservatively treated as live code.

_DIRECTIVE_RE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(.*)$")
_SIMPLE_COND_RE = re.compile(
    r"^(?:defined\s*\(\s*(\w+)\s*\)|IS_ENABLED\s*\(\s*(\w+)\s*\)|(\w+))\s*$"
)


def conditional_guards(text: str) -> list[list[tuple[str | None, bool]]]:
    """For each line (0-based), the stack of (symbol, required_enabled) guards.

    A None symbol marks a condition too complex to model.
    """
    stack: list[tuple[str | None, bool]] = []
    guards: list[list[tuple[str | None, bool]]] = []
    for line in text.split("\n"):
        m = _DIRECTIVE_RE.match(line)
        if m:
            kind, rest = m.group(1), m.group(2).strip()
            if kind in ("ifdef", "ifndef"):
                sym = rest.split()[0] if rest else ""
                sym = sym if sym.isidentifier() else None
                stack.append((sym, kind == "ifdef"))
            elif kind == "if":
                cm = _SIMPLE_COND_RE.match(rest)
                sym = next((g for g in cm.groups() if g), None) if cm else None
                stack.append((sym, True))
            elif kind == "elif":
                if stack:
                    stack[-1] = (None, True)
            elif kind == "else":
                if stack:
                    sym, req = stack[-1]
                    stack[-1] = (sym, not req)
            elif kind == "endif":
                if stack:
                    stack.pop()
        guards.append(list(stack))
    return guards


_config_paths_cache: dict[tuple[str, str], dict[str, list[str]]] = {}
_config_state_cache: dict[tuple[str, str, str], bool | None] = {}


def target_config_state(repo_path: str, ref: str, symbol: str) -> bool | None:
    """Whether a config symbol is enabled on the target, from in-tree config
    fragments (files named after the symbol under a configs/ directory).

    True = enabled, False = disabled, None = unknown or inconsistent.
    """
    skey = (repo_path, ref, symbol)
    if skey in _config_state_cache:
        return _config_state_cache[skey]
    pkey = (repo_path, ref)
    if pkey not in _config_paths_cache:
        result = _run_git(repo_path, ["ls-tree", "-r", "--name-only", ref])
        by_name: dict[str, list[str]] = {}
        if result.success:
            for p in result.stdout.splitlines():
                if "/configs/" in p or p.startswith("configs/"):
                    by_name.setdefault(p.rsplit("/", 1)[-1], []).append(p)
        _config_paths_cache[pkey] = by_name
    enabled = disabled = 0
    for path in _config_paths_cache[pkey].get(symbol, [])[:12]:
        blob = _git_blob(repo_path, ref, path) or ""
        if re.search(rf"^# {re.escape(symbol)} is not set", blob, re.M):
            disabled += 1
        elif re.search(rf"^{re.escape(symbol)}=y", blob, re.M):
            enabled += 1
        # "=m" is deliberately unknown: the plain preprocessor macro is not
        # defined for modules, so #ifdef/#ifndef reasoning would be wrong.
    state = (
        True
        if (enabled and not disabled)
        else False if (disabled and not enabled) else None
    )
    _config_state_cache[skey] = state
    return state


def dead_config_symbol(
    frames: list[tuple[str | None, bool]],
    state_lookup,
) -> str | None:
    """The config symbol proving these guard frames are compiled out, if any."""
    for sym, required_enabled in frames:
        if sym is None:
            continue
        state = state_lookup(sym)
        if state is None:
            continue
        if required_enabled and state is False:
            return sym
        if not required_enabled and state is True:
            return sym
    return None


MAX_LINE_FINDINGS = 30
_TRIVIAL_KEY_LEN = 6

# Conflict-marker lines transiently present during resolution are process
# noise, never provenance evidence.
_MARKER_LINE_RE = re.compile(r"^(<{7}|={7}|>{7}|\|{7})( |$)")


def audit_line_provenance(
    repo_path: str,
    base_head: str,
    fix_commit: str,
    port_diff: str,
    current_texts: dict[str, str] | None = None,
    substitutions: dict[str, str] | None = None,
) -> list[LineFinding]:
    """Three-way discipline check on the port's added and removed lines.

    Every added line should carry either the upstream fix's bytes or the
    target's bytes; every removed line should be one the fix removes (counted
    with multiplicity). Violations are classified for the verification agent:
    ws_drift, adapted_line, novel_line, removed_target_line.

    ``substitutions`` describes the pipeline's own auto-applied token
    substitutions so they are not reported as violations.
    """
    fix_diff_full = _run_git(
        repo_path, ["diff", f"{fix_commit}^1..{fix_commit}"]
    ).stdout
    if len(fix_diff_full) > MAX_FIX_DIFF_BYTES:
        # A truncated fix diff would make faithfully-ported lines from the
        # cut-off files look novel; no audit is better than a wrong one.
        return []
    fix_changes = port_line_changes(fix_diff_full)
    port_changes = port_line_changes(port_diff)

    def apply_subs(text: str) -> str:
        for old, new in (substitutions or {}).items():
            text = _word_re(old).sub(new, text)
        return text

    def state_lookup(sym: str) -> bool | None:
        return target_config_state(repo_path, base_head, sym)

    findings: list[LineFinding] = []
    marker_findings: list[LineFinding] = []
    for f, changes in port_changes.items():
        if len(findings) >= MAX_LINE_FINDINGS:
            break
        upc = fix_changes.get(f, {"added": [], "removed": []})
        up_added = [t for _, t in upc["added"]]
        up_added_bytes = set(up_added)
        up_added_norm = {normalize_ws(t) for t in up_added}
        if substitutions:
            up_added_bytes.update(apply_subs(t) for t in up_added)
            up_added_norm.update(normalize_ws(apply_subs(t)) for t in up_added)
        up_tokens = [TOKEN_RE.findall(t) for t in up_added]
        up_removed_counts = Counter(normalize_ws(t) for t in upc["removed"])

        pre = _git_blob(repo_path, base_head, f) or ""
        pre_lines = set(pre.split("\n"))
        pre_norm = {normalize_ws(line) for line in pre_lines}

        cur = (current_texts or {}).get(f)
        guards = conditional_guards(cur) if cur else []

        def dead_at(lineno: int, guards=guards) -> str | None:  # noqa: B023
            if guards and 0 < lineno <= len(guards):
                return dead_config_symbol(guards[lineno - 1], state_lookup)
            return None

        # Committed conflict-marker lines are their own finding kind: an
        # unambiguous marker (<<<<<<< / >>>>>>> / |||||||) always counts; a
        # lone seven-equals line counts only alongside one of those, since
        # by itself it is legitimate content (doc underlines).
        added_markers = [
            (lineno, a)
            for lineno, a in changes["added"]
            if _MARKER_LINE_RE.match(a.strip())
        ]
        has_unambiguous = any(not a.strip().startswith("=") for _, a in added_markers)
        for _lineno, a in added_markers:
            if a.strip().startswith("=") and not has_unambiguous:
                continue
            marker_findings.append(
                LineFinding(
                    f,
                    "conflict_marker",
                    a,
                    detail="a leftover conflict marker was committed",
                )
            )

        for lineno, a in changes["added"]:
            if len(findings) >= MAX_LINE_FINDINGS:
                break
            key = normalize_ws(a)
            if len(key) < _TRIVIAL_KEY_LEN or a in up_added_bytes:
                continue
            if _MARKER_LINE_RE.match(a.strip()):
                continue  # reported above as conflict_marker findings
            if a in pre_lines:
                continue  # the target's own bytes — three-way OK
            if key in up_added_norm or key in pre_norm:
                findings.append(
                    LineFinding(f, "ws_drift", a, dead_config=dead_at(lineno))
                )
                continue
            a_tokens = TOKEN_RE.findall(a)
            best, best_line = 0.0, ""
            for text, tokens in zip(up_added, up_tokens, strict=False):
                if not tokens:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, a_tokens, tokens, autojunk=False
                ).ratio()
                if ratio > best:
                    best, best_line = ratio, text
            if best >= 0.5:
                findings.append(
                    LineFinding(
                        f,
                        "adapted_line",
                        a,
                        detail=best_line.strip(),
                        dead_config=dead_at(lineno),
                    )
                )
            else:
                findings.append(
                    LineFinding(f, "novel_line", a, dead_config=dead_at(lineno))
                )

        port_removed_counts = Counter(
            normalize_ws(t)
            for t in changes["removed"]
            if len(normalize_ws(t)) >= _TRIVIAL_KEY_LEN
            and not _MARKER_LINE_RE.match(t.strip())
        )
        for key, n in port_removed_counts.items():
            if len(findings) >= MAX_LINE_FINDINGS:
                break
            excess = n - up_removed_counts.get(key, 0)
            if excess > 0:
                sample = next(t for t in changes["removed"] if normalize_ws(t) == key)
                findings.append(
                    LineFinding(
                        f,
                        "removed_target_line",
                        sample,
                        detail=f"removed {n}x, the fix removes {up_removed_counts.get(key, 0)}x",
                    )
                )

        # Stale lines: the fix REPLACES a line the target also had, but the
        # port still carries the old bytes (an incompletely applied hunk —
        # invisible to added/removed auditing because nothing changed).
        if cur is not None:
            pre_counts = Counter(pre.split("\n"))
            cur_counts = Counter(cur.split("\n"))
            up_readd_counts = Counter(t for _, t in upc["added"])
            for r_line, r_n in Counter(upc["removed"]).items():
                if len(findings) >= MAX_LINE_FINDINGS:
                    break
                if len(
                    normalize_ws(r_line)
                ) < _TRIVIAL_KEY_LEN or _MARKER_LINE_RE.match(r_line.strip()):
                    continue
                had = pre_counts.get(r_line, 0)
                if not had:
                    continue  # the target never had these exact bytes
                # A fix that MOVES a line removes and re-adds the same bytes;
                # a faithful port legitimately still carries them.
                expected = max(0, had - r_n) + up_readd_counts.get(r_line, 0)
                if cur_counts.get(r_line, 0) > expected:
                    findings.append(
                        LineFinding(
                            f,
                            "stale_line",
                            r_line,
                            detail=f"the fix removes this line {r_n}x; the port still carries it",
                        )
                    )

    # Marker findings take priority over ordinary findings for the cap.
    return (marker_findings + findings)[:MAX_LINE_FINDINGS]


MAX_RESTORATIONS = 20


def find_pure_rename_restorations(
    repo_path: str,
    base_head: str,
    fix_commit: str,
    port_diff: str,
) -> dict[str, list[tuple[int, str, str]]]:
    """Port lines that are pure token-renames of the fix's added lines.

    A ported line qualifies when substituting its renamed tokens reproduces a
    fix-added line byte-for-byte, every renamed-away token exists in the
    target's version of THE SAME FILE (the rename copied target naming), and
    every fix-side token also exists in that same file's pre-image (tree-wide
    existence is not evidence of per-struct/per-signature validity — a
    same-file requirement is what separates a gratuitous rename from a hard
    adaptation). Lines whose fix counterpart is already ported elsewhere in
    the file are skipped: restoring would manufacture a duplicate statement.
    Returns {file: [(lineno, current_line, fix_line)]}.
    """
    fix_diff = _run_git(repo_path, ["diff", f"{fix_commit}^1..{fix_commit}"]).stdout
    if len(fix_diff) > MAX_FIX_DIFF_BYTES:
        return {}
    fix_changes = port_line_changes(fix_diff)
    port_changes = port_line_changes(port_diff)

    restorations: dict[str, list[tuple[int, str, str]]] = {}
    total = 0
    for f, changes in port_changes.items():
        up_added = [t for _, t in fix_changes.get(f, {"added": []})["added"]]
        if not up_added:
            continue
        up_added_set = set(up_added)
        up_tokens = [TOKEN_RE.findall(t) for t in up_added]
        pre = _git_blob(repo_path, base_head, f) or ""
        port_added_bytes = {t for _, t in changes["added"]}
        for lineno, a in changes["added"]:
            if total >= MAX_RESTORATIONS:
                break
            if (
                a in up_added_set
                or len(normalize_ws(a)) < _TRIVIAL_KEY_LEN
                or _MARKER_LINE_RE.match(a.strip())
            ):
                continue
            a_tokens = TOKEN_RE.findall(a)
            for fix_line, f_tokens in zip(up_added, up_tokens, strict=False):
                if len(f_tokens) != len(a_tokens) or f_tokens == a_tokens:
                    continue
                mapping: dict[str, str] = {}
                consistent = True
                for at, ft in zip(a_tokens, f_tokens, strict=False):
                    if at == ft:
                        continue
                    if mapping.setdefault(at, ft) != ft:
                        consistent = False
                        break
                if not consistent or not mapping:
                    continue
                rebuilt = a
                for at, ft in mapping.items():
                    rebuilt = _word_re(at).sub(ft, rebuilt)
                if rebuilt != fix_line:
                    continue
                if fix_line in port_added_bytes:
                    continue  # the fix line is already ported — no duplicates
                if any(not _word_re(at).search(pre) for at in mapping):
                    continue
                # Every restored token must exist in this same file's
                # pre-image, stopwords included — no tree-wide shortcuts.
                if any(not _word_re(ft).search(pre) for ft in mapping.values()):
                    continue
                restorations.setdefault(f, []).append((lineno, a, fix_line))
                total += 1
                break
    return restorations


def annotate_dead_config(
    token_findings: list[TokenFinding],
    port_changes: dict[str, dict[str, list]],
    current_texts: dict[str, str],
    state_lookup,
) -> None:
    """Mark token findings whose added occurrences all sit in compiled-out code."""
    guards_cache = {f: conditional_guards(text) for f, text in current_texts.items()}
    for finding in token_findings:
        if finding.status != "upstream_only":
            # An unknown/hallucinated token in a dead region is still wrong;
            # only upstream_only findings flip to keep-verbatim.
            continue
        word = _word_re(finding.token)
        dead_syms: set[str] = set()
        all_dead = True
        seen_any = False
        for f in set(finding.files):
            guards = guards_cache.get(f)
            for lineno, text in port_changes.get(f, {}).get("added", []):
                if not word.search(text):
                    continue
                seen_any = True
                sym = None
                if guards and 0 < lineno <= len(guards):
                    sym = dead_config_symbol(guards[lineno - 1], state_lookup)
                if sym:
                    dead_syms.add(sym)
                else:
                    all_dead = False
        if seen_any and all_dead and dead_syms:
            finding.dead_config = sorted(dead_syms)[0]


def scan_followup_fixes(
    repo_path: str,
    fix_commit: str,
    source_ref: str,
    limit: int = 10,
) -> list[str]:
    """Upstream commits whose ``Fixes:`` tag references the ported commit."""
    full_sha = _run_git(repo_path, ["rev-parse", fix_commit]).stdout.strip()
    if not full_sha:
        return []
    result = _run_git(
        repo_path,
        [
            "log",
            "--format=%h %s",
            f"--grep=Fixes: {full_sha[:7]}",
            f"{full_sha}..{source_ref}",
        ],
    )
    if not result.success:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()][:limit]


def run_file_checks(
    repo_path: str,
    files: list[str],
    command_template: str,
    timeout: int | None,
    max_files: int = 20,
) -> list[str]:
    """Run a per-file check command (``{file}`` placeholder) over the files.

    Returns one failure entry per failing file, with the output tail.
    """
    if "{file}" not in command_template:
        return [f"file_check_command has no {{file}} placeholder: {command_template}"]
    failures: list[str] = []
    for f in files[:max_files]:
        abs_path = os.path.join(repo_path, f)
        if not os.path.isfile(abs_path):
            continue
        cmd = command_template.replace("{file}", shlex.quote(abs_path))
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or 120,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{f}: check timed out")
            continue
        if result.returncode != 0:
            tail = (result.stdout + "\n" + result.stderr).strip()[-800:]
            failures.append(f"{f}: exit {result.returncode}\n{tail}")
    return failures


def render_evidence(
    findings: list[TokenFinding],
    applied_substitutions: dict[str, str],
    advisory_substitutions: dict[str, str],
    followup_fixes: list[str],
    file_check_failures: list[str],
    line_findings: list[LineFinding] | None = None,
    conflict_marker_files: list[str] | None = None,
    auto_restored: dict[str, int] | None = None,
) -> str:
    """Render the deterministic evidence as a prompt-ready report."""
    sections: list[str] = []

    if conflict_marker_files:
        lines = ["### CRITICAL: conflict markers committed"]
        lines += [
            f"- {f}: still contains <<<<<<</=======/>>>>>>> markers — resolve them now"
            for f in conflict_marker_files
        ]
        sections.append("\n".join(lines))

    if findings:
        lines = [
            "### Token audit (added identifiers checked against the target branch)"
        ]
        for f in findings:
            where = ", ".join(sorted(set(f.files))[:3])
            if f.dead_config:
                lines.append(
                    f"- `{f.token}` ({where}): not on the target branch, but every added "
                    f"occurrence is inside a region compiled out on the target "
                    f"(`{f.dead_config}` disabled) — dead code; keep the upstream fix's "
                    f"bytes verbatim, do NOT adapt."
                )
            elif f.status == "upstream_only" and f.generated_hint:
                lines.append(
                    f"- `{f.token}` ({where}): exists upstream BEFORE the fix but NOT as "
                    f"a literal identifier on the target branch — however it may be "
                    f"macro-generated on the target via token-pasting "
                    f"(`{f.generated_hint.strip()[:120]}`). Verify the generator before "
                    f"adapting; if the upstream form is produced by the generator, keep it."
                )
            elif f.status == "upstream_only":
                lines.append(
                    f"- `{f.token}` ({where}): exists upstream BEFORE the fix but NOT on "
                    f"the target branch — the target names or structures this differently; "
                    f"it MUST be adapted, UNLESS the identifier is generated on the target "
                    f"by macro token-pasting (e.g. `##` concatenation or a "
                    f"`TRACE_EVENT`/`DEFINE_*` invocation) — if so, grep for the generator "
                    f"and cite it."
                )
            else:
                lines.append(
                    f"- `{f.token}` ({where}): not found on the target branch, upstream "
                    f"before the fix, or in the fix's own additions — verify it is not a "
                    f"hallucinated identifier."
                )
        sections.append("\n".join(lines))

    if line_findings:
        lines = [
            "### Three-way discipline check (port lines vs the upstream fix and the target)"
        ]
        for lf in line_findings:
            text = lf.line.strip()[:100]
            if lf.kind == "removed_target_line":
                lines.append(
                    f"- {lf.file}: the port REMOVES `{text}` which the upstream fix does "
                    f"not remove ({lf.detail}) — restore the target's line unless it lies "
                    f"inside a region the fix rewrites."
                )
            elif lf.kind == "adapted_line":
                note = (
                    f" This line is compiled out on the target (`{lf.dead_config}` "
                    f"disabled) — dead code MUST keep the fix's bytes."
                    if lf.dead_config
                    else ""
                )
                lines.append(
                    f"- {lf.file}: `{text}` diverges from the fix's `{lf.detail[:100]}` — "
                    f"reproduce the fix's bytes unless a hard constraint (symbol missing "
                    f"in compiled code, signature/constant mismatch) forces the change.{note}"
                )
            elif lf.kind == "novel_line":
                lines.append(
                    f"- {lf.file}: `{text}` appears in neither the upstream fix nor the "
                    f"target file — remove it unless the fix cannot function without it."
                )
            elif lf.kind == "stale_line":
                lines.append(
                    f"- {lf.file}: `{text}` is a line the upstream fix REPLACES, but the "
                    f"port still carries the old bytes ({lf.detail}) — apply the fix's "
                    f"replacement for this line (adapted only under hard constraints)."
                )
            elif lf.kind == "conflict_marker":
                lines.append(
                    f"- {lf.file}: CRITICAL — `{text}` is a leftover conflict marker "
                    f"committed into the port; remove the marker and resolve the region."
                )
            else:  # ws_drift
                lines.append(
                    f"- {lf.file}: `{text}` differs only in whitespace from a fix/target "
                    f"line — use the fix's exact bytes if the fix contains this line, "
                    f"otherwise the target's."
                )
        sections.append("\n".join(lines))

    if auto_restored:
        lines = [
            "### Fix lines auto-restored (pure renames reverted to the fix's bytes)"
        ]
        lines += [f"- {f}: {n} lines" for f, n in sorted(auto_restored.items())]
        lines.append(
            "These ported lines had been token-renamed away from the fix; the fix's "
            "exact bytes were restored mechanically (every restored token exists in "
            "the target's version of the same file). Do not re-rename them for style "
            "— but if a restoration violates a hard constraint (a member, type, or "
            "signature absent from the surrounding code), correct it and cite the "
            "evidence."
        )
        sections.append("\n".join(lines))

    if applied_substitutions:
        lines = [
            "### Substitutions auto-applied (mined from the upstream/target file delta)"
        ]
        lines += [f"- `{o}` -> `{n}`" for o, n in sorted(applied_substitutions.items())]
        lines.append("Verify these landed correctly; they were applied mechanically.")
        sections.append("\n".join(lines))

    if advisory_substitutions:
        lines = [
            "### Downstream rename patterns (advisory — both forms exist on target)"
        ]
        lines += [
            f"- upstream `{o}` is usually `{n}` on the target"
            for o, n in sorted(advisory_substitutions.items())
        ]
        sections.append("\n".join(lines))

    if followup_fixes:
        lines = ["### Upstream follow-up fixes to the ported commit"]
        lines += [f"- {f}" for f in followup_fixes]
        lines.append(
            "Downstream backports sometimes fold these in. Do not port them here, but "
            "flag any resolved region that looks broken without one."
        )
        sections.append("\n".join(lines))

    if file_check_failures:
        lines = ["### Per-file check failures"]
        lines += [f"- {f}" for f in file_check_failures]
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
