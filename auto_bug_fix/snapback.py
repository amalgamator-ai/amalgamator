"""Deterministic whitespace snap-back for resolved backports.

LLM conflict resolution can retype lines, drifting tabs-vs-spaces and
alignment even when the token content is right. Real backporters splice
bytes from the upstream commit or the target file instead of retyping them.
This pass restores that property after the fact: any resolved line whose
whitespace-collapsed token stream matches exactly one byte form in the
upstream post-image or the target pre-image is replaced with that source's
exact bytes.

The pass never invents content: replacements always come verbatim from one
of the two source texts, and ambiguous matches (the same token stream
appearing with more than one byte layout in a source) are left untouched.
"""

from __future__ import annotations

import os
import subprocess

from auto_bug_fix.mechanical import _is_ws_sensitive

# Keys shorter than this are too generic to snap safely (e.g. "break;",
# "return 0;" legitimately appear at many indentation depths).
MIN_KEY_LEN = 12

# Files larger than this (in lines) are skipped — huge files are never
# hand-formatted anyway.
MAX_SOURCE_LINES = 30_000


def normalize_ws(line: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return " ".join(line.split())


def _line_forms(text: str) -> dict[str, str | None]:
    """Map normalized token streams to the unique byte form producing them.

    A key produced by more than one distinct byte form is stored as None
    (ambiguous — never snapped).
    """
    lines = text.split("\n")
    if len(lines) > MAX_SOURCE_LINES:
        return {}
    forms: dict[str, str | None] = {}
    for line in lines:
        key = normalize_ws(line)
        if len(key) < MIN_KEY_LEN:
            continue
        if key in forms and forms[key] != line:
            forms[key] = None
        else:
            forms.setdefault(key, line)
    return forms


def snap_back_text(current: str, sources: list[str]) -> tuple[str, int]:
    """Snap drifted lines in ``current`` back to source bytes.

    ``sources`` is ordered by priority (upstream post-image first, then the
    target pre-image). Returns the new text and the number of replacements.
    """
    if not sources:
        return current, 0

    maps = [_line_forms(s) for s in sources]
    present: set[str] = set()
    for s in sources:
        present.update(s.split("\n"))

    out: list[str] = []
    snapped = 0
    for line in current.split("\n"):
        key = normalize_ws(line)
        # A line that already exists byte-for-byte in a source is never drift.
        if line in present or len(key) < MIN_KEY_LEN:
            out.append(line)
            continue
        replacement = None
        for m in maps:
            if key in m:
                # An ambiguous key in a higher-priority source ends the
                # search: we cannot know which layout is intended.
                replacement = m[key]
                break
        if replacement:
            out.append(replacement)
            snapped += 1
        else:
            out.append(line)

    return "\n".join(out), snapped


def _git_blob(repo_path: str, ref: str, path: str) -> str | None:
    """Return the text of ``ref:path``, or None if missing/binary/undecodable.

    Reads bytes and decodes strictly so CRLF endings and encoding problems
    are preserved/detected rather than silently normalized.
    """
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo_path,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    data = result.stdout
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def snap_back_files(
    repo_path: str,
    files: list[str],
    preimage_ref: str,
    upstream_commit: str,
) -> dict[str, int]:
    """Snap back every file in ``files`` against its two source texts.

    Edits the working tree in place. Returns {path: replacements} for files
    that changed.
    """
    counts: dict[str, int] = {}
    for f in files:
        abs_path = os.path.join(repo_path, f)
        # Never snap whitespace-significant files — indentation there is
        # semantics, not style, and mixing source layouts would corrupt them.
        if _is_ws_sensitive(f) or not os.path.isfile(abs_path):
            continue
        try:
            with open(abs_path, encoding="utf-8", newline="") as fh:
                current = fh.read()
        except (UnicodeDecodeError, OSError):
            continue

        sources = []
        upstream = _git_blob(repo_path, upstream_commit, f)
        if upstream is not None:
            sources.append(upstream)
        preimage = _git_blob(repo_path, preimage_ref, f)
        if preimage is not None:
            sources.append(preimage)
        if not sources:
            continue

        new_text, snapped = snap_back_text(current, sources)
        if snapped and new_text != current:
            with open(abs_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
            counts[f] = snapped
    return counts
