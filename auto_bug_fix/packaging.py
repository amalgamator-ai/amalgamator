# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packaging-aware change classification for dist-git-style repositories.

Those checkouts often do not contain the source tree: source changes arrive as
patch files applied by a spec file.  A patch carrier is therefore part of the
effective change, not a production edit in its own right.
"""

from __future__ import annotations

import re

PATCH_SUFFIXES = (".patch", ".diff")
PACKAGING_NAMES = frozenset({"pkgbuild", "apkbuild"})


def is_patch_carrier(path: str) -> bool:
    """True for a file whose content is a unified diff."""
    return path.lower().endswith(PATCH_SUFFIXES)


def is_packaging_path(path: str) -> bool:
    """True for files that package source rather than being source."""
    lower = path.lower()
    parts = lower.split("/")
    return (
        is_patch_carrier(path)
        or lower.endswith(".spec")
        or parts[0] in {"debian", "rpm"}
        or parts[-1] in PACKAGING_NAMES
    )


def _clean_path(raw: str) -> str | None:
    path = raw.split("\t", 1)[0].strip()
    if len(path) > 1 and path[0] == path[-1] == '"':
        path = path[1:-1]
    if not path or path == "/dev/null":
        return None
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[2:]
    return path


def carrier_paths(content: str) -> list[str]:
    """Paths a unified diff touches, in first-seen order."""
    paths: list[str] = []
    for line in content.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = _clean_path(line[4:])
        elif line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            path = match.group(2) if match else None
        else:
            continue
        if path and path not in paths:
            paths.append(path)
    return paths


def carrier_payload(content: str, min_line_len: int) -> set[tuple[str, str]]:
    """The +/- payload lines of a unified diff, as (polarity, content)."""
    payload: set[tuple[str, str]] = set()
    for raw in content.splitlines():
        if raw.startswith(("+++", "---")) or not raw or raw[0] not in "+-":
            continue
        text = " ".join(raw[1:].split())
        if len(text) >= min_line_len:
            payload.add((raw[0], text))
    return payload
