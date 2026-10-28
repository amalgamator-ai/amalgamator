# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate pull-request metadata in CI.

Two checks are available:

- ``dco`` verifies that every non-merge commit between the base and the PR
  head carries a ``Signed-off-by`` trailer matching the commit author.
- ``title`` verifies that the PR title starts with one of the repository's
  commit-type prefixes.

Both checks fail closed: a missing ref, an unusable commit range, or a git
failure is an error, never a pass. Pull requests opened by bots are skipped so
dependency and automation PRs do not need DCO trailers.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TITLE_RE = re.compile(r"(?:feat|fix|docs|test|chore)(?:\([^)]+\))?!?: \S.*")
IDENTITY_RE = re.compile(r"^\s*(?P<name>.*?)\s*<(?P<email>[^<>]+)>\s*$")


class CheckError(Exception):
    """A metadata check could not be completed."""


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run git and return stdout, converting any failure into CheckError."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CheckError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _rev_parse(ref: str) -> str:
    try:
        sha = _git("rev-parse", "--verify", "--end-of-options", ref).strip()
    except CheckError as exc:
        raise CheckError(f"cannot resolve {ref!r}: {exc}") from exc
    if not sha:
        raise CheckError(f"cannot resolve {ref!r}: git returned no commit")
    return sha


def _same_identity(left: str, right: str) -> bool:
    """Compare two ``Name <email>`` identities, ignoring email case."""
    left = left.strip()
    right = right.strip()
    left_match = IDENTITY_RE.match(left)
    right_match = IDENTITY_RE.match(right)
    if left_match and right_match:
        return (
            left_match.group("name").strip() == right_match.group("name").strip()
            and left_match.group("email").casefold()
            == right_match.group("email").casefold()
        )
    return left == right


def _is_bot(author: str) -> bool:
    return author.endswith("[bot]")


def check_dco(base_ref: str, head_ref: str, author: str) -> int:
    """Check that every non-merge PR commit has a matching DCO trailer."""
    if _is_bot(author):
        print(f"Skipping DCO check for bot author {author!r}.")
        return 0

    base = _rev_parse(base_ref)
    head = _rev_parse(head_ref)
    merge_base = _git("merge-base", base, head).strip()
    if not merge_base:
        raise CheckError(f"no common ancestor between {base_ref!r} and {head_ref!r}")

    commits = _git("rev-list", "--no-merges", f"{merge_base}..{head}").split()
    if not commits:
        raise CheckError(
            f"no non-merge commits between {base_ref!r} and {head_ref!r}; "
            "refusing to report success"
        )

    failures = 0
    for sha in commits:
        identity = _git("show", "-s", "--format=%an <%ae>", sha).strip()
        subject = _git("show", "-s", "--format=%s", sha).strip()
        signoffs = _git(
            "show", "-s", "--format=%(trailers:key=Signed-off-by,valueonly)", sha
        ).splitlines()
        if not identity or not any(
            _same_identity(identity, signoff) for signoff in signoffs
        ):
            print(
                f"error: commit {sha[:12]} ({subject}) is missing a "
                f"Signed-off-by matching {identity or 'the commit author'}",
                file=sys.stderr,
            )
            failures += 1

    if failures:
        print(
            f"Add the missing trailers with `git rebase --signoff {base}` "
            "or `git commit --amend -s`.",
            file=sys.stderr,
        )
        return 1

    print(f"All {len(commits)} non-merge commit(s) carry a matching Signed-off-by.")
    return 0


def check_title(title: str, author: str) -> int:
    """Check that the PR title uses a repository commit-type prefix."""
    if _is_bot(author):
        print(f"Skipping PR title check for bot author {author!r}.")
        return 0

    if not TITLE_RE.fullmatch(title.strip()):
        print(
            "error: PR title must start with one of "
            "feat|fix|docs|test|chore, optionally scoped and marked breaking, "
            f"followed by ': '. Got: {title!r}",
            file=sys.stderr,
        )
        return 1

    print(f"PR title looks good: {title.strip()!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Amalgamator pull-request metadata."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dco = subparsers.add_parser("dco", help="check DCO sign-off trailers")
    dco.add_argument("--base-ref", default="HEAD")
    dco.add_argument("--head-ref", default="HEAD")
    dco.add_argument("--author", default="")

    title = subparsers.add_parser("title", help="check the PR title format")
    title.add_argument("--title", required=True)
    title.add_argument("--author", default="")

    args = parser.parse_args(argv)

    try:
        if args.command == "dco":
            return check_dco(args.base_ref, args.head_ref, args.author)
        return check_title(args.title, args.author)
    except (CheckError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
