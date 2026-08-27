#!/bin/bash
#
# Configure the upstream and downstream remotes for this repository.
#
# Upstream is the public GitHub repository; the two downstreams are internal
# Red Hat GitLab mirrors. Remotes live in .git/config, which is never carried
# by a clone, so every fresh clone needs this run once.
#
# Usage: ./script/setup_remotes.sh [options]
#
# Options:
#   --no-fetch  Configure the remotes but skip the initial fetch
#
# Idempotent: safe to re-run. Existing remotes are repointed, not duplicated.

set -euo pipefail

upstream_url="git@github.com:redhat-et/amalgamator.git"
downstream_a_url="git@gitlab.cee.redhat.com:hybrid-platforms-sec/amalgamation.git"
downstream_b_url="git@gitlab.cee.redhat.com:asangior/amalgamator.git"

main_branch="main"
downstream_branch="downstream-main"

do_fetch=true
for arg in "$@"; do
    case "$arg" in
        --no-fetch) do_fetch=false ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

git rev-parse --git-dir >/dev/null 2>&1 || {
    echo "Not a git repository." >&2
    exit 1
}

# Run from the repository root regardless of where the script was invoked.
cd "$(git rev-parse --show-toplevel)"

has_remote() { git remote get-url "$1" >/dev/null 2>&1; }

set_remote() {
    local name="$1" url="$2"
    if has_remote "$name"; then
        git remote set-url "$name" "$url"
        echo "  updated  $name"
    else
        git remote add "$name" "$url"
        echo "  added    $name"
    fi
}

# A fresh clone calls the GitHub remote "origin"; retire that name so all three
# remotes read consistently. Only rename when "upstream" is not already taken.
if has_remote origin && ! has_remote upstream; then
    git remote rename origin upstream
    echo "  renamed  origin -> upstream"
fi

echo "Configuring remotes:"
set_remote upstream "$upstream_url"
set_remote downstream-a "$downstream_a_url"
set_remote downstream-b "$downstream_b_url"

if [ "$do_fetch" = true ]; then
    echo "Fetching all remotes..."
    git fetch --all --quiet
fi

# main follows upstream, so a bare `git pull` / `git push` talks to GitHub.
if git show-ref --verify --quiet "refs/heads/$main_branch"; then
    git branch --set-upstream-to="upstream/$main_branch" "$main_branch" >/dev/null 2>&1 || true
fi

# The downstreams carry commits that must never reach upstream (see
# sync_downstreams.sh). Base the branch on downstream-a, which holds them.
if [ "$do_fetch" = true ] \
   && ! git show-ref --verify --quiet "refs/heads/$downstream_branch" \
   && git show-ref --verify --quiet "refs/remotes/downstream-a/$main_branch"; then
    git branch "$downstream_branch" "downstream-a/$main_branch"
    echo "  created  $downstream_branch (from downstream-a/$main_branch)"
fi

echo
git remote -v
