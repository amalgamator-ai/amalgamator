#!/bin/bash
#
# Sync both downstream mirrors with upstream.
#
# main tracks upstream (GitHub) and is never modified locally. downstream-main
# is main plus the downstream-only commits, and is what gets pushed to main on
# both GitLab remotes. The flow is:
#
#   upstream/main --(fast-forward)--> main --(merge)--> downstream-main --(push)--> downstream-a, downstream-b
#
# Usage: ./script/sync_downstreams.sh [options]
#
# Options:
#   --no-push   Merge locally but do not push to the downstreams
#   --dry-run   Report what would change; modify nothing
#   --only <a|b>  Push to only one downstream

set -euo pipefail

main_branch="main"
downstream_branch="downstream-main"
downstreams=(downstream-a downstream-b)

no_push=false
dry_run=false

while [ $# -gt 0 ]; do
    case "$1" in
        --no-push) no_push=true ;;
        --dry-run) dry_run=true ;;
        --only)
            shift
            case "${1:-}" in
                a) downstreams=(downstream-a) ;;
                b) downstreams=(downstream-b) ;;
                *) echo "--only takes 'a' or 'b'" >&2; exit 1 ;;
            esac
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

git rev-parse --git-dir >/dev/null 2>&1 || {
    echo "Not a git repository." >&2
    exit 1
}
cd "$(git rev-parse --show-toplevel)"

for branch in "$main_branch" "$downstream_branch"; do
    git show-ref --verify --quiet "refs/heads/$branch" || {
        echo "Missing local branch '$branch'. Run ./script/setup_remotes.sh first." >&2
        exit 1
    }
done

# Switching branches mid-sync would clobber uncommitted work.
if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is dirty. Commit or stash first." >&2
    exit 1
fi

starting_branch="$(git rev-parse --abbrev-ref HEAD)"
restore_branch() { git checkout --quiet "$starting_branch" 2>/dev/null || true; }

echo "Fetching upstream..."
git fetch --quiet upstream

incoming="$(git rev-list --count "$main_branch..upstream/$main_branch")"
echo "Upstream is $incoming commit(s) ahead of $main_branch."

if [ "$dry_run" = true ]; then
    [ "$incoming" -gt 0 ] && git log --oneline "$main_branch..upstream/$main_branch"
    echo "(dry run: nothing changed)"
    exit 0
fi

if [ "$incoming" -gt 0 ]; then
    git checkout --quiet "$main_branch"
    # Fast-forward only: main must stay a pristine mirror of upstream.
    git merge --ff-only "upstream/$main_branch"
    echo "Updated $main_branch."
fi

git checkout --quiet "$downstream_branch"

# Conflicts are expected when upstream touches a path the downstream removed;
# resolve with `git rm` and commit, then re-run.
if ! git merge --no-edit "$main_branch"; then
    echo >&2
    echo "Merge conflict in $downstream_branch. Resolve, run 'git commit'," >&2
    echo "then re-run this script to push." >&2
    exit 1
fi

if [ "$no_push" = true ]; then
    echo "Merged; skipping push (--no-push)."
    restore_branch
    exit 0
fi

# --no-verify skips the git-lfs pre-push hook. The four LFS pointers left in
# history have no objects on GitHub (they 404) and LFS is disabled on both
# GitLab projects, so the hook's batch request fails on a push it cannot help.
for remote in "${downstreams[@]}"; do
    echo "Pushing to $remote..."
    git push --no-verify "$remote" "$downstream_branch:$main_branch"
done

restore_branch

echo
echo "Done. Current heads:"
printf '  %-14s %s\n' "$main_branch" "$(git rev-parse --short "$main_branch")"
printf '  %-14s %s\n' "$downstream_branch" "$(git rev-parse --short "$downstream_branch")"
