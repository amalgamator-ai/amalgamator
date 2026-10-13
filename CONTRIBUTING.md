# Contributing to Amalgamator

Thanks for your interest in contributing. This guide covers how to set up a development environment and how to get a change merged.

## Questions and issues

Found a bug or have a question? Open an issue with what you ran, what you expected, and what happened.

## Development setup

You need Python 3.12+. [uv](https://docs.astral.sh/uv/) is recommended (CI uses it), but plain pip works.

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/amalgamator.git
cd amalgamator

# uv path (recommended: locked, reproducible)
uv sync --extra test --extra dev

# plain pip path
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[test,dev]"
```

See the [README](README.md) for an overview of the pipelines and where each one lives.

## Running the tests

```bash
uv run pytest
```

The tests live under `tests/`, in a directory per package (for example `tests/auto_bug_fix/` for `auto_bug_fix/`). Test files are named `test_*.py`. When you change behavior, add or update a test next to the existing ones for that package.

CI runs the full suite on a Python 3.12 / 3.14 matrix.

## Code quality

Before opening a pull request, make sure formatting, lint, and type checks pass — CI runs the same gates.

### Formatting and lint

Install the commit hooks once (black/ruff run hermetically — they need no project environment or uv, only git and pre-commit):

```bash
uv run pre-commit install        # or: pre-commit install (in an active env)
uv run pre-commit run --all-files
```

### Type checking

pyrefly needs the project environment (it resolves third-party imports), so it is a manual-stage pre-commit hook rather than a commit-time hook:

```bash
uv run pyrefly check
# or, via pre-commit:
uv run pre-commit run pyrefly --hook-stage manual --all-files
```

### Tool versions

- black/ruff: pinned revs in `.pre-commit-config.yaml` — the CI authority. CI runs `pre-commit run --all-files`, so it always uses exactly these.
- pyrefly: pinned by `uv.lock` — CI runs it from the locked environment.

### Dependency lockfile

`uv.lock` is committed intentionally: CI syncs with `--locked`, so every run tests the exact same resolved versions and fails loudly if the lock drifts from `pyproject.toml`. It only governs this repo's own environments — installing the project as a package (`pip install -e .`) resolves from `pyproject.toml` and never reads the lock. When adding or bumping dependencies, regenerate the lock (`uv add <pkg>` or `uv lock`) and commit it together with the `pyproject.toml` change. The file is marked `linguist-generated` in `.gitattributes`, so GitHub collapses it in diffs.

### Style notes

- The repo targets **Python 3.12+** (RHEL 10 and Ubuntu 24.04 LTS ship 3.12 as the system `python3`; RHEL 9 offers 3.12 via AppStream). Multi-exception clauses use parentheses, `except (A, B):`.
- Mass-reformat commits are listed in `.git-blame-ignore-revs` so `git blame` (and GitHub's blame UI) keeps pointing at the commit that changed the logic. Enable locally with `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

## Submitting a pull request

1. Create a branch from `main` in your fork.
2. Make your change. Keep it focused: one logical change per pull request is much easier to review than several at once.
3. Add or update tests, and update the documentation if behavior or configuration changes.
4. Run `uv run pytest` and the checks in [Code quality](#code-quality), and make sure they all pass.
5. Open a pull request against `main`. Describe what changed and why, and how you tested it.

### Commit messages

Start the subject line with a type prefix, as in the existing history:

| Prefix | Use for |
|--------|---------|
| `feat:` | A new feature |
| `fix:` | A bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or changing tests |
| `chore:` | Maintenance that does not change behavior |

Keep the subject short and write it in the imperative mood, for example `fix: handle BlockingIOError in Tee.write() for piped output`.

### Sign your work (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Every commit needs a `Signed-off-by: Your Name <you@example.com>` line, certifying that you wrote the change or otherwise have the right to submit it. The easiest way is to sign off while committing:

```bash
git commit -s
git commit --amend -s --no-edit   # add a forgotten sign-off to the last commit
```

## Releases and versioning

Releases follow [Semantic Versioning 2.0.0](https://semver.org/). While the version is `0.y.z`, the CLI, configuration format, and Python APIs may change in a minor release — pin a specific tag if you need stability. There is no fixed release cadence; a release is cut when enough meaningful change has accumulated on `main`.

- **Version source of truth:** `version` in `pyproject.toml`. The same version is recorded in `uv.lock`, so a version bump must regenerate the lock (`uv lock`) in the same change — CI's `uv sync --locked` step fails otherwise.
- **Cutting a release:** open a pull request that bumps `version` and regenerates `uv.lock`. Once it merges, tag the merge commit `vX.Y.Z` and publish a GitHub Release from the tag summarizing what changed. Release notes live on GitHub; there is no separate changelog file.
- **Support:** the latest release tag and `main`. The project does not maintain release branches of its own.

## Credentials and local files

Never commit API keys, tokens, or other credentials. Keep local model registries and case files that contain secrets out of your commits.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold it — please report unacceptable behavior to the contact listed there.

## License

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). By contributing, you agree that your contributions are licensed under the same terms.
