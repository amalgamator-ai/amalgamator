# Contributing to Amalgamator

Thanks for your interest in contributing. This guide covers how to set up a development environment and how to get a change merged.

## Development setup

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

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

## Credentials and local files

Never commit API keys, tokens, or other credentials. Keep local model registries and case files that contain secrets out of your commits.

## License

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). By contributing, you agree that your contributions are licensed under the same terms.
