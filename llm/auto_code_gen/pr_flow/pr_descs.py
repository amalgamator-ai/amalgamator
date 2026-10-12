"""Generate / refresh the PR title and description files.

commit_desc.txt (first line = PR title) and pr_desc.txt are the single source
of truth for the PR text: step 2 creates the PR from them, step 4 re-publishes
them whenever their sha256 no longer matches pr_state.descs_sha256. This module
keeps those two files fresh.

Run automatically at the end of every passing runtime iteration (rebase flow:
run_rebase_runtime_iters; review flow: run_pr_review_iters) and manually via

    python -m llm.auto_code_gen.run_pr_descs_update --config <code_gen.json>

The generation itself is the same agent step the normal pipeline uses
(UpdateCommitAndPRDescsPrompt), which already knows to update the files in
place rather than regenerate from scratch.
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os

from llm.auto_code_gen.agents import agent_run
from llm.auto_code_gen.code_gen_prompts import (
    PR_COMMANDS_DIR,
    PR_COMMIT_DESC_FILE,
    PR_PR_DESC_FILE,
    RUNTIME_DIR,
    RUNTIME_SUCCESS_FILE,
)


class PrDescsError(Exception):
    """No passing result to build a description from (or the agent wrote nothing)."""


def find_success_proof(output_dir):
    """Return a human-readable proof of success, or raise PrDescsError.

    The normal pipeline writes runtime/success_result.txt; the PR-flow runtime
    iterations write runtime/V<n>/verify_verdict.json per iteration. Either is
    enough -- but a rebase/review verdict only counts when overall == PASS.
    """
    success_file = os.path.join(output_dir, RUNTIME_DIR, RUNTIME_SUCCESS_FILE)
    if os.path.isfile(success_file):
        return success_file

    verdict = _latest_passing_verdict(output_dir)
    if verdict:
        return verdict

    raise PrDescsError(
        "no passing result found in {} (looked for {} and a {} with "
        "overall == PASS). Run the runtime iterations first.".format(
            os.path.join(output_dir, RUNTIME_DIR),
            RUNTIME_SUCCESS_FILE,
            os.path.join("V*", "verify_verdict.json"),
        )
    )


def _latest_passing_verdict(output_dir):
    """Newest verify_verdict.json under runtime/V*/ whose overall is PASS."""
    from llm.auto_code_gen.pr_flow import gates

    root = os.path.join(output_dir, RUNTIME_DIR)
    if not os.path.isdir(root):
        return None
    best = None
    for name in os.listdir(root):
        if not (name.startswith("V") and name[1:].isdigit()):
            continue
        candidate = os.path.join(root, name, gates.VERDICT_FILE)
        if not os.path.isfile(candidate):
            continue
        try:
            import json

            with open(candidate) as f:
                verdict = json.load(f)
        except OSError, ValueError:
            continue
        if verdict.get("overall") == "PASS":
            number = int(name[1:])
            if best is None or number > best[0]:
                best = (number, candidate)
    return best[1] if best else None


def _ensure_attribution(body_path):
    """The PR body always ends with the campaign's AI-assistance notice.

    The refresh agent rewrites pr_desc.txt from the prompt, which knows
    nothing about this footer -- append it here so every refresh keeps it.
    """
    line = (
        "AI-based Amalgamator was used to prepare this PR. " "Human reviewed changes."
    )
    try:
        with open(body_path) as f:
            body = f.read()
        if line in body:
            return
        with open(body_path, "a") as f:
            if body and not body.endswith("\n"):
                f.write("\n")
            f.write(f"\n{line}\n")
    except OSError as exc:
        print(f"  WARNING: could not append attribution to {body_path}: {exc}")


async def update_pr_descs(config, claude_config, use_case, client=None):
    """(Re)generate pr-generate/commit_desc.txt + pr_desc.txt.

    Creates the files if absent, updates them in place otherwise (the prompt's
    Step 0). Returns the two file paths. Never raises on agent failure --
    returns (None, None) after printing how to run it manually, so callers can
    use it as a best-effort post-PASS hook.
    """
    from common.claude_utils import PipelineStep
    from llm.auto_code_gen.code_gen_prompts import gen_UpdateCommitAndPRDescsPrompt

    output_dir = config.output_dir
    try:
        proof = find_success_proof(output_dir)
    except PrDescsError as exc:
        print(f"  PR descriptions NOT refreshed: {exc}")
        return None, None

    pr_dir = os.path.join(output_dir, PR_COMMANDS_DIR)
    desc_path = os.path.join(pr_dir, PR_COMMIT_DESC_FILE)
    body_path = os.path.join(pr_dir, PR_PR_DESC_FILE)
    desc_existed = os.path.isfile(desc_path)
    body_existed = os.path.isfile(body_path)

    context = use_case.create_context_str(claude_config, config)
    prompt = gen_UpdateCommitAndPRDescsPrompt(
        context=context, code_gen_output_dir=output_dir
    )
    await agent_run(
        claude_config,
        [
            PipelineStep(name="update_commit_and_pr_descs", prompt=prompt.prompt()),
        ],
        client=client,
    )

    missing = [p for p in (desc_path, body_path) if not os.path.isfile(p)]
    if missing:
        print(
            "  WARNING: PR description refresh wrote nothing "
            "({} missing). Run manually:".format(
                ", ".join(os.path.basename(p) for p in missing)
            )
        )
        print(
            "    python -m llm.auto_code_gen.run_pr_descs_update --config {}".format(
                getattr(config, "config_json_path", "<config.json>")
            )
        )
        return None, None

    _ensure_attribution(body_path)

    print(
        "  PR descriptions {} from {} ({} / {})".format(
            "updated" if (desc_existed or body_existed) else "generated",
            proof,
            PR_COMMIT_DESC_FILE,
            PR_PR_DESC_FILE,
        )
    )
    return desc_path, body_path


async def refresh_after_pass(config, claude_config, use_case, client=None):
    """Best-effort refresh of the description files after a PASS verdict.

    A description that lags the measurements would publish stale numbers, so
    the runtime-iteration runners call this right after a passing iteration.
    Never fatal: the verdict is already green, and the manual entrypoint
    (llm.auto_code_gen.run_pr_descs_update) can finish the job.
    """
    try:
        await update_pr_descs(config, claude_config, use_case, client=client)
    except Exception as exc:  # noqa: BLE001 -- the PASS verdict must survive this
        print(f"  WARNING: PR description refresh failed ({exc}). Run manually:")
        print(
            "    python -m llm.auto_code_gen.run_pr_descs_update --config {}".format(
                getattr(config, "config_json_path", "<config.json>")
            )
        )
