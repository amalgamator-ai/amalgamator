"""Shared gen => review iteration infrastructure.

Used by both auto_analyze and auto_code_gen pipelines.
"""

import glob
import os
import re

CONVERGENCE_MARKER = "CONVERGED"


def prev_iteration_section(
    prev_output_file: str | None,
    prev_output_summary_file: str | None,
) -> str:
    if prev_output_file is None:
        return ""
    return f"""
<prev_output_file>
{prev_output_file}
</prev_output_file>
<prev_output_summary_file>
{prev_output_summary_file}
</prev_output_summary_file>

Read and analyze in-detail <prev_output_file> and <prev_output_summary_file> which contain the results from the previous iterations. Understand what was done, what issues were found and fixed, and what the iteration evolution looks like. The current attempt is done from scratch, but must incorporate all learnings from the previous iterations to avoid repeating mistakes and to build on what worked.
"""


def check_converged(output_dir: str, summary_file: str) -> bool:
    path = os.path.join(output_dir, summary_file)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        first_line = f.readline().strip()
    return first_line == CONVERGENCE_MARKER


def resolve_output_file(output_dir: str, canonical_filename: str) -> str:
    """Find the latest versioned output file for a given canonical filename.

    Looks for {prefix}_V{N}.txt files and returns the highest version.  When
    that version also has a non-empty {prefix}_review_V{N}.txt artifact, the
    reviewed artifact is authoritative and is returned instead of the draft.
    Falls back to the canonical filename for backward compatibility with
    older result directories.
    """
    prefix = canonical_filename.rsplit(".", 1)[0]
    ext = (
        "." + canonical_filename.rsplit(".", 1)[1] if "." in canonical_filename else ""
    )

    pattern = os.path.join(output_dir, f"{prefix}_V*{ext}")
    candidates = []
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        if "_review_" in basename or "_summary_" in basename:
            continue
        match = re.search(r"_V(\d+)" + re.escape(ext) + "$", basename)
        if match:
            candidates.append((int(match.group(1)), path))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        version, draft_path = candidates[0]
        review_path = os.path.join(output_dir, f"{prefix}_review_V{version}{ext}")
        if os.path.isfile(review_path) and os.path.getsize(review_path) > 0:
            return review_path
        return draft_path

    return os.path.join(output_dir, canonical_filename)
