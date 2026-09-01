"""
Benchmark evaluation metrics.

Based on the evaluation criteria from:
  - JavaBackports (2026): TYPE-I…V complexity classification + semantic equivalence
  - BackportBench (2024): pass@k, semantic diff comparison, cost tracking

Metrics computed per case, then aggregated across the run:

  CORRECTNESS
  -----------
  semantic_equiv   (primary)  LLM judge: is the generated patch semantically
                              equivalent to the ground-truth backport?
                              Values: "yes" | "no" | "uncertain"

  exact_match      (secondary) Syntactic exact match after normalizing whitespace.
                               A subset of semantic_equiv=yes; rarely achievable.

  COMPLEXITY STRATIFICATION
  -------------------------
  Metrics are always reported broken down by JavaBackports TYPE (I–V):
    TYPE-I   Direct cherry-pick, no conflicts
    TYPE-II  Minor context differences (surrounding code changed)
    TYPE-III API or signature differences between versions
    TYPE-IV  Significant refactoring required
    TYPE-V   Major architectural divergence

  COST & EFFICIENCY
  -----------------
  input_tokens     Tokens sent to the model
  output_tokens    Tokens received
  cost_usd         Estimated cost (Claude Opus pricing: $15/$75 per M tokens)
  latency_s        Wall-clock seconds for the evaluator to run

  ROBUSTNESS
  ----------
  error_rate       Fraction of cases where the evaluator crashed / produced no diff
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

log = logging.getLogger(__name__)



SemanticVerdict = Literal["yes", "no", "uncertain", "error"]


@dataclass
class CaseResult:
    """Result for a single (evaluator, backport case) pair."""
    evaluator:       str
    project:         str
    backport_type:   str        # TYPE-I … TYPE-V
    orig_sha:        str
    target_branch:   str

    # Correctness
    semantic_equiv:  SemanticVerdict = "error"
    judge_reasoning: str = ""
    exact_match:     bool = False

    # Cost & efficiency
    input_tokens:    int   = 0
    output_tokens:   int   = 0
    cost_usd:        float = 0.0
    latency_s:       float = 0.0

    # Artefacts
    generated_diff_path:  str = ""
    groundtruth_diff_path: str = ""
    error_msg:            str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across a set of CaseResults for one evaluator."""
    evaluator:    str
    n_total:      int = 0
    n_error:      int = 0

    # Overall rates
    semantic_equiv_rate: float = 0.0   # fraction of non-error cases that are "yes"
    exact_match_rate:    float = 0.0
    uncertain_rate:      float = 0.0
    error_rate:          float = 0.0

    # By complexity type
    by_type: dict[str, dict] = field(default_factory=dict)

    # Cost & efficiency
    mean_input_tokens:  float = 0.0
    mean_output_tokens: float = 0.0
    total_cost_usd:     float = 0.0
    mean_cost_usd:      float = 0.0
    mean_latency_s:     float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost(input_tokens: int, output_tokens: int,
                   model: str = "claude-opus-4-6") -> float:
    """Compute cost using the pipeline's canonical pricing table."""
    from auto_bug_fix.tracker import compute_cost
    return compute_cost(model, input_tokens, output_tokens)


def normalize_diff(diff: str) -> str:
    """Strip commit headers, normalize whitespace for comparison."""
    lines = []
    for line in diff.splitlines():
        if line.startswith(("commit ", "Author:", "Date:", "index ")):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def check_exact_match(generated: str, groundtruth: str) -> bool:
    return normalize_diff(generated) == normalize_diff(groundtruth)


def _judge_completion(model: str, messages: list[dict], max_tokens: int = 256):
    """Model-agnostic completion for the semantic judge."""
    from common.llm_client import completion as llm_completion
    return llm_completion(model=model, messages=messages, max_tokens=max_tokens)


def semantic_judge(
    generated_diff: str,
    groundtruth_diff: str,
    project: str,
    orig_sha: str,
    target_branch: str,
    model: str = "claude-opus-4-6",
) -> tuple[SemanticVerdict, str, int, int]:
    """
    Ask the LLM judge whether two diffs are semantically equivalent.

    Returns (verdict, reasoning, input_tokens, output_tokens).

    Semantic equivalence means:
      - The fix addresses the same root cause
      - The resulting behaviour is identical
      - Syntactic differences (variable names, spacing, style) are acceptable
    """
    prompt = f"""You are a senior code reviewer evaluating an automated backporting tool.

PROJECT: {project}
ORIGINAL FIX COMMIT: {orig_sha}
TARGET BRANCH: {target_branch}

## Ground-Truth Backport (what a human developer actually committed)
```diff
{groundtruth_diff[:6000]}
```

## Generated Backport (produced by the automated tool)
```diff
{generated_diff[:6000]}
```

## Your Task
Determine whether the generated backport is **semantically equivalent** to the ground-truth backport.

Semantic equivalence means:
- Both fixes address the same root cause
- The resulting runtime behaviour is identical
- Minor syntactic differences are acceptable (style, variable names, whitespace)
- If the generated diff is a superset or subset that still fully preserves the fix intent, that counts as equivalent

Answer with exactly one of:
  YES       — semantically equivalent
  NO        — different logic, missing changes, or introduces errors
  UNCERTAIN — cannot determine without running the code

Then on the next line, write one sentence explaining your verdict."""

    t0 = time.time()
    result = _judge_completion(model, messages=[{"role": "user", "content": prompt}])
    elapsed = time.time() - t0

    text = result.text.strip()
    first_word = text.split()[0].upper() if text else ""
    reasoning = " ".join(text.split("\n")[1:]).strip()

    if first_word == "YES":
        verdict: SemanticVerdict = "yes"
    elif first_word == "NO":
        verdict = "no"
    else:
        verdict = "uncertain"

    log.info("Judge [%s/%s → %s]: %s — %s", project, orig_sha[:8], target_branch, verdict, reasoning[:80])
    return verdict, reasoning, result.input_tokens, result.output_tokens


def aggregate(results: list[CaseResult], evaluator: str) -> AggregatedMetrics:
    """Compute aggregated metrics for one evaluator from its CaseResults."""
    agg = AggregatedMetrics(evaluator=evaluator)
    if not results:
        return agg

    agg.n_total = len(results)
    agg.n_error  = sum(1 for r in results if r.semantic_equiv == "error")

    valid = [r for r in results if r.semantic_equiv != "error"]
    n_valid = len(valid)

    if n_valid > 0:
        agg.semantic_equiv_rate = sum(1 for r in valid if r.semantic_equiv == "yes") / n_valid
        agg.exact_match_rate    = sum(1 for r in valid if r.exact_match) / n_valid
        agg.uncertain_rate      = sum(1 for r in valid if r.semantic_equiv == "uncertain") / n_valid

    agg.error_rate = agg.n_error / agg.n_total

    # Cost & latency (include all cases, including errored ones)
    agg.mean_input_tokens  = sum(r.input_tokens  for r in results) / agg.n_total
    agg.mean_output_tokens = sum(r.output_tokens for r in results) / agg.n_total
    agg.total_cost_usd     = sum(r.cost_usd      for r in results)
    agg.mean_cost_usd      = agg.total_cost_usd / agg.n_total
    agg.mean_latency_s     = sum(r.latency_s     for r in results) / agg.n_total

    # Stratify by complexity type
    all_types = sorted({r.backport_type for r in results})
    for t in all_types:
        t_results = [r for r in results if r.backport_type == t]
        t_valid   = [r for r in t_results if r.semantic_equiv != "error"]
        n_tv = len(t_valid)
        agg.by_type[t] = {
            "n": len(t_results),
            "semantic_equiv_rate": sum(1 for r in t_valid if r.semantic_equiv == "yes") / n_tv if n_tv else 0.0,
            "exact_match_rate":    sum(1 for r in t_valid if r.exact_match) / n_tv if n_tv else 0.0,
            "error_rate":          sum(1 for r in t_results if r.semantic_equiv == "error") / len(t_results),
            "mean_cost_usd":       sum(r.cost_usd for r in t_results) / len(t_results),
            "mean_latency_s":      sum(r.latency_s for r in t_results) / len(t_results),
        }

    return agg
