# Backport Benchmark Evaluation

Evaluates and compares automated backporting strategies against real-world datasets, producing a leaderboard ranked by semantic equivalence.

## What We're Measuring

Two backporting strategies are compared head-to-head:

| Strategy | Description |
|---|---|
| **Amalgamator Pipeline** | Full git cherry-pick + LLM conflict resolution + iterative build-fix loop |
| **Raw Claude (Single-Shot)** | One-shot Claude prompt with the original diff + target branch context; no iteration |

The goal is to understand **when the full pipeline outperforms a single LLM call**, and at what cost and latency trade-off. The model (e.g., Claude Opus, Claude Sonnet, DeepSeek) can be swapped independently for both strategies.

---

## Dataset: JavaBackports

The [JavaBackports](https://github.com/Javabackports/javabackports) dataset contains 491 validated real-world backport instances across 13 major Java projects.

### Complexity Types

Each case is classified by how hard the backport is:

| Type | What it means | Count |
|---|---|---|
| **TYPE-I** | Direct cherry-pick, no conflicts — code context is identical | 233 |
| **TYPE-II** | Minor context differences — surrounding lines changed between versions | 151 |
| **TYPE-III** | API or method signature differences between branches | 26 |
| **TYPE-IV** | Significant refactoring required — logic restructured in target | 4 |
| **TYPE-V** | Major architectural divergence — fundamentally different implementation | 77 |

**Hypothesis**: The full pipeline should outperform raw Claude on TYPE-III through TYPE-V (complex cases), while a single-shot prompt may be competitive on TYPE-I/II at much lower cost.

### Projects Included

CrateDB, Elasticsearch, Apache Hadoop, Apache Druid, Apache HBase, OpenJDK (8/11/17/21/25), Graylog, Hibernate ORM, Spring Framework, gRPC-Java, Logstash

### Dataset Setup

Each row in the CSV contains:
- `Original Commit` — SHA of the fix on the source branch (main/master)
- `Backport Commit` — SHA of the human backport on the target branch (the **already-fixed** state)
- `Backport Version` — target branch name (e.g., `6.1`, `3.6`)

To evaluate fairly, the target branch is checked out at **one commit before** the backport (`BACKPORT_SHA~1`), so the tool has to do the actual backport — then the output is compared against the ground-truth `git show BACKPORT_SHA`.

---

## Metrics

Based on evaluation criteria from [JavaBackports (2026)](https://github.com/Javabackports/javabackports) and [BackportBench (2024)](https://arxiv.org/abs/2408.11227):

### Primary
| Metric | Description |
|---|---|
| **Semantic Equivalence Rate** | Fraction of cases where the generated patch is semantically equivalent to the human ground-truth (LLM-as-judge) |

### Secondary
| Metric | Description |
|---|---|
| **Exact Match Rate** | Syntactic exact match after whitespace normalization (a strict subset of semantic equiv) |
| **Uncertain Rate** | Judge could not determine equivalence — needs human review |
| **Error Rate** | Evaluator crashed or produced no diff |

### Cost & Efficiency
| Metric | Description |
|---|---|
| **Avg Latency** | Wall-clock seconds per backport |
| **Avg Cost / Case** | Estimated USD (Claude Opus: $15/$75 per M input/output tokens) |
| **Total Cost** | Total run cost for the evaluated set |

### Stratified by Complexity Type
All metrics are broken down by TYPE-I through TYPE-V to show exactly where each strategy wins or loses.

---

## Semantic Equivalence Judge

Rather than running full build+test suites (which can take hours per project), an **LLM-as-judge** approach is used:

> *"Is the generated backport semantically equivalent to the ground-truth human backport?"*

The judge receives both diffs and reasons about intent, not syntax. This is:
- Fast (seconds vs hours)
- Scalable (no Docker or complex build environments required)
- Aligned with real-world acceptance criteria (would a maintainer accept this patch?)

Build/test verification can be added as an optional next step for a subset of cases.

---

## Folder Structure

```
benchmark/
├── README.md                        ← This file
├── __init__.py
├── generic_claude_backport.py       ← Raw Claude evaluator (single-shot)
└── evaluate/
    ├── __init__.py
    ├── config.yaml                  ← Main config: dataset, repos, evaluators, models
    ├── run_evaluation.py            ← Main runner (entry point)
    ├── metrics.py                   ← Metric computation + LLM judge
    └── report.py                    ← Leaderboard output (terminal + JSON + Markdown)
```

Output is written to the path in `config.yaml → output.root`:

```
benchmark-results/
├── amalgamator_pipeline_opus/
│   ├── crate/<sha12>/
│   │   ├── generated.diff
│   │   ├── groundtruth.diff
│   │   └── result.json
│   └── aggregated.json
├── raw_claude_opus/
│   └── ...
├── leaderboard.json
└── leaderboard.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd /path/to/amalgamator
uv sync
```

### 2. Edit the config

Open `benchmark/evaluate/config.yaml` and set:
- `dataset.csv_path` → path to `javabackports/dataset/all_projects_final.csv`
- `repos.root` → where repos will be cloned (needs ~10–50 GB depending on projects)
- `output.root` → where results go
- `dataset.projects` → which projects to test (start small: `crate`, `grpc-java`)
- `dataset.sample_per_type` → cases per type (3 = quick, 10 = more statistically meaningful)

### 3. Dry run (preview cases without running anything)

```bash
uv run python -m benchmark.evaluate.run_evaluation \
    --config benchmark/evaluate/config.yaml \
    --dry-run
```

### 4. Run only the single-shot baseline first (fast, cheap)

```bash
uv run python -m benchmark.evaluate.run_evaluation \
    --config benchmark/evaluate/config.yaml \
    --evaluator raw_claude_opus \
    --limit 10
```

### 5. Run both evaluators

```bash
uv run python -m benchmark.evaluate.run_evaluation \
    --config benchmark/evaluate/config.yaml
```

### 6. Resume interrupted runs

```bash
uv run python -m benchmark.evaluate.run_evaluation \
    --config benchmark/evaluate/config.yaml \
    --resume
```

### 7. Swap the model (e.g., test with Sonnet instead of Opus)

Edit `config.yaml → evaluators → raw_claude_opus → model: "claude-sonnet-4-6"` and re-run. Or add a new evaluator entry with a different model key (e.g., `raw_claude_sonnet`) to compare both in one run.

---

## Example Leaderboard Output

```
================================================================================
  BACKPORT BENCHMARK LEADERBOARD
================================================================================

  Amalgamator Pipeline (claude-opus-4-6)
  Cherry-pick + LLM conflict resolution + iterative build-fix loop
  Cases: 18  |  Semantic Equiv: 77.8%  |  Avg Latency: 142s  |  Avg Cost: $0.089

  Type        Semantic Equiv    Exact    Errors    Avg Cost    Avg Lat
  TYPE-I            94.4%      44.4%      0.0%       $0.041     38s
  TYPE-II           72.2%       5.6%      5.6%       $0.093    152s
  TYPE-III          55.6%       0.0%     11.1%       $0.178    287s

  Raw Claude Single-Shot (claude-opus-4-6)
  Single-shot prompt, no cherry-pick pipeline or iteration
  Cases: 18  |  Semantic Equiv: 61.1%  |  Avg Latency: 12s  |  Avg Cost: $0.008

  Type        Semantic Equiv    Exact    Errors    Avg Cost    Avg Lat
  TYPE-I            83.3%       5.6%      0.0%       $0.006     11s
  TYPE-II           55.6%       0.0%      0.0%       $0.008     13s
  TYPE-III          22.2%       0.0%      0.0%       $0.011     14s
```

---

## Adding More Evaluators / Models

To add a new evaluator, add an entry under `config.yaml → evaluators`:

```yaml
evaluators:
  raw_claude_sonnet:
    enabled: true
    label: "Raw Claude Single-Shot (claude-sonnet-4-6)"
    model: "claude-sonnet-4-6"
    strategy: "raw_claude"

  pipeline_sonnet:
    enabled: true
    label: "Amalgamator Pipeline (claude-sonnet-4-6)"
    model: "claude-sonnet-4-6"
    strategy: "pipeline"
```

This makes it easy to build a full model × strategy comparison matrix.

---

## References

- [JavaBackports Dataset](https://github.com/Javabackports/javabackports) — 491 validated Java backports (TYPE-I through TYPE-V)
- [BackportBench (2024)](https://arxiv.org/abs/2408.11227) — 202 multi-language backport tasks with pass@k evaluation
