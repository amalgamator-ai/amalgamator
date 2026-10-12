"""LLM agent that analyzes raw Prometheus metrics to determine
the pure-decode phase and profile capture window."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLES_TO_SEND = 15

RELEVANT_METRIC_PATTERNS = [
    "generation_tokens",
    "prompt_tokens",
    "num_requests_running",
    "num_requests_waiting",
    "cache_usage",
    "kv_cache",
    "token_usage",
    "throughput",
    "request_success",
    "num_running",
    "num_queue",
]


def _filter_block(block: str) -> str:
    """Keep only relevant metrics from a Prometheus block."""
    lines = block.split("\n")
    header = lines[0] if lines else ""
    kept = [header]
    for line in lines[1:]:
        if line.startswith("#"):
            continue
        if any(p in line for p in RELEVANT_METRIC_PATTERNS):
            kept.append(line)
    return "\n".join(kept)


def _downsample_raw_metrics(raw_text: str) -> str:
    """Downsample raw metrics to a manageable size for the LLM.

    Keeps only relevant metrics and evenly samples across the timeline.
    """
    blocks = raw_text.split("=== ")
    blocks = [f"=== {b}" for b in blocks if b.strip()]

    if not blocks:
        return raw_text

    # Sample evenly
    if len(blocks) > SAMPLES_TO_SEND:
        step = max(1, (len(blocks) - 2) // (SAMPLES_TO_SEND - 2))
        indices = (
            [0]
            + list(range(1, len(blocks) - 1, step))[: SAMPLES_TO_SEND - 2]
            + [len(blocks) - 1]
        )
        blocks = [blocks[i] for i in indices]

    # Filter each block to relevant metrics only
    filtered = [_filter_block(b) for b in blocks]

    log.info(
        "Downsampled raw metrics: %d snapshots, %d chars",
        len(filtered),
        sum(len(b) for b in filtered),
    )
    return "\n".join(filtered)


def _call_llm(prompt: str) -> str:
    """Call the LLM via the claude CLI."""
    import subprocess
    import tempfile

    # Write prompt to file, then have claude read it via shell cat
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        log.info("Calling claude CLI (%d chars prompt)", len(prompt))
        result = subprocess.run(
            f"cat {prompt_file} | claude -p - --model sonnet --output-format text 2>/dev/null",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout.strip()
        if not output:
            raise RuntimeError(
                f"claude CLI returned empty output (exit {result.returncode})"
            )
        return output
    finally:
        Path(prompt_file).unlink(missing_ok=True)


ANALYSIS_PROMPT = """\
You are analyzing raw Prometheus metrics from a GPU LLM inference benchmark.

The benchmark ran {num_prompts} prompts with input_len={input_len}, \
output_len={output_len}, concurrency={concurrency} on framework "{framework}".

Below is the raw /metrics output captured every ~1 second during the run. \
Each block starts with "=== <timestamp> ===" followed by the Prometheus metrics.

YOUR TASKS:
1. Find the metric that tracks cumulative generated tokens (e.g. \
vllm:generation_tokens_total, sglang_generation_tokens_total, or similar).
2. Compute throughput (tokens/sec) between consecutive timestamps.
3. Identify the PURE-DECODE phase: the continuous period where:
   - Throughput is stable (not ramping up or winding down)
   - No new prefills are starting (prompt token counter is flat)
   - All requests are actively generating tokens
4. Determine the midpoint of the pure-decode phase.
5. Compute profile_delay_sec = midpoint - 2.5 (so the 5-second capture \
is centered on the midpoint).

Return ONLY valid JSON with this exact schema:
{{
  "token_metric_name": "<the metric name you used>",
  "throughput_series": [
    {{"timestamp": <float>, "tokens_total": <float>, "throughput_tok_s": <float>}},
    ...
  ],
  "pure_decode_phase": {{
    "start_sec": <float>,
    "end_sec": <float>,
    "duration_sec": <float>
  }},
  "profile_window": {{
    "profile_delay_sec": <float>,
    "profile_duration_sec": 5.0
  }},
  "throughput_stats": {{
    "mean": <float>,
    "std": <float>,
    "cv_pct": <float>
  }}
}}

All *_sec values are relative to the first timestamp in the file.

RAW METRICS:
{raw_metrics}
"""


@dataclass
class ProfileWindow:
    profile_delay_sec: float
    profile_duration_sec: float = 5.0


@dataclass
class MetricsAnalysis:
    token_metric_name: str
    throughput_series: list[dict]
    pure_decode_phase: dict
    profile_window: ProfileWindow
    throughput_stats: dict
    raw_json: dict


def analyze_metrics(
    raw_metrics_path: str,
    framework: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    concurrency: int,
) -> MetricsAnalysis:
    """Invoke the LLM agent to analyze raw metrics and determine the profile window.

    Raises on any failure — no fallback.
    """
    raw_text = Path(raw_metrics_path).read_text()
    if not raw_text.strip():
        raise RuntimeError(f"Raw metrics file is empty: {raw_metrics_path}")

    # Downsample: the raw file can be huge (MB). Send first 5 + middle 5 +
    # last 5 metric snapshots so the LLM sees the full progression without
    # exceeding context limits.
    raw_text = _downsample_raw_metrics(raw_text)

    prompt = ANALYSIS_PROMPT.format(
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        concurrency=concurrency,
        framework=framework,
        raw_metrics=raw_text,
    )

    log.info("Invoking metrics analysis agent (%d chars of raw metrics)", len(raw_text))

    response = _call_llm(prompt)

    # Extract JSON from response (may have preamble text + markdown fencing)
    response = response.strip()
    # Find JSON block inside markdown code fences
    if "```" in response:
        import re

        m = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
        if m:
            response = m.group(1).strip()
    # Or find raw JSON object
    elif not response.startswith("{"):
        idx = response.find("{")
        if idx >= 0:
            response = response[idx:]

    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Metrics agent returned invalid JSON: {e}\nResponse: {response[:500]}"
        )

    # Validate required fields
    for key in (
        "token_metric_name",
        "throughput_series",
        "pure_decode_phase",
        "profile_window",
        "throughput_stats",
    ):
        if key not in data:
            raise RuntimeError(f"Metrics agent response missing field: {key}")

    pw = data["profile_window"]
    if "profile_delay_sec" not in pw:
        raise RuntimeError(
            "Metrics agent response missing profile_window.profile_delay_sec"
        )
    if pw["profile_delay_sec"] < 0:
        raise RuntimeError(
            f"Metrics agent computed negative profile_delay_sec: {pw['profile_delay_sec']}"
        )

    analysis = MetricsAnalysis(
        token_metric_name=data["token_metric_name"],
        throughput_series=data["throughput_series"],
        pure_decode_phase=data["pure_decode_phase"],
        profile_window=ProfileWindow(
            profile_delay_sec=pw["profile_delay_sec"],
            profile_duration_sec=pw.get("profile_duration_sec", 5.0),
        ),
        throughput_stats=data["throughput_stats"],
        raw_json=data,
    )

    log.info(
        "Metrics analysis: decode_phase=[%.1f, %.1f]s, profile_delay=%.1fs, "
        "throughput=%.1f±%.1f tok/s (cv=%.1f%%)",
        analysis.pure_decode_phase["start_sec"],
        analysis.pure_decode_phase["end_sec"],
        analysis.profile_window.profile_delay_sec,
        analysis.throughput_stats["mean"],
        analysis.throughput_stats["std"],
        analysis.throughput_stats["cv_pct"],
    )

    return analysis
