"""Lightweight pipeline analytics tracker — records timing, tokens, costs, and gate outcomes."""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":   {"input": 5.0,  "output": 25.0},
    "claude-opus-4-7":   {"input": 5.0,  "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0,  "output": 15.0},
    "claude-haiku-4-5":  {"input": 1.0,  "output": 5.0},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts using per-model pricing."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-opus-4-6"])
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


@dataclass
class QueryRecord:
    phase: str
    prompt_name: str
    start_time: float
    end_time: float
    duration_s: float
    duration_api_ms: int = 0
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    is_error: bool = False


@dataclass
class PhaseRecord:
    phase: str
    start_time: float
    end_time: float = 0.0
    duration_s: float = 0.0
    outcome: str = "in_progress"
    gate_decisions: dict[str, str] = field(default_factory=dict)
    queries: list[QueryRecord] = field(default_factory=list)


@dataclass
class PipelineRun:
    run_id: str
    issue_id: str
    config: dict
    start_time: str
    end_time: str = ""
    total_duration_s: float = 0.0
    outcome: str = "in_progress"
    phases: list[PhaseRecord] = field(default_factory=list)
    total_queries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


class Tracker:
    def __init__(self, issue_id: str, config_dict: dict, model: str = "claude-opus-4-6", output_dir: str = "runs"):
        """Initialize a new pipeline run tracker."""
        self._model = model
        self._output_dir = output_dir
        self._current_phase: PhaseRecord | None = None
        self._run = PipelineRun(
            run_id=str(uuid.uuid4()),
            issue_id=issue_id,
            config=config_dict,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

    @contextmanager
    def phase(self, name: str):
        """Context manager that times a pipeline phase and records its outcome."""
        phase = PhaseRecord(phase=name, start_time=time.monotonic())
        self._current_phase = phase
        try:
            yield phase
            phase.outcome = "proceed"
        except Exception as e:
            phase.outcome = type(e).__name__
            raise
        finally:
            phase.end_time = time.monotonic()
            phase.duration_s = round(phase.end_time - phase.start_time, 3)
            self._run.phases.append(phase)
            self._current_phase = None

    def record_gate(self, name: str, decision: str) -> None:
        """Record a deterministic gate decision in the current phase."""
        if self._current_phase is not None:
            self._current_phase.gate_decisions[name] = decision

    def record_query(
        self,
        prompt_name: str,
        start_time: float,
        end_time: float,
        duration_api_ms: int = 0,
        num_turns: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        is_error: bool = False,
    ) -> QueryRecord:
        """Record an LLM query with timing, token usage, and cost."""
        duration_s = round(end_time - start_time, 3)
        cost = compute_cost(self._model, input_tokens, output_tokens)

        record = QueryRecord(
            phase=self._current_phase.phase if self._current_phase else "unknown",
            prompt_name=prompt_name,
            start_time=start_time,
            end_time=end_time,
            duration_s=duration_s,
            duration_api_ms=duration_api_ms,
            num_turns=num_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=round(cost, 6),
            is_error=is_error,
        )

        if self._current_phase is not None:
            self._current_phase.queries.append(record)
        return record

    def set_outcome(self, outcome: str) -> None:
        """Set the final pipeline outcome (e.g. 'success', 'escalate')."""
        self._run.outcome = outcome

    def _compute_rollups(self) -> None:
        """Aggregate token counts and costs across all phases."""
        total_queries = 0
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for phase in self._run.phases:
            for q in phase.queries:
                total_queries += 1
                total_input += q.input_tokens
                total_output += q.output_tokens
                total_cost += q.cost_usd

        self._run.total_queries = total_queries
        self._run.total_input_tokens = total_input
        self._run.total_output_tokens = total_output
        self._run.total_cost_usd = round(total_cost, 6)

    def save(self) -> str:
        """Write the run record to a timestamped JSON file and return its path."""
        self._run.end_time = datetime.now(timezone.utc).isoformat()
        if self._run.phases:
            self._run.total_duration_s = round(
                sum(p.duration_s for p in self._run.phases), 3
            )
        self._compute_rollups()

        os.makedirs(self._output_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self._run.issue_id}_{ts}.json"
        path = os.path.join(self._output_dir, filename)

        with open(path, "w") as f:
            json.dump(asdict(self._run), f, indent=2)
        return path

    def summary(self) -> str:
        """Return a formatted table of all phases, queries, gates, and totals."""
        self._compute_rollups()
        run = self._run
        lines = [
            "",
            f"{'=' * 70}",
            f"  Pipeline Run: {run.issue_id}  ({run.run_id[:8]})",
            f"  Outcome: {run.outcome}",
            f"{'=' * 70}",
            "",
            f"  {'Phase':<35} {'Duration':>10} {'Queries':>8} {'Tokens':>12} {'Cost':>10}",
            f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 12} {'-' * 10}",
        ]

        for phase in run.phases:
            phase_tokens = sum(q.input_tokens + q.output_tokens for q in phase.queries)
            phase_cost = sum(q.cost_usd for q in phase.queries)
            lines.append(
                f"  {phase.phase:<35} {phase.duration_s:>9.1f}s {len(phase.queries):>8} "
                f"{phase_tokens:>12,} ${phase_cost:>9.4f}"
            )

            for q in phase.queries:
                lines.append(
                    f"    -> {q.prompt_name:<31} {q.duration_s:>9.1f}s "
                    f"{'':>8} {q.input_tokens + q.output_tokens:>12,} ${q.cost_usd:>9.4f}"
                )

            if phase.gate_decisions:
                gates = ", ".join(f"{k}={v}" for k, v in phase.gate_decisions.items())
                lines.append(f"    Gates: {gates}")

        lines.extend([
            "",
            f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 12} {'-' * 10}",
            f"  {'TOTAL':<35} {run.total_duration_s:>9.1f}s {run.total_queries:>8} "
            f"{run.total_input_tokens + run.total_output_tokens:>12,} ${run.total_cost_usd:>9.4f}",
            "",
        ])
        return "\n".join(lines)
