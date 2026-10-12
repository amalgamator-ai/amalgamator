"""Prometheus /metrics endpoint poller — background thread that samples server metrics."""

from __future__ import annotations

import csv
import logging
import re
import subprocess
import threading
import time
from pathlib import Path

from llm.auto_profile_server_client.models import MetricsSample

log = logging.getLogger(__name__)

# Prometheus metric name patterns per framework
_METRIC_PATTERNS = {
    "generation_tokens": re.compile(
        r"(?:vllm:generation_tokens_total|sglang:generation_tokens_total|trtllm_generation_tokens_total)\{[^}]*\}\s+([\d.eE+\-]+)"
    ),
    "requests_running": re.compile(
        r"(?:vllm:num_requests_running|sglang:num_running_reqs|trtllm_num_requests_running)\{[^}]*\}\s+([\d.eE+\-]+)"
    ),
    "kv_cache_usage": re.compile(
        r"(?:vllm:gpu_cache_usage_perc|vllm:kv_cache_usage_perc|sglang:token_usage|trtllm_kv_cache_utilization)\{[^}]*\}\s+([\d.eE+\-]+)"
    ),
    "queue_size": re.compile(
        r"(?:vllm:num_requests_waiting|sglang:num_queue_reqs)\{[^}]*\}\s+([\d.eE+\-]+)"
    ),
}


def _fetch_metrics(url: str) -> str | None:
    try:
        r = subprocess.run(
            ["curl", "-sf", "--max-time", "3", url],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout
    except Exception:
        pass
    return None


def _parse_metrics_text(text: str, prev_tokens_total: float) -> MetricsSample:
    sample = MetricsSample(timestamp=time.time())

    for metric_key, pattern in _METRIC_PATTERNS.items():
        m = pattern.search(text)
        if m:
            val = float(m.group(1))
            if metric_key == "generation_tokens":
                sample.generation_tokens_total = val
            elif metric_key == "requests_running":
                sample.requests_running = val
            elif metric_key == "kv_cache_usage":
                sample.kv_cache_usage_pct = val
            elif metric_key == "queue_size":
                sample.queue_size = val

    return sample


class MetricsCollector:
    """Background thread that polls Prometheus metrics endpoint."""

    def __init__(
        self,
        metrics_url: str,
        poll_interval_sec: float = 1.0,
        raw_log_path: str | None = None,
    ):
        self.metrics_url = metrics_url
        self.poll_interval_sec = poll_interval_sec
        self.raw_log_path = raw_log_path
        self.samples: list[MetricsSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._prev_tokens_total = 0.0
        self._raw_log_file = None

    def start(self) -> None:
        self._stop_event.clear()
        self.samples = []
        self._prev_tokens_total = 0.0
        if self.raw_log_path:
            Path(self.raw_log_path).parent.mkdir(parents=True, exist_ok=True)
            self._raw_log_file = (
                open(  # noqa: SIM115 (thread-lifetime; closed in stop())
                    self.raw_log_path, "w"
                )
            )
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info(
            "Metrics collector started: %s (interval=%.1fs)",
            self.metrics_url,
            self.poll_interval_sec,
        )

    def stop(self) -> list[MetricsSample]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._raw_log_file:
            self._raw_log_file.close()
            self._raw_log_file = None
        log.info("Metrics collector stopped: %d samples collected", len(self.samples))
        return list(self.samples)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            text = _fetch_metrics(self.metrics_url)
            if text:
                if self._raw_log_file:
                    self._raw_log_file.write(f"=== {time.time():.3f} ===\n{text}\n")
                    self._raw_log_file.flush()
                sample = _parse_metrics_text(text, self._prev_tokens_total)
                self._prev_tokens_total = sample.generation_tokens_total
                self.samples.append(sample)
            self._stop_event.wait(self.poll_interval_sec)

    def save_to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "generation_tokens_total",
                    "requests_running",
                    "kv_cache_usage_pct",
                    "queue_size",
                ]
            )
            for s in self.samples:
                writer.writerow(
                    [
                        f"{s.timestamp:.3f}",
                        f"{s.generation_tokens_total:.1f}",
                        f"{s.requests_running:.0f}",
                        f"{s.kv_cache_usage_pct:.4f}",
                        f"{s.queue_size:.0f}",
                    ]
                )
        log.info("Saved %d metrics samples to %s", len(self.samples), path)
