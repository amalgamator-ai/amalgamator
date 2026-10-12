"""Constants and defaults for the server-client benchmark pipeline."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

VALID_FRAMEWORKS = ("vllm", "sgl", "trt")

# Docker defaults
DOCKER_BASE_DIR = "/app"
DOCKER_HF_HUB_CACHE = "/app/hf_hub_cache"
DOCKER_PORT = 8000
DOCKER_SHM_SIZE = "32g"

# Server defaults
SERVER_HEALTH_TIMEOUT_SEC = 1800
SERVER_HEALTH_POLL_SEC = 5

# Benchmark defaults
DEFAULT_NUM_PROMPTS_MULTIPLIER = 4
DEFAULT_STABILITY_THRESHOLD_PCT = 5.0
DEFAULT_MAX_STABILITY_RETRIES = 3
DEFAULT_METRICS_POLL_INTERVAL_SEC = 1.0

# NSYS profiling
NSYS_DEFAULT_FLAGS = [
    "-t",
    "cuda,nvtx",
    "-c",
    "cudaProfilerApi",
    "--capture-range-end=stop",
    "--cuda-graph-trace=node",
    "--trace-fork-before-exec=true",
]
NUM_TRACE_ITERS = 50
PROFILING_BUFFER = 256

# Framework-specific defaults
FRAMEWORK_PORT_POOLS = {
    "vllm": list(range(8123, 8133)),
    "sgl": list(range(30123, 30133)),
    "trt": list(range(8534, 8544)),
}

# Deprecated — use get_next_port() instead
FRAMEWORK_PORTS = {fw: pools[0] for fw, pools in FRAMEWORK_PORT_POOLS.items()}

MAX_MODEL_LEN_DEFAULT = 128 * 1024

PORT_INDEX_FILE = Path.home() / ".auto_profile_port_index.json"


def get_next_port(framework: str) -> int:
    """Get the next port for a framework, rotating through the pool."""
    pool = FRAMEWORK_PORT_POOLS.get(framework)
    if not pool:
        return 8000

    index = {}
    if PORT_INDEX_FILE.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            index = json.loads(PORT_INDEX_FILE.read_text())

    idx = index.get(framework, 0) % len(pool)
    port = pool[idx]

    index[framework] = (idx + 1) % len(pool)
    with contextlib.suppress(OSError):
        PORT_INDEX_FILE.write_text(json.dumps(index))

    _log.info("Port for %s: %d (index %d/%d)", framework, port, idx, len(pool))
    return port
