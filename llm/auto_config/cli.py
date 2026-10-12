"""CLI for auto_config: generate profiling configs from framework recipes."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from llm.auto_config.generate import generate_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="auto_config",
        description="Generate auto_profile_server_client configs from framework recipes.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. nvidia/GLM-5.2-NVFP4)",
    )
    parser.add_argument(
        "--hardware", required=True, help="GPU target (e.g. b200, h200, mi355x)"
    )
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["vllm"],
        help="Frameworks to include: vllm, sgl (default: vllm)",
    )
    parser.add_argument(
        "--strategy",
        default="low-latency",
        choices=["low-latency", "balanced", "high-throughput"],
        help="SGLang serving strategy (default: low-latency)",
    )
    parser.add_argument(
        "--spec-decoding",
        default="none",
        help="Speculative decoding: none, mtp, dspark, keep (default: none)",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1],
        help="Concurrency levels (default: 1)",
    )
    parser.add_argument(
        "--exec-mode",
        default="pure_decode",
        help="Execution mode (default: pure_decode)",
    )
    parser.add_argument(
        "--input-len", type=int, default=4, help="Input token length (default: 4)"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Output token length (default: 1024)",
    )
    parser.add_argument(
        "--enable-trace",
        action="store_true",
        default=True,
        help="Enable NSYS tracing (default: true)",
    )
    parser.add_argument("--no-trace", action="store_true", help="Disable NSYS tracing")
    parser.add_argument(
        "--extra-flags",
        default="",
        help="Additional server flags (override recipe on conflict)",
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file path (prints to stdout if omitted)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    enable_trace = not args.no_trace

    try:
        config = generate_config(
            model=args.model,
            hardware=args.hardware,
            num_gpus=args.num_gpus,
            frameworks=args.frameworks,
            strategy=args.strategy,
            spec_decoding=args.spec_decoding,
            concurrencies=args.concurrencies,
            exec_mode=args.exec_mode,
            input_len=args.input_len,
            output_len=args.output_len,
            enable_trace=enable_trace,
            extra_flags=args.extra_flags,
            output_path=args.output,
        )
    except Exception as e:
        logging.error("%s", e)
        sys.exit(1)

    if not args.output:
        print(json.dumps(config, indent=4))
