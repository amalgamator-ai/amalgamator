"""
Entry point for the branch scanner.

Usage:
    python -m branch_scanner.run_scanner --config configs/examples/openssl_scan.yaml
    python -m branch_scanner.run_scanner --config configs/examples/openssl_scan.yaml --output report.json
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .report import print_report, save_json
from .scanner import ScanConfig, run_scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(config_path: Path) -> ScanConfig:
    """Load a YAML config file and return a ScanConfig."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    repo_cfg = raw.get("repository", {})
    fix_cfg = raw.get("fix", {})
    scan_cfg = raw.get("scan", {})
    llm_cfg = raw.get("llm", {})

    repo_path = Path(repo_cfg.get("local_path", "."))
    if not repo_path.exists():
        log.error("Repository path does not exist: %s", repo_path)
        sys.exit(1)

    return ScanConfig(
        repo_path=repo_path,
        fix_commit=fix_cfg["commit"],
        vuln_commit=fix_cfg.get("vuln_commit"),
        branches=scan_cfg.get("branches", []),
        branch_pattern=scan_cfg.get("branch_pattern"),
        use_llm=llm_cfg.get("enabled", False),
        llm_model=llm_cfg.get("model", "claude-opus-4-6"),
        fix_description=fix_cfg.get("description", ""),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scan repo branches to identify which need a bug fix backported."
    )
    parser.add_argument(
        "--config", "-c", required=True, type=Path,
        help="Path to YAML scan config file"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path to save JSON report (optional)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    log.info("Starting branch scan for commit %s", config.fix_commit)
    log.info("Repo: %s", config.repo_path)

    report = run_scan(config)
    print_report(report)

    if args.output:
        save_json(report, args.output)

    # Exit code: 0 if all branches fixed, 1 if any need fix
    sys.exit(0 if not report.needs_fix else 1)


if __name__ == "__main__":
    main()
