"""
Entry point for the backport scanner.

Usage:
    python -m backport_scanner.run_scanner --config configs/examples/openssl_scan.yaml
    python -m backport_scanner.run_scanner --config configs/examples/openssl_scan.yaml --output report.json
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from common.model_registry import RegistryError, find_registry

from .report import print_report, save_json
from .scanner import ScanConfig, run_scan

log = logging.getLogger(__name__)


def load_config(config_path: Path, models_path: str | Path | None = None) -> ScanConfig:
    """Load scan settings and resolve llm.model through an optional registry."""
    config_path = Path(config_path).expanduser()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Scan configuration must be a YAML mapping")
    for key in ("repository", "fix", "scan", "llm"):
        if key in raw and not isinstance(raw[key], dict):
            raise ValueError(f"Scan configuration '{key}' must be a mapping")
    repo_cfg = raw.get("repository", {})
    fix_cfg = raw.get("fix", {})
    scan_cfg = raw.get("scan", {})
    llm_cfg = raw.get("llm", {})

    repo_path = Path(repo_cfg.get("local_path", ".")).expanduser()
    if not repo_path.exists():
        log.error("Repository path does not exist: %s", repo_path)
        sys.exit(1)

    model_ref = llm_cfg.get("model")
    if model_ref is not None and (
        not isinstance(model_ref, str) or not model_ref.strip()
    ):
        raise ValueError("llm.model must be a non-empty profile ID or model name")
    registry = find_registry(
        config_dir=str(config_path.resolve().parent),
        explicit_path=str(models_path) if models_path is not None else None,
    )
    profile = None
    model = model_ref or "claude-opus-4-6"
    if registry is not None:
        resolved = registry.resolve(model_ref)
        model = resolved.model
        if model_ref is None or model_ref in registry.profiles:
            profile = resolved
            model_ref = resolved.id
        # Bare-model callers retain legacy environment/global transport lookup.

    return ScanConfig(
        repo_path=repo_path,
        fix_commit=fix_cfg["commit"],
        vuln_commit=fix_cfg.get("vuln_commit"),
        branches=scan_cfg.get("branches", []),
        branch_pattern=scan_cfg.get("branch_pattern"),
        tags=scan_cfg.get("tags", []),
        tag_pattern=scan_cfg.get("tag_pattern"),
        llm_model=model,
        fix_description=fix_cfg.get("description", ""),
        discover_commands=llm_cfg.get("discover_commands", False),
        profile=profile,
        model_ref=model_ref,
        models_path=registry.source_path if registry else None,
    )


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Scan repo branches to identify which need a bug fix backported."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Path to YAML scan config file (supports branches and/or tags)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save JSON report (optional)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--models", type=Path, help="Explicit model registry (overrides discovery)"
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = load_config(args.config, models_path=args.models)
    except (RegistryError, ValueError, OSError, yaml.YAMLError) as e:
        parser.error(str(e))
    log.info("Starting backport scan for commit %s", config.fix_commit)
    log.info("Repo: %s", config.repo_path)

    report = run_scan(config)

    if not report.results:
        log.warning("No refs were scanned — config has neither branches nor tags.")
        sys.exit(2)

    print_report(report)

    if args.output:
        save_json(report, args.output)

    if report.errors:
        log.error("%d ref(s) failed validation — see report above", len(report.errors))
        sys.exit(2)
    if report.ambiguous:
        log.warning(
            "%d ref(s) remain AMBIGUOUS after LLM check — manual review needed",
            len(report.ambiguous),
        )
        sys.exit(3)
    sys.exit(0 if not report.needs_fix else 1)


if __name__ == "__main__":
    main()
