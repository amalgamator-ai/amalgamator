"""YAML/JSON configuration loader with validation for bug fix porting pipeline."""
from __future__ import annotations

import logging
import math
import os
import re
import yaml
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_bug_fix.bug_fix_config import BugFixConfig
from auto_bug_fix.claude_utils import BugFixClaudeConfig, ClaudeConfig
from common.model_registry import resolve_api_key as _resolve_api_key
from common.model_registry import ModelProfile, ModelRegistry, build_legacy_profile

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


def advanced_float(
    advanced: dict[str, Any],
    key: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a float from the ``advanced:`` section, warning on unusable values.

    Accepts YAML numbers and numeric strings (``0.7`` and ``"0.7"`` alike).
    A key that is absent uses the default silently. A key that is present but
    unusable warns and falls back rather than crashing the run at load time,
    or — worse — silently disabling the check it configures. Unusable means:

    - unparseable (null, text, a list)
    - a bool, since ``float(True)`` is ``1.0`` and almost certainly not what
      a config that says ``true`` intended
    - NaN or an infinity, which make every comparison against the value false
    - outside [minimum, maximum] when bounds are given
    """
    if key not in advanced:
        return default
    raw = advanced[key]
    if isinstance(raw, bool):
        log.warning(
            "advanced.%s=%r is a boolean, not a number — falling back to %s",
            key, raw, default,
        )
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "advanced.%s=%r is not a number — falling back to %s",
            key, raw, default,
        )
        return default
    if not math.isfinite(value):
        log.warning(
            "advanced.%s=%r is not finite — falling back to %s",
            key, raw, default,
        )
        return default
    if (minimum is not None and value < minimum) or \
       (maximum is not None and value > maximum):
        log.warning(
            "advanced.%s=%r is outside [%s, %s] — falling back to %s",
            key, raw, minimum, maximum, default,
        )
        return default
    return value


def advanced_str(advanced: dict[str, Any], key: str, default: str) -> str:
    """Read a non-empty string from the ``advanced:`` section.

    A key that is absent uses the default silently. A key that is present but
    empty, blank, or not a string warns and falls back — an empty model name
    would otherwise surface as an opaque API error deep inside the phase it
    configures.
    """
    if key not in advanced:
        return default
    raw = advanced[key]
    if not isinstance(raw, str) or not raw.strip():
        log.warning(
            "advanced.%s=%r is not a non-empty string — falling back to %s",
            key, raw, default,
        )
        return default
    return raw.strip()


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load YAML or JSON config file."""
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()

    try:
        with open(path) as f:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ConfigError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_path}: {e}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML/JSON mapping, got {type(data).__name__}")
    return data


def _require_mapping(config: dict, key: str) -> dict:
    """Get a sub-section from config, raising ConfigError if it's not a mapping."""
    val = config.get(key, {})
    if not isinstance(val, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(val).__name__}")
    return val


def _resolve_patch_path(raw: str, config_dir: str) -> str:
    """Expand and resolve a patch path, making relative paths relative to config_dir."""
    expanded = expand_path(raw)
    if not os.path.isabs(expanded) and config_dir:
        expanded = os.path.join(config_dir, expanded)
    return os.path.normpath(os.path.abspath(expanded))


def validate_config(config: dict[str, Any], config_dir: str = "") -> None:
    """Validate configuration structure and required fields."""
    required = ["issue_id", "bug_description", "repository", "branches", "fix", "build"]
    missing = [f for f in required if f not in config]
    if missing:
        raise ConfigError(f"Missing required fields: {', '.join(missing)}")

    repo = _require_mapping(config, "repository")
    if repo.get("source_path") is None:
        raise ConfigError("repository.source_path is required")

    branches = _require_mapping(config, "branches")
    if branches.get("target") is None:
        raise ConfigError("branches.target is required")

    fix = _require_mapping(config, "fix")
    has_commit = fix.get("commit") is not None
    has_patch = fix.get("patch") is not None

    if has_commit and has_patch:
        raise ConfigError("fix.commit and fix.patch are mutually exclusive — specify one, not both")
    if not has_commit and not has_patch:
        raise ConfigError("fix.commit or fix.patch is required")

    if has_patch:
        resolved = _resolve_patch_path(str(fix["patch"]), config_dir)
        if not os.path.isfile(resolved):
            raise ConfigError(f"fix.patch file not found: {resolved}")
        if branches.get("source") is None:
            branches["source"] = "origin/main"
            log.info("branches.source defaulting to 'origin/main' in patch mode")
    else:
        if branches.get("source") is None:
            raise ConfigError("branches.source is required")
        if fix.get("upstream_ref") is not None:
            log.warning("fix.upstream_ref is only used in patch mode — ignored in commit mode")

    _require_mapping(config, "build")


def expand_path(path: str | None, default: str = "") -> str:
    """Expand ~ and environment variables in path. Returns default for None."""
    if path is None:
        return default
    return os.path.expanduser(os.path.expandvars(path))


def _pipeline_model_name(config: dict[str, Any]) -> str | None:
    """Read the selection before applying registry or legacy defaults."""
    claude = config.get("claude", {})
    if isinstance(claude, dict):
        m = claude.get("model")
        if m:
            return str(m)
    return str(config["model"]) if config.get("model") else None


def _legacy_profile(config: dict[str, Any], model: str) -> ModelProfile:
    claude = config.get("claude", {})
    if not isinstance(claude, dict):
        claude = {}
    api_base = config.get("api_base") or claude.get("api_base")
    params = config.get("model_params")
    if params is not None and not isinstance(params, dict):
        log.warning("model_params must be a mapping, ignoring")
        params = None
    return build_legacy_profile(
        model, api_base=str(api_base) if api_base else None,
        api_key=_resolve_api_key(config) or _resolve_api_key(claude), params=params,
    )


def resolve_pipeline_profiles(
    config: dict[str, Any], registry: ModelRegistry | None,
) -> tuple[ModelProfile, ModelProfile]:
    """Resolve the pipeline and Phase 4b together, inheriting whole profiles."""
    def legacy(model_name: str) -> ModelProfile:
        return _legacy_profile(config, model_name)

    model = _pipeline_model_name(config)
    named = registry is not None and (model is None or model in registry.profiles)
    inline_fields = _inline_model_fields(config)
    pipeline = registry.resolve(model, legacy_factory=legacy) if registry else legacy(model or "claude-sonnet-4-6")
    if named and inline_fields:
        log.warning("Named profile ignores inline fields: %s", ", ".join(inline_fields))
    advanced = config.get("advanced", {})
    if not isinstance(advanced, dict):
        advanced = {}
    override = advanced_str(advanced, "phase_4b_model", "")
    if not override:
        phase_4b = pipeline
    elif registry and (registry.strict or override in registry.profiles):
        phase_4b = registry.resolve(override)
    elif override == pipeline.model:
        phase_4b = pipeline
    else:
        phase_4b = registry.resolve(override, legacy_factory=legacy) if registry else legacy(override)
        if inline_fields:
            log.warning(
                "Different bare Phase 4b model inherits inline fields: %s; "
                "prefer an explicit named profile for its endpoint",
                ", ".join(inline_fields),
            )
    return pipeline, phase_4b


def _inline_model_fields(config: dict[str, Any]) -> list[str]:
    """Field names only: warnings must not disclose transport values."""
    names = [key for key in ("api_base", "api_key", "api_key_file", "model_params") if key in config]
    claude = config.get("claude", {})
    if isinstance(claude, dict):
        names.extend(f"claude.{key}" for key in ("api_base", "api_key", "api_key_file") if key in claude)
    return names


def parse_bug_fix_config(
    config: dict[str, Any],
    config_dir: str = "",
    registry: "ModelRegistry | None" = None,
    *,
    profiles: tuple[ModelProfile, ModelProfile] | None = None,
) -> tuple[BugFixConfig, str, str]:
    """Parse YAML/JSON config into BugFixConfig dataclass."""
    validate_config(config, config_dir=config_dir)

    source_path = os.path.normpath(expand_path(config["repository"]["source_path"]))
    workdir = expand_path(config["repository"].get("workdir"), "/tmp/bugfix-workdir")
    repo_name = os.path.basename(source_path)
    repo_path = os.path.join(workdir, repo_name)

    build_subdir = config["repository"].get("build_subdir", "")
    if build_subdir:
        build_dir = os.path.join(repo_path, build_subdir)
    else:
        build_dir = repo_path

    advanced = config.get("advanced", {})
    if not isinstance(advanced, dict):
        advanced = {}

    # Timeout is specified in minutes in the YAML; 0/unset/negative = no limit.
    timeout_minutes = advanced.get("timeout_minutes")
    command_timeout = (
        int(timeout_minutes * 60)
        if isinstance(timeout_minutes, (int, float)) and timeout_minutes > 0
        else None
    )

    fix = config["fix"]
    is_patch_mode = fix.get("patch") is not None

    fix_patch = ""
    fix_upstream_ref = ""
    source_fix_commit = ""

    if is_patch_mode:
        fix_patch = _resolve_patch_path(str(fix["patch"]), config_dir)
        fix_upstream_ref = str(fix.get("upstream_ref", "") or "")
    else:
        source_fix_commit = str(fix["commit"])

    _, phase_4b_profile = profiles or resolve_pipeline_profiles(config, registry)

    bug_fix_config = BugFixConfig(
        repo_path=repo_path,
        build_dir=build_dir,
        source_branch=str(config["branches"]["source"]),
        target_branch=str(config["branches"]["target"]),
        source_fix_commit=source_fix_commit,
        bug_description=str(config["bug_description"]),
        issue_id=str(config["issue_id"]),
        build_command=config["build"].get("command", ""),
        test_command=config["build"].get("test_command", ""),
        lint_command=config["build"].get("lint_command", ""),
        file_check_command=config["build"].get("file_check_command", ""),
        auto_discover_tests=config["build"].get("auto_discover_tests", False),
        max_build_test_retries=advanced.get("max_build_test_retries", 3),
        max_resolution_retries=advanced.get("max_resolution_retries", 3),
        forward_patch_id_lookback=advanced.get("forward_patch_id_lookback", "12 months"),
        allowlist_rename_expansion_cap=advanced.get("allowlist_rename_expansion_cap", 3),
        command_timeout=command_timeout,
        context_length=advanced.get("context_length"),
        mechanical_resolution=bool(advanced.get("mechanical_resolution", True)),
        snapback=bool(advanced.get("snapback", True)),
        evidence_checks=bool(advanced.get("evidence_checks", True)),
        post_resolution_check=bool(advanced.get("post_resolution_check", True)),
        blind_review=bool(advanced.get("blind_review", True)),
        partial_success_skip_ratio=advanced_float(
            advanced, "partial_success_skip_ratio", 0.5,
            minimum=0.0, maximum=1.0),
        phase_4b_model=phase_4b_profile.model,
        phase_4b_profile=phase_4b_profile,
        fix_patch=fix_patch,
        fix_upstream_ref=fix_upstream_ref,
    )

    return bug_fix_config, workdir, source_path


def parse_claude_config(
    config: dict[str, Any],
    output_dir: str,
    registry: "ModelRegistry | None" = None,
    *,
    profiles: tuple[ModelProfile, ModelProfile] | None = None,
) -> ClaudeConfig:
    """Parse model/agent configuration section.

    When a *registry* is provided, resolves the ``model:`` value as a
    profile id.  Otherwise builds a legacy profile from inline fields.

    Reads the model name from ``model:`` (top-level) or ``claude.model:``.
    If ``api_base:`` is present, configures the LLM client to route
    ``openai/`` prefixed models to that endpoint (e.g. a vLLM server).

    Additional optional fields:

    * ``model_params:`` — free-form dict merged into every LLM call
      (temperature, top_p, reasoning_effort, …).
    * ``api_key:`` / ``api_key_file:`` — explicit API key or path to a
      key file (``~`` expansion supported).
    """
    claude = config.get("claude", {})
    if not isinstance(claude, dict):
        claude = {}
    profile, _ = profiles or resolve_pipeline_profiles(config, registry)

    # Deprecated: set module-level state in llm_client for callers that
    # haven't adopted profiles yet. Named selections never seed legacy globals.
    model_ref = _pipeline_model_name(config)
    if registry is None or (model_ref is not None and model_ref not in registry.profiles):
        from common.llm_client import configure
        configure(api_base=profile.api_base, api_key=profile.api_key, model_params=profile.params)

    return BugFixClaudeConfig(
        model=profile.model,
        allowed_tools=claude.get("allowed_tools", ["Read", "Write", "Bash"]),
        perm_mode=claude.get("perm_mode", "acceptEdits"),
        cwd=output_dir,
        profile=profile,
    )


def load_pipeline_config(
    config_path: str,
    models_path: str | None = None,
) -> tuple[BugFixConfig, ClaudeConfig, str, str]:
    """Load and parse full pipeline configuration."""
    from common.model_registry import find_registry

    config = load_config_file(config_path)
    config_dir = str(Path(config_path).resolve().parent)
    output_section = config.get("output", {})
    if not isinstance(output_section, dict):
        output_section = {}
    output_dir = expand_path(output_section.get("directory"), "./runs")

    registry = find_registry(config_dir=config_dir, explicit_path=models_path)
    profiles = resolve_pipeline_profiles(config, registry)

    bug_fix_config, workdir, source_path = parse_bug_fix_config(
        config, config_dir=config_dir, registry=registry, profiles=profiles,
    )
    claude_config = parse_claude_config(config, output_dir, registry=registry, profiles=profiles)
    claude_config.command_timeout = bug_fix_config.command_timeout
    claude_config.context_length = bug_fix_config.context_length

    return bug_fix_config, claude_config, workdir, source_path


# ---------------------------------------------------------------------------
# Multi-target support
# ---------------------------------------------------------------------------

@dataclass
class TargetSpec:
    """Per-target overrides for multi-branch porting."""
    branch: str
    issue_id: str | None = None
    bug_description: str | None = None
    build_command: str | None = None
    test_command: str | None = None
    lint_command: str | None = None


def _sanitize_ref(ref: str) -> str:
    """Sanitize a git ref for use in filesystem paths.

    Strips ``origin/`` prefix and replaces ``/`` with ``_``.
    Commit SHAs pass through unchanged.
    """
    s = re.sub(r"^origin/", "", ref)
    return s.replace("/", "_")


def parse_targets(config: dict[str, Any]) -> list[TargetSpec]:
    """Normalize ``branches.target`` or ``branches.targets`` into a list of TargetSpec.

    Raises ConfigError if both or neither are present, or if two targets
    collide after sanitization.
    """
    branches = config.get("branches", {})
    if not isinstance(branches, dict):
        raise ConfigError("'branches' must be a mapping")

    has_singular = "target" in branches
    has_plural = "targets" in branches

    if has_singular and has_plural:
        raise ConfigError("branches must have 'target' or 'targets', not both")
    if not has_singular and not has_plural:
        raise ConfigError("branches must have 'target' or 'targets'")

    if has_singular:
        return [TargetSpec(branch=str(branches["target"]))]

    raw_targets = branches["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ConfigError("branches.targets must be a non-empty list")

    specs: list[TargetSpec] = []
    for entry in raw_targets:
        if isinstance(entry, (str, int, float)):
            specs.append(TargetSpec(branch=str(entry)))
        elif isinstance(entry, dict):
            if "branch" not in entry:
                raise ConfigError("each target mapping must have a 'branch' key")
            specs.append(TargetSpec(
                branch=str(entry["branch"]),
                issue_id=entry.get("issue_id"),
                bug_description=entry.get("bug_description"),
                build_command=entry.get("build_command"),
                test_command=entry.get("test_command"),
                lint_command=entry.get("lint_command"),
            ))
        else:
            raise ConfigError(
                f"each target must be a string or mapping, got {type(entry).__name__}"
            )

    seen: dict[str, str] = {}
    for spec in specs:
        key = _sanitize_ref(spec.branch)
        if key in seen:
            raise ConfigError(
                f"duplicate target after sanitization: '{spec.branch}' and "
                f"'{seen[key]}' both resolve to '{key}'"
            )
        seen[key] = spec.branch

    return specs


def load_multi_target_config(
    config_path: str,
    models_path: str | None = None,
) -> tuple[list[TargetSpec], BugFixConfig, ClaudeConfig, str, str]:
    """Load config with multi-target support.

    Returns (targets, base_config, claude_config, workdir, source_path).
    The base_config has ``target_branch`` set to the first target's branch
    to satisfy the required field.
    """
    from common.model_registry import find_registry

    config = load_config_file(config_path)
    config_dir = str(Path(config_path).resolve().parent)

    targets = parse_targets(config)

    # Pop 'targets' and inject synthetic 'target' so parse_bug_fix_config
    # (which reads config["branches"]["target"]) doesn't KeyError.
    # parse_targets already rejected having both, so no false positive.
    config.setdefault("branches", {})
    config["branches"].pop("targets", None)
    config["branches"]["target"] = targets[0].branch

    output_section = config.get("output", {})
    if not isinstance(output_section, dict):
        output_section = {}
    output_dir = expand_path(output_section.get("directory"), "./runs")

    registry = find_registry(config_dir=config_dir, explicit_path=models_path)
    profiles = resolve_pipeline_profiles(config, registry)

    bug_fix_config, workdir, source_path = parse_bug_fix_config(
        config, config_dir=config_dir, registry=registry, profiles=profiles,
    )
    claude_config = parse_claude_config(config, output_dir, registry=registry, profiles=profiles)
    claude_config.command_timeout = bug_fix_config.command_timeout
    claude_config.context_length = bug_fix_config.context_length

    return targets, bug_fix_config, claude_config, workdir, source_path
