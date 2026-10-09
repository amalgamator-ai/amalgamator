import json
import os
import sys

VALID_FRAMEWORKS = ("vllm", "sgl", "trt")

# Extra output tokens beyond kv_cache_fill to ensure the profiling window fits.
# Must be >= NUM_TRACE_ITERS (50) defined in config.sh.
PROFILING_BUFFER = 256

DEFAULT_NUM_WAVES = 4

PROFILE_REQUIRED_FIELDS = {
    "model": str,
    "gpu_ids": (str, list),
}


def parse_frameworks(frameworks_list, profile_idx, config_dir, errors):
    parsed = {}
    if not isinstance(frameworks_list, list):
        errors.append(f'"profiles[{profile_idx}].frameworks" must be a list.')
        return parsed

    for entry in frameworks_list:
        if not isinstance(entry, dict):
            errors.append(
                f'"profiles[{profile_idx}].frameworks" entries must be objects '
                f'with at least a "name" field. Got: {entry!r}.'
            )
            continue

        name = entry.get("name")
        if not isinstance(name, str) or name not in VALID_FRAMEWORKS:
            errors.append(
                f'"profiles[{profile_idx}].frameworks" entry has invalid '
                f'"name": {name!r}. Must be one of: {list(VALID_FRAMEWORKS)}.'
            )
            continue

        if name in parsed:
            errors.append(
                f'"profiles[{profile_idx}].frameworks" has duplicate '
                f'framework "{name}".'
            )
            continue

        mode = entry.get("mode")
        if mode is not None:
            if not isinstance(mode, str) or len(mode) == 0:
                errors.append(
                    f'"profiles[{profile_idx}].frameworks" entry for "{name}" '
                    f'has invalid "mode": must be a non-empty string.'
                )
                mode = None
            else:
                mode_path = f"{config_dir}/modes/{name}/{mode}"
                if not os.path.isfile(mode_path):
                    errors.append(
                        f'"profiles[{profile_idx}].frameworks" entry for "{name}" '
                        f'references mode "{mode}" but file not found '
                        f"({mode_path})."
                    )

        enable_trace = entry.get("enable_trace", False)
        if not isinstance(enable_trace, bool):
            errors.append(
                f'"profiles[{profile_idx}].frameworks" entry for "{name}" '
                f'has invalid "enable_trace": must be a boolean.'
            )
            enable_trace = False

        parsed[name] = {"mode": mode, "enable_trace": enable_trace}

    return parsed


def validate_gpu_configs(gpu_configs, errors):
    if not isinstance(gpu_configs, dict):
        errors.append('"gpu_configs" must be an object mapping names to GPU ID lists.')
        return
    for name, ids in gpu_configs.items():
        if not isinstance(ids, list) or len(ids) == 0:
            errors.append(f'"gpu_configs.{name}" must be a non-empty list of GPU numbers.')
        elif not all(isinstance(g, int) and g >= 0 for g in ids):
            errors.append(f'"gpu_configs.{name}" must contain non-negative integers.')


def load_json_file(field_name, file_path, errors):
    if not os.path.isfile(file_path):
        errors.append(f'"{field_name}" path not found: {file_path}')
        return None

    with open(file_path) as f:
        return json.load(f)


def validate_docker_images(docker_images, errors):
    if not isinstance(docker_images, dict):
        errors.append('"docker_images_file" must contain an object mapping framework names to image strings.')
        return
    for name, image in docker_images.items():
        if name not in VALID_FRAMEWORKS:
            errors.append(
                f'docker_images contains invalid key "{name}". '
                f"Valid keys are: {list(VALID_FRAMEWORKS)}."
            )
        elif not isinstance(image, str) or len(image) == 0:
            errors.append(f'docker_images.{name} must be a non-empty string.')


def load_docker_images(docker_images_file, errors):
    docker_images = load_json_file("docker_images_file", docker_images_file, errors)
    if docker_images is not None:
        validate_docker_images(docker_images, errors)
    return docker_images


def validate_exec_modes(exec_modes, errors):
    if not isinstance(exec_modes, dict):
        errors.append('"exec_modes_file" must contain an object mapping names to {input_len, output_len} or {input_len, kv_cache_fill}.')
        return
    for name, cfg in exec_modes.items():
        if not isinstance(cfg, dict):
            errors.append(f'exec_modes.{name} must be an object with "input_len" and either "output_len" or "kv_cache_fill".')
            continue
        if "input_len" not in cfg or not isinstance(cfg["input_len"], int) or cfg["input_len"] <= 0:
            errors.append(f'exec_modes.{name}.input_len must be a positive integer.')
        has_output = "output_len" in cfg
        has_kvcf = "kv_cache_fill" in cfg
        if has_output and has_kvcf:
            errors.append(f'exec_modes.{name} must have either "output_len" or "kv_cache_fill", not both.')
        elif not has_output and not has_kvcf:
            errors.append(f'exec_modes.{name} must have either "output_len" or "kv_cache_fill".')
        elif has_output:
            if not isinstance(cfg["output_len"], int) or cfg["output_len"] <= 0:
                errors.append(f'exec_modes.{name}.output_len must be a positive integer.')
        elif has_kvcf:
            if not isinstance(cfg["kv_cache_fill"], int) or cfg["kv_cache_fill"] <= 0:
                errors.append(f'exec_modes.{name}.kv_cache_fill must be a positive integer.')
            elif isinstance(cfg.get("input_len"), int) and cfg["kv_cache_fill"] <= cfg["input_len"]:
                errors.append(f'exec_modes.{name}.kv_cache_fill must be greater than input_len.')


def resolve_exec_mode(i, p, exec_modes, errors):
    exec_mode = p.get("exec_mode")
    if exec_mode is None:
        return
    if not isinstance(exec_mode, str):
        errors.append(f'"profiles[{i}].exec_mode" must be a string.')
        return
    if exec_modes is None:
        errors.append(
            f'"profiles[{i}].exec_mode" references "{exec_mode}" '
            f'but no "exec_modes_file" is specified.'
        )
        return
    if exec_mode not in exec_modes:
        errors.append(
            f'"profiles[{i}].exec_mode" references "{exec_mode}" '
            f"which is not defined in exec_modes. "
            f"Available: {list(exec_modes.keys())}."
        )
        return
    cfg = exec_modes[exec_mode]
    p["input_len"] = cfg["input_len"]
    if "output_len" in cfg:
        p["output_len"] = cfg["output_len"]
        p["kv_cache_fill"] = 0
    else:
        p["output_len"] = cfg["kv_cache_fill"] + PROFILING_BUFFER
        p["kv_cache_fill"] = cfg["kv_cache_fill"]


def load_gpu_configs(gpu_configs_file, errors):
    gpu_configs = load_json_file("gpu_configs_file", gpu_configs_file, errors)
    if gpu_configs is not None:
        validate_gpu_configs(gpu_configs, errors)
    return gpu_configs


def load_exec_modes(exec_modes_file, errors):
    exec_modes = load_json_file("exec_modes_file", exec_modes_file, errors)
    if exec_modes is not None:
        validate_exec_modes(exec_modes, errors)
    return exec_modes


def resolve_gpu_ids(i, p, gpu_configs, errors):
    gpu_ids = p.get("gpu_ids")
    if gpu_ids is None:
        return
    if isinstance(gpu_ids, str):
        if gpu_configs is None:
            errors.append(
                f'"profiles[{i}].gpu_ids" references "{gpu_ids}" '
                f'but no "gpu_configs_file" is specified.'
            )
        elif gpu_ids not in gpu_configs:
            errors.append(
                f'"profiles[{i}].gpu_ids" references "{gpu_ids}" '
                f"which is not defined in gpu_configs. "
                f"Available: {list(gpu_configs.keys())}."
            )
        else:
            p["gpu_ids"] = gpu_configs[gpu_ids]


def validate_profile(i, p, config_dir, errors):
    if not isinstance(p, dict):
        errors.append(
            f'"profiles[{i}]" must be an object with fields: '
            f'{", ".join(PROFILE_REQUIRED_FIELDS)}.'
        )
        return

    for field, ftype in PROFILE_REQUIRED_FIELDS.items():
        if field not in p:
            errors.append(f'"profiles[{i}].{field}" is required.')
        elif not isinstance(p[field], ftype):
            errors.append(f'"profiles[{i}].{field}" must be of type {ftype.__name__}.')

    if "model" in p and isinstance(p["model"], str) and len(p["model"]) == 0:
        errors.append(f'"profiles[{i}].model" must be a non-empty string.')

    if "gpu_ids" in p and isinstance(p["gpu_ids"], list):
        if len(p["gpu_ids"]) == 0:
            errors.append(f'"profiles[{i}].gpu_ids" must be a non-empty list of GPU numbers.')
        elif not all(isinstance(g, int) and g >= 0 for g in p["gpu_ids"]):
            errors.append(f'"profiles[{i}].gpu_ids" must contain non-negative integers.')

    has_exec_mode = "exec_mode" in p
    has_inline = "input_len" in p or "output_len" in p or "kv_cache_fill" in p
    if not has_exec_mode and not has_inline:
        errors.append(
            f'"profiles[{i}]" must specify either "exec_mode" '
            f'or "input_len" with "output_len" or "kv_cache_fill".'
        )
    elif not has_exec_mode:
        if "input_len" not in p:
            errors.append(f'"profiles[{i}].input_len" is required when "exec_mode" is not set.')
        elif not isinstance(p["input_len"], int) or p["input_len"] <= 0:
            errors.append(f'"profiles[{i}].input_len" must be a positive integer.')

        has_output = "output_len" in p
        has_kvcf = "kv_cache_fill" in p
        if has_output and has_kvcf:
            errors.append(f'"profiles[{i}]" must have either "output_len" or "kv_cache_fill", not both.')
        elif not has_output and not has_kvcf:
            errors.append(f'"profiles[{i}]" must have either "output_len" or "kv_cache_fill".')
        elif has_output:
            if not isinstance(p["output_len"], int) or p["output_len"] <= 0:
                errors.append(f'"profiles[{i}].output_len" must be a positive integer.')
        elif has_kvcf:
            if not isinstance(p["kv_cache_fill"], int) or p["kv_cache_fill"] <= 0:
                errors.append(f'"profiles[{i}].kv_cache_fill" must be a positive integer.')
            elif isinstance(p.get("input_len"), int) and p["kv_cache_fill"] <= p["input_len"]:
                errors.append(f'"profiles[{i}].kv_cache_fill" must be greater than input_len.')
            else:
                p["output_len"] = p["kv_cache_fill"] + PROFILING_BUFFER

    if "frameworks" in p:
        parse_frameworks(p["frameworks"], i, config_dir, errors)


def shell_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def load_infra_config(infra_path, errors):
    infra = load_json_file("infra_config", infra_path, errors)
    if infra is None:
        return None, None, None

    if not isinstance(infra, dict):
        errors.append("infra_config must be a JSON object.")
        return None, None, None

    docker_images = None
    docker_images_file = infra.get("docker_images_file")
    if docker_images_file is not None:
        if not isinstance(docker_images_file, str):
            errors.append('"docker_images_file" must be a string path.')
        else:
            docker_images = load_docker_images(docker_images_file, errors)

    gpu_configs = None
    gpu_configs_file = infra.get("gpu_configs_file")
    if gpu_configs_file is not None:
        if not isinstance(gpu_configs_file, str):
            errors.append('"gpu_configs_file" must be a string path.')
        else:
            gpu_configs = load_gpu_configs(gpu_configs_file, errors)

    exec_modes = None
    exec_modes_file = infra.get("exec_modes_file")
    if exec_modes_file is not None:
        if not isinstance(exec_modes_file, str):
            errors.append('"exec_modes_file" must be a string path.')
        else:
            exec_modes = load_exec_modes(exec_modes_file, errors)

    return docker_images, gpu_configs, exec_modes


def parse_config(infra_config_path, run_config_path):
    with open(run_config_path) as f:
        c = json.load(f)

    errors = []

    config_dir = os.path.dirname(run_config_path)

    docker_images, gpu_configs, exec_modes = load_infra_config(
        infra_config_path, errors
    )

    profiles = c.get("profiles", [])
    if not isinstance(profiles, list) or len(profiles) == 0:
        errors.append('"profiles" must be a non-empty list.')
    else:
        for i, p in enumerate(profiles):
            resolve_gpu_ids(i, p, gpu_configs, errors)
            resolve_exec_mode(i, p, exec_modes, errors)
            validate_profile(i, p, config_dir, errors)

    if "concurrencies" not in c or not isinstance(c["concurrencies"], list) or len(c["concurrencies"]) == 0:
        errors.append('"concurrencies" must be a non-empty list of integers.')

    num_waves = c.get("num_waves", DEFAULT_NUM_WAVES)
    if not isinstance(num_waves, int) or num_waves <= 0:
        errors.append('"num_waves" must be a positive integer.')
        num_waves = DEFAULT_NUM_WAVES

    all_parsed_frameworks = []
    all_framework_names = set()
    all_trace_frameworks = set()

    for i, p in enumerate(profiles):
        parsed_fw = parse_frameworks(p.get("frameworks", []), i, config_dir, [])
        all_parsed_frameworks.append(parsed_fw)
        for fw_name, fw_cfg in parsed_fw.items():
            all_framework_names.add(fw_name)
            if fw_cfg["enable_trace"]:
                all_trace_frameworks.add(fw_name)

    frameworks_ordered = [fw for fw in VALID_FRAMEWORKS if fw in all_framework_names]
    traces_ordered = [fw for fw in VALID_FRAMEWORKS if fw in all_trace_frameworks]

    if docker_images is not None and frameworks_ordered:
        fw_missing_image = [fw for fw in frameworks_ordered if fw not in docker_images]
        if fw_missing_image:
            errors.append(
                f'Profiles reference frameworks {fw_missing_image} but no docker image '
                f"is defined for them in docker_images_file."
            )
    elif docker_images is None and frameworks_ordered:
        errors.append(
            'Profiles reference frameworks but no "docker_images_file" is specified in infra config.'
        )

    if errors:
        for e in errors:
            print(f'log_error "Config error: {e}"')
        print("return 1")
        return

    concurrencies = " ".join(str(x) for x in c["concurrencies"])

    def make_assoc(value_fn):
        entries = " ".join(
            f'["{shell_escape(p["model"])}"]="{shell_escape(value_fn(p, pf))}"'
            for p, pf in zip(profiles, all_parsed_frameworks)
        )
        return entries

    profiles_arr = " ".join(f'"{shell_escape(p["model"])}"' for p in profiles)

    for fw in VALID_FRAMEWORKS:
        var_name = f"{fw.upper()}_DOCKER_IMAGE"
        image = docker_images.get(fw, "")
        print(f'{var_name}="{shell_escape(image)}"')

    print(f"PROFILES=({profiles_arr})")
    print(f"declare -gA PROFILE_GPU_IDS=({make_assoc(lambda p, _: ','.join(str(g) for g in p['gpu_ids']))})")
    print(f"declare -gA PROFILE_INPUT_LENS=({make_assoc(lambda p, _: str(p['input_len']))})")
    print(f"declare -gA PROFILE_OUTPUT_LENS=({make_assoc(lambda p, _: str(p['output_len']))})")
    print(f"declare -gA PROFILE_KV_CACHE_FILLS=({make_assoc(lambda p, _: str(p.get('kv_cache_fill', 0)))})")
    print(f"declare -gA PROFILE_VLLM_MODES=({make_assoc(lambda _, pf: pf.get('vllm', {}).get('mode') or 'none')})")
    print(f"declare -gA PROFILE_SGL_MODES=({make_assoc(lambda _, pf: pf.get('sgl', {}).get('mode') or 'none')})")
    print(f"declare -gA PROFILE_TRT_MODES=({make_assoc(lambda _, pf: pf.get('trt', {}).get('mode') or 'none')})")
    print(f'PROFILE_CONCURRENCIES="{concurrencies}"')
    print(f"NUM_WAVES={num_waves}")
    print(f'RUN_FRAMEWORKS="{" ".join(frameworks_ordered)}"')
    print(f'ENABLE_TRACES="{" ".join(traces_ordered)}"')
    print(f'RUN_CONFIG_DIR="{shell_escape(config_dir)}"')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('log_error "Usage: parse_run_config.py <infra_config.json> <run_config.json>"')
        print("return 1")
        sys.exit(1)
    parse_config(sys.argv[1], sys.argv[2])
