from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def apply_cli_overrides(
    config: dict[str, Any],
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if data_root is not None:
        resolved["data"]["root"] = str(Path(data_root).resolve())
    if output_dir is not None:
        resolved["experiment"]["output_dir"] = str(Path(output_dir).resolve())
    if not resolved.get("data", {}).get("root"):
        raise ValueError("ImageNet root is missing. Pass --data-root.")
    return resolved


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    for dotted in keys:
        value: Any = config
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"Missing required configuration key: {dotted}")
            value = value[part]

