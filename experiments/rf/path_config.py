"""Path resolution helpers for local RadioML data.

The RadioML archive is gated and usually lives outside the repo. This module
lets scripts resolve that location from CLI flags, environment variables, or a
local JSON config that is intentionally git-ignored.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RADIOML_DATA_PATH = Path("data/GOLD_XYZ_OSC.0001_1024.hdf5")
DEFAULT_RADIOML_PATHS_CONFIG = Path("config/radioml_paths.json")

RADIOML_PATHS_CONFIG_ENV = "RADIOML_PATHS_CONFIG"
RADIOML_DATA_PATH_ENV = "RADIOML_DATA_PATH"
RADIOML_CLASSES_PATH_ENV = "RADIOML_CLASSES_PATH"


@dataclass(frozen=True)
class RadioMLPathResolution:
    """Resolved RadioML paths and where they came from."""

    data_path: Path
    classes_path: Path | None
    data_path_source: str
    classes_path_source: str
    config_path: Path | None


def resolve_radioml_paths(
    *,
    data_path: Path | None,
    classes_path: Path | None,
    config_path: Path | None = None,
) -> RadioMLPathResolution:
    """Resolve RadioML archive paths.

    Precedence:
      1. explicit CLI arguments
      2. environment variables (`RADIOML_DATA_PATH`, `RADIOML_CLASSES_PATH`)
      3. JSON config (`config/radioml_paths.json` by default, or
         `RADIOML_PATHS_CONFIG` / `--paths-config`)
      4. the historical repo default under `data/`
    """

    loaded_config_path, config = _load_config(config_path)
    config_base_dir = loaded_config_path.parent if loaded_config_path else None

    env_data_path = _nonempty_env(RADIOML_DATA_PATH_ENV)
    env_classes_path = _nonempty_env(RADIOML_CLASSES_PATH_ENV)
    config_data_path = _nonempty_config_string(config, "data_path")
    config_classes_path = _nonempty_config_string(config, "classes_path")

    if data_path is not None:
        resolved_data_path = _coerce_path(data_path)
        data_source = "cli:--data-path"
    elif env_data_path is not None:
        resolved_data_path = _coerce_path(env_data_path)
        data_source = f"env:{RADIOML_DATA_PATH_ENV}"
    elif config_data_path is not None:
        resolved_data_path = _coerce_path(config_data_path, base_dir=config_base_dir)
        source_path = loaded_config_path or DEFAULT_RADIOML_PATHS_CONFIG
        data_source = f"config:{source_path}:data_path"
    else:
        resolved_data_path = DEFAULT_RADIOML_DATA_PATH
        data_source = "default"

    if classes_path is not None:
        resolved_classes_path = _coerce_path(classes_path)
        classes_source = "cli:--classes-path"
    elif env_classes_path is not None:
        resolved_classes_path = _coerce_path(env_classes_path)
        classes_source = f"env:{RADIOML_CLASSES_PATH_ENV}"
    elif config_classes_path is not None:
        resolved_classes_path = _coerce_path(
            config_classes_path, base_dir=config_base_dir
        )
        source_path = loaded_config_path or DEFAULT_RADIOML_PATHS_CONFIG
        classes_source = f"config:{source_path}:classes_path"
    else:
        resolved_classes_path = None
        classes_source = "loader-sidecar-default"

    return RadioMLPathResolution(
        data_path=resolved_data_path,
        classes_path=resolved_classes_path,
        data_path_source=data_source,
        classes_path_source=classes_source,
        config_path=loaded_config_path,
    )


def _load_config(
    config_path: Path | None,
) -> tuple[Path | None, Mapping[str, Any]]:
    explicit_config_path = config_path is not None
    env_config_path = _nonempty_env(RADIOML_PATHS_CONFIG_ENV)
    if config_path is None:
        if env_config_path is not None:
            config_path = Path(env_config_path)
            explicit_config_path = True
        else:
            config_path = DEFAULT_RADIOML_PATHS_CONFIG

    if not config_path.exists():
        if explicit_config_path:
            msg = f"RadioML paths config not found at {config_path}"
            raise FileNotFoundError(msg)
        return None, {}

    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        msg = f"RadioML paths config must be a JSON object: {config_path}"
        raise TypeError(msg)

    section = payload.get("radioml_2018_01a", payload)
    if not isinstance(section, dict):
        msg = (
            "RadioML paths config key `radioml_2018_01a` must be a JSON object: "
            f"{config_path}"
        )
        raise TypeError(msg)
    return config_path, section


def _nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _nonempty_config_string(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"RadioML paths config `{key}` must be a string or null"
        raise TypeError(msg)
    if not value.strip():
        return None
    return value


def _coerce_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path


__all__ = [
    "DEFAULT_RADIOML_DATA_PATH",
    "DEFAULT_RADIOML_PATHS_CONFIG",
    "RADIOML_CLASSES_PATH_ENV",
    "RADIOML_DATA_PATH_ENV",
    "RADIOML_PATHS_CONFIG_ENV",
    "RadioMLPathResolution",
    "resolve_radioml_paths",
]
