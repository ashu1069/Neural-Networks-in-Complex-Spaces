from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rf.path_config import (
    DEFAULT_RADIOML_DATA_PATH,
    RADIOML_CLASSES_PATH_ENV,
    RADIOML_DATA_PATH_ENV,
    RADIOML_PATHS_CONFIG_ENV,
    resolve_radioml_paths,
)


def test_resolve_radioml_paths_uses_repo_default_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RADIOML_PATHS_CONFIG_ENV, raising=False)
    monkeypatch.delenv(RADIOML_DATA_PATH_ENV, raising=False)
    monkeypatch.delenv(RADIOML_CLASSES_PATH_ENV, raising=False)

    resolved = resolve_radioml_paths(data_path=None, classes_path=None)

    assert resolved.data_path == DEFAULT_RADIOML_DATA_PATH
    assert resolved.classes_path is None
    assert resolved.data_path_source == "default"
    assert resolved.classes_path_source == "loader-sidecar-default"


def test_resolve_radioml_paths_reads_config_relative_to_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RADIOML_DATA_PATH_ENV, raising=False)
    monkeypatch.delenv(RADIOML_CLASSES_PATH_ENV, raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "radioml_paths.json"
    config_path.write_text(
        json.dumps(
            {
                "radioml_2018_01a": {
                    "data_path": "../drive/GOLD_XYZ_OSC.0001_1024.hdf5",
                    "classes_path": "../drive/classes-fixed.json",
                }
            }
        )
    )

    resolved = resolve_radioml_paths(
        data_path=None,
        classes_path=None,
        config_path=config_path,
    )

    assert resolved.data_path == config_dir / "../drive/GOLD_XYZ_OSC.0001_1024.hdf5"
    assert resolved.classes_path == config_dir / "../drive/classes-fixed.json"
    assert resolved.config_path == config_path


def test_resolve_radioml_paths_prefers_cli_over_env_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "radioml_paths.json"
    config_path.write_text(json.dumps({"data_path": "/from/config.hdf5"}))
    monkeypatch.setenv(RADIOML_DATA_PATH_ENV, "/from/env.hdf5")

    resolved = resolve_radioml_paths(
        data_path=Path("/from/cli.hdf5"),
        classes_path=None,
        config_path=config_path,
    )

    assert resolved.data_path == Path("/from/cli.hdf5")
    assert resolved.data_path_source == "cli:--data-path"


def test_resolve_radioml_paths_uses_env_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "radioml_paths.json"
    config_path.write_text(
        json.dumps({"radioml_2018_01a": {"data_path": "/from/env-config.hdf5"}})
    )
    monkeypatch.setenv(RADIOML_PATHS_CONFIG_ENV, str(config_path))
    monkeypatch.delenv(RADIOML_DATA_PATH_ENV, raising=False)

    resolved = resolve_radioml_paths(data_path=None, classes_path=None)

    assert resolved.data_path == Path("/from/env-config.hdf5")
    assert resolved.config_path == config_path
