"""Helpers for recording reproducible experiment metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

type JsonValue = (
    str | int | float | bool | None | dict[str, JsonValue] | list[JsonValue]
)
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True)
class Environment:
    """Runtime metadata needed to interpret an experiment result."""

    python: str
    platform: str
    macos: str | None
    torch: str | None
    device: str
    dtype: str
    git_commit: str | None
    git_dirty: bool | None = None
    mps_available: bool | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    cuda_device: str | None = None


@dataclass(frozen=True)
class ResultManifest:
    """Serializable record for a single experiment run."""

    schema_version: str
    run_id: str
    created_at: str
    environment: Environment
    config: JsonObject
    seeds: list[int]
    metrics: JsonObject
    dataset: JsonObject | None = None
    artifacts: JsonObject | None = None
    notes: str | None = None

    def to_dict(self) -> JsonObject:
        """Return a JSON-compatible representation."""

        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write the manifest with stable formatting."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def collect_environment(device: str = "cpu", dtype: str = "complex64") -> Environment:
    """Collect local environment details for a result manifest."""

    return Environment(
        python=sys.version.split()[0],
        platform=platform.platform(),
        macos=_macos_version(),
        torch=_torch_version(),
        device=device,
        dtype=dtype,
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        mps_available=_mps_available(),
        cuda_available=_cuda_available(),
        cuda_version=_cuda_version(),
        cuda_device=_cuda_device(),
    )


def new_manifest(
    *,
    run_id: str,
    config: JsonObject,
    seeds: list[int],
    metrics: JsonObject,
    device: str = "cpu",
    dtype: str = "complex64",
    dataset: JsonObject | None = None,
    artifacts: JsonObject | None = None,
    notes: str | None = None,
) -> ResultManifest:
    """Create a result manifest using the current environment."""

    return ResultManifest(
        schema_version="0.1.0",
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        environment=collect_environment(device=device, dtype=dtype),
        config=config,
        seeds=seeds,
        metrics=metrics,
        dataset=dataset,
        artifacts=artifacts,
        notes=notes,
    )


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return str(torch.__version__)


def _mps_available() -> bool | None:
    try:
        import torch
    except ImportError:
        return None
    return bool(torch.backends.mps.is_available())


def _cuda_available() -> bool | None:
    try:
        import torch
    except ImportError:
        return None
    return bool(torch.cuda.is_available())


def _cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return None if torch.version.cuda is None else str(torch.version.cuda)


def _cuda_device() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _macos_version() -> str | None:
    version = platform.mac_ver()[0]
    return version or None
