"""RadioML 2018.01A loader.

The dataset is distributed by DeepSig (https://www.deepsig.ai/datasets) under a
registration-required licence. We don't bundle it; this module loads from a
local HDF5 file that the user has obtained separately. See
[`docs/radioml.md`](../../docs/radioml.md) for acquisition instructions and the
expected file layout.

The 2018.01A archive ships as `GOLD_XYZ_OSC.0001_1024.hdf5` (or similar) with
three top-level datasets:

- `X` — `(N, 1024, 2)` float32, the (real, imag) IQ sequence per example
- `Y` — `(N, 24)` int8/float32 one-hot modulation labels
- `Z` — `(N, 1)` int8 SNR in dB (-20 to +30 in 2 dB steps)

The loader returns an `RFModulationData` (the same dataclass the synthetic
benchmark uses), so the rest of the experiment harness works unchanged.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.rf.synthetic_modulation import RFModulationData

RADIOML_2018_01A_MODULATIONS: tuple[str, ...] = (
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
)

RADIOML_2018_01A_SNR_DB: tuple[int, ...] = tuple(range(-20, 32, 2))

_FIXED_CLASS_SIDECARS: tuple[str, ...] = ("classes-fixed.json", "classes-fixed.txt")

_MODULATION_ALIASES: Mapping[str, str] = {
    "ASK4": "4ASK",
    "ASK8": "8ASK",
    "PSK8": "8PSK",
    "PSK16": "16PSK",
    "PSK32": "32PSK",
    "APSK16": "16APSK",
    "APSK32": "32APSK",
    "APSK64": "64APSK",
    "APSK128": "128APSK",
    "QAM16": "16QAM",
    "QAM32": "32QAM",
    "QAM64": "64QAM",
    "QAM128": "128QAM",
    "QAM256": "256QAM",
    "AM_SSB_WC": "AM-SSB-WC",
    "AM_SSB_SC": "AM-SSB-SC",
    "AM_DSB_WC": "AM-DSB-WC",
    "AM_DSB_SC": "AM-DSB-SC",
    "OQPS": "OQPSK",
}


def load_radioml_2018_01a(
    path: str | Path,
    *,
    modulations: Sequence[str] | None = None,
    snr_db_levels: Sequence[int] | None = None,
    max_per_class_per_snr: int | None = None,
    train_fraction: float = 0.8,
    seed: int = 0,
    dtype: torch.dtype = torch.complex64,
    sample_length: int = 1024,
    classes_path: str | Path | None = None,
    strict_snr: bool = True,
) -> RFModulationData:
    """Load a (filtered) subset of RadioML 2018.01A from a local HDF5 file.

    Filtering is done at load time so we don't materialize the full 2.55M-sample
    archive in memory; only the rows matching the requested `modulations` and
    `snr_db_levels` are read. `max_per_class_per_snr` further caps per-bucket
    sample count for fast smoke runs.

    Returns the same `RFModulationData` shape the synthetic benchmark produces,
    so `train_rf_classifier` etc. work unchanged - just point them at the
    output of this function instead of `make_synthetic_rf_modulation_dataset`.
    """

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guarded
        msg = "h5py is required to load RadioML; install with `uv add h5py`"
        raise ImportError(msg) from exc

    path = Path(path)
    if not path.exists():
        msg = (
            f"RadioML HDF5 not found at {path}. "
            "See docs/radioml.md for acquisition instructions."
        )
        raise FileNotFoundError(msg)
    if not (0.0 < train_fraction < 1.0):
        msg = "train_fraction must lie in (0, 1)"
        raise ValueError(msg)
    if not dtype.is_complex:
        msg = f"RadioML inputs require a complex dtype, got {dtype}"
        raise TypeError(msg)
    if sample_length <= 0:
        msg = "sample_length must be positive"
        raise ValueError(msg)

    requested_mods = _normalize_modulation_names(
        tuple(modulations) if modulations is not None else RADIOML_2018_01A_MODULATIONS
    )
    requested_snrs = (
        tuple(int(s) for s in snr_db_levels)
        if snr_db_levels is not None
        else RADIOML_2018_01A_SNR_DB
    )
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    with h5py.File(path, "r") as handle:
        x_dataset = handle["X"]
        y_dataset = handle["Y"]
        z_dataset = handle["Z"]
        archive_modulations = _resolve_archive_modulations(
            handle, archive_path=path, classes_path=classes_path
        )

        mod_to_archive_index = {
            name: idx for idx, name in enumerate(archive_modulations)
        }
        if len(mod_to_archive_index) != len(archive_modulations):
            msg = (
                "archive class names are not unique after alias normalization; "
                f"resolved names: {archive_modulations}"
            )
            raise ValueError(msg)
        for mod in requested_mods:
            if mod not in mod_to_archive_index:
                msg = (
                    f"modulation {mod!r} is not in the archive's class set; "
                    f"available: {archive_modulations}"
                )
                raise ValueError(msg)

        labels_full = _argmax_one_hot(y_dataset[:])
        snrs_full = _flatten_snr(z_dataset[:])

        gen = torch.Generator(device="cpu").manual_seed(seed)
        train_inputs_chunks: list[torch.Tensor] = []
        train_labels_chunks: list[torch.Tensor] = []
        train_snr_chunks: list[torch.Tensor] = []
        test_inputs_chunks: list[torch.Tensor] = []
        test_labels_chunks: list[torch.Tensor] = []
        test_snr_chunks: list[torch.Tensor] = []

        empty_buckets: list[tuple[str, int]] = []
        for output_label, mod in enumerate(requested_mods):
            archive_label = mod_to_archive_index[mod]
            for snr in requested_snrs:
                mask = (labels_full == archive_label) & (snrs_full == snr)
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                if indices.numel() == 0:
                    empty_buckets.append((mod, int(snr)))
                    continue
                if max_per_class_per_snr is not None:
                    cap = min(max_per_class_per_snr, int(indices.numel()))
                    pick = indices[torch.randperm(indices.numel(), generator=gen)[:cap]]
                else:
                    pick = indices
                pick_sorted, _ = torch.sort(pick)
                # h5py wants sorted indices; round-trip through numpy.
                rows = x_dataset[pick_sorted.numpy(), :, :]
                trimmed = _trim_sample_length(rows, sample_length=sample_length)
                complex_rows = torch.complex(
                    torch.tensor(trimmed[..., 0], dtype=real_dtype),
                    torch.tensor(trimmed[..., 1], dtype=real_dtype),
                )
                n_total = complex_rows.shape[0]
                permutation = torch.randperm(n_total, generator=gen)
                shuffled = complex_rows[permutation]
                n_train = int(math.floor(n_total * train_fraction))
                train_chunk = shuffled[:n_train]
                test_chunk = shuffled[n_train:]
                train_inputs_chunks.append(train_chunk.to(dtype))
                test_inputs_chunks.append(test_chunk.to(dtype))
                train_labels_chunks.append(
                    torch.full((train_chunk.shape[0],), output_label, dtype=torch.long)
                )
                test_labels_chunks.append(
                    torch.full((test_chunk.shape[0],), output_label, dtype=torch.long)
                )
                train_snr_chunks.append(
                    torch.full((train_chunk.shape[0],), int(snr), dtype=torch.long)
                )
                test_snr_chunks.append(
                    torch.full((test_chunk.shape[0],), int(snr), dtype=torch.long)
                )

    if not train_inputs_chunks:
        msg = (
            "no examples matched the requested filters - check that "
            f"modulations={list(requested_mods)} and "
            f"snr_db_levels={list(requested_snrs)} are present in the archive"
        )
        raise ValueError(msg)

    if empty_buckets:
        # RadioML 2018.01A only has even SNRs (-20, -18, ..., +28, +30). Asking
        # for an odd SNR like -5 silently used to drop the bucket and produced
        # a per-SNR table with mysteriously missing rows. Either raise so the
        # user catches the typo, or merely warn and continue.
        bucket_str = ", ".join(f"({mod}, {snr} dB)" for mod, snr in empty_buckets)
        msg = (
            f"requested (mod, SNR) buckets had zero examples in the archive: "
            f"{bucket_str}. RadioML 2018.01A uses 2 dB steps starting from -20 "
            f"so odd SNRs do not exist. Pass strict_snr=False to silently skip "
            f"empty buckets instead of raising."
        )
        if strict_snr:
            raise ValueError(msg)
        import warnings

        warnings.warn(msg, stacklevel=2)

    train_inputs = torch.cat(train_inputs_chunks, dim=0)
    train_labels = torch.cat(train_labels_chunks, dim=0)
    train_snr_db = torch.cat(train_snr_chunks, dim=0)
    test_inputs = torch.cat(test_inputs_chunks, dim=0)
    test_labels = torch.cat(test_labels_chunks, dim=0)
    test_snr_db = torch.cat(test_snr_chunks, dim=0)

    train_perm = torch.randperm(train_inputs.shape[0], generator=gen)
    test_perm = torch.randperm(test_inputs.shape[0], generator=gen)
    return RFModulationData(
        train_inputs=train_inputs[train_perm],
        train_labels=train_labels[train_perm],
        train_snr_db=train_snr_db[train_perm],
        test_inputs=test_inputs[test_perm],
        test_labels=test_labels[test_perm],
        test_snr_db=test_snr_db[test_perm],
        modulation_names=tuple(requested_mods),  # type: ignore[arg-type]
        snr_db_levels=tuple(requested_snrs),
        sample_length=sample_length,
    )


def _resolve_archive_modulations(
    handle: Any,
    *,
    archive_path: Path,
    classes_path: str | Path | None,
) -> tuple[str, ...]:
    """Return the modulation names ordered by their one-hot index in `Y`.

    DeepSig's original `classes.txt` is known to use the wrong order for
    2018.01A. Prefer the fixed sidecars distributed with this project/dataset,
    then fall back to an HDF5-embedded class list or the documented paper order.
    """

    if classes_path is not None:
        return _load_modulation_sidecar(Path(classes_path))

    for sidecar_name in _FIXED_CLASS_SIDECARS:
        candidate = archive_path.with_name(sidecar_name)
        if candidate.exists():
            return _load_modulation_sidecar(candidate)

    if "classes" in handle:
        raw = handle["classes"][:]
        return _normalize_modulation_names(
            tuple(
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in raw
            )
        )
    return RADIOML_2018_01A_MODULATIONS


def _load_modulation_sidecar(path: Path) -> tuple[str, ...]:
    if not path.exists():
        msg = f"RadioML class sidecar not found at {path}"
        raise FileNotFoundError(msg)

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return _modulation_names_from_payload(payload, path=path)
    return _modulation_names_from_text(path.read_text(), path=path)


def _modulation_names_from_payload(payload: Any, *, path: Path) -> tuple[str, ...]:
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        return _coerce_modulation_names(payload, path=path)
    if isinstance(payload, Mapping):
        for key in ("classes", "modulations", "labels"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                return _coerce_modulation_names(value, path=path)

        numeric_items: list[tuple[int, Any]] = []
        for key, value in payload.items():
            try:
                numeric_items.append((int(str(key)), value))
            except ValueError:
                numeric_items.clear()
                break
        if numeric_items:
            return _coerce_modulation_names(
                [value for _, value in sorted(numeric_items)], path=path
            )

    msg = f"could not parse RadioML class names from {path}"
    raise ValueError(msg)


def _modulation_names_from_text(text: str, *, path: Path) -> tuple[str, ...]:
    try:
        module = ast.parse(text, filename=str(path))
    except SyntaxError:
        module = None

    if module is not None:
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "classes"
                for target in node.targets
            ):
                return _coerce_modulation_names(ast.literal_eval(node.value), path=path)

    lines = [
        line.strip().strip(",").strip("'\"")
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _coerce_modulation_names(lines, path=path)


def _coerce_modulation_names(values: Sequence[Any], *, path: Path) -> tuple[str, ...]:
    names = _normalize_modulation_names(tuple(str(value) for value in values))
    if not names or any(not name for name in names):
        msg = f"empty RadioML class name in {path}"
        raise ValueError(msg)
    return names


def _normalize_modulation_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalize_modulation_name(name) for name in names)


def _normalize_modulation_name(name: str) -> str:
    key = str(name).strip().upper()
    return _MODULATION_ALIASES.get(key, key)


def _argmax_one_hot(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    return tensor.argmax(dim=-1).to(torch.long)


def _flatten_snr(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(values).reshape(-1).to(torch.long)
    return tensor


def _trim_sample_length(rows: Any, *, sample_length: int) -> Any:
    if rows.shape[1] < sample_length:
        msg = (
            f"requested sample_length={sample_length} exceeds archive length "
            f"{rows.shape[1]}"
        )
        raise ValueError(msg)
    if rows.shape[1] == sample_length:
        return rows
    return rows[:, :sample_length, :]


__all__ = [
    "RADIOML_2018_01A_MODULATIONS",
    "RADIOML_2018_01A_SNR_DB",
    "load_radioml_2018_01a",
]
