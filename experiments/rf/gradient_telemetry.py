"""Per-step gradient/loss telemetry for the matched-shared-trial config.

Goal: explain *why* real baselines collapse on `crelu` / `cardioid` / `siglog`
under the matched-shared-trial selection rule but stay stable on `modrelu`
and `zrelu`. The hypothesis (see `docs/report.md` §3.4) is that complex's
selected config lives in a high-LR regime under the unstable activations
and a low-LR regime under the stable ones; under high LR, real baselines
see exploding gradients and a fraction of seeds never recover.

This module reads the selected hyperparameters from a sweep's `summary.json`
and re-runs the same `(family, hyperparameters, seed)` combination with
per-step instrumentation:

- `step` — training step index
- `train_loss` — cross-entropy on the current batch
- `total_grad_norm` — L2 norm of all parameter gradients post-backward
- `max_param_abs` — max absolute parameter value (drift / explosion)
- `per_layer_grad_norm` — dict keyed by parameter name (e.g. conv1.weight)

One JSONL file per `(family, seed)` pair. The driver script
`scripts/run_gradient_telemetry.py` loops over (activation, family, seed)
combinations and is the intended entry point.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

from cvnn.baselines import count_real_parameters
from experiments.rf.radioml import load_radioml_2018_01a
from experiments.rf.synthetic_modulation import (
    ArchitectureName,
    ModelFamily,
    _features_for_family,
    _make_model,
    make_synthetic_rf_modulation_dataset,
)

DataSource = Literal["radioml", "synthetic"]


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration captured from the matching sweep's summary.json."""

    activation: str
    family: str
    seed: int
    hyperparameters: dict[str, Any]
    architecture: ArchitectureName
    kernel_size: int
    modulations: tuple[str, ...]
    snr_db_levels: tuple[int, ...]
    max_per_class_per_snr: int
    sample_length: int
    val_fraction: float
    real_activation: str
    data_source: DataSource
    dtype: torch.dtype
    # Only used when data_source == "radioml"
    data_path: Path | None = None
    classes_path: Path | None = None


@dataclass
class StepRecord:
    """One per-step row of the JSONL log."""

    step: int
    train_loss: float
    total_grad_norm: float
    max_param_abs: float
    per_layer_grad_norm: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "train_loss": self.train_loss,
            "total_grad_norm": self.total_grad_norm,
            "max_param_abs": self.max_param_abs,
            "per_layer_grad_norm": dict(self.per_layer_grad_norm),
        }


def telemetry_config_from_sweep_summary(
    summary_path: Path,
    *,
    activation: str,
    family: str,
    seed: int,
    data_path: Path | None = None,
    classes_path: Path | None = None,
) -> TelemetryConfig:
    """Build a TelemetryConfig from a sweep's summary.json + selected family.

    The data source is inferred from the sweep's `experiment` field in
    config:
      - `radioml_modulation_sweep` → loads from a local HDF5 archive
        (requires `data_path`).
      - `rf_synthetic_modulation_sweep` → generates symbols on the fly via
        `make_synthetic_rf_modulation_dataset`; `data_path` is unused.
    """

    summary = json.loads(summary_path.read_text())
    config = summary["config"]
    selections = summary.get("matched_selections") or summary.get("selections")
    if not selections:
        msg = f"summary at {summary_path} has no matched_selections/selections"
        raise ValueError(msg)
    by_family = {sel["family"]: sel for sel in selections}
    if family not in by_family:
        msg = f"family {family!r} not in {list(by_family)}"
        raise ValueError(msg)

    experiment = str(config.get("experiment", ""))
    if "synthetic" in experiment:
        data_source: DataSource = "synthetic"
    elif "radioml" in experiment:
        data_source = "radioml"
        if data_path is None:
            msg = (
                f"summary at {summary_path} is a RadioML sweep; pass "
                "data_path pointing at the HDF5 archive."
            )
            raise ValueError(msg)
    else:
        msg = f"could not infer data_source from experiment={experiment!r}"
        raise ValueError(msg)

    sample_count_key = (
        "max_per_class_per_snr"
        if "max_per_class_per_snr" in config
        else "n_per_class_per_snr"
    )
    dtype = torch.complex64 if config["dtype"] == "complex64" else torch.complex128
    return TelemetryConfig(
        activation=activation,
        family=family,
        seed=seed,
        hyperparameters=dict(by_family[family]["selected_hyperparameters"]),
        architecture=cast(ArchitectureName, config["architecture"]),
        kernel_size=int(config["kernel_size"]),
        modulations=tuple(config["modulations"]),
        snr_db_levels=tuple(int(s) for s in config["snr_db_levels"]),
        max_per_class_per_snr=int(config[sample_count_key]),
        sample_length=int(config["sample_length"]),
        val_fraction=float(config["val_fraction"]),
        real_activation=str(config.get("real_activation", "relu")),
        data_source=data_source,
        dtype=dtype,
        data_path=data_path,
        classes_path=classes_path,
    )


_DATA_CACHE: dict[tuple[Any, ...], Any] = {}


def run_instrumented_training(
    config: TelemetryConfig,
    *,
    device: torch.device | str = "cpu",
    log_every_n: int = 1,
    log_per_layer: bool = True,
    max_steps: int | None = None,
) -> tuple[list[StepRecord], dict[str, Any]]:
    """Re-train the configured (family, hp, seed) combo with per-step logging.

    `max_steps` caps training to the first N steps regardless of what the
    sweep selected. Useful when you only need to surface the early-training
    divergence pattern rather than reproduce final accuracy.

    Returns (records, summary_metrics) where summary_metrics matches the
    final-step values plus the test/val accuracies that the sweep itself
    would have reported (so we can confirm we reproduced the run).
    """

    device_obj = torch.device(device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    cache_key = (
        config.data_source,
        str(config.data_path) if config.data_path else None,
        str(config.classes_path) if config.classes_path else None,
        config.modulations,
        config.snr_db_levels,
        config.max_per_class_per_snr,
        config.sample_length,
        config.seed,
        str(config.dtype),
    )
    if cache_key in _DATA_CACHE:
        data = _DATA_CACHE[cache_key]
    elif config.data_source == "synthetic":
        data = make_synthetic_rf_modulation_dataset(
            seed=config.seed,
            modulations=cast(Any, tuple(config.modulations)),
            snr_db_levels=config.snr_db_levels,
            n_per_class_per_snr=config.max_per_class_per_snr,
            sample_length=config.sample_length,
            train_fraction=0.8,
            dtype=config.dtype,
        )
        _DATA_CACHE[cache_key] = data
    else:
        if config.data_path is None:
            msg = "RadioML telemetry requires data_path"
            raise ValueError(msg)
        data = load_radioml_2018_01a(
            config.data_path,
            modulations=config.modulations,
            snr_db_levels=config.snr_db_levels,
            max_per_class_per_snr=config.max_per_class_per_snr,
            sample_length=config.sample_length,
            train_fraction=0.8,
            seed=config.seed,
            dtype=config.dtype,
            classes_path=config.classes_path,
        )
        _DATA_CACHE[cache_key] = data
    n_train_total = data.train_inputs.shape[0]
    n_val = int(round(n_train_total * config.val_fraction))
    if n_val < 1 or n_val >= n_train_total:
        msg = "val_fraction yields zero or full validation set"
        raise ValueError(msg)
    train_actual_inputs = data.train_inputs[:-n_val].to(device_obj)
    train_actual_labels = data.train_labels[:-n_val].to(device_obj)
    val_inputs_complex = data.train_inputs[-n_val:].to(device_obj)
    val_labels = data.train_labels[-n_val:].to(device_obj)
    test_inputs_complex = data.test_inputs.to(device_obj)
    test_labels = data.test_labels.to(device_obj)
    n_classes = len(data.modulation_names)

    model_family = cast(ModelFamily, config.family)
    model, _, _ = _make_model(
        model_family,
        architecture=config.architecture,
        sample_length=config.sample_length,
        complex_hidden_features=int(config.hyperparameters["hidden_features"]),
        n_classes=n_classes,
        activation=config.activation,  # type: ignore[arg-type]
        real_activation=config.real_activation,  # type: ignore[arg-type]
        kernel_size=config.kernel_size,
        device=device_obj,
        dtype=config.dtype,
    )

    train_inputs = _features_for_family(
        train_actual_inputs, model_family, config.architecture
    )
    val_inputs = _features_for_family(
        val_inputs_complex, model_family, config.architecture
    )
    test_inputs = _features_for_family(
        test_inputs_complex, model_family, config.architecture
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.hyperparameters["learning_rate"]),
        weight_decay=0.0,
    )
    n_train = train_inputs.shape[0]
    batch_size = min(int(config.hyperparameters["batch_size"]), n_train)
    full_steps = int(config.hyperparameters["steps"])
    n_steps = full_steps if max_steps is None else min(max_steps, full_steps)
    batch_gen = torch.Generator(device="cpu").manual_seed(config.seed + 1_000_000)

    records: list[StepRecord] = []
    start = time.perf_counter()
    model.train()
    last_loss = float("nan")
    for step in range(n_steps):
        indices = torch.randint(0, n_train, (batch_size,), generator=batch_gen)
        optimizer.zero_grad()
        logits = model(train_inputs[indices])
        loss = F.cross_entropy(logits, train_actual_labels[indices])
        loss.backward()  # type: ignore[no-untyped-call]

        if step % log_every_n == 0 or step == n_steps - 1:
            total_sq = 0.0
            max_param_abs = 0.0
            per_layer_norms: dict[str, float] = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if torch.is_complex(parameter):
                    grad_sq = float(parameter.grad.abs().square().sum().item())
                    abs_max = float(parameter.abs().max().item())
                else:
                    grad_sq = float(parameter.grad.square().sum().item())
                    abs_max = float(parameter.abs().max().item())
                total_sq += grad_sq
                if abs_max > max_param_abs:
                    max_param_abs = abs_max
                if log_per_layer:
                    per_layer_norms[name] = float(grad_sq**0.5)
            last_loss = float(loss.detach().cpu().item())
            records.append(
                StepRecord(
                    step=step,
                    train_loss=last_loss,
                    total_grad_norm=float(total_sq**0.5),
                    max_param_abs=max_param_abs,
                    per_layer_grad_norm=per_layer_norms,
                )
            )

        optimizer.step()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        val_predictions = model(val_inputs).argmax(dim=-1)
        test_predictions = model(test_inputs).argmax(dim=-1)
        val_acc = float((val_predictions == val_labels).float().mean().item())
        test_acc = float((test_predictions == test_labels).float().mean().item())

    summary_metrics: dict[str, Any] = {
        "activation": config.activation,
        "family": config.family,
        "seed": config.seed,
        "hyperparameters": dict(config.hyperparameters),
        "n_steps": n_steps,
        "train_seconds": train_seconds,
        "final_train_loss": last_loss,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "parameter_count": count_real_parameters(model),
    }
    return records, summary_metrics


def write_telemetry_jsonl(
    output_path: Path,
    *,
    records: Sequence[StepRecord],
    summary: dict[str, Any],
) -> None:
    """Write one JSONL with summary as the first line, step records after."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write(json.dumps({"summary": summary}) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")


__all__ = [
    "StepRecord",
    "TelemetryConfig",
    "run_instrumented_training",
    "telemetry_config_from_sweep_summary",
    "write_telemetry_jsonl",
]
