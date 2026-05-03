"""Synthetic RF modulation classification benchmark.

This is a **stand-in** for a future RadioML 2018.01A benchmark. It generates
i.i.d. PSK/QAM symbols, adds AWGN at controlled SNRs, and trains the four
baseline families on a per-sample classification task. It uses *symbol-level*
inputs (no pulse shaping, no carrier offset, no fading), so absolute numbers
are not comparable to RadioML literature - but the relative differences
between baseline families are what this benchmark exists to measure.

When the real RadioML loader lands, the four-family scaffolding here can be
reused as-is.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cvnn.activations import ComplexCardioid, CReLU, ModReLU, Siglog, ZReLU
from cvnn.baselines import count_real_parameters
from cvnn.nn import ComplexMLP
from cvnn.repro import JsonObject, new_manifest
from experiments.synthetic.phase_classification import bootstrap_mean_ci

ActivationName = Literal["crelu", "zrelu", "modrelu", "cardioid", "siglog"]
RealActivationName = Literal["relu", "leaky_relu", "gelu", "tanh"]
ModelFamily = Literal[
    "complex",
    "real_stacked",
    "real_matched_params",
    "real_matched_flops",
]
ModulationName = Literal["bpsk", "qpsk", "8psk", "qam16", "qam64"]

DEFAULT_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "complex",
    "real_stacked",
    "real_matched_params",
    "real_matched_flops",
)
DEFAULT_MODULATIONS: tuple[ModulationName, ...] = (
    "bpsk",
    "qpsk",
    "8psk",
)
DEFAULT_SNR_DB: tuple[int, ...] = (-10, -5, 0, 5, 10, 15, 20)


@dataclass(frozen=True)
class RFModulationData:
    """Deterministic synthetic RF modulation split."""

    train_inputs: Tensor
    train_labels: Tensor
    train_snr_db: Tensor
    test_inputs: Tensor
    test_labels: Tensor
    test_snr_db: Tensor
    modulation_names: tuple[ModulationName, ...]
    snr_db_levels: tuple[int, ...]
    sample_length: int


@dataclass(frozen=True)
class RFBenchmarkConfig:
    """Serializable benchmark configuration."""

    seeds: tuple[int, ...]
    model_families: tuple[ModelFamily, ...]
    modulations: tuple[ModulationName, ...]
    snr_db_levels: tuple[int, ...]
    n_per_class_per_snr: int
    sample_length: int
    train_fraction: float
    hidden_features: int
    steps: int
    batch_size: int
    learning_rate: float
    activation: ActivationName
    real_activation: RealActivationName
    device: str
    dtype: str
    bootstrap_samples: int
    confidence: float

    def to_dict(self) -> JsonObject:
        return {
            "experiment": "rf_synthetic_modulation",
            "seeds": list(self.seeds),
            "model_families": list(self.model_families),
            "modulations": list(self.modulations),
            "snr_db_levels": list(self.snr_db_levels),
            "n_per_class_per_snr": self.n_per_class_per_snr,
            "sample_length": self.sample_length,
            "train_fraction": self.train_fraction,
            "hidden_features": self.hidden_features,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "activation": self.activation,
            "real_activation": self.real_activation,
            "device": self.device,
            "dtype": self.dtype,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RFRunResult:
    """Raw metrics for one seed and one model family."""

    model_family: ModelFamily
    seed: int
    steps: int
    learning_rate: float
    hidden_features: int
    parameter_count: int
    estimated_forward_madds: int
    train_seconds: float
    train_loss: float
    test_loss: float
    test_accuracy: float
    accuracy_by_snr_db: dict[str, float]

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


@dataclass(frozen=True)
class RFSummary:
    """Aggregate metrics for one model family across seeds."""

    model_family: ModelFamily
    n_runs: int
    hidden_features: int
    parameter_count: int
    estimated_forward_madds: int
    test_accuracy_mean: float
    test_accuracy_std: float
    test_accuracy_ci_low: float
    test_accuracy_ci_high: float
    test_loss_mean: float
    train_seconds_mean: float
    accuracy_by_snr_db_mean: dict[str, float]

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


class ComplexRFClassifier(nn.Module):
    """Two-hidden-layer complex MLP over flattened complex IQ sequence.

    Inputs: complex tensor of shape `(B, sample_length)`. Logits are taken
    as `|z_c|` for each output class - the standard CVNN classification
    convention; see `ComplexPhaseClassifier`'s docstring for why `.real`
    truncation is avoided.

    The flatten-then-MLP architecture loses the temporal structure of the
    IQ sequence; both complex and real baselines have the same handicap so
    the comparison stays fair. A `ComplexConv1d`-based variant is the
    natural follow-up once that layer lands.
    """

    def __init__(
        self,
        *,
        sample_length: int,
        hidden_features: int,
        n_classes: int,
        activation: ActivationName,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.network = ComplexMLP(
            sample_length,
            [hidden_features, hidden_features],
            n_classes,
            activation_factory=_complex_activation_factory(activation),
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.network(input).abs()
        return output


class RealRFClassifier(nn.Module):
    """Two-hidden-layer real MLP over stacked real/imag flattened IQ.

    Inputs: real tensor of shape `(B, 2 * sample_length)` formed by
    concatenating real and imaginary parts of the IQ sequence along the
    last dim.
    """

    def __init__(
        self,
        *,
        sample_length: int,
        hidden_features: int,
        n_classes: int,
        activation: RealActivationName,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(2 * sample_length, hidden_features, device=device, dtype=dtype),
            _real_activation_module(activation),
            nn.Linear(hidden_features, hidden_features, device=device, dtype=dtype),
            _real_activation_module(activation),
            nn.Linear(hidden_features, n_classes, device=device, dtype=dtype),
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.network(input)
        return output


def make_synthetic_rf_modulation_dataset(
    *,
    seed: int = 0,
    modulations: Sequence[ModulationName] = DEFAULT_MODULATIONS,
    snr_db_levels: Sequence[int] = DEFAULT_SNR_DB,
    n_per_class_per_snr: int = 256,
    sample_length: int = 128,
    train_fraction: float = 0.8,
    dtype: torch.dtype = torch.complex64,
) -> RFModulationData:
    """Generate a deterministic synthetic IQ-modulation classification dataset.

    For each `(modulation, snr)` pair, draws `n_per_class_per_snr` examples
    of `sample_length` i.i.d. complex symbols from the named constellation,
    then adds AWGN whose variance matches the requested SNR. Symbols are
    *unit average symbol energy*; no pulse shaping or carrier offset.

    Each example carries one modulation label and one SNR (dB); the test set
    is split off after a per-modulation, per-SNR shuffle so the per-SNR
    breakdown is not biased toward easier or harder classes.
    """

    if not modulations:
        msg = "at least one modulation is required"
        raise ValueError(msg)
    if not snr_db_levels:
        msg = "at least one SNR level is required"
        raise ValueError(msg)
    if n_per_class_per_snr <= 1:
        msg = "n_per_class_per_snr must be > 1 (need at least 1 train + 1 test)"
        raise ValueError(msg)
    if sample_length <= 0:
        msg = "sample_length must be positive"
        raise ValueError(msg)
    if not (0.0 < train_fraction < 1.0):
        msg = "train_fraction must lie in (0, 1)"
        raise ValueError(msg)
    if not dtype.is_complex:
        msg = f"RF modulation inputs require a complex dtype, got {dtype}"
        raise TypeError(msg)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    train_inputs_chunks: list[Tensor] = []
    train_labels_chunks: list[Tensor] = []
    train_snr_chunks: list[Tensor] = []
    test_inputs_chunks: list[Tensor] = []
    test_labels_chunks: list[Tensor] = []
    test_snr_chunks: list[Tensor] = []

    for label_index, modulation in enumerate(modulations):
        constellation = _constellation(modulation, real_dtype=real_dtype)
        for snr_db in snr_db_levels:
            symbols = _draw_symbols(
                constellation,
                n_examples=n_per_class_per_snr,
                sample_length=sample_length,
                generator=generator,
            )
            noisy = _add_awgn(symbols, snr_db=snr_db, generator=generator)
            permutation = torch.randperm(n_per_class_per_snr, generator=generator)
            shuffled = noisy[permutation]
            n_train = int(math.floor(n_per_class_per_snr * train_fraction))
            train_chunk = shuffled[:n_train]
            test_chunk = shuffled[n_train:]
            labels_train = torch.full((n_train,), label_index, dtype=torch.long)
            labels_test = torch.full(
                (n_per_class_per_snr - n_train,), label_index, dtype=torch.long
            )
            snr_train = torch.full((n_train,), snr_db, dtype=torch.long)
            snr_test = torch.full(
                (n_per_class_per_snr - n_train,), snr_db, dtype=torch.long
            )
            train_inputs_chunks.append(train_chunk.to(dtype))
            train_labels_chunks.append(labels_train)
            train_snr_chunks.append(snr_train)
            test_inputs_chunks.append(test_chunk.to(dtype))
            test_labels_chunks.append(labels_test)
            test_snr_chunks.append(snr_test)

    train_inputs = torch.cat(train_inputs_chunks, dim=0)
    train_labels = torch.cat(train_labels_chunks, dim=0)
    train_snr_db = torch.cat(train_snr_chunks, dim=0)
    test_inputs = torch.cat(test_inputs_chunks, dim=0)
    test_labels = torch.cat(test_labels_chunks, dim=0)
    test_snr_db = torch.cat(test_snr_chunks, dim=0)

    train_perm = torch.randperm(train_inputs.shape[0], generator=generator)
    test_perm = torch.randperm(test_inputs.shape[0], generator=generator)
    return RFModulationData(
        train_inputs=train_inputs[train_perm],
        train_labels=train_labels[train_perm],
        train_snr_db=train_snr_db[train_perm],
        test_inputs=test_inputs[test_perm],
        test_labels=test_labels[test_perm],
        test_snr_db=test_snr_db[test_perm],
        modulation_names=tuple(modulations),
        snr_db_levels=tuple(snr_db_levels),
        sample_length=sample_length,
    )


def train_rf_classifier(
    *,
    seed: int,
    model_family: ModelFamily,
    modulations: Sequence[ModulationName] = DEFAULT_MODULATIONS,
    snr_db_levels: Sequence[int] = DEFAULT_SNR_DB,
    n_per_class_per_snr: int = 256,
    sample_length: int = 128,
    train_fraction: float = 0.8,
    hidden_features: int = 64,
    steps: int = 200,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    activation: ActivationName = "crelu",
    real_activation: RealActivationName = "relu",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> RFRunResult:
    """Train one model family on one deterministic seed."""

    if hidden_features <= 0:
        msg = "hidden_features must be positive"
        raise ValueError(msg)
    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    device_obj = torch.device(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = make_synthetic_rf_modulation_dataset(
        seed=seed,
        modulations=modulations,
        snr_db_levels=snr_db_levels,
        n_per_class_per_snr=n_per_class_per_snr,
        sample_length=sample_length,
        train_fraction=train_fraction,
        dtype=dtype,
    )
    n_classes = len(data.modulation_names)
    model, effective_hidden, estimated_madds = _make_model(
        model_family,
        sample_length=sample_length,
        complex_hidden_features=hidden_features,
        n_classes=n_classes,
        activation=activation,
        real_activation=real_activation,
        device=device_obj,
        dtype=dtype,
    )
    parameter_count = count_real_parameters(model)
    train_inputs = _features_for_family(data.train_inputs.to(device_obj), model_family)
    test_inputs = _features_for_family(data.test_inputs.to(device_obj), model_family)
    train_labels = data.train_labels.to(device_obj)
    test_labels = data.test_labels.to(device_obj)
    test_snr_db = data.test_snr_db

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    n_train = train_inputs.shape[0]
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(seed + 1_000_000)
    start = time.perf_counter()
    train_loss = torch.tensor(float("nan"), device=device_obj)
    model.train()
    for _ in range(steps):
        indices = torch.randint(0, n_train, (batch_size,), generator=batch_generator)
        batch_inputs = train_inputs[indices]
        batch_labels = train_labels[indices]
        optimizer.zero_grad()
        logits = model(batch_inputs)
        train_loss = F.cross_entropy(logits, batch_labels)
        train_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        test_logits = model(test_inputs)
        test_loss = F.cross_entropy(test_logits, test_labels)
        predictions = test_logits.argmax(dim=-1)
        correct = (predictions == test_labels).float()
        test_accuracy = correct.mean()
        accuracy_by_snr_db: dict[str, float] = {}
        for snr in data.snr_db_levels:
            mask = test_snr_db == snr
            if mask.any():
                accuracy_by_snr_db[str(int(snr))] = float(
                    correct[mask.to(correct.device)].mean().item()
                )

    return RFRunResult(
        model_family=model_family,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        hidden_features=effective_hidden,
        parameter_count=parameter_count,
        estimated_forward_madds=estimated_madds,
        train_seconds=train_seconds,
        train_loss=float(train_loss.detach().cpu().item()),
        test_loss=float(test_loss.detach().cpu().item()),
        test_accuracy=float(test_accuracy.detach().cpu().item()),
        accuracy_by_snr_db=accuracy_by_snr_db,
    )


def run_rf_modulation_benchmark(
    config: RFBenchmarkConfig,
) -> tuple[list[RFRunResult], list[RFSummary]]:
    """Run all configured seeds and model families."""

    dtype = _parse_complex_dtype(config.dtype)
    runs: list[RFRunResult] = []
    for model_family in config.model_families:
        for seed in config.seeds:
            runs.append(
                train_rf_classifier(
                    seed=seed,
                    model_family=model_family,
                    modulations=config.modulations,
                    snr_db_levels=config.snr_db_levels,
                    n_per_class_per_snr=config.n_per_class_per_snr,
                    sample_length=config.sample_length,
                    train_fraction=config.train_fraction,
                    hidden_features=config.hidden_features,
                    steps=config.steps,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    activation=config.activation,
                    real_activation=config.real_activation,
                    device=config.device,
                    dtype=dtype,
                )
            )
    summaries = summarize_rf_runs(
        runs,
        model_order=config.model_families,
        snr_db_levels=config.snr_db_levels,
        bootstrap_samples=config.bootstrap_samples,
        confidence=config.confidence,
    )
    return runs, summaries


def summarize_rf_runs(
    runs: Sequence[RFRunResult],
    *,
    model_order: Sequence[ModelFamily] = DEFAULT_MODEL_FAMILIES,
    snr_db_levels: Sequence[int] = DEFAULT_SNR_DB,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> list[RFSummary]:
    """Aggregate per-seed metrics into paper-table rows."""

    summaries: list[RFSummary] = []
    for model_family in model_order:
        family_runs = [run for run in runs if run.model_family == model_family]
        if not family_runs:
            continue
        accuracies = [run.test_accuracy for run in family_runs]
        losses = [run.test_loss for run in family_runs]
        train_seconds = [run.train_seconds for run in family_runs]
        ci_low, ci_high = bootstrap_mean_ci(
            accuracies,
            samples=bootstrap_samples,
            confidence=confidence,
        )
        snr_means: dict[str, float] = {}
        for snr in snr_db_levels:
            key = str(int(snr))
            per_seed = [
                run.accuracy_by_snr_db[key]
                for run in family_runs
                if key in run.accuracy_by_snr_db
            ]
            if per_seed:
                snr_means[key] = statistics.mean(per_seed)
        summaries.append(
            RFSummary(
                model_family=model_family,
                n_runs=len(family_runs),
                hidden_features=family_runs[0].hidden_features,
                parameter_count=family_runs[0].parameter_count,
                estimated_forward_madds=family_runs[0].estimated_forward_madds,
                test_accuracy_mean=statistics.mean(accuracies),
                test_accuracy_std=_sample_std(accuracies),
                test_accuracy_ci_low=ci_low,
                test_accuracy_ci_high=ci_high,
                test_loss_mean=statistics.mean(losses),
                train_seconds_mean=statistics.mean(train_seconds),
                accuracy_by_snr_db_mean=snr_means,
            )
        )
    return summaries


def write_rf_benchmark_outputs(
    output_dir: Path,
    *,
    config: RFBenchmarkConfig,
    runs: Sequence[RFRunResult],
    summaries: Sequence[RFSummary],
) -> None:
    """Write raw runs, aggregate summaries, manifest, and human summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_path = output_dir / "raw_runs.json"
    summary_path = output_dir / "summary.json"
    summary_markdown_path = output_dir / "summary.md"
    manifest_path = output_dir / "manifest.json"

    raw_runs_path.write_text(
        json.dumps([run.to_dict() for run in runs], indent=2, sort_keys=True) + "\n"
    )
    summary_payload: JsonObject = {
        "config": config.to_dict(),
        "summaries": [summary.to_dict() for summary in summaries],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    summary_markdown_path.write_text(
        _format_summary_markdown(summaries, config=config) + "\n"
    )

    manifest = new_manifest(
        run_id="rf-synthetic-modulation",
        config=config.to_dict(),
        seeds=list(config.seeds),
        metrics=summary_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "rf_synthetic_modulation",
            "version": "0.1.0",
            "description": (
                "synthetic IQ PSK/QAM symbols + AWGN; stand-in for RadioML"
            ),
        },
        artifacts={
            "raw_runs": str(raw_runs_path),
            "summary_json": str(summary_path),
            "summary_markdown": str(summary_markdown_path),
        },
    )
    manifest.write_json(manifest_path)


def _constellation(modulation: ModulationName, *, real_dtype: torch.dtype) -> Tensor:
    if modulation == "bpsk":
        points = torch.tensor([1.0 + 0.0j, -1.0 + 0.0j])
    elif modulation == "qpsk":
        scale = 1.0 / math.sqrt(2.0)
        points = scale * torch.tensor(
            [1.0 + 1.0j, 1.0 - 1.0j, -1.0 + 1.0j, -1.0 - 1.0j]
        )
    elif modulation == "8psk":
        angles = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
        points = torch.complex(angles.cos(), angles.sin()).to(torch.complex128)
    elif modulation == "qam16":
        coords = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        real_part, imag_part = torch.meshgrid(coords, coords, indexing="ij")
        flat = torch.complex(real_part.flatten(), imag_part.flatten())
        points = flat / torch.sqrt((flat.real**2 + flat.imag**2).mean())
    elif modulation == "qam64":
        coords = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        real_part, imag_part = torch.meshgrid(coords, coords, indexing="ij")
        flat = torch.complex(real_part.flatten(), imag_part.flatten())
        points = flat / torch.sqrt((flat.real**2 + flat.imag**2).mean())
    else:
        msg = f"unsupported modulation: {modulation}"
        raise ValueError(msg)
    target_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    return points.to(target_dtype)


def _draw_symbols(
    constellation: Tensor,
    *,
    n_examples: int,
    sample_length: int,
    generator: torch.Generator,
) -> Tensor:
    indices = torch.randint(
        0,
        constellation.shape[0],
        (n_examples, sample_length),
        generator=generator,
    )
    return constellation[indices]


def _add_awgn(
    signal: Tensor,
    *,
    snr_db: int,
    generator: torch.Generator,
) -> Tensor:
    """Add complex AWGN to a unit-energy signal at the requested SNR (dB)."""

    real_dtype = signal.real.dtype
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance_per_component = 1.0 / (2.0 * snr_linear)
    noise_std = math.sqrt(noise_variance_per_component)
    noise_real = torch.randn(signal.shape, generator=generator, dtype=real_dtype)
    noise_imag = torch.randn(signal.shape, generator=generator, dtype=real_dtype)
    noise = torch.complex(noise_real, noise_imag) * noise_std
    return signal + noise


def _make_model(
    model_family: ModelFamily,
    *,
    sample_length: int,
    complex_hidden_features: int,
    n_classes: int,
    activation: ActivationName,
    real_activation: RealActivationName,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, int, int]:
    if model_family == "complex":
        model: nn.Module = ComplexRFClassifier(
            sample_length=sample_length,
            hidden_features=complex_hidden_features,
            n_classes=n_classes,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        madds = _complex_mlp_madds(
            in_features=sample_length,
            hidden_features=complex_hidden_features,
            out_features=n_classes,
        )
        return model, complex_hidden_features, madds

    complex_madds = _complex_mlp_madds(
        in_features=sample_length,
        hidden_features=complex_hidden_features,
        out_features=n_classes,
    )
    complex_parameter_budget = _complex_mlp_parameter_count(
        in_features=sample_length,
        hidden_features=complex_hidden_features,
        out_features=n_classes,
    )
    if model_family == "real_stacked":
        real_hidden = complex_hidden_features
    elif model_family == "real_matched_params":
        real_hidden = _choose_real_hidden_for_parameter_budget(
            sample_length=sample_length,
            n_classes=n_classes,
            budget=complex_parameter_budget,
        )
    elif model_family == "real_matched_flops":
        real_hidden = _choose_real_hidden_for_madds_budget(
            sample_length=sample_length,
            n_classes=n_classes,
            budget=complex_madds,
        )
    else:
        msg = f"unsupported model family: {model_family}"
        raise ValueError(msg)

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    model = RealRFClassifier(
        sample_length=sample_length,
        hidden_features=real_hidden,
        n_classes=n_classes,
        activation=real_activation,
        device=device,
        dtype=real_dtype,
    )
    madds = _real_mlp_madds(
        sample_length=sample_length, hidden_features=real_hidden, n_classes=n_classes
    )
    return model, real_hidden, madds


def _choose_real_hidden_for_parameter_budget(
    *,
    sample_length: int,
    n_classes: int,
    budget: int,
    max_hidden: int = 4096,
) -> int:
    return min(
        range(1, max_hidden + 1),
        key=lambda hidden: abs(
            _real_mlp_parameter_count(
                sample_length=sample_length,
                hidden_features=hidden,
                n_classes=n_classes,
            )
            - budget
        ),
    )


def _choose_real_hidden_for_madds_budget(
    *,
    sample_length: int,
    n_classes: int,
    budget: int,
    max_hidden: int = 4096,
) -> int:
    return min(
        range(1, max_hidden + 1),
        key=lambda hidden: abs(
            _real_mlp_madds(
                sample_length=sample_length,
                hidden_features=hidden,
                n_classes=n_classes,
            )
            - budget
        ),
    )


def _complex_mlp_parameter_count(
    *,
    in_features: int,
    hidden_features: int,
    out_features: int,
) -> int:
    complex_slots = (
        in_features * hidden_features
        + hidden_features
        + hidden_features * hidden_features
        + hidden_features
        + hidden_features * out_features
        + out_features
    )
    return 2 * complex_slots


def _real_mlp_parameter_count(
    *, sample_length: int, hidden_features: int, n_classes: int
) -> int:
    return (
        2 * sample_length * hidden_features
        + hidden_features
        + hidden_features * hidden_features
        + hidden_features
        + hidden_features * n_classes
        + n_classes
    )


def _complex_mlp_madds(
    *, in_features: int, hidden_features: int, out_features: int
) -> int:
    return 4 * (
        in_features * hidden_features
        + hidden_features * hidden_features
        + hidden_features * out_features
    )


def _real_mlp_madds(*, sample_length: int, hidden_features: int, n_classes: int) -> int:
    return (
        2 * sample_length * hidden_features
        + hidden_features * hidden_features
        + hidden_features * n_classes
    )


def _features_for_family(inputs: Tensor, model_family: ModelFamily) -> Tensor:
    if model_family == "complex":
        return inputs
    return torch.cat([inputs.real, inputs.imag], dim=-1)


def _complex_activation_factory(activation: ActivationName) -> Callable[[], nn.Module]:
    if activation == "crelu":
        return CReLU
    if activation == "zrelu":
        return ZReLU
    if activation == "modrelu":
        return lambda: ModReLU(init_bias=-0.1)
    if activation == "cardioid":
        return ComplexCardioid
    if activation == "siglog":
        return Siglog
    msg = f"unsupported activation: {activation}"
    raise ValueError(msg)


def _real_activation_module(activation: RealActivationName) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "leaky_relu":
        return nn.LeakyReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "tanh":
        return nn.Tanh()
    msg = f"unsupported real activation: {activation}"
    raise ValueError(msg)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _format_summary_markdown(
    summaries: Sequence[RFSummary],
    *,
    config: RFBenchmarkConfig | None = None,
) -> str:
    rows = [
        "# Synthetic RF Modulation Classification",
        "",
        (
            "Stand-in for a future RadioML 2018.01A benchmark. Inputs are "
            "i.i.d. PSK/QAM symbols with AWGN at controlled SNR; no pulse "
            "shaping, no carrier offset, no fading. Numbers are not "
            "comparable to RadioML literature - they exist to compare "
            "baseline families on a sequence-shaped task."
        ),
        "",
        (
            "Snapshot of one configuration. Numbers depend on platform BLAS "
            "and `git_commit`/`git_dirty` recorded in the manifest. Re-runs "
            "will not be byte-identical; see `docs/baselines.md` and "
            "`docs/tuning_budget.md` for the comparison rules."
        ),
        "",
    ]
    if config is not None:
        rows.extend(
            [
                f"Activation (complex): `{config.activation}`. "
                f"Activation (real baselines): `{config.real_activation}`. "
                f"Seeds: `{list(config.seeds)}`. "
                f"Steps: `{config.steps}`. "
                f"Modulations: `{list(config.modulations)}`. "
                f"SNR (dB): `{list(config.snr_db_levels)}`. "
                f"Sample length: `{config.sample_length}`.",
                "",
            ]
        )

    rows.extend(
        [
            (
                "| model | hidden | params | est. forward MAdds | accuracy mean | "
                "accuracy std | 95% CI | loss mean | train s/run |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for summary in summaries:
        rows.append(
            " | ".join(
                [
                    f"| `{summary.model_family}`",
                    f"{summary.hidden_features}",
                    f"{summary.parameter_count}",
                    f"{summary.estimated_forward_madds}",
                    f"{summary.test_accuracy_mean:.4f}",
                    f"{summary.test_accuracy_std:.4f}",
                    (
                        f"[{summary.test_accuracy_ci_low:.4f}, "
                        f"{summary.test_accuracy_ci_high:.4f}]"
                    ),
                    f"{summary.test_loss_mean:.3g}",
                    f"{summary.train_seconds_mean:.2g} |",
                ]
            )
        )

    if summaries and any(s.accuracy_by_snr_db_mean for s in summaries):
        snr_keys = sorted(
            {key for s in summaries for key in s.accuracy_by_snr_db_mean},
            key=lambda value: int(value),
        )
        rows.extend(
            [
                "",
                "## Accuracy by SNR (dB)",
                "",
                "| model | " + " | ".join(f"{snr} dB" for snr in snr_keys) + " |",
                "| --- | " + " | ".join("---:" for _ in snr_keys) + " |",
            ]
        )
        for summary in summaries:
            cells = [
                f"{summary.accuracy_by_snr_db_mean.get(snr, float('nan')):.3f}"
                for snr in snr_keys
            ]
            row = "| `" + summary.model_family + "` | " + " | ".join(cells) + " |"
            rows.append(row)
    return "\n".join(rows)


def _parse_complex_dtype(name: str) -> torch.dtype:
    if name == "complex64":
        return torch.complex64
    if name == "complex128":
        return torch.complex128
    msg = f"unsupported complex dtype: {name}"
    raise ValueError(msg)


def _parse_model_families(values: Sequence[str]) -> tuple[ModelFamily, ...]:
    parsed: list[ModelFamily] = []
    for value in values:
        if value not in {
            "complex",
            "real_stacked",
            "real_matched_params",
            "real_matched_flops",
        }:
            msg = f"unsupported model family: {value}"
            raise ValueError(msg)
        parsed.append(cast(ModelFamily, value))
    return tuple(parsed)


def _parse_modulations(values: Sequence[str]) -> tuple[ModulationName, ...]:
    parsed: list[ModulationName] = []
    for value in values:
        if value not in {"bpsk", "qpsk", "8psk", "qam16", "qam64"}:
            msg = f"unsupported modulation: {value}"
            raise ValueError(msg)
        parsed.append(cast(ModulationName, value))
    return tuple(parsed)


def _parse_activation(value: str) -> ActivationName:
    if value not in {"crelu", "zrelu", "modrelu", "cardioid", "siglog"}:
        msg = f"unsupported activation: {value}"
        raise ValueError(msg)
    return cast(ActivationName, value)


def _parse_real_activation(value: str) -> RealActivationName:
    if value not in {"relu", "leaky_relu", "gelu", "tanh"}:
        msg = f"unsupported real activation: {value}"
        raise ValueError(msg)
    return cast(RealActivationName, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--model-families",
        nargs="+",
        choices=list(DEFAULT_MODEL_FAMILIES),
        default=list(DEFAULT_MODEL_FAMILIES),
    )
    parser.add_argument(
        "--modulations",
        nargs="+",
        choices=list(DEFAULT_MODULATIONS),
        default=list(DEFAULT_MODULATIONS),
    )
    parser.add_argument(
        "--snr-db-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_SNR_DB),
    )
    parser.add_argument("--n-per-class-per-snr", type=int, default=256)
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--activation",
        choices=["crelu", "zrelu", "modrelu", "cardioid", "siglog"],
        default="crelu",
    )
    parser.add_argument(
        "--real-activation",
        choices=["relu", "leaky_relu", "gelu", "tanh"],
        default="relu",
        help="activation for real baselines (asymmetric vs --activation)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex64",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rf_synthetic_modulation"),
    )
    args = parser.parse_args()

    config = RFBenchmarkConfig(
        seeds=tuple(args.seeds),
        model_families=_parse_model_families(args.model_families),
        modulations=_parse_modulations(args.modulations),
        snr_db_levels=tuple(args.snr_db_levels),
        n_per_class_per_snr=args.n_per_class_per_snr,
        sample_length=args.sample_length,
        train_fraction=args.train_fraction,
        hidden_features=args.hidden_features,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        activation=_parse_activation(args.activation),
        real_activation=_parse_real_activation(args.real_activation),
        device=args.device,
        dtype=args.dtype,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
    )
    runs, summaries = run_rf_modulation_benchmark(config)
    write_rf_benchmark_outputs(
        args.output_dir,
        config=config,
        runs=runs,
        summaries=summaries,
    )
    print(_format_summary_markdown(summaries, config=config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
