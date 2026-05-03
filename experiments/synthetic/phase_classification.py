"""Phase-classification benchmark for Phase 4 evidence runs."""

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

ActivationName = Literal["crelu", "zrelu", "modrelu", "cardioid", "siglog"]
ModelFamily = Literal[
    "complex",
    "real_stacked",
    "real_matched_params",
    "real_matched_flops",
]

DEFAULT_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "complex",
    "real_stacked",
    "real_matched_params",
    "real_matched_flops",
)


@dataclass(frozen=True)
class PhaseClassificationData:
    """Deterministic complex-valued classification split."""

    train_inputs: Tensor
    train_labels: Tensor
    test_inputs: Tensor
    test_labels: Tensor
    n_classes: int
    class_spread: float
    noise_std: float


@dataclass(frozen=True)
class PhaseBenchmarkConfig:
    """Serializable benchmark configuration."""

    seeds: tuple[int, ...]
    model_families: tuple[ModelFamily, ...]
    n_train: int
    n_test: int
    n_classes: int
    hidden_features: int
    steps: int
    learning_rate: float
    class_spread: float | None
    noise_std: float
    activation: ActivationName
    real_activation: RealActivationName
    device: str
    dtype: str
    bootstrap_samples: int
    confidence: float

    def to_dict(self) -> JsonObject:
        return {
            "experiment": "synthetic_phase_classification",
            "seeds": list(self.seeds),
            "model_families": list(self.model_families),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_classes": self.n_classes,
            "hidden_features": self.hidden_features,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "class_spread": self.class_spread,
            "noise_std": self.noise_std,
            "activation": self.activation,
            "real_activation": self.real_activation,
            "device": self.device,
            "dtype": self.dtype,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PhaseRunResult:
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

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


@dataclass(frozen=True)
class PhaseSummary:
    """Aggregate metrics for one model family."""

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
    test_loss_std: float
    train_seconds_mean: float

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


RealActivationName = Literal["relu", "leaky_relu", "gelu", "tanh"]


class ComplexPhaseClassifier(nn.Module):
    """Complex MLP classifier with `|z|`-magnitude logits.

    The complex MLP outputs `n_classes` complex values; the per-class logit is
    the magnitude `|z_c|`, which is symmetric in real and imaginary parts and
    preserves both halves of the complex output. This is the standard CVNN
    classification convention; using `.real` truncates half the signal and
    silently handicaps the complex model in `complex vs. real` comparisons.
    """

    def __init__(
        self,
        *,
        hidden_features: int,
        n_classes: int,
        activation: ActivationName,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.network = ComplexMLP(
            1,
            [hidden_features],
            n_classes,
            activation_factory=_activation_factory(activation),
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.network(input).abs()
        return output


class RealPhaseClassifier(nn.Module):
    """Real MLP over stacked real/imaginary inputs.

    The activation is selectable so a sweep over complex activations does not
    silently force the comparison "complex+X vs. real+ReLU"; pick the real
    activation closest in spirit (or just hold it fixed across the sweep)
    and disclose the choice in the result table.
    """

    def __init__(
        self,
        *,
        hidden_features: int,
        n_classes: int,
        activation: RealActivationName,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_features, device=device, dtype=dtype),
            _real_activation_module(activation),
            nn.Linear(hidden_features, n_classes, device=device, dtype=dtype),
        )

    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.network(input)
        return output


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


def make_phase_classification(
    *,
    seed: int = 0,
    n_train: int = 512,
    n_test: int = 512,
    n_classes: int = 4,
    class_spread: float | None = None,
    noise_std: float = 0.05,
    dtype: torch.dtype = torch.complex64,
) -> PhaseClassificationData:
    """Create a balanced classification task where labels are phase sectors."""

    if n_train <= 0 or n_test <= 0:
        msg = "n_train and n_test must be positive"
        raise ValueError(msg)
    if n_classes < 2:
        msg = "n_classes must be at least 2"
        raise ValueError(msg)
    if noise_std < 0:
        msg = f"noise_std must be non-negative, got {noise_std}"
        raise ValueError(msg)
    if not dtype.is_complex:
        msg = f"phase classification inputs require a complex dtype, got {dtype}"
        raise TypeError(msg)

    resolved_spread = (
        0.35 * math.pi / n_classes if class_spread is None else class_spread
    )
    if resolved_spread < 0 or resolved_spread >= math.pi / n_classes:
        msg = (
            "class_spread must be non-negative and smaller than half a sector, "
            f"got {resolved_spread}"
        )
        raise ValueError(msg)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_inputs, train_labels = _make_phase_split(
        n_train,
        n_classes=n_classes,
        class_spread=resolved_spread,
        noise_std=noise_std,
        generator=generator,
        dtype=dtype,
    )
    test_inputs, test_labels = _make_phase_split(
        n_test,
        n_classes=n_classes,
        class_spread=resolved_spread,
        noise_std=noise_std,
        generator=generator,
        dtype=dtype,
    )
    return PhaseClassificationData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
        n_classes=n_classes,
        class_spread=resolved_spread,
        noise_std=noise_std,
    )


def train_phase_classifier(
    *,
    seed: int,
    model_family: ModelFamily,
    n_train: int = 512,
    n_test: int = 512,
    n_classes: int = 4,
    hidden_features: int = 16,
    steps: int = 300,
    learning_rate: float = 0.02,
    class_spread: float | None = None,
    noise_std: float = 0.05,
    activation: ActivationName = "crelu",
    real_activation: RealActivationName = "relu",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> PhaseRunResult:
    """Train one model family on one deterministic seed."""

    if hidden_features <= 0:
        msg = "hidden_features must be positive"
        raise ValueError(msg)
    if steps <= 0:
        msg = "steps must be positive"
        raise ValueError(msg)

    device_obj = torch.device(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = make_phase_classification(
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        n_classes=n_classes,
        class_spread=class_spread,
        noise_std=noise_std,
        dtype=dtype,
    )
    model, effective_hidden, estimated_madds = _make_model(
        model_family,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    start = time.perf_counter()
    train_loss = torch.tensor(float("nan"), device=device_obj)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(train_inputs)
        train_loss = F.cross_entropy(logits, train_labels)
        train_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        test_logits = model(test_inputs)
        test_loss = F.cross_entropy(test_logits, test_labels)
        predictions = test_logits.argmax(dim=-1)
        test_accuracy = (predictions == test_labels).float().mean()

    return PhaseRunResult(
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
    )


def run_phase_classification_benchmark(
    config: PhaseBenchmarkConfig,
) -> tuple[list[PhaseRunResult], list[PhaseSummary]]:
    """Run all configured seeds and model families."""

    dtype = _parse_complex_dtype(config.dtype)
    runs: list[PhaseRunResult] = []
    for model_family in config.model_families:
        for seed in config.seeds:
            runs.append(
                train_phase_classifier(
                    seed=seed,
                    model_family=model_family,
                    n_train=config.n_train,
                    n_test=config.n_test,
                    n_classes=config.n_classes,
                    hidden_features=config.hidden_features,
                    steps=config.steps,
                    learning_rate=config.learning_rate,
                    class_spread=config.class_spread,
                    noise_std=config.noise_std,
                    activation=config.activation,
                    real_activation=config.real_activation,
                    device=config.device,
                    dtype=dtype,
                )
            )
    summaries = summarize_phase_runs(
        runs,
        model_order=config.model_families,
        bootstrap_samples=config.bootstrap_samples,
        confidence=config.confidence,
    )
    return runs, summaries


def summarize_phase_runs(
    runs: Sequence[PhaseRunResult],
    *,
    model_order: Sequence[ModelFamily] = DEFAULT_MODEL_FAMILIES,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> list[PhaseSummary]:
    """Aggregate raw per-seed metrics into paper-table rows."""

    summaries: list[PhaseSummary] = []
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
        summaries.append(
            PhaseSummary(
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
                test_loss_std=_sample_std(losses),
                train_seconds_mean=statistics.mean(train_seconds),
            )
        )
    return summaries


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""

    if not values:
        msg = "values must be non-empty"
        raise ValueError(msg)
    if samples <= 0:
        msg = "samples must be positive"
        raise ValueError(msg)
    if confidence <= 0.0 or confidence >= 1.0:
        msg = "confidence must be in (0, 1)"
        raise ValueError(msg)
    if len(values) == 1:
        value = float(values[0])
        return value, value

    values_tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        0,
        len(values),
        (samples, len(values)),
        generator=generator,
    )
    means = values_tensor[indices].mean(dim=1)
    alpha = (1.0 - confidence) / 2.0
    low = torch.quantile(means, alpha)
    high = torch.quantile(means, 1.0 - alpha)
    return float(low.item()), float(high.item())


def write_phase_benchmark_outputs(
    output_dir: Path,
    *,
    config: PhaseBenchmarkConfig,
    runs: Sequence[PhaseRunResult],
    summaries: Sequence[PhaseSummary],
) -> None:
    """Write raw runs, aggregate summaries, and a manifest."""

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
        run_id="synthetic-phase-classification",
        config=config.to_dict(),
        seeds=list(config.seeds),
        metrics=summary_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "synthetic_phase_classification",
            "version": "0.1.0",
            "description": "balanced complex phase-sector classification",
        },
        artifacts={
            "raw_runs": str(raw_runs_path),
            "summary_json": str(summary_path),
            "summary_markdown": str(summary_markdown_path),
        },
    )
    manifest.write_json(manifest_path)


def _make_phase_split(
    n_samples: int,
    *,
    n_classes: int,
    class_spread: float,
    noise_std: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    labels = _balanced_labels(n_samples, n_classes=n_classes, generator=generator)
    sector_width = 2.0 * math.pi / n_classes
    centers = labels.to(real_dtype) * sector_width
    jitter = torch.empty(n_samples, dtype=real_dtype).uniform_(
        -class_spread,
        class_spread,
        generator=generator,
    )
    radius = torch.empty(n_samples, dtype=real_dtype).uniform_(
        0.75,
        1.25,
        generator=generator,
    )
    inputs = torch.polar(radius, centers + jitter).unsqueeze(-1).to(dtype)
    if noise_std:
        inputs = inputs + noise_std * _complex_randn(
            inputs.shape,
            generator=generator,
            dtype=dtype,
        )
    return inputs, labels


def _balanced_labels(
    n_samples: int,
    *,
    n_classes: int,
    generator: torch.Generator,
) -> Tensor:
    labels = torch.arange(n_samples, dtype=torch.long) % n_classes
    permutation = torch.randperm(n_samples, generator=generator)
    return labels[permutation]


def _complex_randn(
    shape: torch.Size,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(shape, generator=generator, dtype=real_dtype)
    imag = torch.randn(shape, generator=generator, dtype=real_dtype)
    return torch.complex(real, imag)


def _make_model(
    model_family: ModelFamily,
    *,
    complex_hidden_features: int,
    n_classes: int,
    activation: ActivationName,
    real_activation: RealActivationName,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, int, int]:
    if model_family == "complex":
        model: nn.Module = ComplexPhaseClassifier(
            hidden_features=complex_hidden_features,
            n_classes=n_classes,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        madds = _complex_mlp_madds(
            in_features=1,
            hidden_features=complex_hidden_features,
            out_features=n_classes,
        )
        return model, complex_hidden_features, madds

    complex_madds = _complex_mlp_madds(
        in_features=1,
        hidden_features=complex_hidden_features,
        out_features=n_classes,
    )
    complex_parameter_budget = _complex_mlp_parameter_count(
        in_features=1,
        hidden_features=complex_hidden_features,
        out_features=n_classes,
    )
    if model_family == "real_stacked":
        real_hidden = complex_hidden_features
    elif model_family == "real_matched_params":
        real_hidden = choose_real_hidden_for_parameter_budget(
            budget=complex_parameter_budget,
            n_classes=n_classes,
        )
    elif model_family == "real_matched_flops":
        real_hidden = choose_real_hidden_for_madds_budget(
            budget=complex_madds,
            n_classes=n_classes,
        )
    else:
        msg = f"unsupported model family: {model_family}"
        raise ValueError(msg)

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    model = RealPhaseClassifier(
        hidden_features=real_hidden,
        n_classes=n_classes,
        activation=real_activation,
        device=device,
        dtype=real_dtype,
    )
    madds = _real_mlp_madds(hidden_features=real_hidden, n_classes=n_classes)
    return model, real_hidden, madds


def choose_real_hidden_for_parameter_budget(
    *,
    budget: int,
    n_classes: int,
    max_hidden: int = 4096,
) -> int:
    """Choose the real hidden width closest to a scalar parameter budget."""

    return min(
        range(1, max_hidden + 1),
        key=lambda hidden: abs(
            _real_mlp_parameter_count(hidden_features=hidden, n_classes=n_classes)
            - budget
        ),
    )


def choose_real_hidden_for_madds_budget(
    *,
    budget: int,
    n_classes: int,
    max_hidden: int = 4096,
) -> int:
    """Choose the real hidden width closest to an estimated forward budget."""

    return min(
        range(1, max_hidden + 1),
        key=lambda hidden: abs(
            _real_mlp_madds(hidden_features=hidden, n_classes=n_classes) - budget
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
        + hidden_features * out_features
        + out_features
    )
    return 2 * complex_slots


def _real_mlp_parameter_count(*, hidden_features: int, n_classes: int) -> int:
    return (
        2 * hidden_features + hidden_features + hidden_features * n_classes + n_classes
    )


def _complex_mlp_madds(
    *,
    in_features: int,
    hidden_features: int,
    out_features: int,
) -> int:
    return 4 * (in_features * hidden_features + hidden_features * out_features)


def _real_mlp_madds(*, hidden_features: int, n_classes: int) -> int:
    return 2 * hidden_features + hidden_features * n_classes


def _features_for_family(inputs: Tensor, model_family: ModelFamily) -> Tensor:
    if model_family == "complex":
        return inputs
    return torch.cat([inputs.real, inputs.imag], dim=-1)


def _activation_factory(activation: ActivationName) -> Callable[[], nn.Module]:
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


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _format_summary_markdown(
    summaries: Sequence[PhaseSummary],
    *,
    config: PhaseBenchmarkConfig | None = None,
) -> str:
    rows = [
        "# Synthetic Phase Classification",
        "",
        (
            "Snapshot of one configuration. Numbers depend on platform BLAS, "
            "wall-clock load, and `git_commit`/`git_dirty` recorded in the "
            "manifest. Re-runs will not be byte-identical; see "
            "`docs/baselines.md` and `docs/tuning_budget.md` for the "
            "comparison rules."
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
                f"`n_classes={config.n_classes}`, "
                f"`noise_std={config.noise_std}`.",
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
        if value == "complex":
            parsed.append("complex")
        elif value == "real_stacked":
            parsed.append("real_stacked")
        elif value == "real_matched_params":
            parsed.append("real_matched_params")
        elif value == "real_matched_flops":
            parsed.append("real_matched_flops")
        else:
            msg = f"unsupported model family: {value}"
            raise ValueError(msg)
    return tuple(parsed)


def _parse_activation(value: str) -> ActivationName:
    allowed = {"crelu", "zrelu", "modrelu", "cardioid", "siglog"}
    if value not in allowed:
        msg = f"unsupported activation: {value}"
        raise ValueError(msg)
    return cast(ActivationName, value)


def _parse_real_activation(value: str) -> RealActivationName:
    allowed = {"relu", "leaky_relu", "gelu", "tanh"}
    if value not in allowed:
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
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--class-spread", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument(
        "--activation",
        choices=["crelu", "zrelu", "modrelu", "cardioid", "siglog"],
        default="crelu",
    )
    parser.add_argument(
        "--real-activation",
        choices=["relu", "leaky_relu", "gelu", "tanh"],
        default="relu",
        help="activation for real-valued baselines (asymmetric vs --activation)",
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
        default=Path("results/synthetic_phase_classification"),
    )
    args = parser.parse_args()

    config = PhaseBenchmarkConfig(
        seeds=tuple(args.seeds),
        model_families=_parse_model_families(args.model_families),
        n_train=args.n_train,
        n_test=args.n_test,
        n_classes=args.n_classes,
        hidden_features=args.hidden_features,
        steps=args.steps,
        learning_rate=args.learning_rate,
        class_spread=args.class_spread,
        noise_std=args.noise_std,
        activation=_parse_activation(args.activation),
        real_activation=_parse_real_activation(args.real_activation),
        device=args.device,
        dtype=args.dtype,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
    )
    runs, summaries = run_phase_classification_benchmark(config)
    write_phase_benchmark_outputs(
        args.output_dir,
        config=config,
        runs=runs,
        summaries=summaries,
    )
    print(_format_summary_markdown(summaries, config=config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
