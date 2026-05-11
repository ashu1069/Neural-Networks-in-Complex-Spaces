"""Quantum wavefunction pilot for complex-valued representation claims.

The pilot keeps the physics deliberately small and inspectable: one task asks
models to infer wavepacket momentum from the phase gradient of a complex state,
and one task asks models to infer the potential class after 1D Schrodinger
evolution. Global-phase stress tests probe whether a model has learned a
physical symmetry or merely a coordinate convention.

Examples:

    uv run python experiments/physics/quantum_wavefunction.py --preset smoke

    uv run python experiments/physics/quantum_wavefunction.py \\
        --preset standard --device cuda --resume
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from cvnn.baselines import count_real_parameters
from cvnn.repro import Environment, JsonObject, collect_environment, new_manifest
from experiments.rf.synthetic_modulation import (
    ActivationName,
    ArchitectureName,
    ModelFamily,
    RealActivationName,
    _features_for_family,
    _make_model,
)
from experiments.synthetic.phase_classification import bootstrap_mean_ci

PresetName = Literal["smoke", "standard", "full"]
TaskName = Literal["momentum_phase", "potential_inverse"]
TransformName = Literal["none", "fixed_global_phase", "random_global_phase"]

PHYSICS_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "complex",
    "real_stacked",
    "real_polar",
    "real_phase",
    "real_magnitude",
)
MODEL_COLORS: dict[str, str] = {
    "complex": "#355c9c",
    "real_stacked": "#4c956c",
    "real_polar": "#2a9d8f",
    "real_phase": "#e76f51",
    "real_magnitude": "#6c757d",
}
MOMENTUM_LABELS: tuple[str, ...] = (
    "k=-3",
    "k=-1.5",
    "k=1.5",
    "k=3",
)
MOMENTA: tuple[float, ...] = (-3.0, -1.5, 1.5, 3.0)
POTENTIAL_LABELS: tuple[str, ...] = (
    "free",
    "barrier",
    "well",
    "double_barrier",
    "harmonic",
)


@dataclass(frozen=True)
class QuantumWavefunctionData:
    """Balanced complex-wavefunction train/test split."""

    train_inputs: Tensor
    train_labels: Tensor
    test_inputs: Tensor
    test_labels: Tensor
    class_names: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class InputTransform:
    """Serializable transform applied to wavefunctions."""

    name: TransformName = "none"
    rotation_radians: float = 0.0

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "rotation_radians": self.rotation_radians,
        }


@dataclass(frozen=True)
class QuantumCondition:
    """One physics pilot condition."""

    condition_id: str
    task: TaskName
    question: str
    expected_signal: str
    model_families: tuple[ModelFamily, ...] = PHYSICS_MODEL_FAMILIES
    train_transform: InputTransform = InputTransform()
    test_transform: InputTransform = InputTransform()

    def to_dict(self) -> JsonObject:
        return {
            "condition_id": self.condition_id,
            "task": self.task,
            "question": self.question,
            "expected_signal": self.expected_signal,
            "model_families": list(self.model_families),
            "train_transform": self.train_transform.to_dict(),
            "test_transform": self.test_transform.to_dict(),
        }


@dataclass(frozen=True)
class QuantumRunConfig:
    """Shared knobs for the quantum pilot suite."""

    preset: PresetName
    progress: bool
    seeds: tuple[int, ...]
    examples_per_class: int
    grid_size: int
    x_min: float
    x_max: float
    train_fraction: float
    evolution_steps: int
    evolution_dt: float
    observation_noise_std: float
    hidden_features: int
    train_steps: int
    batch_size: int
    learning_rate: float
    architecture: ArchitectureName
    kernel_size: int
    activation: ActivationName
    real_activation: RealActivationName
    device: str
    dtype: str
    bootstrap_samples: int
    confidence: float

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


@dataclass(frozen=True)
class QuantumRunResult:
    """Raw metrics for one condition/model/seed."""

    condition_id: str
    task: TaskName
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
    accuracy_by_class: dict[str, float]

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


@dataclass(frozen=True)
class QuantumSummary:
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
    train_seconds_mean: float
    accuracy_by_class_mean: dict[str, float]

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


def build_quantum_conditions() -> tuple[QuantumCondition, ...]:
    """Return the ordered physics pilot suite."""

    return (
        QuantumCondition(
            condition_id="momentum_phase",
            task="momentum_phase",
            question=(
                "Can models infer wavepacket momentum when the label is stored "
                "in the phase gradient rather than in |psi|?"
            ),
            expected_signal=(
                "magnitude-only should sit near chance; phase-aware and "
                "Cartesian encodings should recover the momentum classes."
            ),
        ),
        QuantumCondition(
            condition_id="potential_inverse",
            task="potential_inverse",
            question=(
                "Can models infer which potential generated the observed final "
                "wavefunction after 1D Schrodinger evolution?"
            ),
            expected_signal=(
                "full complex/Cartesian/polar inputs should outperform "
                "phase-only or density-only views when both amplitude and "
                "phase carry scattering information."
            ),
        ),
        QuantumCondition(
            condition_id="global_phase_shift",
            task="potential_inverse",
            question=(
                "Do models trained in one global-phase convention respect the "
                "physical invariance psi -> exp(i theta) psi?"
            ),
            expected_signal=(
                "coordinate-dependent models may degrade under an unseen "
                "global phase; magnitude-only is invariant but information-poor."
            ),
            test_transform=InputTransform("fixed_global_phase", math.pi / 2.0),
        ),
        QuantumCondition(
            condition_id="global_phase_augmented",
            task="potential_inverse",
            question=(
                "Does random global-phase augmentation recover robustness to an "
                "unseen fixed global phase?"
            ),
            expected_signal=(
                "phase-aware models should improve relative to the unaugmented "
                "global-phase stress test."
            ),
            train_transform=InputTransform("random_global_phase"),
            test_transform=InputTransform("fixed_global_phase", math.pi / 2.0),
        ),
    )


def make_momentum_phase_dataset(
    *,
    seed: int = 0,
    examples_per_class: int = 128,
    grid_size: int = 96,
    x_min: float = -8.0,
    x_max: float = 8.0,
    train_fraction: float = 0.8,
    dtype: torch.dtype = torch.complex64,
) -> QuantumWavefunctionData:
    """Generate Gaussian wavepackets whose class is encoded by phase gradient."""

    _validate_dataset_args(
        examples_per_class=examples_per_class,
        grid_size=grid_size,
        train_fraction=train_fraction,
        dtype=dtype,
    )
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.linspace(x_min, x_max, grid_size, dtype=real_dtype)
    dx = _grid_spacing(x)
    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    for label, momentum in enumerate(MOMENTA):
        x0 = _uniform(generator, (examples_per_class, 1), -0.9, 0.9, real_dtype)
        width = _uniform(generator, (examples_per_class, 1), 0.75, 1.35, real_dtype)
        global_phase = _uniform(
            generator,
            (examples_per_class, 1),
            -math.pi,
            math.pi,
            real_dtype,
        )
        amplitude = torch.exp(-0.5 * ((x.unsqueeze(0) - x0) / width) ** 2)
        angle = momentum * x.unsqueeze(0) + global_phase
        wavefunction = torch.complex(
            amplitude * torch.cos(angle),
            amplitude * torch.sin(angle),
        )
        wavefunction = _normalize_wavefunction(wavefunction.to(dtype), dx=dx)
        inputs.append(wavefunction)
        labels.append(torch.full((examples_per_class,), label, dtype=torch.long))
    return _balanced_split(inputs, labels, MOMENTUM_LABELS, train_fraction, generator)


def make_potential_inverse_dataset(
    *,
    seed: int = 0,
    examples_per_class: int = 128,
    grid_size: int = 96,
    x_min: float = -8.0,
    x_max: float = 8.0,
    train_fraction: float = 0.8,
    evolution_steps: int = 40,
    evolution_dt: float = 0.03,
    observation_noise_std: float = 0.0,
    dtype: torch.dtype = torch.complex64,
) -> QuantumWavefunctionData:
    """Generate final states from a 1D time-dependent Schrodinger simulation."""

    _validate_dataset_args(
        examples_per_class=examples_per_class,
        grid_size=grid_size,
        train_fraction=train_fraction,
        dtype=dtype,
    )
    if evolution_steps <= 0:
        msg = "evolution_steps must be positive"
        raise ValueError(msg)
    if evolution_dt <= 0:
        msg = "evolution_dt must be positive"
        raise ValueError(msg)
    if observation_noise_std < 0:
        msg = "observation_noise_std must be non-negative"
        raise ValueError(msg)

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.linspace(x_min, x_max, grid_size, dtype=real_dtype)
    dx = _grid_spacing(x)
    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    for label, class_name in enumerate(POTENTIAL_LABELS):
        initial = _initial_scattering_states(
            x,
            examples_per_class=examples_per_class,
            generator=generator,
            dtype=dtype,
        )
        potentials = _potential_batch(
            class_name,
            x,
            examples_per_class=examples_per_class,
            generator=generator,
        )
        final = _split_step_propagate(
            initial,
            potentials,
            dx=dx,
            dt=evolution_dt,
            steps=evolution_steps,
        )
        if observation_noise_std > 0:
            final = _add_complex_noise(
                final,
                std=observation_noise_std,
                generator=generator,
            )
        final = _normalize_wavefunction(final.to(dtype), dx=dx)
        inputs.append(final)
        labels.append(torch.full((examples_per_class,), label, dtype=torch.long))
    return _balanced_split(inputs, labels, POTENTIAL_LABELS, train_fraction, generator)


def apply_input_transform(
    inputs: Tensor,
    transform: InputTransform,
    *,
    seed: int,
) -> Tensor:
    """Apply a deterministic global-phase transform."""

    if transform.name == "none":
        return inputs
    real_dtype = inputs.real.dtype
    if transform.name == "fixed_global_phase":
        angle = torch.tensor(transform.rotation_radians, dtype=real_dtype)
        phasor = torch.complex(torch.cos(angle), torch.sin(angle)).to(inputs.dtype)
        return inputs * phasor
    if transform.name == "random_global_phase":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 7_000_000)
        angles = _uniform(
            generator,
            (inputs.shape[0], 1),
            -math.pi,
            math.pi,
            real_dtype,
        )
        phasor = torch.complex(torch.cos(angles), torch.sin(angles)).to(inputs.dtype)
        return inputs * phasor
    msg = f"unsupported transform: {transform.name}"
    raise ValueError(msg)


def run_condition(
    condition: QuantumCondition,
    config: QuantumRunConfig,
    *,
    output_dir: Path,
    environment: Environment,
    resume: bool = False,
    progress_bar: tqdm[Any] | None = None,
) -> tuple[list[QuantumRunResult], list[QuantumSummary]]:
    """Run one physics condition and write condition artifacts."""

    condition_dir = output_dir / condition.condition_id
    summary_path = condition_dir / "summary.json"
    if resume and summary_path.exists():
        payload = json.loads(summary_path.read_text())
        resumed_runs = [
            _run_result_from_dict(cast(dict[str, Any], item))
            for item in cast(list[Any], payload["raw_runs"])
        ]
        resumed_summaries = [
            _summary_from_dict(cast(dict[str, Any], item))
            for item in cast(list[Any], payload["summaries"])
        ]
        return resumed_runs, resumed_summaries

    runs: list[QuantumRunResult] = []
    for model_family in condition.model_families:
        for seed in config.seeds:
            latest = train_quantum_classifier(
                condition,
                config=config,
                seed=seed,
                model_family=model_family,
            )
            runs.append(latest)
            if progress_bar is not None:
                progress_bar.set_description_str(condition.condition_id)
                progress_bar.set_postfix_str(
                    f"{model_family}/s{seed} acc={latest.test_accuracy:.4f}",
                    refresh=False,
                )
                progress_bar.update(1)
            else:
                print(
                    f"[{condition.condition_id}] {model_family}/seed{seed}: "
                    f"acc={latest.test_accuracy:.4f}"
                )

    summaries = summarize_quantum_runs(
        runs,
        model_order=condition.model_families,
        class_names=_class_names_for_task(condition.task),
        bootstrap_samples=config.bootstrap_samples,
        confidence=config.confidence,
    )
    write_condition_outputs(
        condition_dir,
        condition=condition,
        config=config,
        runs=runs,
        summaries=summaries,
        environment=environment,
    )
    return runs, summaries


def train_quantum_classifier(
    condition: QuantumCondition,
    *,
    config: QuantumRunConfig,
    seed: int,
    model_family: ModelFamily,
) -> QuantumRunResult:
    """Train one model family on one deterministic physics split."""

    dtype = _parse_complex_dtype(config.dtype)
    device = torch.device(config.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = _make_dataset_for_condition(condition, config=config, seed=seed, dtype=dtype)
    train_inputs_complex = apply_input_transform(
        data.train_inputs,
        condition.train_transform,
        seed=seed,
    )
    test_inputs_complex = apply_input_transform(
        data.test_inputs,
        condition.test_transform,
        seed=seed,
    )
    model, effective_hidden, estimated_madds = _make_model(
        model_family,
        architecture=config.architecture,
        sample_length=config.grid_size,
        complex_hidden_features=config.hidden_features,
        n_classes=data.n_classes,
        activation=config.activation,
        real_activation=config.real_activation,
        kernel_size=config.kernel_size,
        device=device,
        dtype=dtype,
    )
    parameter_count = count_real_parameters(model)
    train_inputs = _features_for_family(
        train_inputs_complex.to(device), model_family, config.architecture
    )
    test_inputs = _features_for_family(
        test_inputs_complex.to(device), model_family, config.architecture
    )
    train_labels = data.train_labels.to(device)
    test_labels = data.test_labels.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    n_train = train_inputs.shape[0]
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(seed + 1_000_000)
    train_loss = torch.tensor(float("nan"), device=device)
    start = time.perf_counter()
    model.train()
    for _ in range(config.train_steps):
        indices = torch.randint(
            0,
            n_train,
            (config.batch_size,),
            generator=batch_generator,
        )
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
        accuracy_by_class = {}
        for label, class_name in enumerate(data.class_names):
            mask = test_labels == label
            if mask.any():
                accuracy_by_class[class_name] = float(correct[mask].mean().item())

    return QuantumRunResult(
        condition_id=condition.condition_id,
        task=condition.task,
        model_family=model_family,
        seed=seed,
        steps=config.train_steps,
        learning_rate=config.learning_rate,
        hidden_features=effective_hidden,
        parameter_count=parameter_count,
        estimated_forward_madds=estimated_madds,
        train_seconds=train_seconds,
        train_loss=float(train_loss.detach().cpu().item()),
        test_loss=float(test_loss.detach().cpu().item()),
        test_accuracy=float(test_accuracy.detach().cpu().item()),
        accuracy_by_class=accuracy_by_class,
    )


def summarize_quantum_runs(
    runs: Sequence[QuantumRunResult],
    *,
    model_order: Sequence[ModelFamily],
    class_names: Sequence[str],
    bootstrap_samples: int,
    confidence: float,
) -> list[QuantumSummary]:
    """Aggregate per-seed physics runs."""

    summaries: list[QuantumSummary] = []
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
        class_means: dict[str, float] = {}
        for class_name in class_names:
            per_seed = [
                run.accuracy_by_class[class_name]
                for run in family_runs
                if class_name in run.accuracy_by_class
            ]
            if per_seed:
                class_means[class_name] = statistics.mean(per_seed)
        summaries.append(
            QuantumSummary(
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
                accuracy_by_class_mean=class_means,
            )
        )
    return summaries


def write_condition_outputs(
    output_dir: Path,
    *,
    condition: QuantumCondition,
    config: QuantumRunConfig,
    runs: Sequence[QuantumRunResult],
    summaries: Sequence[QuantumSummary],
    environment: Environment,
) -> None:
    """Write condition-level JSON, Markdown, plots, and manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_runs_payload = [run.to_dict() for run in runs]
    summary_payload = cast(
        JsonObject,
        {
            "condition": condition.to_dict(),
            "config": config.to_dict(),
            "raw_runs": raw_runs_payload,
            "summaries": [summary.to_dict() for summary in summaries],
        },
    )
    (output_dir / "raw_runs.json").write_text(
        json.dumps(raw_runs_payload, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    plot_artifacts = write_condition_plots(output_dir, condition, summaries=summaries)
    (output_dir / "summary.md").write_text(
        format_condition_markdown(
            condition,
            config=config,
            summaries=summaries,
            plot_artifacts=plot_artifacts,
        )
        + "\n"
    )
    manifest = new_manifest(
        run_id=f"quantum-wavefunction-{condition.condition_id}",
        config={"condition": condition.to_dict(), "run_config": config.to_dict()},
        seeds=list(config.seeds),
        metrics=summary_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "quantum_wavefunction_pilot",
            "version": "0.1.0",
            "description": (
                "1D complex wavefunctions for phase-gradient and Schrodinger "
                "potential-inverse pilot tasks"
            ),
        },
        artifacts={
            "raw_runs": str(output_dir / "raw_runs.json"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_markdown": str(output_dir / "summary.md"),
            **plot_artifacts,
        },
        environment=environment,
    )
    manifest.write_json(output_dir / "manifest.json")


def write_suite_index(
    output_dir: Path,
    *,
    config: QuantumRunConfig,
    condition_results: Sequence[tuple[QuantumCondition, Sequence[QuantumSummary]]],
    environment: Environment,
) -> None:
    """Write root-level suite index and cross-condition plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [
        {
            "condition_id": condition.condition_id,
            "task": condition.task,
            "question": condition.question,
            "expected_signal": condition.expected_signal,
            "best_model": _best_summary(summaries).model_family if summaries else None,
            "best_accuracy": _best_summary(summaries).test_accuracy_mean
            if summaries
            else None,
            "complex_accuracy": _accuracy_for("complex", summaries),
            "real_stacked_accuracy": _accuracy_for("real_stacked", summaries),
            "real_phase_accuracy": _accuracy_for("real_phase", summaries),
            "real_polar_accuracy": _accuracy_for("real_polar", summaries),
            "real_magnitude_accuracy": _accuracy_for("real_magnitude", summaries),
            "summary_markdown": str(output_dir / condition.condition_id / "summary.md"),
        }
        for condition, summaries in condition_results
    ]
    index_payload = cast(
        JsonObject,
        {
            "config": config.to_dict(),
            "conditions": [condition.to_dict() for condition, _ in condition_results],
            "results": rows,
        },
    )
    (output_dir / "index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n"
    )
    plot_artifacts = write_suite_plots(
        output_dir,
        condition_results=condition_results,
    )
    (output_dir / "index.md").write_text(
        format_index_markdown(rows, plot_artifacts=plot_artifacts) + "\n"
    )
    manifest = new_manifest(
        run_id="quantum-wavefunction-pilot-suite",
        config=config.to_dict(),
        seeds=list(config.seeds),
        metrics=index_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "quantum_wavefunction_pilot",
            "version": "0.1.0",
            "description": "sequential quantum-wavefunction representation pilot",
        },
        artifacts={
            "index_json": str(output_dir / "index.json"),
            "index_markdown": str(output_dir / "index.md"),
            **plot_artifacts,
        },
        environment=environment,
    )
    manifest.write_json(output_dir / "manifest.json")


def write_condition_plots(
    output_dir: Path,
    condition: QuantumCondition,
    *,
    summaries: Sequence[QuantumSummary],
) -> dict[str, str]:
    """Write plots for one physics condition."""

    if not summaries:
        return {}
    plt = _load_pyplot()
    artifacts: dict[str, str] = {}
    labels = [summary.model_family for summary in summaries]
    accuracies = [summary.test_accuracy_mean for summary in summaries]
    yerr_low = [
        max(0.0, summary.test_accuracy_mean - summary.test_accuracy_ci_low)
        for summary in summaries
    ]
    yerr_high = [
        max(0.0, summary.test_accuracy_ci_high - summary.test_accuracy_mean)
        for summary in summaries
    ]
    colors = [MODEL_COLORS.get(label, "#495057") for label in labels]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 * len(labels)), 4.4))
    ax.bar(
        labels,
        accuracies,
        yerr=[yerr_low, yerr_high],
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        capsize=3,
    )
    ax.set_title(condition.condition_id)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    bar_path = output_dir / "accuracy_bar.png"
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)
    artifacts["accuracy_bar"] = str(bar_path)

    class_names = sorted(
        {
            class_name
            for summary in summaries
            for class_name in summary.accuracy_by_class_mean
        }
    )
    if class_names:
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        x_positions = torch.arange(len(class_names), dtype=torch.float32).tolist()
        width = 0.8 / max(1, len(summaries))
        for index, summary in enumerate(summaries):
            offset = (index - (len(summaries) - 1) / 2.0) * width
            values = [
                summary.accuracy_by_class_mean.get(class_name, float("nan"))
                for class_name in class_names
            ]
            ax.bar(
                [position + offset for position in x_positions],
                values,
                width=width,
                label=summary.model_family,
                color=MODEL_COLORS.get(summary.model_family),
            )
        ax.set_title(f"{condition.condition_id}: accuracy by class")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(class_names, rotation=25, ha="right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        class_path = output_dir / "accuracy_by_class.png"
        fig.savefig(class_path, dpi=180)
        plt.close(fig)
        artifacts["accuracy_by_class"] = str(class_path)

    return artifacts


def write_suite_plots(
    output_dir: Path,
    *,
    condition_results: Sequence[tuple[QuantumCondition, Sequence[QuantumSummary]]],
) -> dict[str, str]:
    """Write cross-condition pilot plots."""

    if not condition_results:
        return {}
    plt = _load_pyplot()
    artifacts: dict[str, str] = {}
    families = [
        family
        for family in PHYSICS_MODEL_FAMILIES
        if any(
            any(summary.model_family == family for summary in summaries)
            for _, summaries in condition_results
        )
    ]
    matrix: list[list[float]] = []
    for _, summaries in condition_results:
        by_family = {summary.model_family: summary for summary in summaries}
        matrix.append(
            [
                by_family[family].test_accuracy_mean
                if family in by_family
                else float("nan")
                for family in families
            ]
        )

    fig, ax = plt.subplots(
        figsize=(max(7.0, 1.0 * len(families)), max(4.0, 0.65 * len(matrix) + 1.8))
    )
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_yticks(range(len(condition_results)))
    ax.set_yticklabels([condition.condition_id for condition, _ in condition_results])
    ax.set_title("Quantum wavefunction pilot: test accuracy")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if math.isfinite(value):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="test accuracy")
    fig.tight_layout()
    heatmap_path = output_dir / "accuracy_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    artifacts["accuracy_heatmap"] = str(heatmap_path)

    labels = [condition.condition_id for condition, _ in condition_results]
    best_values = [
        _best_summary(summaries).test_accuracy_mean
        for _, summaries in condition_results
    ]
    best_labels = [
        _best_summary(summaries).model_family for _, summaries in condition_results
    ]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.35 * len(labels)), 4.4))
    colors = [MODEL_COLORS.get(label, "#495057") for label in best_labels]
    ax.bar(labels, best_values, color=colors, edgecolor="#222222", linewidth=0.6)
    for index, (value, family) in enumerate(zip(best_values, best_labels, strict=True)):
        ax.text(
            index,
            min(0.98, value + 0.025),
            family,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Best model by quantum condition")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    best_path = output_dir / "best_accuracy_by_condition.png"
    fig.savefig(best_path, dpi=180)
    plt.close(fig)
    artifacts["best_accuracy_by_condition"] = str(best_path)
    return artifacts


def format_condition_markdown(
    condition: QuantumCondition,
    *,
    config: QuantumRunConfig,
    summaries: Sequence[QuantumSummary],
    plot_artifacts: dict[str, str],
) -> str:
    lines = [
        f"# {condition.condition_id}",
        "",
        f"Task: `{condition.task}`.",
        "",
        f"Question: {condition.question}",
        "",
        f"Expected signal: {condition.expected_signal}",
        "",
        (
            f"Preset: `{config.preset}`. Seeds: `{list(config.seeds)}`. "
            f"Examples/class: `{config.examples_per_class}`. Grid: "
            f"`{config.grid_size}`. Train steps: `{config.train_steps}`."
        ),
        "",
    ]
    if plot_artifacts:
        lines.extend(_plot_markdown_lines(plot_artifacts))
    lines.extend(
        [
            "| family | acc | 95% CI | loss | params | madds | seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        lines.append(
            " | ".join(
                [
                    f"| `{summary.model_family}`",
                    f"{summary.test_accuracy_mean:.4f}",
                    (
                        f"[{summary.test_accuracy_ci_low:.4f}, "
                        f"{summary.test_accuracy_ci_high:.4f}]"
                    ),
                    f"{summary.test_loss_mean:.4f}",
                    str(summary.parameter_count),
                    str(summary.estimated_forward_madds),
                    f"{summary.train_seconds_mean:.2f} |",
                ]
            )
        )
    return "\n".join(lines)


def format_index_markdown(
    rows: Sequence[dict[str, object]],
    *,
    plot_artifacts: dict[str, str],
) -> str:
    lines = ["# Quantum Wavefunction Pilot", ""]
    if plot_artifacts:
        lines.extend(_plot_markdown_lines(plot_artifacts))
    lines.extend(
        [
            (
                "| condition | best | acc | complex | stacked | phase | polar | "
                "magnitude |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"| `{row['condition_id']}`",
                    f"`{row['best_model']}`",
                    _format_optional_float(row["best_accuracy"]),
                    _format_optional_float(row["complex_accuracy"]),
                    _format_optional_float(row["real_stacked_accuracy"]),
                    _format_optional_float(row["real_phase_accuracy"]),
                    _format_optional_float(row["real_polar_accuracy"]),
                    _format_optional_float(row["real_magnitude_accuracy"]) + " |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "Each condition directory contains `raw_runs.json`, `summary.json`, "
            "`summary.md`, `manifest.json`, and plots.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["smoke", "standard", "full"],
        default="smoke",
        help="shared pilot budget; individual knobs can override the preset",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help="condition ids to run, or all",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--evolution-steps", type=int, default=None)
    parser.add_argument("--evolution-dt", type=float, default=0.03)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    parser.add_argument("--hidden-features", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--architecture", choices=["mlp", "conv"], default="conv")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--activation", default="zrelu")
    parser.add_argument("--real-activation", default="relu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex64",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/physics_quantum_wavefunction_pilot"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)
    conditions = _select_conditions(args.tests, build_quantum_conditions())
    environment = collect_environment(device=config.device, dtype=config.dtype)
    condition_results: list[tuple[QuantumCondition, Sequence[QuantumSummary]]] = []
    progress_bar = _progress_bar(
        _unfinished_run_count(
            conditions,
            config=config,
            output_dir=args.output_dir,
            resume=args.resume,
        ),
        enabled=config.progress,
        desc="quantum pilot",
    )
    try:
        for condition in conditions:
            _, summaries = run_condition(
                condition,
                config,
                output_dir=args.output_dir,
                environment=environment,
                resume=args.resume,
                progress_bar=progress_bar,
            )
            condition_results.append((condition, summaries))
    finally:
        if progress_bar is not None:
            progress_bar.close()
    write_suite_index(
        args.output_dir,
        config=config,
        condition_results=condition_results,
        environment=environment,
    )
    print((args.output_dir / "index.md").read_text())
    return 0


def _config_from_args(args: argparse.Namespace) -> QuantumRunConfig:
    preset = cast(PresetName, args.preset)
    defaults = _preset_defaults(preset)
    return QuantumRunConfig(
        preset=preset,
        progress=not args.no_progress,
        seeds=tuple(args.seeds)
        if args.seeds is not None
        else cast(tuple[int, ...], defaults["seeds"]),
        examples_per_class=args.examples_per_class
        if args.examples_per_class is not None
        else cast(int, defaults["examples_per_class"]),
        grid_size=args.grid_size
        if args.grid_size is not None
        else cast(int, defaults["grid_size"]),
        x_min=args.x_min,
        x_max=args.x_max,
        train_fraction=args.train_fraction,
        evolution_steps=args.evolution_steps
        if args.evolution_steps is not None
        else cast(int, defaults["evolution_steps"]),
        evolution_dt=args.evolution_dt,
        observation_noise_std=args.observation_noise_std,
        hidden_features=args.hidden_features
        if args.hidden_features is not None
        else cast(int, defaults["hidden_features"]),
        train_steps=args.train_steps
        if args.train_steps is not None
        else cast(int, defaults["train_steps"]),
        batch_size=args.batch_size
        if args.batch_size is not None
        else cast(int, defaults["batch_size"]),
        learning_rate=args.learning_rate,
        architecture=cast(ArchitectureName, args.architecture),
        kernel_size=args.kernel_size,
        activation=cast(ActivationName, args.activation),
        real_activation=cast(RealActivationName, args.real_activation),
        device=args.device,
        dtype=args.dtype,
        bootstrap_samples=args.bootstrap_samples
        if args.bootstrap_samples is not None
        else cast(int, defaults["bootstrap_samples"]),
        confidence=args.confidence,
    )


def _preset_defaults(preset: PresetName) -> dict[str, object]:
    if preset == "smoke":
        return {
            "seeds": (0,),
            "examples_per_class": 24,
            "grid_size": 48,
            "evolution_steps": 12,
            "hidden_features": 8,
            "train_steps": 30,
            "batch_size": 48,
            "bootstrap_samples": 100,
        }
    if preset == "standard":
        return {
            "seeds": (0, 1, 2),
            "examples_per_class": 128,
            "grid_size": 96,
            "evolution_steps": 40,
            "hidden_features": 16,
            "train_steps": 140,
            "batch_size": 128,
            "bootstrap_samples": 1000,
        }
    if preset == "full":
        return {
            "seeds": (0, 1, 2, 3, 4),
            "examples_per_class": 512,
            "grid_size": 128,
            "evolution_steps": 80,
            "hidden_features": 32,
            "train_steps": 400,
            "batch_size": 256,
            "bootstrap_samples": 2000,
        }
    msg = f"unsupported preset: {preset}"
    raise ValueError(msg)


def _make_dataset_for_condition(
    condition: QuantumCondition,
    *,
    config: QuantumRunConfig,
    seed: int,
    dtype: torch.dtype,
) -> QuantumWavefunctionData:
    if condition.task == "momentum_phase":
        return make_momentum_phase_dataset(
            seed=seed,
            examples_per_class=config.examples_per_class,
            grid_size=config.grid_size,
            x_min=config.x_min,
            x_max=config.x_max,
            train_fraction=config.train_fraction,
            dtype=dtype,
        )
    if condition.task == "potential_inverse":
        return make_potential_inverse_dataset(
            seed=seed,
            examples_per_class=config.examples_per_class,
            grid_size=config.grid_size,
            x_min=config.x_min,
            x_max=config.x_max,
            train_fraction=config.train_fraction,
            evolution_steps=config.evolution_steps,
            evolution_dt=config.evolution_dt,
            observation_noise_std=config.observation_noise_std,
            dtype=dtype,
        )
    msg = f"unsupported task: {condition.task}"
    raise ValueError(msg)


def _initial_scattering_states(
    x: Tensor,
    *,
    examples_per_class: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    real_dtype = x.dtype
    x0 = _uniform(generator, (examples_per_class, 1), -4.8, -3.2, real_dtype)
    width = _uniform(generator, (examples_per_class, 1), 0.55, 0.9, real_dtype)
    momentum = _uniform(generator, (examples_per_class, 1), 2.0, 3.0, real_dtype)
    amplitude = torch.exp(-0.5 * ((x.unsqueeze(0) - x0) / width) ** 2)
    angle = momentum * x.unsqueeze(0)
    wavefunction = torch.complex(
        amplitude * torch.cos(angle),
        amplitude * torch.sin(angle),
    )
    return _normalize_wavefunction(wavefunction.to(dtype), dx=_grid_spacing(x))


def _potential_batch(
    class_name: str,
    x: Tensor,
    *,
    examples_per_class: int,
    generator: torch.Generator,
) -> Tensor:
    real_dtype = x.dtype
    batch_x = x.unsqueeze(0)
    if class_name == "free":
        return torch.zeros((examples_per_class, x.shape[0]), dtype=real_dtype)
    if class_name == "barrier":
        height = _uniform(generator, (examples_per_class, 1), 3.0, 5.0, real_dtype)
        center = _uniform(generator, (examples_per_class, 1), -0.25, 0.25, real_dtype)
        width = _uniform(generator, (examples_per_class, 1), 0.35, 0.6, real_dtype)
        return height * torch.exp(-(((batch_x - center) / width) ** 8))
    if class_name == "well":
        depth = _uniform(generator, (examples_per_class, 1), 3.0, 5.0, real_dtype)
        center = _uniform(generator, (examples_per_class, 1), -0.25, 0.25, real_dtype)
        width = _uniform(generator, (examples_per_class, 1), 0.35, 0.6, real_dtype)
        return -depth * torch.exp(-(((batch_x - center) / width) ** 8))
    if class_name == "double_barrier":
        height = _uniform(generator, (examples_per_class, 1), 2.2, 4.2, real_dtype)
        sep = _uniform(generator, (examples_per_class, 1), 0.7, 1.2, real_dtype)
        width = _uniform(generator, (examples_per_class, 1), 0.22, 0.38, real_dtype)
        left = torch.exp(-(((batch_x + sep / 2.0) / width) ** 8))
        right = torch.exp(-(((batch_x - sep / 2.0) / width) ** 8))
        return height * (left + right)
    if class_name == "harmonic":
        omega = _uniform(generator, (examples_per_class, 1), 0.22, 0.36, real_dtype)
        center = _uniform(generator, (examples_per_class, 1), -0.2, 0.2, real_dtype)
        return 0.5 * (omega**2) * (batch_x - center) ** 2
    msg = f"unsupported potential class: {class_name}"
    raise ValueError(msg)


def _split_step_propagate(
    wavefunction: Tensor,
    potential: Tensor,
    *,
    dx: float,
    dt: float,
    steps: int,
) -> Tensor:
    n_grid = wavefunction.shape[-1]
    real_dtype = wavefunction.real.dtype
    k = 2.0 * math.pi * torch.fft.fftfreq(n_grid, d=dx, dtype=real_dtype)
    kinetic_angle = -0.5 * (k**2) * dt
    kinetic_phase = torch.complex(torch.cos(kinetic_angle), torch.sin(kinetic_angle))
    potential_angle = -0.5 * potential * dt
    potential_phase = torch.complex(
        torch.cos(potential_angle),
        torch.sin(potential_angle),
    )

    psi = wavefunction
    for _ in range(steps):
        psi = potential_phase * psi
        psi = torch.fft.ifft(torch.fft.fft(psi, dim=-1) * kinetic_phase, dim=-1)
        psi = potential_phase * psi
    return _normalize_wavefunction(psi, dx=dx)


def _add_complex_noise(
    inputs: Tensor,
    *,
    std: float,
    generator: torch.Generator,
) -> Tensor:
    real_dtype = inputs.real.dtype
    noise_real = torch.randn(inputs.shape, generator=generator, dtype=real_dtype)
    noise_imag = torch.randn(inputs.shape, generator=generator, dtype=real_dtype)
    return inputs + torch.complex(noise_real, noise_imag).to(inputs.dtype) * std


def _normalize_wavefunction(wavefunction: Tensor, *, dx: float) -> Tensor:
    norm = torch.sqrt(wavefunction.abs().square().sum(dim=-1, keepdim=True) * dx)
    return wavefunction / norm.clamp_min(torch.finfo(wavefunction.real.dtype).eps)


def _balanced_split(
    inputs: Sequence[Tensor],
    labels: Sequence[Tensor],
    class_names: Sequence[str],
    train_fraction: float,
    generator: torch.Generator,
) -> QuantumWavefunctionData:
    train_inputs: list[Tensor] = []
    train_labels: list[Tensor] = []
    test_inputs: list[Tensor] = []
    test_labels: list[Tensor] = []
    for class_inputs, class_labels in zip(inputs, labels, strict=True):
        n_examples = class_inputs.shape[0]
        n_train = int(round(n_examples * train_fraction))
        if n_train <= 0 or n_train >= n_examples:
            msg = "train_fraction leaves an empty train or test split"
            raise ValueError(msg)
        permutation = torch.randperm(n_examples, generator=generator)
        train_idx = permutation[:n_train]
        test_idx = permutation[n_train:]
        train_inputs.append(class_inputs[train_idx])
        train_labels.append(class_labels[train_idx])
        test_inputs.append(class_inputs[test_idx])
        test_labels.append(class_labels[test_idx])
    return QuantumWavefunctionData(
        train_inputs=torch.cat(train_inputs, dim=0),
        train_labels=torch.cat(train_labels, dim=0),
        test_inputs=torch.cat(test_inputs, dim=0),
        test_labels=torch.cat(test_labels, dim=0),
        class_names=tuple(class_names),
    )


def _validate_dataset_args(
    *,
    examples_per_class: int,
    grid_size: int,
    train_fraction: float,
    dtype: torch.dtype,
) -> None:
    if examples_per_class < 4:
        msg = "examples_per_class must be at least 4"
        raise ValueError(msg)
    if grid_size < 16:
        msg = "grid_size must be at least 16"
        raise ValueError(msg)
    if not 0.0 < train_fraction < 1.0:
        msg = "train_fraction must be in (0, 1)"
        raise ValueError(msg)
    if dtype not in {torch.complex64, torch.complex128}:
        msg = "dtype must be a complex torch dtype"
        raise TypeError(msg)


def _uniform(
    generator: torch.Generator,
    shape: tuple[int, ...],
    low: float,
    high: float,
    dtype: torch.dtype,
) -> Tensor:
    return torch.empty(shape, dtype=dtype).uniform_(low, high, generator=generator)


def _grid_spacing(x: Tensor) -> float:
    return float((x[1] - x[0]).item())


def _class_names_for_task(task: TaskName) -> tuple[str, ...]:
    if task == "momentum_phase":
        return MOMENTUM_LABELS
    if task == "potential_inverse":
        return POTENTIAL_LABELS
    msg = f"unsupported task: {task}"
    raise ValueError(msg)


def _select_conditions(
    requested: Sequence[str],
    available: Sequence[QuantumCondition],
) -> tuple[QuantumCondition, ...]:
    by_id = {condition.condition_id: condition for condition in available}
    if requested == ["all"]:
        return tuple(available)
    selected: list[QuantumCondition] = []
    for condition_id in requested:
        if condition_id not in by_id:
            msg = f"unknown condition id {condition_id!r}; available: {sorted(by_id)}"
            raise ValueError(msg)
        selected.append(by_id[condition_id])
    return tuple(selected)


def _unfinished_run_count(
    conditions: Sequence[QuantumCondition],
    *,
    config: QuantumRunConfig,
    output_dir: Path,
    resume: bool,
) -> int:
    if not resume:
        return sum(
            len(condition.model_families) * len(config.seeds)
            for condition in conditions
        )
    return sum(
        len(condition.model_families) * len(config.seeds)
        for condition in conditions
        if not (output_dir / condition.condition_id / "summary.json").exists()
    )


def _progress_bar(
    total: int,
    *,
    enabled: bool,
    desc: str,
) -> tqdm[Any] | None:
    if not enabled:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)


def _best_summary(summaries: Sequence[QuantumSummary]) -> QuantumSummary:
    return max(summaries, key=lambda summary: summary.test_accuracy_mean)


def _accuracy_for(
    model_family: ModelFamily,
    summaries: Sequence[QuantumSummary],
) -> float | None:
    for summary in summaries:
        if summary.model_family == model_family:
            return summary.test_accuracy_mean
    return None


def _format_optional_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    msg = f"expected numeric value, got {type(value).__name__}"
    raise TypeError(msg)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _plot_markdown_lines(plot_artifacts: dict[str, str]) -> list[str]:
    lines = ["## Plots", ""]
    for name, path in plot_artifacts.items():
        title = name.replace("_", " ")
        lines.append(f"![{title}]({Path(path).name})")
        lines.append("")
    return lines


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_complex_dtype(name: str) -> torch.dtype:
    if name == "complex64":
        return torch.complex64
    if name == "complex128":
        return torch.complex128
    msg = f"unsupported complex dtype: {name}"
    raise ValueError(msg)


def _run_result_from_dict(payload: dict[str, Any]) -> QuantumRunResult:
    return QuantumRunResult(
        condition_id=str(payload["condition_id"]),
        task=cast(TaskName, payload["task"]),
        model_family=cast(ModelFamily, payload["model_family"]),
        seed=int(payload["seed"]),
        steps=int(payload["steps"]),
        learning_rate=float(payload["learning_rate"]),
        hidden_features=int(payload["hidden_features"]),
        parameter_count=int(payload["parameter_count"]),
        estimated_forward_madds=int(payload["estimated_forward_madds"]),
        train_seconds=float(payload["train_seconds"]),
        train_loss=float(payload["train_loss"]),
        test_loss=float(payload["test_loss"]),
        test_accuracy=float(payload["test_accuracy"]),
        accuracy_by_class={
            str(key): float(value)
            for key, value in cast(dict[str, Any], payload["accuracy_by_class"]).items()
        },
    )


def _summary_from_dict(payload: dict[str, Any]) -> QuantumSummary:
    return QuantumSummary(
        model_family=cast(ModelFamily, payload["model_family"]),
        n_runs=int(payload["n_runs"]),
        hidden_features=int(payload["hidden_features"]),
        parameter_count=int(payload["parameter_count"]),
        estimated_forward_madds=int(payload["estimated_forward_madds"]),
        test_accuracy_mean=float(payload["test_accuracy_mean"]),
        test_accuracy_std=float(payload["test_accuracy_std"]),
        test_accuracy_ci_low=float(payload["test_accuracy_ci_low"]),
        test_accuracy_ci_high=float(payload["test_accuracy_ci_high"]),
        test_loss_mean=float(payload["test_loss_mean"]),
        train_seconds_mean=float(payload["train_seconds_mean"]),
        accuracy_by_class_mean={
            str(key): float(value)
            for key, value in cast(
                dict[str, Any],
                payload["accuracy_by_class_mean"],
            ).items()
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
