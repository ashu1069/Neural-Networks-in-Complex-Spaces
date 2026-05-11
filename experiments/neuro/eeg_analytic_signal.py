"""EEG analytic-signal pilot for complex-valued representation claims.

The pilot is synthetic, but it mirrors common EEG/MEG feature pipelines:
real sensor traces are transformed into complex analytic signals whose
magnitude is an amplitude envelope and whose angle is instantaneous phase.
The conditions separate phase locking, amplitude events, phase-amplitude
coupling, and reference/global-phase robustness.

Examples:

    uv run python experiments/neuro/eeg_analytic_signal.py --preset smoke

    uv run python experiments/neuro/eeg_analytic_signal.py \\
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
TaskName = Literal["phase_locking", "amplitude_event", "phase_amplitude_coupling"]
TransformName = Literal["none", "fixed_reference_phase", "random_reference_phase"]

EEG_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
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
PHASE_LOCKING_LABELS: tuple[str, ...] = ("lag0", "lag90", "lag180", "lag270")
PHASE_LOCKING_LAGS: tuple[float, ...] = (
    0.0,
    math.pi / 2.0,
    math.pi,
    3.0 * math.pi / 2.0,
)
AMPLITUDE_EVENT_LABELS: tuple[str, ...] = (
    "burst_ch0",
    "burst_ch1",
    "burst_ch2",
    "burst_ch3",
)
PAC_LABELS: tuple[str, ...] = ("pac0", "pac90", "pac180", "pac270")
PAC_OFFSETS: tuple[float, ...] = PHASE_LOCKING_LAGS


@dataclass(frozen=True)
class EEGAnalyticSignalData:
    """Balanced analytic-signal train/test split."""

    train_inputs: Tensor
    train_labels: Tensor
    test_inputs: Tensor
    test_labels: Tensor
    class_names: tuple[str, ...]
    n_channels: int
    time_steps: int

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class InputTransform:
    """Serializable transform applied to analytic signals."""

    name: TransformName = "none"
    rotation_radians: float = 0.0

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "rotation_radians": self.rotation_radians,
        }


@dataclass(frozen=True)
class EEGCondition:
    """One neuroscience pilot condition."""

    condition_id: str
    task: TaskName
    question: str
    expected_signal: str
    model_families: tuple[ModelFamily, ...] = EEG_MODEL_FAMILIES
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
class EEGRunConfig:
    """Shared knobs for the EEG analytic-signal pilot."""

    preset: PresetName
    progress: bool
    seeds: tuple[int, ...]
    examples_per_class: int
    n_channels: int
    time_steps: int
    train_fraction: float
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

    @property
    def sample_length(self) -> int:
        return self.n_channels * self.time_steps

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


@dataclass(frozen=True)
class EEGRunResult:
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
class EEGSummary:
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


def build_eeg_conditions() -> tuple[EEGCondition, ...]:
    """Return the ordered EEG analytic-signal pilot suite."""

    return (
        EEGCondition(
            condition_id="phase_locking",
            task="phase_locking",
            question=(
                "Can models classify inter-channel phase locking when amplitude "
                "envelopes are randomized independently of the label?"
            ),
            expected_signal=(
                "phase-aware and Cartesian views should learn; magnitude-only "
                "should remain near chance."
            ),
        ),
        EEGCondition(
            condition_id="amplitude_event",
            task="amplitude_event",
            question=(
                "Can models classify which sensor carries an amplitude burst "
                "when phase is independent of the label?"
            ),
            expected_signal=(
                "magnitude, polar, Cartesian, and complex views should learn; "
                "phase-only should remain near chance."
            ),
        ),
        EEGCondition(
            condition_id="phase_amplitude_coupling",
            task="phase_amplitude_coupling",
            question=(
                "Can models detect phase-amplitude coupling when the high-band "
                "amplitude is locked to a low-band phase offset?"
            ),
            expected_signal=(
                "full complex/Cartesian/polar views should outperform pure "
                "phase or pure magnitude because the label is relational."
            ),
        ),
        EEGCondition(
            condition_id="reference_phase_shift",
            task="phase_locking",
            question=(
                "Does a model trained in one sensor-reference phase convention "
                "respect a common unseen analytic-signal rotation?"
            ),
            expected_signal=(
                "coordinate-dependent models may degrade under the reference "
                "shift; magnitude-only remains invariant but information-poor."
            ),
            test_transform=InputTransform("fixed_reference_phase", math.pi / 2.0),
        ),
        EEGCondition(
            condition_id="reference_phase_augmented",
            task="phase_locking",
            question=(
                "Does random reference-phase augmentation recover robustness to "
                "the unseen fixed reference shift?"
            ),
            expected_signal=(
                "phase-aware models should improve relative to the unaugmented "
                "reference-shift condition."
            ),
            train_transform=InputTransform("random_reference_phase"),
            test_transform=InputTransform("fixed_reference_phase", math.pi / 2.0),
        ),
    )


def make_phase_locking_dataset(
    *,
    seed: int = 0,
    examples_per_class: int = 128,
    n_channels: int = 4,
    time_steps: int = 64,
    train_fraction: float = 0.8,
    observation_noise_std: float = 0.03,
    dtype: torch.dtype = torch.complex64,
) -> EEGAnalyticSignalData:
    """Generate analytic signals whose class is inter-channel phase lag."""

    _validate_dataset_args(
        examples_per_class=examples_per_class,
        n_channels=n_channels,
        time_steps=time_steps,
        train_fraction=train_fraction,
        dtype=dtype,
    )
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    time = _normalized_time(time_steps, real_dtype)
    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    for label, lag in enumerate(PHASE_LOCKING_LAGS):
        theta = _base_oscillation(
            time,
            examples_per_class=examples_per_class,
            generator=generator,
            low=3.0,
            high=5.0,
        )
        phases = []
        for channel in range(n_channels):
            if channel == 0:
                channel_phase = theta
            elif channel == 1:
                jitter = _normal(
                    generator,
                    (examples_per_class, time_steps),
                    real_dtype,
                )
                channel_phase = theta + lag + 0.08 * jitter
            else:
                random_offset = _uniform(
                    generator,
                    (examples_per_class, 1),
                    -math.pi,
                    math.pi,
                    real_dtype,
                )
                channel_phase = theta + random_offset
            phases.append(channel_phase)
        phase_tensor = torch.stack(phases, dim=1)
        amplitude = _random_envelope(
            generator,
            examples_per_class=examples_per_class,
            n_channels=n_channels,
            time_steps=time_steps,
            dtype=real_dtype,
        )
        signal = _complex_from_amp_phase(amplitude, phase_tensor, dtype=dtype)
        signal = _add_complex_noise(
            signal,
            std=observation_noise_std,
            generator=generator,
        )
        inputs.append(_flatten_channels(signal))
        labels.append(torch.full((examples_per_class,), label, dtype=torch.long))
    return _balanced_split(
        inputs,
        labels,
        PHASE_LOCKING_LABELS,
        n_channels=n_channels,
        time_steps=time_steps,
        train_fraction=train_fraction,
        generator=generator,
    )


def make_amplitude_event_dataset(
    *,
    seed: int = 0,
    examples_per_class: int = 128,
    n_channels: int = 4,
    time_steps: int = 64,
    train_fraction: float = 0.8,
    observation_noise_std: float = 0.03,
    dtype: torch.dtype = torch.complex64,
) -> EEGAnalyticSignalData:
    """Generate analytic signals whose class is the burst-carrying channel."""

    _validate_dataset_args(
        examples_per_class=examples_per_class,
        n_channels=n_channels,
        time_steps=time_steps,
        train_fraction=train_fraction,
        dtype=dtype,
    )
    if n_channels < len(AMPLITUDE_EVENT_LABELS):
        msg = "amplitude-event task needs at least 4 channels"
        raise ValueError(msg)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    time = _normalized_time(time_steps, real_dtype)
    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    for label in range(len(AMPLITUDE_EVENT_LABELS)):
        phase = _independent_phases(
            generator,
            examples_per_class=examples_per_class,
            n_channels=n_channels,
            time=time,
        )
        amplitude = _random_envelope(
            generator,
            examples_per_class=examples_per_class,
            n_channels=n_channels,
            time_steps=time_steps,
            dtype=real_dtype,
        )
        center = _uniform(generator, (examples_per_class, 1), 0.35, 0.65, real_dtype)
        width = _uniform(generator, (examples_per_class, 1), 0.055, 0.095, real_dtype)
        burst = 1.25 * torch.exp(-0.5 * ((time.unsqueeze(0) - center) / width) ** 2)
        amplitude[:, label, :] = amplitude[:, label, :] + burst
        signal = _complex_from_amp_phase(amplitude, phase, dtype=dtype)
        signal = _add_complex_noise(
            signal,
            std=observation_noise_std,
            generator=generator,
        )
        inputs.append(_flatten_channels(signal))
        labels.append(torch.full((examples_per_class,), label, dtype=torch.long))
    return _balanced_split(
        inputs,
        labels,
        AMPLITUDE_EVENT_LABELS,
        n_channels=n_channels,
        time_steps=time_steps,
        train_fraction=train_fraction,
        generator=generator,
    )


def make_pac_dataset(
    *,
    seed: int = 0,
    examples_per_class: int = 128,
    n_channels: int = 4,
    time_steps: int = 64,
    train_fraction: float = 0.8,
    observation_noise_std: float = 0.03,
    dtype: torch.dtype = torch.complex64,
) -> EEGAnalyticSignalData:
    """Generate phase-amplitude coupling labels in synthetic analytic EEG."""

    _validate_dataset_args(
        examples_per_class=examples_per_class,
        n_channels=n_channels,
        time_steps=time_steps,
        train_fraction=train_fraction,
        dtype=dtype,
    )
    if n_channels < 2:
        msg = "phase-amplitude coupling task needs at least 2 channels"
        raise ValueError(msg)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    time = _normalized_time(time_steps, real_dtype)
    inputs: list[Tensor] = []
    labels: list[Tensor] = []
    for label, offset in enumerate(PAC_OFFSETS):
        low_phase = _base_oscillation(
            time,
            examples_per_class=examples_per_class,
            generator=generator,
            low=1.0,
            high=2.0,
        )
        high_phase = _base_oscillation(
            time,
            examples_per_class=examples_per_class,
            generator=generator,
            low=8.0,
            high=11.0,
        )
        phase = _independent_phases(
            generator,
            examples_per_class=examples_per_class,
            n_channels=n_channels,
            time=time,
        )
        phase[:, 0, :] = low_phase
        phase[:, 1, :] = high_phase
        amplitude = _random_envelope(
            generator,
            examples_per_class=examples_per_class,
            n_channels=n_channels,
            time_steps=time_steps,
            dtype=real_dtype,
        )
        amplitude[:, 0, :] = 1.0
        coupling_noise = 0.03 * _normal(
            generator,
            (examples_per_class, time_steps),
            real_dtype,
        )
        coupling = 1.15 + 0.85 * torch.cos(low_phase - offset) + coupling_noise
        amplitude[:, 1, :] = coupling.clamp_min(0.05)
        signal = _complex_from_amp_phase(amplitude, phase, dtype=dtype)
        signal = _add_complex_noise(
            signal,
            std=observation_noise_std,
            generator=generator,
        )
        inputs.append(_flatten_channels(signal))
        labels.append(torch.full((examples_per_class,), label, dtype=torch.long))
    return _balanced_split(
        inputs,
        labels,
        PAC_LABELS,
        n_channels=n_channels,
        time_steps=time_steps,
        train_fraction=train_fraction,
        generator=generator,
    )


def apply_input_transform(
    inputs: Tensor,
    transform: InputTransform,
    *,
    seed: int,
) -> Tensor:
    """Apply a deterministic common reference-phase transform."""

    if transform.name == "none":
        return inputs
    real_dtype = inputs.real.dtype
    if transform.name == "fixed_reference_phase":
        angle = torch.tensor(transform.rotation_radians, dtype=real_dtype)
        phasor = torch.complex(torch.cos(angle), torch.sin(angle)).to(inputs.dtype)
        return inputs * phasor
    if transform.name == "random_reference_phase":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 8_000_000)
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
    condition: EEGCondition,
    config: EEGRunConfig,
    *,
    output_dir: Path,
    environment: Environment,
    resume: bool = False,
    progress_bar: tqdm[Any] | None = None,
) -> tuple[list[EEGRunResult], list[EEGSummary]]:
    """Run one EEG condition and write artifacts."""

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

    runs: list[EEGRunResult] = []
    for model_family in condition.model_families:
        for seed in config.seeds:
            latest = train_eeg_classifier(
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

    summaries = summarize_eeg_runs(
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


def train_eeg_classifier(
    condition: EEGCondition,
    *,
    config: EEGRunConfig,
    seed: int,
    model_family: ModelFamily,
) -> EEGRunResult:
    """Train one model family on one deterministic analytic-signal split."""

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
        sample_length=config.sample_length,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
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
        optimizer.zero_grad()
        logits = model(train_inputs[indices])
        train_loss = F.cross_entropy(logits, train_labels[indices])
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

    return EEGRunResult(
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


def summarize_eeg_runs(
    runs: Sequence[EEGRunResult],
    *,
    model_order: Sequence[ModelFamily],
    class_names: Sequence[str],
    bootstrap_samples: int,
    confidence: float,
) -> list[EEGSummary]:
    """Aggregate per-seed EEG runs."""

    summaries: list[EEGSummary] = []
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
            EEGSummary(
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
    condition: EEGCondition,
    config: EEGRunConfig,
    runs: Sequence[EEGRunResult],
    summaries: Sequence[EEGSummary],
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
        run_id=f"eeg-analytic-signal-{condition.condition_id}",
        config={"condition": condition.to_dict(), "run_config": config.to_dict()},
        seeds=list(config.seeds),
        metrics=summary_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "eeg_analytic_signal_pilot",
            "version": "0.1.0",
            "description": (
                "synthetic complex analytic EEG signals for phase locking, "
                "amplitude events, and phase-amplitude coupling"
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
    config: EEGRunConfig,
    condition_results: Sequence[tuple[EEGCondition, Sequence[EEGSummary]]],
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
        run_id="eeg-analytic-signal-pilot-suite",
        config=config.to_dict(),
        seeds=list(config.seeds),
        metrics=index_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "eeg_analytic_signal_pilot",
            "version": "0.1.0",
            "description": "sequential EEG analytic-signal representation pilot",
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
    condition: EEGCondition,
    *,
    summaries: Sequence[EEGSummary],
) -> dict[str, str]:
    """Write plots for one EEG condition."""

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
    return artifacts


def write_suite_plots(
    output_dir: Path,
    *,
    condition_results: Sequence[tuple[EEGCondition, Sequence[EEGSummary]]],
) -> dict[str, str]:
    """Write cross-condition EEG pilot plots."""

    if not condition_results:
        return {}
    plt = _load_pyplot()
    artifacts: dict[str, str] = {}
    families = [
        family
        for family in EEG_MODEL_FAMILIES
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
        figsize=(max(7.0, 1.0 * len(families)), max(4.2, 0.65 * len(matrix) + 1.8))
    )
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_yticks(range(len(condition_results)))
    ax.set_yticklabels([condition.condition_id for condition, _ in condition_results])
    ax.set_title("EEG analytic-signal pilot: test accuracy")
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
    ax.set_title("Best model by EEG condition")
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
    condition: EEGCondition,
    *,
    config: EEGRunConfig,
    summaries: Sequence[EEGSummary],
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
            f"Examples/class: `{config.examples_per_class}`. Channels: "
            f"`{config.n_channels}`. Time steps: `{config.time_steps}`. "
            f"Train steps: `{config.train_steps}`."
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
    lines = ["# EEG Analytic-Signal Pilot", ""]
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
    parser.add_argument("--tests", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=None)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--observation-noise-std", type=float, default=0.03)
    parser.add_argument("--hidden-features", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--architecture", choices=["mlp", "conv"], default="mlp")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--activation", default="crelu")
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
        default=Path("results/neuro_eeg_analytic_signal_pilot"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)
    conditions = _select_conditions(args.tests, build_eeg_conditions())
    environment = collect_environment(device=config.device, dtype=config.dtype)
    condition_results: list[tuple[EEGCondition, Sequence[EEGSummary]]] = []
    progress_bar = _progress_bar(
        _unfinished_run_count(
            conditions,
            config=config,
            output_dir=args.output_dir,
            resume=args.resume,
        ),
        enabled=config.progress,
        desc="eeg pilot",
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


def _config_from_args(args: argparse.Namespace) -> EEGRunConfig:
    preset = cast(PresetName, args.preset)
    defaults = _preset_defaults(preset)
    return EEGRunConfig(
        preset=preset,
        progress=not args.no_progress,
        seeds=tuple(args.seeds)
        if args.seeds is not None
        else cast(tuple[int, ...], defaults["seeds"]),
        examples_per_class=args.examples_per_class
        if args.examples_per_class is not None
        else cast(int, defaults["examples_per_class"]),
        n_channels=args.n_channels,
        time_steps=args.time_steps
        if args.time_steps is not None
        else cast(int, defaults["time_steps"]),
        train_fraction=args.train_fraction,
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
            "time_steps": 32,
            "hidden_features": 8,
            "train_steps": 25,
            "batch_size": 48,
            "bootstrap_samples": 100,
        }
    if preset == "standard":
        return {
            "seeds": (0, 1, 2),
            "examples_per_class": 128,
            "time_steps": 64,
            "hidden_features": 24,
            "train_steps": 120,
            "batch_size": 128,
            "bootstrap_samples": 1000,
        }
    if preset == "full":
        return {
            "seeds": (0, 1, 2, 3, 4),
            "examples_per_class": 512,
            "time_steps": 128,
            "hidden_features": 48,
            "train_steps": 350,
            "batch_size": 256,
            "bootstrap_samples": 2000,
        }
    msg = f"unsupported preset: {preset}"
    raise ValueError(msg)


def _make_dataset_for_condition(
    condition: EEGCondition,
    *,
    config: EEGRunConfig,
    seed: int,
    dtype: torch.dtype,
) -> EEGAnalyticSignalData:
    if condition.task == "phase_locking":
        return make_phase_locking_dataset(
            seed=seed,
            examples_per_class=config.examples_per_class,
            n_channels=config.n_channels,
            time_steps=config.time_steps,
            train_fraction=config.train_fraction,
            observation_noise_std=config.observation_noise_std,
            dtype=dtype,
        )
    if condition.task == "amplitude_event":
        return make_amplitude_event_dataset(
            seed=seed,
            examples_per_class=config.examples_per_class,
            n_channels=config.n_channels,
            time_steps=config.time_steps,
            train_fraction=config.train_fraction,
            observation_noise_std=config.observation_noise_std,
            dtype=dtype,
        )
    if condition.task == "phase_amplitude_coupling":
        return make_pac_dataset(
            seed=seed,
            examples_per_class=config.examples_per_class,
            n_channels=config.n_channels,
            time_steps=config.time_steps,
            train_fraction=config.train_fraction,
            observation_noise_std=config.observation_noise_std,
            dtype=dtype,
        )
    msg = f"unsupported task: {condition.task}"
    raise ValueError(msg)


def _complex_from_amp_phase(
    amplitude: Tensor,
    phase: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    return torch.complex(
        amplitude * torch.cos(phase),
        amplitude * torch.sin(phase),
    ).to(dtype)


def _flatten_channels(signal: Tensor) -> Tensor:
    return signal.flatten(start_dim=1)


def _normalized_time(time_steps: int, dtype: torch.dtype) -> Tensor:
    return torch.linspace(0.0, 1.0, time_steps, dtype=dtype)


def _base_oscillation(
    time: Tensor,
    *,
    examples_per_class: int,
    generator: torch.Generator,
    low: float,
    high: float,
) -> Tensor:
    real_dtype = time.dtype
    frequency = _uniform(generator, (examples_per_class, 1), low, high, real_dtype)
    phase0 = _uniform(generator, (examples_per_class, 1), -math.pi, math.pi, real_dtype)
    return 2.0 * math.pi * frequency * time.unsqueeze(0) + phase0


def _independent_phases(
    generator: torch.Generator,
    *,
    examples_per_class: int,
    n_channels: int,
    time: Tensor,
) -> Tensor:
    phases = []
    for _ in range(n_channels):
        phases.append(
            _base_oscillation(
                time,
                examples_per_class=examples_per_class,
                generator=generator,
                low=3.0,
                high=8.0,
            )
        )
    return torch.stack(phases, dim=1)


def _random_envelope(
    generator: torch.Generator,
    *,
    examples_per_class: int,
    n_channels: int,
    time_steps: int,
    dtype: torch.dtype,
) -> Tensor:
    baseline = _uniform(
        generator,
        (examples_per_class, n_channels, 1),
        0.75,
        1.15,
        dtype,
    )
    slope = _uniform(
        generator,
        (examples_per_class, n_channels, 1),
        -0.08,
        0.08,
        dtype,
    )
    time = _normalized_time(time_steps, dtype).view(1, 1, time_steps)
    ripple_phase = _uniform(
        generator,
        (examples_per_class, n_channels, 1),
        -math.pi,
        math.pi,
        dtype,
    )
    ripple = 0.08 * torch.sin(2.0 * math.pi * time + ripple_phase)
    envelope = baseline + slope * (time - 0.5) + ripple
    return envelope.clamp_min(0.05)


def _add_complex_noise(
    inputs: Tensor,
    *,
    std: float,
    generator: torch.Generator,
) -> Tensor:
    if std == 0.0:
        return inputs
    real_dtype = inputs.real.dtype
    noise_real = _normal(generator, tuple(inputs.shape), real_dtype)
    noise_imag = _normal(generator, tuple(inputs.shape), real_dtype)
    return inputs + torch.complex(noise_real, noise_imag).to(inputs.dtype) * std


def _balanced_split(
    inputs: Sequence[Tensor],
    labels: Sequence[Tensor],
    class_names: Sequence[str],
    *,
    n_channels: int,
    time_steps: int,
    train_fraction: float,
    generator: torch.Generator,
) -> EEGAnalyticSignalData:
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
    return EEGAnalyticSignalData(
        train_inputs=torch.cat(train_inputs, dim=0),
        train_labels=torch.cat(train_labels, dim=0),
        test_inputs=torch.cat(test_inputs, dim=0),
        test_labels=torch.cat(test_labels, dim=0),
        class_names=tuple(class_names),
        n_channels=n_channels,
        time_steps=time_steps,
    )


def _validate_dataset_args(
    *,
    examples_per_class: int,
    n_channels: int,
    time_steps: int,
    train_fraction: float,
    dtype: torch.dtype,
) -> None:
    if examples_per_class < 4:
        msg = "examples_per_class must be at least 4"
        raise ValueError(msg)
    if n_channels < 2:
        msg = "n_channels must be at least 2"
        raise ValueError(msg)
    if time_steps < 16:
        msg = "time_steps must be at least 16"
        raise ValueError(msg)
    if not 0.0 < train_fraction < 1.0:
        msg = "train_fraction must be in (0, 1)"
        raise ValueError(msg)
    if dtype not in {torch.complex64, torch.complex128}:
        msg = "dtype must be a complex torch dtype"
        raise TypeError(msg)


def _class_names_for_task(task: TaskName) -> tuple[str, ...]:
    if task == "phase_locking":
        return PHASE_LOCKING_LABELS
    if task == "amplitude_event":
        return AMPLITUDE_EVENT_LABELS
    if task == "phase_amplitude_coupling":
        return PAC_LABELS
    msg = f"unsupported task: {task}"
    raise ValueError(msg)


def _uniform(
    generator: torch.Generator,
    shape: tuple[int, ...],
    low: float,
    high: float,
    dtype: torch.dtype,
) -> Tensor:
    return torch.empty(shape, dtype=dtype).uniform_(low, high, generator=generator)


def _normal(
    generator: torch.Generator,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    return torch.randn(shape, generator=generator, dtype=dtype)


def _select_conditions(
    requested: Sequence[str],
    available: Sequence[EEGCondition],
) -> tuple[EEGCondition, ...]:
    by_id = {condition.condition_id: condition for condition in available}
    if requested == ["all"]:
        return tuple(available)
    selected: list[EEGCondition] = []
    for condition_id in requested:
        if condition_id not in by_id:
            msg = f"unknown condition id {condition_id!r}; available: {sorted(by_id)}"
            raise ValueError(msg)
        selected.append(by_id[condition_id])
    return tuple(selected)


def _unfinished_run_count(
    conditions: Sequence[EEGCondition],
    *,
    config: EEGRunConfig,
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


def _best_summary(summaries: Sequence[EEGSummary]) -> EEGSummary:
    return max(summaries, key=lambda summary: summary.test_accuracy_mean)


def _accuracy_for(
    model_family: ModelFamily,
    summaries: Sequence[EEGSummary],
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


def _run_result_from_dict(payload: dict[str, Any]) -> EEGRunResult:
    return EEGRunResult(
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


def _summary_from_dict(payload: dict[str, Any]) -> EEGSummary:
    return EEGSummary(
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
