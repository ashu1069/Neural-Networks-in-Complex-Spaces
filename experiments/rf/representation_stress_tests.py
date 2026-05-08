"""Sequential RF synthetic stress tests for representation claims.

This runner turns the "could this contradict us?" notes into a reproducible
suite. Each condition writes its own raw runs and summary table, while the
root output directory gets an index across conditions.

Examples:

    uv run python experiments/rf/representation_stress_tests.py --preset smoke

    uv run python experiments/rf/representation_stress_tests.py \\
        --preset standard --device cuda --resume
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

from cvnn.baselines import count_real_parameters
from cvnn.repro import Environment, JsonObject, collect_environment, new_manifest
from experiments.rf.synthetic_modulation import (
    ActivationName,
    ArchitectureName,
    ModelFamily,
    ModulationName,
    RealActivationName,
    RFRunResult,
    RFSummary,
    _features_for_family,
    _make_model,
    make_synthetic_rf_modulation_dataset,
    summarize_rf_runs,
)

PresetName = Literal["smoke", "standard", "full"]
TransformName = Literal[
    "none",
    "fixed_rotation",
    "random_rotation",
    "unit_magnitude",
    "unit_power",
]

REPRESENTATION_FAMILIES: tuple[ModelFamily, ...] = (
    "complex",
    "real_stacked",
    "real_matched_params",
    "real_matched_flops",
    "real_polar",
    "real_phase",
    "real_magnitude",
)
REPRESENTATION_ONLY_FAMILIES: tuple[ModelFamily, ...] = (
    "complex",
    "real_stacked",
    "real_polar",
    "real_phase",
    "real_magnitude",
)
ACTIVATION_SWEEP_FAMILIES: tuple[ModelFamily, ...] = ("complex", "real_stacked")
PLOTTED_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "complex",
    "real_stacked",
    "real_matched_params",
    "real_matched_flops",
    "real_polar",
    "real_phase",
    "real_magnitude",
)
MODEL_COLORS: dict[str, str] = {
    "complex": "#355c9c",
    "real_stacked": "#4c956c",
    "real_matched_params": "#7b2cbf",
    "real_matched_flops": "#f77f00",
    "real_polar": "#2a9d8f",
    "real_phase": "#e76f51",
    "real_magnitude": "#6c757d",
}


@dataclass(frozen=True)
class InputTransform:
    """Serializable transform applied before feature extraction."""

    name: TransformName = "none"
    rotation_radians: float = 0.0

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "rotation_radians": self.rotation_radians,
        }


@dataclass(frozen=True)
class StressCondition:
    """One contradiction test in the sequential suite."""

    condition_id: str
    question: str
    contradiction_signal: str
    modulations: tuple[ModulationName, ...]
    snr_db_levels: tuple[int, ...]
    model_families: tuple[ModelFamily, ...]
    activation: ActivationName = "zrelu"
    real_activation: RealActivationName = "relu"
    train_transform: InputTransform = InputTransform()
    test_transform: InputTransform = InputTransform()

    def to_dict(self) -> JsonObject:
        return {
            "condition_id": self.condition_id,
            "question": self.question,
            "contradiction_signal": self.contradiction_signal,
            "modulations": list(self.modulations),
            "snr_db_levels": list(self.snr_db_levels),
            "model_families": list(self.model_families),
            "activation": self.activation,
            "real_activation": self.real_activation,
            "train_transform": self.train_transform.to_dict(),
            "test_transform": self.test_transform.to_dict(),
        }


@dataclass(frozen=True)
class StressRunConfig:
    """Shared training knobs for every condition."""

    preset: PresetName
    progress: bool
    seeds: tuple[int, ...]
    n_per_class_per_snr: int
    sample_length: int
    train_fraction: float
    hidden_features: int
    steps: int
    batch_size: int
    learning_rate: float
    architecture: ArchitectureName
    kernel_size: int
    device: str
    dtype: str
    bootstrap_samples: int
    confidence: float

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


def build_stress_conditions() -> tuple[StressCondition, ...]:
    """Return the full ordered suite of RF representation stress tests."""

    psk: tuple[ModulationName, ...] = ("bpsk", "qpsk", "8psk")
    qam: tuple[ModulationName, ...] = ("qam16", "qam64")
    mixed: tuple[ModulationName, ...] = (
        "bpsk",
        "qpsk",
        "8psk",
        "qam16",
        "qam64",
    )
    activations: tuple[ActivationName, ...] = (
        "crelu",
        "zrelu",
        "modrelu",
        "cardioid",
        "siglog",
    )
    return (
        StressCondition(
            condition_id="psk_representation",
            question="Do phase-aware encodings explain PSK-family performance?",
            contradiction_signal=(
                "magnitude-only approaches phase/Cartesian accuracy, or real "
                "encodings consistently beat complex."
            ),
            modulations=psk,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_FAMILIES,
        ),
        StressCondition(
            condition_id="qam_representation",
            question="Does amplitude structure change the story on QAM-only data?",
            contradiction_signal=(
                "magnitude-only becomes competitive, or phase-only collapses "
                "relative to Cartesian/polar."
            ),
            modulations=qam,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_FAMILIES,
        ),
        StressCondition(
            condition_id="mixed_representation",
            question="Does the representation conclusion survive PSK+QAM together?",
            contradiction_signal=(
                "a hand-chosen real coordinate system matches or beats the "
                "complex model over mixed modulation families."
            ),
            modulations=mixed,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_FAMILIES,
        ),
        StressCondition(
            condition_id="low_snr_psk",
            question="Does phase become too noisy in the low-SNR regime?",
            contradiction_signal=(
                "Cartesian or magnitude-heavy encodings beat phase/polar at "
                "low SNR, suggesting phase singularity/noise sensitivity."
            ),
            modulations=psk,
            snr_db_levels=(-10, -5, 0),
            model_families=REPRESENTATION_FAMILIES,
        ),
        StressCondition(
            condition_id="high_snr_psk",
            question="Is the complex advantage only a high-SNR effect?",
            contradiction_signal=(
                "complex only separates from real baselines when the phase "
                "estimate is clean."
            ),
            modulations=psk,
            snr_db_levels=(10, 15, 20),
            model_families=REPRESENTATION_FAMILIES,
        ),
        StressCondition(
            condition_id="unit_magnitude_mixed",
            question="What happens if per-symbol amplitude is removed?",
            contradiction_signal=(
                "complex or phase-only still performs on QAM after amplitude "
                "is removed, implying hidden leakage or a too-easy task."
            ),
            modulations=mixed,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_ONLY_FAMILIES,
            train_transform=InputTransform("unit_magnitude"),
            test_transform=InputTransform("unit_magnitude"),
        ),
        StressCondition(
            condition_id="unit_power_mixed",
            question="Does per-example energy normalization change the ranking?",
            contradiction_signal=(
                "rankings change substantially, suggesting models were using "
                "energy/SNR scale rather than modulation geometry."
            ),
            modulations=mixed,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_ONLY_FAMILIES,
            train_transform=InputTransform("unit_power"),
            test_transform=InputTransform("unit_power"),
        ),
        StressCondition(
            condition_id="fixed_rotation_psk",
            question="Are models robust to an unseen global carrier phase offset?",
            contradiction_signal=(
                "all coordinate-dependent models fail under a fixed test "
                "rotation, weakening claims about native phase handling."
            ),
            modulations=psk,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_ONLY_FAMILIES,
            test_transform=InputTransform("fixed_rotation", math.pi / 4.0),
        ),
        StressCondition(
            condition_id="rotation_augmented_psk",
            question=(
                "Can real baselines recover rotation robustness with augmentation?"
            ),
            contradiction_signal=(
                "random train rotations close the gap to complex under fixed "
                "rotated test data, making augmentation the key ingredient."
            ),
            modulations=psk,
            snr_db_levels=(0, 10, 20),
            model_families=REPRESENTATION_ONLY_FAMILIES,
            train_transform=InputTransform("random_rotation"),
            test_transform=InputTransform("fixed_rotation", math.pi / 4.0),
        ),
        *tuple(
            StressCondition(
                condition_id=f"activation_{activation}",
                question=f"How much does complex activation `{activation}` matter?",
                contradiction_signal=(
                    "the complex result changes enough across activations that "
                    "the broad 'complex NN' claim is underspecified."
                ),
                modulations=psk,
                snr_db_levels=(0, 10, 20),
                model_families=ACTIVATION_SWEEP_FAMILIES,
                activation=activation,
            )
            for activation in activations
        ),
    )


def train_rf_stress_condition(
    *,
    condition: StressCondition,
    config: StressRunConfig,
    seed: int,
    model_family: ModelFamily,
) -> RFRunResult:
    """Train one seed/model under a condition-specific input transform."""

    dtype = _parse_complex_dtype(config.dtype)
    device = torch.device(config.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = make_synthetic_rf_modulation_dataset(
        seed=seed,
        modulations=condition.modulations,
        snr_db_levels=condition.snr_db_levels,
        n_per_class_per_snr=config.n_per_class_per_snr,
        sample_length=config.sample_length,
        train_fraction=config.train_fraction,
        dtype=dtype,
    )
    train_complex = apply_input_transform(
        data.train_inputs,
        condition.train_transform,
        seed=seed + 10_000,
    ).to(device)
    test_complex = apply_input_transform(
        data.test_inputs,
        condition.test_transform,
        seed=seed + 20_000,
    ).to(device)
    train_inputs = _features_for_family(
        train_complex,
        model_family,
        config.architecture,
    )
    test_inputs = _features_for_family(
        test_complex,
        model_family,
        config.architecture,
    )
    train_labels = data.train_labels.to(device)
    test_labels = data.test_labels.to(device)
    test_snr_db = data.test_snr_db

    model, effective_hidden, estimated_madds = _make_model(
        model_family,
        architecture=config.architecture,
        sample_length=config.sample_length,
        complex_hidden_features=config.hidden_features,
        n_classes=len(data.modulation_names),
        activation=condition.activation,
        real_activation=condition.real_activation,
        kernel_size=config.kernel_size,
        device=device,
        dtype=dtype,
    )
    parameter_count = count_real_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    n_train = train_inputs.shape[0]
    batch_size = min(config.batch_size, n_train)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_000)
    start = time.perf_counter()
    train_loss = torch.tensor(float("nan"), device=device)
    inner_bar = _step_progress_bar(
        config.steps,
        enabled=config.progress,
        desc=f"{condition.condition_id}:{model_family}/s{seed}",
    )
    model.train()
    for step in range(config.steps):
        indices = torch.randint(0, n_train, (batch_size,), generator=batch_generator)
        optimizer.zero_grad()
        logits = model(train_inputs[indices])
        train_loss = F.cross_entropy(logits, train_labels[indices])
        train_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if inner_bar is not None:
            if step == config.steps - 1 or step % max(1, config.steps // 50) == 0:
                inner_bar.set_postfix_str(
                    f"loss={float(train_loss.detach().cpu().item()):.4f}",
                    refresh=False,
                )
            inner_bar.update(1)
    if inner_bar is not None:
        inner_bar.close()
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
        steps=config.steps,
        learning_rate=config.learning_rate,
        hidden_features=effective_hidden,
        parameter_count=parameter_count,
        estimated_forward_madds=estimated_madds,
        train_seconds=train_seconds,
        train_loss=float(train_loss.detach().cpu().item()),
        test_loss=float(test_loss.detach().cpu().item()),
        test_accuracy=float(test_accuracy.detach().cpu().item()),
        accuracy_by_snr_db=accuracy_by_snr_db,
    )


def apply_input_transform(
    inputs: torch.Tensor,
    transform: InputTransform,
    *,
    seed: int,
) -> torch.Tensor:
    """Apply a deterministic complex-input transform on CPU or GPU tensors."""

    if transform.name == "none":
        return inputs
    if transform.name == "fixed_rotation":
        return _rotate(inputs, transform.rotation_radians)
    if transform.name == "random_rotation":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        angles = torch.rand(
            (inputs.shape[0], 1),
            generator=generator,
            dtype=inputs.real.dtype,
        )
        angles = angles.to(inputs.device) * (2.0 * math.pi)
        phasor = torch.polar(torch.ones_like(angles), angles).to(inputs.dtype)
        return inputs * phasor
    if transform.name == "unit_magnitude":
        magnitude = inputs.abs()
        denominator = magnitude.clamp_min(torch.finfo(magnitude.dtype).eps)
        return inputs / denominator
    if transform.name == "unit_power":
        power = inputs.abs().square().mean(dim=-1, keepdim=True).sqrt()
        denominator = power.clamp_min(torch.finfo(power.dtype).eps)
        return inputs / denominator
    msg = f"unsupported transform: {transform.name}"
    raise ValueError(msg)


def run_condition(
    condition: StressCondition,
    config: StressRunConfig,
    *,
    output_dir: Path,
    environment: Environment,
    resume: bool,
    progress_bar: Any | None = None,
) -> tuple[list[RFRunResult], list[RFSummary]]:
    """Run or load one condition directory."""

    condition_dir = output_dir / condition.condition_id
    summary_path = condition_dir / "summary.json"
    if resume and summary_path.exists():
        payload = json.loads(summary_path.read_text())
        loaded_runs = [
            _run_result_from_dict(cast(dict[str, Any], item))
            for item in cast(list[object], payload["raw_runs"])
        ]
        loaded_summaries = [
            _summary_from_dict(cast(dict[str, Any], item))
            for item in cast(list[object], payload["summaries"])
        ]
        plot_artifacts = write_condition_plots(
            condition_dir,
            condition=condition,
            summaries=loaded_summaries,
        )
        (condition_dir / "summary.md").write_text(
            format_condition_markdown(
                condition,
                config=config,
                summaries=loaded_summaries,
                plot_artifacts=plot_artifacts,
            )
            + "\n"
        )
        return loaded_runs, loaded_summaries

    if progress_bar is not None:
        progress_bar.set_description_str(condition.condition_id)
    else:
        print(f"[{condition.condition_id}] starting")
    runs: list[RFRunResult] = []
    for model_family in condition.model_families:
        for seed in config.seeds:
            runs.append(
                train_rf_stress_condition(
                    condition=condition,
                    config=config,
                    seed=seed,
                    model_family=model_family,
                )
            )
            latest = runs[-1]
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

    summaries = summarize_rf_runs(
        runs,
        model_order=condition.model_families,
        snr_db_levels=condition.snr_db_levels,
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


def write_condition_outputs(
    output_dir: Path,
    *,
    condition: StressCondition,
    config: StressRunConfig,
    runs: Sequence[RFRunResult],
    summaries: Sequence[RFSummary],
    environment: Environment,
) -> None:
    """Write condition-level artifacts."""

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
    plot_artifacts = write_condition_plots(
        output_dir,
        condition=condition,
        summaries=summaries,
    )
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
        run_id=f"rf-representation-stress-{condition.condition_id}",
        config={
            "condition": condition.to_dict(),
            "run_config": config.to_dict(),
        },
        seeds=list(config.seeds),
        metrics=summary_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "rf_synthetic_modulation",
            "version": "0.1.0",
            "description": (
                "synthetic IQ PSK/QAM symbols + AWGN with optional input transforms"
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
    config: StressRunConfig,
    condition_results: Sequence[tuple[StressCondition, Sequence[RFSummary]]],
    environment: Environment,
) -> None:
    """Write the root suite index."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = [
        {
            "condition_id": condition.condition_id,
            "question": condition.question,
            "contradiction_signal": condition.contradiction_signal,
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
        run_id="rf-representation-stress-suite",
        config=config.to_dict(),
        seeds=list(config.seeds),
        metrics=index_payload,
        device=config.device,
        dtype=config.dtype,
        dataset={
            "name": "rf_synthetic_modulation",
            "version": "0.1.0",
            "description": "sequential synthetic RF contradiction tests",
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
    *,
    condition: StressCondition,
    summaries: Sequence[RFSummary],
) -> dict[str, str]:
    """Write plots for one stress-test condition."""

    if not summaries:
        return {}
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(labels)), 4.6))
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

    snr_keys = sorted(
        {key for summary in summaries for key in summary.accuracy_by_snr_db_mean},
        key=lambda value: int(value),
    )
    if snr_keys:
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        snr_values = [int(key) for key in snr_keys]
        for summary in summaries:
            values = [
                summary.accuracy_by_snr_db_mean.get(key, float("nan"))
                for key in snr_keys
            ]
            ax.plot(
                snr_values,
                values,
                marker="o",
                linewidth=2.0,
                label=summary.model_family,
                color=MODEL_COLORS.get(summary.model_family),
            )
        ax.set_title(f"{condition.condition_id}: accuracy by SNR")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(frameon=False, fontsize=8, ncols=2)
        fig.tight_layout()
        snr_path = output_dir / "accuracy_by_snr.png"
        fig.savefig(snr_path, dpi=180)
        plt.close(fig)
        artifacts["accuracy_by_snr"] = str(snr_path)

    return artifacts


def write_suite_plots(
    output_dir: Path,
    *,
    condition_results: Sequence[tuple[StressCondition, Sequence[RFSummary]]],
) -> dict[str, str]:
    """Write root-level plots across all conditions."""

    if not condition_results:
        return {}
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    condition_labels = [condition.condition_id for condition, _ in condition_results]
    model_labels = [
        model
        for model in PLOTTED_MODEL_FAMILIES
        if any(
            _accuracy_for(model, summaries) is not None
            for _, summaries in condition_results
        )
    ]
    if model_labels:
        import numpy as np

        matrix = np.array(
            [
                [
                    _accuracy_for(model, summaries)
                    if _accuracy_for(model, summaries) is not None
                    else np.nan
                    for model in model_labels
                ]
                for _, summaries in condition_results
            ],
            dtype=float,
        )
        fig, ax = plt.subplots(
            figsize=(
                max(8.5, 1.15 * len(model_labels)),
                max(4.8, 0.42 * len(condition_labels)),
            )
        )
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(model_labels)), model_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(condition_labels)), condition_labels)
        ax.set_title("RF stress tests: accuracy heatmap")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("test accuracy")
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                if not np.isnan(value):
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.65 else "black",
                        fontsize=7,
                    )
        fig.tight_layout()
        heatmap_path = output_dir / "accuracy_heatmap.png"
        fig.savefig(heatmap_path, dpi=180)
        plt.close(fig)
        artifacts["accuracy_heatmap"] = str(heatmap_path)

    best_values = [
        _best_summary(summaries).test_accuracy_mean if summaries else 0.0
        for _, summaries in condition_results
    ]
    best_models = [
        _best_summary(summaries).model_family if summaries else ""
        for _, summaries in condition_results
    ]
    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(condition_labels)), 4.8))
    bar_colors = [MODEL_COLORS.get(model, "#495057") for model in best_models]
    ax.bar(condition_labels, best_values, color=bar_colors, edgecolor="#222222")
    ax.set_title("Best model accuracy by stress condition")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    for index, (value, model) in enumerate(zip(best_values, best_models, strict=True)):
        ax.text(
            index,
            min(0.98, value + 0.025),
            model,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    best_path = output_dir / "best_accuracy_by_condition.png"
    fig.savefig(best_path, dpi=180)
    plt.close(fig)
    artifacts["best_accuracy_by_condition"] = str(best_path)

    return artifacts


def format_condition_markdown(
    condition: StressCondition,
    *,
    config: StressRunConfig,
    summaries: Sequence[RFSummary],
    plot_artifacts: dict[str, str] | None = None,
) -> str:
    """Format a condition-level markdown table."""

    rows = [
        f"# {condition.condition_id}",
        "",
        f"Question: {condition.question}",
        "",
        f"Contradiction signal: {condition.contradiction_signal}",
        "",
        (
            f"Modulations: `{list(condition.modulations)}`. "
            f"SNR (dB): `{list(condition.snr_db_levels)}`. "
            f"Architecture: `{config.architecture}`. "
            f"Activation: `{condition.activation}`. "
            f"Train transform: `{condition.train_transform.name}`. "
            f"Test transform: `{condition.test_transform.name}`."
        ),
        "",
    ]
    if plot_artifacts:
        rows.extend(_plot_markdown_lines(plot_artifacts))
        rows.append("")
    rows.extend(
        [
            (
                "| model | hidden | params | MAdds | accuracy | std | 95% CI | "
                "loss | s/run |"
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
    if summaries and any(summary.accuracy_by_snr_db_mean for summary in summaries):
        snr_keys = sorted(
            {key for summary in summaries for key in summary.accuracy_by_snr_db_mean},
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
            rows.append(
                "| `" + summary.model_family + "` | " + " | ".join(cells) + " |"
            )
    return "\n".join(rows)


def format_index_markdown(
    rows: Sequence[dict[str, object]],
    *,
    plot_artifacts: dict[str, str] | None = None,
) -> str:
    """Format the root suite index."""

    lines = [
        "# RF Synthetic Representation Stress Tests",
        "",
        (
            "Sequential contradiction tests for whether complex-valued RF "
            "models help because of native complex arithmetic, coordinate "
            "choice, phase information, augmentation, or compute budget."
        ),
        "",
    ]
    if plot_artifacts:
        lines.extend(_plot_markdown_lines(plot_artifacts))
        lines.append("")
    lines.extend(
        [
            (
                "| condition | best | acc | complex | real_stack | phase | polar | "
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
            "`summary.md`, and `manifest.json`.",
        ]
    )
    return "\n".join(lines)


def _plot_markdown_lines(plot_artifacts: dict[str, str]) -> list[str]:
    lines = ["## Plots", ""]
    for name, path in plot_artifacts.items():
        title = name.replace("_", " ")
        lines.append(f"![{title}]({Path(path).name})")
        lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["smoke", "standard", "full"],
        default="smoke",
        help="shared training budget; individual knobs can override the preset",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help="condition ids to run, or all",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-per-class-per-snr", type=int, default=None)
    parser.add_argument("--sample-length", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--hidden-features", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--architecture", choices=["mlp", "conv"], default="conv")
    parser.add_argument("--kernel-size", type=int, default=5)
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
        default=Path("results/rf_synthetic_representation_stress_tests"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip condition directories that already have summary.json",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable tqdm progress bars",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _config_from_args(args)
    conditions = _select_conditions(args.tests, build_stress_conditions())
    environment = collect_environment(device=config.device, dtype=config.dtype)
    condition_results: list[tuple[StressCondition, Sequence[RFSummary]]] = []
    progress_bar = _progress_bar(
        _unfinished_run_count(
            conditions,
            config=config,
            output_dir=args.output_dir,
            resume=args.resume,
        ),
        enabled=config.progress,
        desc="rf stress",
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


def _config_from_args(args: argparse.Namespace) -> StressRunConfig:
    preset = cast(PresetName, args.preset)
    defaults = _preset_defaults(preset)
    return StressRunConfig(
        preset=preset,
        progress=not args.no_progress,
        seeds=tuple(args.seeds)
        if args.seeds is not None
        else cast(tuple[int, ...], defaults["seeds"]),
        n_per_class_per_snr=args.n_per_class_per_snr
        if args.n_per_class_per_snr is not None
        else cast(int, defaults["n_per_class_per_snr"]),
        sample_length=args.sample_length
        if args.sample_length is not None
        else cast(int, defaults["sample_length"]),
        train_fraction=args.train_fraction,
        hidden_features=args.hidden_features
        if args.hidden_features is not None
        else cast(int, defaults["hidden_features"]),
        steps=args.steps if args.steps is not None else cast(int, defaults["steps"]),
        batch_size=args.batch_size
        if args.batch_size is not None
        else cast(int, defaults["batch_size"]),
        learning_rate=args.learning_rate,
        architecture=cast(ArchitectureName, args.architecture),
        kernel_size=args.kernel_size,
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
            "n_per_class_per_snr": 24,
            "sample_length": 32,
            "hidden_features": 8,
            "steps": 40,
            "batch_size": 48,
            "bootstrap_samples": 100,
        }
    if preset == "standard":
        return {
            "seeds": (0, 1, 2),
            "n_per_class_per_snr": 128,
            "sample_length": 64,
            "hidden_features": 16,
            "steps": 200,
            "batch_size": 128,
            "bootstrap_samples": 1000,
        }
    if preset == "full":
        return {
            "seeds": (0, 1, 2, 3, 4),
            "n_per_class_per_snr": 512,
            "sample_length": 128,
            "hidden_features": 32,
            "steps": 600,
            "batch_size": 256,
            "bootstrap_samples": 2000,
        }
    msg = f"unsupported preset: {preset}"
    raise ValueError(msg)


def _select_conditions(
    requested: Sequence[str],
    available: Sequence[StressCondition],
) -> tuple[StressCondition, ...]:
    by_id = {condition.condition_id: condition for condition in available}
    if requested == ["all"]:
        return tuple(available)
    selected: list[StressCondition] = []
    for condition_id in requested:
        if condition_id not in by_id:
            msg = f"unknown condition id {condition_id!r}; available: {sorted(by_id)}"
            raise ValueError(msg)
        selected.append(by_id[condition_id])
    return tuple(selected)


def _unfinished_run_count(
    conditions: Sequence[StressCondition],
    *,
    config: StressRunConfig,
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


def _progress_bar(total: int, *, enabled: bool, desc: str) -> Any:
    if not enabled or total <= 0:
        return None
    tqdm = _load_tqdm()
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)


def _step_progress_bar(
    total: int,
    *,
    enabled: bool,
    desc: str,
    min_steps: int = 50,
    mininterval: float = 0.5,
) -> Any:
    if not enabled or total < min_steps:
        return None
    tqdm = _load_tqdm()
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        mininterval=mininterval,
    )


def _load_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]
    except ImportError:
        return None
    return tqdm


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


def _run_result_from_dict(payload: dict[str, Any]) -> RFRunResult:
    return RFRunResult(
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
        accuracy_by_snr_db={
            str(key): float(value)
            for key, value in cast(
                dict[str, Any],
                payload["accuracy_by_snr_db"],
            ).items()
        },
    )


def _summary_from_dict(payload: dict[str, Any]) -> RFSummary:
    return RFSummary(
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
        accuracy_by_snr_db_mean={
            str(key): float(value)
            for key, value in cast(
                dict[str, Any],
                payload["accuracy_by_snr_db_mean"],
            ).items()
        },
    )


def _rotate(inputs: torch.Tensor, radians: float) -> torch.Tensor:
    angle = torch.tensor(radians, dtype=inputs.real.dtype, device=inputs.device)
    phasor = torch.polar(
        torch.ones((), dtype=inputs.real.dtype, device=inputs.device),
        angle,
    )
    return inputs * phasor.to(inputs.dtype)


def _best_summary(summaries: Sequence[RFSummary]) -> RFSummary:
    return max(summaries, key=lambda summary: summary.test_accuracy_mean)


def _accuracy_for(
    model_family: ModelFamily,
    summaries: Sequence[RFSummary],
) -> float | None:
    for summary in summaries:
        if summary.model_family == model_family:
            return summary.test_accuracy_mean
    return None


def _format_optional_float(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "-"


if __name__ == "__main__":
    raise SystemExit(main())
