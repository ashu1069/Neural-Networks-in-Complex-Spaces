"""16-trial random search over RadioML 2018.01A hyperparameters.

Mirror of `experiments/rf/sweep_synthetic_modulation.py`, but loads the
real DeepSig dataset via `experiments.rf.radioml.load_radioml_2018_01a`
instead of generating synthetic IQ symbols. The four-family scaffolding
and the matched-shared-trial selection rule are unchanged.

## Cost note

The default modulation/SNR subset is small (BPSK / QPSK / 8PSK at 7 SNR
levels, 256 examples per bucket, sample_length 128). The full RadioML
24-class / 26-SNR / 1024-sample setup is much larger and a sweep at the
upper end of the search space takes hours even on an A100. Start small.

For GPU:

    uv run python experiments/rf/sweep_radioml.py --device cuda \\
        --data-path data/GOLD_XYZ_OSC.0001_1024.hdf5
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F

from cvnn.baselines import count_real_parameters
from cvnn.repro import collect_environment, new_manifest
from experiments._sweep import (
    FamilySelection,
    JsonObject,
    SearchSpace,
    TrialResult,
    TrialSeedOutcome,
    random_search,
    select_best_per_family,
    select_reference_trial_for_all_families,
    write_tuning_log,
)
from experiments.rf.radioml import load_radioml_2018_01a
from experiments.rf.synthetic_modulation import (
    DEFAULT_MODEL_FAMILIES,
    ArchitectureName,
    ModelFamily,
    RFModulationData,
    _features_for_family,
    _make_model,
)

DEFAULT_MODULATIONS_RADIOML: tuple[str, ...] = ("BPSK", "QPSK", "8PSK")
DEFAULT_SNR_DB_RADIOML: tuple[int, ...] = (-10, -5, 0, 5, 10, 15, 20)


def _train_one(
    family: str,
    hp: dict[str, Any],
    seed: int,
    *,
    data_path: Path,
    classes_path: Path | None,
    data_cache: dict[tuple[Any, ...], RFModulationData] | None,
    architecture: ArchitectureName,
    kernel_size: int,
    modulations: Sequence[str],
    snr_db_levels: Sequence[int],
    max_per_class_per_snr: int | None,
    sample_length: int,
    val_fraction: float,
    activation: str,
    real_activation: str,
    device: torch.device,
    dtype: torch.dtype,
) -> TrialSeedOutcome:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = _load_data_for_run(
        data_path=data_path,
        classes_path=classes_path,
        data_cache=data_cache,
        modulations=modulations,
        snr_db_levels=snr_db_levels,
        max_per_class_per_snr=max_per_class_per_snr,
        sample_length=sample_length,
        seed=seed,
        dtype=dtype,
    )
    n_train_total = data.train_inputs.shape[0]
    n_val = int(round(n_train_total * val_fraction))
    if n_val < 1 or n_val >= n_train_total:
        msg = "val_fraction yields zero or full validation set"
        raise ValueError(msg)
    train_actual_inputs = data.train_inputs[:-n_val].to(device)
    train_actual_labels = data.train_labels[:-n_val].to(device)
    val_inputs_complex = data.train_inputs[-n_val:].to(device)
    val_labels = data.train_labels[-n_val:].to(device)
    test_inputs_complex = data.test_inputs.to(device)
    test_labels = data.test_labels.to(device)
    test_snr_db = data.test_snr_db.to(device)
    n_classes = len(data.modulation_names)

    model_family = cast(ModelFamily, family)
    model, _, _ = _make_model(
        model_family,
        architecture=architecture,
        sample_length=sample_length,
        complex_hidden_features=hp["hidden_features"],
        n_classes=n_classes,
        activation=activation,  # type: ignore[arg-type]
        real_activation=real_activation,  # type: ignore[arg-type]
        kernel_size=kernel_size,
        device=device,
        dtype=dtype,
    )

    train_inputs = _features_for_family(train_actual_inputs, model_family, architecture)
    val_inputs = _features_for_family(val_inputs_complex, model_family, architecture)
    test_inputs = _features_for_family(test_inputs_complex, model_family, architecture)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=0.0
    )
    n_train = train_inputs.shape[0]
    batch_size = min(int(hp["batch_size"]), n_train)
    batch_gen = torch.Generator(device="cpu").manual_seed(seed + 1_000_000)
    start = time.perf_counter()
    model.train()
    for _ in range(int(hp["steps"])):
        indices = torch.randint(0, n_train, (batch_size,), generator=batch_gen)
        optimizer.zero_grad()
        logits = model(train_inputs[indices])
        loss = F.cross_entropy(logits, train_actual_labels[indices])
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        val_predictions = model(val_inputs).argmax(dim=-1)
        test_predictions = model(test_inputs).argmax(dim=-1)
        val_acc = float((val_predictions == val_labels).float().mean().item())
        test_acc = float((test_predictions == test_labels).float().mean().item())
        test_accuracy_by_snr_db = _accuracy_by_snr_db(
            predictions=test_predictions,
            labels=test_labels,
            snr_db=test_snr_db,
            snr_db_levels=data.snr_db_levels,
        )
    parameter_count = count_real_parameters(model)
    return TrialSeedOutcome(
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        train_seconds=train_seconds,
        extra={
            "parameter_count": parameter_count,
            "test_accuracy_by_snr_db": test_accuracy_by_snr_db,
        },
    )


def _load_data_for_run(
    *,
    data_path: Path,
    classes_path: Path | None,
    data_cache: dict[tuple[Any, ...], RFModulationData] | None,
    modulations: Sequence[str],
    snr_db_levels: Sequence[int],
    max_per_class_per_snr: int | None,
    sample_length: int,
    seed: int,
    dtype: torch.dtype,
) -> RFModulationData:
    cache_key = (
        str(data_path),
        str(classes_path) if classes_path else None,
        tuple(modulations),
        tuple(int(snr) for snr in snr_db_levels),
        max_per_class_per_snr,
        sample_length,
        seed,
        str(dtype),
    )
    if data_cache is not None and cache_key in data_cache:
        return data_cache[cache_key]

    data = load_radioml_2018_01a(
        data_path,
        modulations=modulations,
        snr_db_levels=snr_db_levels,
        max_per_class_per_snr=max_per_class_per_snr,
        sample_length=sample_length,
        train_fraction=0.8,
        seed=seed,
        dtype=dtype,
        classes_path=classes_path,
    )
    if data_cache is not None:
        data_cache[cache_key] = data
    return data


def _accuracy_by_snr_db(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    snr_db: torch.Tensor,
    snr_db_levels: Sequence[int],
) -> dict[str, float]:
    accuracy_by_snr: dict[str, float] = {}
    for snr in snr_db_levels:
        mask = snr_db == int(snr)
        if bool(mask.any().item()):
            accuracy_by_snr[str(int(snr))] = float(
                (predictions[mask] == labels[mask]).float().mean().item()
            )
    return accuracy_by_snr


def _selection_table(selections: Sequence[FamilySelection]) -> list[str]:
    lines = [
        (
            "| family | trial | val acc | test acc (mean) | test std | params | "
            "hyperparameters |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for sel in selections:
        hp_str = ", ".join(
            f"{k}={_format_value(v)}"
            for k, v in sorted(sel.selected_hyperparameters.items())
        )
        params = sel.selected_extra.get("parameter_count_mean", "-")
        params_str = (
            f"{int(params)}" if isinstance(params, int | float) else str(params)
        )
        lines.append(
            " | ".join(
                [
                    f"| `{sel.family}`",
                    str(sel.selected_trial_index),
                    f"{sel.selected_val_accuracy_mean:.4f}",
                    f"{sel.selected_test_accuracy_mean:.4f}",
                    f"{sel.selected_test_accuracy_std:.4f}",
                    params_str,
                    f"{hp_str} |",
                ]
            )
        )
    return lines


def _summary_markdown(
    matched_selections: Sequence[FamilySelection],
    *,
    independent_selections: Sequence[FamilySelection],
    sweep_config: JsonObject,
) -> str:
    lines = [
        "# RadioML 2018.01A Modulation Classification (Swept)",
        "",
        (
            "Random-search sweep of "
            f"`{sweep_config['n_trials']}` trials x "
            f"`{len(cast(list[int], sweep_config['seeds']))}` seeds, following "
            "`docs/tuning_budget.md`. Real-data benchmark on the DeepSig "
            "RadioML 2018.01A archive (see `docs/radioml.md` for acquisition)."
        ),
        "",
        (
            f"Architecture: `{sweep_config['architecture']}`. "
            f"Activation (complex): `{sweep_config['activation']}`. "
            f"Activation (real baselines): `{sweep_config['real_activation']}`. "
            f"Modulations: `{sweep_config['modulations']}`. "
            f"SNR (dB): `{sweep_config['snr_db_levels']}`. "
            f"Sample length: `{sweep_config['sample_length']}`. "
            f"Cap per class per SNR: `{sweep_config['max_per_class_per_snr']}`."
        ),
        "",
        "## Matched shared-trial comparison",
        "",
        (
            "Primary paper table. The trial index is selected by the complex "
            "family's mean validation accuracy, then every real baseline is "
            "reported at that same trial index so parameter/FLOP matching is "
            "with respect to the selected complex model."
        ),
        "",
    ]
    lines.extend(_selection_table(matched_selections))
    per_snr_lines = _per_snr_table(matched_selections)
    if per_snr_lines:
        lines.extend(["", "## Matched per-SNR test accuracy", ""])
        lines.extend(per_snr_lines)
    lines.extend(
        [
            "",
            "## Independent family winners",
            "",
            (
                "Diagnostic only. These rows show each family's own best "
                "validation trial, so their parameter counts are not "
                "guaranteed to be matched to the selected complex model."
            ),
            "",
        ]
    )
    lines.extend(_selection_table(independent_selections))
    return "\n".join(lines)


def _per_snr_table(selections: Sequence[FamilySelection]) -> list[str]:
    per_family: dict[str, dict[str, float]] = {}
    for selection in selections:
        accuracies = _mean_accuracy_by_snr(selection)
        if accuracies:
            per_family[selection.family] = accuracies
    if not per_family:
        return []
    snr_keys = sorted(
        {snr for accuracies in per_family.values() for snr in accuracies},
        key=int,
    )
    lines = [
        "| family | " + " | ".join(f"{snr} dB" for snr in snr_keys) + " |",
        "| --- | " + " | ".join("---:" for _ in snr_keys) + " |",
    ]
    for family, accuracies in per_family.items():
        cells = [
            f"{accuracies[snr]:.3f}" if snr in accuracies else "-" for snr in snr_keys
        ]
        lines.append(f"| `{family}` | " + " | ".join(cells) + " |")
    return lines


def _mean_accuracy_by_snr(selection: FamilySelection) -> dict[str, float]:
    per_seed = selection.selected_extra.get("test_accuracy_by_snr_db_per_seed")
    if not isinstance(per_seed, list):
        return {}
    by_snr: dict[str, list[float]] = {}
    for item in per_seed:
        if not isinstance(item, dict):
            continue
        for snr, value in item.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                by_snr.setdefault(str(snr), []).append(float(value))
    return {snr: sum(values) / len(values) for snr, values in by_snr.items() if values}


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/GOLD_XYZ_OSC.0001_1024.hdf5"),
    )
    parser.add_argument(
        "--classes-path",
        type=Path,
        default=None,
        help=(
            "optional fixed class-order sidecar; defaults to classes-fixed.json "
            "or classes-fixed.txt next to the HDF5"
        ),
    )
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--model-families",
        nargs="+",
        choices=list(DEFAULT_MODEL_FAMILIES),
        default=list(DEFAULT_MODEL_FAMILIES),
    )
    parser.add_argument(
        "--modulations", nargs="+", default=list(DEFAULT_MODULATIONS_RADIOML)
    )
    parser.add_argument(
        "--snr-db-levels",
        nargs="+",
        type=int,
        default=list(DEFAULT_SNR_DB_RADIOML),
    )
    parser.add_argument(
        "--max-per-class-per-snr",
        type=int,
        default=256,
        help="cap per (modulation, SNR) bucket; pass 0 to disable",
    )
    parser.add_argument(
        "--no-cache-data",
        action="store_true",
        help=(
            "reload the HDF5 subset for every run; by default capped subsets "
            "are cached per seed/filter split"
        ),
    )
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--architecture", choices=["mlp", "conv"], default="conv")
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--activation", default="crelu")
    parser.add_argument("--real-activation", default="relu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=["complex64", "complex128"], default="complex64"
    )
    parser.add_argument("--sweep-seed", type=int, default=20260503)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/radioml_modulation_sweep"),
    )
    args = parser.parse_args()

    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    device = torch.device(args.device)
    architecture = cast(ArchitectureName, args.architecture)
    max_per_class = (
        None if args.max_per_class_per_snr <= 0 else args.max_per_class_per_snr
    )
    cache_data = (not args.no_cache_data) and max_per_class is not None
    data_cache: dict[tuple[Any, ...], RFModulationData] | None = (
        {} if cache_data else None
    )

    if architecture == "conv":
        hidden_choices = [16, 32, 64]
    else:
        hidden_choices = [32, 64, 128]
    space = SearchSpace(
        distributions={
            "learning_rate": ("loguniform", 1e-3, 5e-2),
            "hidden_features": ("choice", hidden_choices),
            "steps": ("choice", [200, 400, 800]),
            "batch_size": ("choice", [128, 256, 512]),
        }
    )

    def train_fn(family: str, hp: dict[str, Any], seed: int) -> TrialSeedOutcome:
        return _train_one(
            family,
            hp,
            seed,
            data_path=args.data_path,
            classes_path=args.classes_path,
            data_cache=data_cache,
            architecture=architecture,
            kernel_size=args.kernel_size,
            modulations=args.modulations,
            snr_db_levels=args.snr_db_levels,
            max_per_class_per_snr=max_per_class,
            sample_length=args.sample_length,
            val_fraction=args.val_fraction,
            activation=args.activation,
            real_activation=args.real_activation,
            device=device,
            dtype=dtype,
        )

    environment = collect_environment(device=args.device, dtype=args.dtype)
    trials = random_search(
        families=list(args.model_families),
        search_space=space,
        seeds=list(args.seeds),
        n_trials=args.n_trials,
        sweep_seed=args.sweep_seed,
        train_fn=train_fn,
    )
    independent_selections = select_best_per_family(trials)
    matched_selections = select_reference_trial_for_all_families(trials)

    sweep_config: JsonObject = {
        "experiment": "radioml_modulation_sweep",
        "data_path": str(args.data_path),
        "classes_path": str(args.classes_path) if args.classes_path else None,
        "n_trials": args.n_trials,
        "seeds": list(args.seeds),
        "model_families": list(args.model_families),
        "modulations": list(args.modulations),
        "snr_db_levels": list(args.snr_db_levels),
        "max_per_class_per_snr": args.max_per_class_per_snr,
        "cache_data": cache_data,
        "sample_length": args.sample_length,
        "val_fraction": args.val_fraction,
        "architecture": args.architecture,
        "kernel_size": args.kernel_size,
        "activation": args.activation,
        "real_activation": args.real_activation,
        "device": args.device,
        "dtype": args.dtype,
        "sweep_seed": args.sweep_seed,
        "search_space": {
            name: list(spec) for name, spec in space.distributions.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tuning_log(
        args.output_dir,
        task_name="RadioML 2018.01A Modulation Classification",
        sweep_config=sweep_config,
        trials=trials,
        selections=matched_selections,
    )
    summary_md = _summary_markdown(
        matched_selections,
        independent_selections=independent_selections,
        sweep_config=sweep_config,
    )
    (args.output_dir / "summary.md").write_text(summary_md + "\n")

    summary_json: JsonObject = {
        "config": sweep_config,
        "matched_selections": [sel.to_dict() for sel in matched_selections],
        "independent_selections": [sel.to_dict() for sel in independent_selections],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
    )

    manifest = new_manifest(
        run_id="radioml-modulation-sweep",
        config=sweep_config,
        seeds=list(args.seeds),
        metrics={
            "matched_selections": [sel.to_dict() for sel in matched_selections],
            "independent_selections": [sel.to_dict() for sel in independent_selections],
        },
        device=args.device,
        dtype=args.dtype,
        dataset={
            "name": "radioml_2018_01a",
            "version": "2018.01A",
            "description": "DeepSig RadioML 2018.01A modulation classification",
        },
        artifacts={
            "trials_json": str(args.output_dir / "trials.json"),
            "tuning_log_markdown": str(args.output_dir / "tuning_log.md"),
            "summary_markdown": str(args.output_dir / "summary.md"),
            "summary_json": str(args.output_dir / "summary.json"),
        },
        environment=environment,
    )
    manifest.write_json(args.output_dir / "manifest.json")
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _ensure_unused() -> tuple[type[TrialResult], type[FamilySelection]]:
    return TrialResult, FamilySelection
