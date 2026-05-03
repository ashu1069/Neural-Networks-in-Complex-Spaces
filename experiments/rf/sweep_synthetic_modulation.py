"""16-trial random search over RF synthetic modulation hyperparameters.

Honors `docs/tuning_budget.md`: shared search space across all four model
families, 3 seeds per trial, selection by mean validation accuracy. Default
architecture is `conv` (the natural fit for sequence-shaped IQ inputs);
`mlp` remains supported for direct comparison against the snapshot run.

## Cost note

Defaults target a GPU run. On CPU each conv trial seed can take ~2 minutes
at the upper end of the search space, so a full 16x3x4 sweep is multi-hour.
For a tractable CPU smoke run, override:

    uv run python experiments/rf/sweep_synthetic_modulation.py \\
        --sample-length 64 --n-per-class-per-snr 128

For GPU:

    uv run python experiments/rf/sweep_synthetic_modulation.py --device cuda
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
from experiments.rf.synthetic_modulation import (
    DEFAULT_MODEL_FAMILIES,
    DEFAULT_MODULATIONS,
    DEFAULT_SNR_DB,
    ArchitectureName,
    ModelFamily,
    _features_for_family,
    _make_model,
    make_synthetic_rf_modulation_dataset,
)


def _train_one(
    family: str,
    hp: dict[str, Any],
    seed: int,
    *,
    architecture: ArchitectureName,
    kernel_size: int,
    modulations: Sequence[str],
    snr_db_levels: Sequence[int],
    n_per_class_per_snr: int,
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

    data = make_synthetic_rf_modulation_dataset(
        seed=seed,
        modulations=cast(Any, tuple(modulations)),
        snr_db_levels=tuple(snr_db_levels),
        n_per_class_per_snr=n_per_class_per_snr,
        sample_length=sample_length,
        train_fraction=0.8,
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
        val_acc = float(
            (model(val_inputs).argmax(dim=-1) == val_labels).float().mean().item()
        )
        test_acc = float(
            (model(test_inputs).argmax(dim=-1) == test_labels).float().mean().item()
        )
    parameter_count = count_real_parameters(model)
    return TrialSeedOutcome(
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        train_seconds=train_seconds,
        extra={"parameter_count": parameter_count},
    )


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
        "# Synthetic RF Modulation Classification (Swept)",
        "",
        (
            "Random-search sweep of "
            f"`{sweep_config['n_trials']}` trials x "
            f"`{len(cast(list[int], sweep_config['seeds']))}` seeds, following "
            "`docs/tuning_budget.md`. Stand-in for a future RadioML 2018.01A "
            "benchmark; numbers reflect the synthetic IQ + AWGN distribution, "
            "not the real dataset."
        ),
        "",
        (
            f"Architecture: `{sweep_config['architecture']}`. "
            f"Activation (complex): `{sweep_config['activation']}`. "
            f"Activation (real baselines): `{sweep_config['real_activation']}`. "
            f"Modulations: `{sweep_config['modulations']}`. "
            f"SNR (dB): `{sweep_config['snr_db_levels']}`. "
            f"Sample length: `{sweep_config['sample_length']}`."
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
    lines.extend(
        [
            "",
            "## Independent family winners",
            "",
            (
                "Diagnostic only. These rows show each family's own best "
                "validation trial, so their parameter counts are not guaranteed "
                "to be matched to the selected complex model."
            ),
            "",
        ]
    )
    lines.extend(_selection_table(independent_selections))
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=16)
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
        default=list(DEFAULT_MODULATIONS),
    )
    parser.add_argument(
        "--snr-db-levels", nargs="+", type=int, default=list(DEFAULT_SNR_DB)
    )
    parser.add_argument("--n-per-class-per-snr", type=int, default=256)
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
        default=Path("results/rf_synthetic_modulation_sweep"),
    )
    args = parser.parse_args()

    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    device = torch.device(args.device)
    architecture = cast(ArchitectureName, args.architecture)

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
            architecture=architecture,
            kernel_size=args.kernel_size,
            modulations=args.modulations,
            snr_db_levels=args.snr_db_levels,
            n_per_class_per_snr=args.n_per_class_per_snr,
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
        "experiment": "rf_synthetic_modulation_sweep",
        "n_trials": args.n_trials,
        "seeds": list(args.seeds),
        "model_families": list(args.model_families),
        "modulations": list(args.modulations),
        "snr_db_levels": list(args.snr_db_levels),
        "n_per_class_per_snr": args.n_per_class_per_snr,
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
        task_name="Synthetic RF Modulation Classification",
        sweep_config=sweep_config,
        trials=trials,
        selections=independent_selections,
    )
    summary_md = _summary_markdown(
        matched_selections,
        independent_selections=independent_selections,
        sweep_config=sweep_config,
    )
    (args.output_dir / "summary.md").write_text(summary_md + "\n")

    summary_json: JsonObject = {
        "config": sweep_config,
        "reference_family": "complex",
        "selections": [sel.to_dict() for sel in matched_selections],
        "independent_selections": [sel.to_dict() for sel in independent_selections],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
    )

    manifest = new_manifest(
        run_id="rf-synthetic-modulation-sweep",
        config=sweep_config,
        seeds=list(args.seeds),
        metrics={
            "reference_family": "complex",
            "selections": [sel.to_dict() for sel in matched_selections],
            "independent_selections": [sel.to_dict() for sel in independent_selections],
        },
        device=args.device,
        dtype=args.dtype,
        dataset={
            "name": "rf_synthetic_modulation",
            "version": "0.1.0",
            "description": (
                "synthetic IQ PSK/QAM symbols + AWGN; stand-in for RadioML"
            ),
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
