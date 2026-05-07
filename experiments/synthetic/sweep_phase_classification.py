"""16-trial random search over phase-classification hyperparameters.

Honors `docs/tuning_budget.md`: shared search space across all four model
families, 3 seeds per trial, selection by mean validation accuracy. Writes
`tuning_log.md` and `trials.json` plus a `summary.md` and `manifest.json`
that report the *selected* configuration's test metric per family.
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
    step_progress_bar,
    write_loss_curve_plots,
    write_training_params,
    write_tuning_log,
)
from experiments.synthetic.phase_classification import (
    DEFAULT_MODEL_FAMILIES,
    ModelFamily,
    _make_model,
    make_phase_classification,
)


def _train_one(
    family: str,
    hp: dict[str, Any],
    seed: int,
    *,
    n_classes: int,
    n_train: int,
    n_test: int,
    val_fraction: float,
    noise_std: float,
    activation: str,
    real_activation: str,
    device: torch.device,
    dtype: torch.dtype,
) -> TrialSeedOutcome:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = make_phase_classification(
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        n_classes=n_classes,
        noise_std=noise_std,
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

    model_family = cast(ModelFamily, family)
    model, _, _ = _make_model(
        model_family,
        complex_hidden_features=hp["hidden_features"],
        n_classes=n_classes,
        activation=activation,  # type: ignore[arg-type]
        real_activation=real_activation,  # type: ignore[arg-type]
        device=device,
        dtype=dtype,
    )

    def _features(x: torch.Tensor) -> torch.Tensor:
        if family == "complex":
            return x
        return torch.cat([x.real, x.imag], dim=-1)

    train_inputs = _features(train_actual_inputs)
    val_inputs = _features(val_inputs_complex)
    test_inputs = _features(test_inputs_complex)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=0.0
    )
    n_steps = int(hp["steps"])
    inner_bar = step_progress_bar(n_steps, desc=f"{family}/s{seed}")
    start = time.perf_counter()
    model.train()
    last_loss = float("nan")
    loss_curve: list[float] = []
    for step in range(n_steps):
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = F.cross_entropy(logits, train_actual_labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
        loss_curve.append(last_loss)
        if inner_bar is not None:
            if step == n_steps - 1 or step % max(1, n_steps // 50) == 0:
                inner_bar.set_postfix_str(f"loss={last_loss:.4f}", refresh=False)
            inner_bar.update(1)
    if inner_bar is not None:
        inner_bar.close()
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
        extra={
            "parameter_count": parameter_count,
            "final_train_loss": last_loss,
            "train_loss_curve": loss_curve,
        },
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
        "# Synthetic Phase Classification (Swept)",
        "",
        (
            "Random-search sweep of "
            f"`{sweep_config['n_trials']}` trials x "
            f"`{len(cast(list[int], sweep_config['seeds']))}` seeds, following "
            "`docs/tuning_budget.md`. See `tuning_log.md` for the per-trial "
            "log and `trials.json` for the full record."
        ),
        "",
        (
            f"Activation (complex): `{sweep_config['activation']}`. "
            f"Activation (real baselines): `{sweep_config['real_activation']}`. "
            f"`n_classes={sweep_config['n_classes']}`, "
            f"`noise_std={sweep_config['noise_std']}`."
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
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--n-classes", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.3)
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
        default=Path("results/synthetic_phase_classification_sweep"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "JSON seed-level checkpoint. Defaults to checkpoint.json inside "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the checkpoint and skip completed seed runs",
    )
    args = parser.parse_args()

    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint_path or (args.output_dir / "checkpoint.json")
    space = SearchSpace(
        distributions={
            "learning_rate": ("loguniform", 5e-3, 1e-1),
            "hidden_features": ("choice", [8, 16, 32, 64]),
            "steps": ("choice", [200, 400, 800]),
        }
    )

    def train_fn(family: str, hp: dict[str, Any], seed: int) -> TrialSeedOutcome:
        return _train_one(
            family,
            hp,
            seed,
            n_classes=args.n_classes,
            n_train=args.n_train,
            n_test=args.n_test,
            val_fraction=args.val_fraction,
            noise_std=args.noise_std,
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
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
    independent_selections = select_best_per_family(trials)
    matched_selections = select_reference_trial_for_all_families(trials)

    sweep_config: JsonObject = {
        "experiment": "synthetic_phase_classification_sweep",
        "n_trials": args.n_trials,
        "seeds": list(args.seeds),
        "model_families": list(args.model_families),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "val_fraction": args.val_fraction,
        "n_classes": args.n_classes,
        "noise_std": args.noise_std,
        "activation": args.activation,
        "real_activation": args.real_activation,
        "device": args.device,
        "dtype": args.dtype,
        "sweep_seed": args.sweep_seed,
        "checkpoint_path": str(checkpoint_path),
        "resume": args.resume,
        "search_space": {
            name: list(spec) for name, spec in space.distributions.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tuning_log(
        args.output_dir,
        task_name="Synthetic Phase Classification",
        sweep_config=sweep_config,
        trials=trials,
        selections=independent_selections,
    )
    training_params_path = write_training_params(
        args.output_dir,
        sweep_config=sweep_config,
        trials=trials,
    )
    plot_artifacts = write_loss_curve_plots(
        args.output_dir,
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
        "reference_family": "complex",
        "selections": [sel.to_dict() for sel in matched_selections],
        "independent_selections": [sel.to_dict() for sel in independent_selections],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n"
    )

    manifest = new_manifest(
        run_id="synthetic-phase-classification-sweep",
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
            "name": "synthetic_phase_classification",
            "version": "0.1.0",
            "description": "balanced complex phase-sector classification",
        },
        artifacts={
            "checkpoint_json": str(checkpoint_path),
            "trials_json": str(args.output_dir / "trials.json"),
            "training_params_json": str(training_params_path),
            "tuning_log_markdown": str(args.output_dir / "tuning_log.md"),
            "summary_markdown": str(args.output_dir / "summary.md"),
            "summary_json": str(args.output_dir / "summary.json"),
            **plot_artifacts,
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
