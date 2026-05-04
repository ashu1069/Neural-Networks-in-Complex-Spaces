"""Synthetic-RF version of `run_gradient_telemetry.py`.

Loads each activation's selected hyperparameters from the RadioML sweep
summaries and re-runs the same `(activation, family, seed)` combinations
against the **synthetic** RF modulation generator (no pulse shaping, no
channel effects) instead of the gated RadioML archive.

The point: confirm whether the explosion-into-dead-region mechanism we
documented on RadioML in §3.4.4 also shows up on the synthetic stand-in
under the same hp regime. If yes → mechanism is hp-regime-driven and not
RadioML-data-specific. If no → channel effects matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from experiments.rf.gradient_telemetry import (
    run_instrumented_training,
    telemetry_config_from_sweep_summary,
    write_telemetry_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/GOLD_XYZ_OSC.0001_1024.hdf5"),
        help=(
            "RadioML HDF5 path; only used to source the per-activation hp via "
            "telemetry_config_from_sweep_summary. Synthetic data is generated "
            "fresh from `make_synthetic_rf_modulation_dataset`."
        ),
    )
    parser.add_argument(
        "--classes-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["crelu", "modrelu", "cardioid", "siglog", "zrelu"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=[
            "complex",
            "real_stacked",
            "real_matched_params",
            "real_matched_flops",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-n", type=int, default=1)
    parser.add_argument("--no-per-layer", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="cap training to first N steps; pass 0 to disable",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/synthetic_rf_telemetry"),
    )
    args = parser.parse_args()

    activations = list(args.activations)
    families = list(args.families)
    seeds = list(args.seeds)
    total = len(activations) * len(families) * len(seeds)
    print(
        f"Synthetic RF telemetry: {len(activations)} acts × {len(families)} "
        f"families × {len(seeds)} seeds = {total} runs",
        flush=True,
    )

    summaries: list[dict[str, object]] = []
    counter = 0
    for activation in activations:
        sweep_dir = Path(f"results/radioml_modulation_sweep_{activation}")
        summary_path = sweep_dir / "summary.json"
        if not summary_path.exists():
            print(f"  skipping {activation}: {summary_path} missing", flush=True)
            continue
        for family in families:
            for seed in seeds:
                counter += 1
                radioml_config = telemetry_config_from_sweep_summary(
                    summary_path,
                    activation=activation,
                    family=family,
                    seed=seed,
                    data_path=args.data_path,
                    classes_path=args.classes_path,
                )
                # Override: synthetic data source, lowercase modulations
                # (synthetic generator's vocabulary differs from RadioML's).
                synthetic_modulations = tuple(
                    m.lower() for m in radioml_config.modulations
                )
                config = replace(
                    radioml_config,
                    data_source="synthetic",
                    modulations=synthetic_modulations,
                    data_path=None,
                    classes_path=None,
                )
                output_path = (
                    args.output_root / activation / f"{family}_seed{seed}.jsonl"
                )
                print(
                    f"  [{counter}/{total}] {activation}/{family}/seed={seed} "
                    f"hp={config.hyperparameters}",
                    flush=True,
                )
                records, run_summary = run_instrumented_training(
                    config,
                    device=args.device,
                    log_every_n=args.log_every_n,
                    log_per_layer=not args.no_per_layer,
                    max_steps=args.max_steps if args.max_steps > 0 else None,
                )
                run_summary["data_source"] = "synthetic"
                write_telemetry_jsonl(output_path, records=records, summary=run_summary)
                summaries.append(
                    {
                        "activation": activation,
                        "family": family,
                        "seed": seed,
                        "test_accuracy": run_summary["test_accuracy"],
                        "final_train_loss": run_summary["final_train_loss"],
                        "n_steps": run_summary["n_steps"],
                        "output_path": str(output_path),
                    }
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "index.json"
    index_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Silence unused-import warning when torch is required only for caller signatures.
_ = torch
