"""Drive `experiments.rf.gradient_telemetry` across (activation, family, seed).

Default first-cut: `crelu` and `modrelu` × all 4 families × seeds 0,1,2 = 24
runs. Add `--full` to extend to all 5 activations × 4 × 3 = 60 runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    )
    parser.add_argument("--classes-path", type=Path, default=None)
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["crelu", "modrelu"],
        choices=["crelu", "modrelu", "cardioid", "siglog", "zrelu"],
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 5 activations × 4 families × 3 seeds = 60 runs",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log-every-n",
        type=int,
        default=1,
        help="log per-step (1) or every Nth step",
    )
    parser.add_argument(
        "--no-per-layer",
        action="store_true",
        help="skip per-parameter grad norm logging (smaller files)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help=(
            "cap training to first N steps (default 200). The early-training "
            "divergence pattern is what we're after; full step budgets are "
            "wasteful for telemetry. Pass 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/radioml_telemetry"),
    )
    args = parser.parse_args()

    activations = (
        ["crelu", "modrelu", "cardioid", "siglog", "zrelu"]
        if args.full
        else list(args.activations)
    )
    families = list(args.families)
    seeds = list(args.seeds)
    total = len(activations) * len(families) * len(seeds)
    print(
        f"Telemetry: {len(activations)} acts × {len(families)} families × "
        f"{len(seeds)} seeds = {total} runs",
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
                config = telemetry_config_from_sweep_summary(
                    summary_path,
                    activation=activation,
                    family=family,
                    seed=seed,
                    data_path=args.data_path,
                    classes_path=args.classes_path,
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

    index_path = args.output_root / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
