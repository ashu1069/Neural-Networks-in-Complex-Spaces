"""Re-run synthetic-RF telemetry without the 200-step cap.

Drives `experiments.rf.gradient_telemetry` with `max_steps=None` so each
trajectory runs the activation's full matched-shared-trial selected step
budget (200–800). Output goes to `results/synthetic_rf_telemetry_full/`
to keep the original 200-step traces (used by the early-divergence
analysis in the paper) intact.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from experiments.rf.gradient_telemetry import (
    run_instrumented_training,
    telemetry_config_from_sweep_summary,
    write_telemetry_jsonl,
)

ACTIVATIONS = ["crelu", "cardioid", "siglog", "modrelu", "zrelu"]
FAMILIES = ["real_stacked", "real_matched_params", "real_matched_flops"]
SEEDS = [0, 1, 2]
DEVICE = "mps"
OUTPUT_ROOT = Path("results/synthetic_rf_telemetry_full")


def main() -> int:
    summaries = []
    total = len(ACTIVATIONS) * len(FAMILIES) * len(SEEDS)
    counter = 0
    for activation in ACTIVATIONS:
        sweep_summary = (
            Path(f"results/radioml_modulation_sweep_{activation}") / "summary.json"
        )
        for family in FAMILIES:
            for seed in SEEDS:
                counter += 1
                radioml_config = telemetry_config_from_sweep_summary(
                    sweep_summary,
                    activation=activation,
                    family=family,
                    seed=seed,
                    data_path=Path("/dev/null"),
                )
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
                output_path = OUTPUT_ROOT / activation / f"{family}_seed{seed}.jsonl"
                print(
                    f"  [{counter}/{total}] {activation}/{family}/seed={seed} "
                    f"steps={config.hyperparameters['steps']} "
                    f"lr={config.hyperparameters['learning_rate']:.4f}",
                    flush=True,
                )
                records, run_summary = run_instrumented_training(
                    config,
                    device=DEVICE,
                    log_every_n=1,
                    log_per_layer=True,
                    max_steps=None,  # full step budget
                )
                run_summary["data_source"] = "synthetic"
                write_telemetry_jsonl(
                    output_path, records=records, summary=run_summary
                )
                summaries.append(
                    {
                        "activation": activation,
                        "family": family,
                        "seed": seed,
                        "n_steps": run_summary["n_steps"],
                        "test_accuracy": run_summary["test_accuracy"],
                        "final_train_loss": run_summary["final_train_loss"],
                        "output_path": str(output_path),
                    }
                )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "index.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nWrote {OUTPUT_ROOT / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
