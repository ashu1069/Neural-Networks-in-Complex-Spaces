"""LR-vs-activation disambiguation for the dead-seed mechanism.

Hypothesis to test: the dead-seed asymmetry across activations
(crelu/cardioid/siglog → 3/9 dead; modrelu/zrelu → 0/9) reported in
section 4.4.4 is driven by the *learning rate* the matched-shared-trial
rule selects, not by the *activation*. The unstable activations select
lr in [0.024, 0.040]; the stable ones select lr in [0.0024, 0.008].

Design: a 2x2 factorial on synthetic AWGN RF data. We pick crelu (the
unstable headline activation) and zrelu (the stable counter-example),
and run each with both the high-lr config (lr=0.0236, the crelu
selection) and the low-lr config (lr=0.0024, the zrelu selection). All
other hyperparameters are taken from each activation's own
matched-shared-trial selection so that only lr changes within an
activation row.

If the asymmetry is *lr-driven*: zrelu@high-lr should produce dead
seeds; crelu@low-lr should not.
If the asymmetry is *activation-driven*: crelu should stay unstable at
low lr; zrelu should stay stable at high lr.

Output: results/lr_activation_disambiguation/{cell}/{family}_seed{N}.jsonl
plus an index.json with dead-seed counts for the paper.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

from experiments.rf.gradient_telemetry import (
    run_instrumented_training,
    telemetry_config_from_sweep_summary,
    write_telemetry_jsonl,
)

DEAD_LOSS_THRESHOLD = math.log(3) - 0.05  # 3-class CE chance level - epsilon

CELLS = [
    # (cell_name, source_activation_for_hp, lr_override)
    ("crelu_highlr", "crelu", 0.0236),  # baseline (re-run for parity, same lr)
    ("crelu_lowlr", "crelu", 0.0024),  # crelu with zrelu's lr
    ("zrelu_lowlr", "zrelu", 0.0024),  # baseline (re-run for parity)
    ("zrelu_highlr", "zrelu", 0.0236),  # zrelu with crelu's lr
]
REAL_FAMILIES = ["real_stacked", "real_matched_params", "real_matched_flops"]
SEEDS = [0, 1, 2]
OUTPUT_ROOT = Path("results/lr_activation_disambiguation")
DEVICE = "mps"
INCLUDE_COMPLEX = False  # disambiguation only needs real families


def main() -> int:
    summaries: list[dict[str, object]] = []
    counter = 0
    families_run = (["complex"] if INCLUDE_COMPLEX else []) + REAL_FAMILIES
    total = len(CELLS) * len(families_run) * len(SEEDS)
    for cell_name, source_act, lr in CELLS:
        sweep_summary = (
            Path(f"results/radioml_modulation_sweep_{source_act}") / "summary.json"
        )
        if not sweep_summary.exists():
            print(f"  skip {cell_name}: {sweep_summary} missing", flush=True)
            continue
        for family in families_run:
            for seed in SEEDS:
                counter += 1
                radioml_config = telemetry_config_from_sweep_summary(
                    sweep_summary,
                    activation=source_act,
                    family=family,
                    seed=seed,
                    data_path=Path("/dev/null"),  # synthetic; not read
                )
                hp = dict(radioml_config.hyperparameters)
                hp["learning_rate"] = lr
                synthetic_modulations = tuple(
                    m.lower() for m in radioml_config.modulations
                )
                config = replace(
                    radioml_config,
                    data_source="synthetic",
                    modulations=synthetic_modulations,
                    data_path=None,
                    classes_path=None,
                    hyperparameters=hp,
                )
                output_path = OUTPUT_ROOT / cell_name / f"{family}_seed{seed}.jsonl"
                print(
                    f"  [{counter}/{total}] {cell_name}/{family}/seed={seed} "
                    f"lr={lr} hp={hp}",
                    flush=True,
                )
                records, run_summary = run_instrumented_training(
                    config,
                    device=DEVICE,
                    log_every_n=1,
                    log_per_layer=True,
                    max_steps=200,
                )
                run_summary["data_source"] = "synthetic"
                run_summary["cell"] = cell_name
                run_summary["lr_override"] = lr
                write_telemetry_jsonl(
                    output_path, records=records, summary=run_summary
                )
                summaries.append(
                    {
                        "cell": cell_name,
                        "source_activation": source_act,
                        "lr": lr,
                        "family": family,
                        "seed": seed,
                        "test_accuracy": run_summary["test_accuracy"],
                        "final_train_loss": run_summary["final_train_loss"],
                        "dead": run_summary["final_train_loss"]
                        >= DEAD_LOSS_THRESHOLD,
                        "output_path": str(output_path),
                    }
                )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_ROOT / "index.json"
    index_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {index_path}", flush=True)

    print("\n=== Dead-seed counts (real families only) ===", flush=True)
    for cell_name, _, lr in CELLS:
        n_dead = sum(
            1
            for s in summaries
            if s["cell"] == cell_name
            and s["family"] in REAL_FAMILIES
            and s["dead"]
        )
        n_total = sum(
            1
            for s in summaries
            if s["cell"] == cell_name and s["family"] in REAL_FAMILIES
        )
        print(f"  {cell_name} (lr={lr}): {n_dead}/{n_total} dead", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
