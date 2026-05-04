"""Generate paper figures from committed result artifacts.

Outputs go to ``results/figures/`` and are regenerated deterministically from
the JSON summaries under ``results/`` and ``notebooks/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NOTEBOOKS = ROOT / "notebooks"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ["complex", "real_stacked", "real_matched_params", "real_matched_flops"]
FAMILY_COLOR = {
    "complex": "#1f77b4",
    "real_stacked": "#ff7f0e",
    "real_matched_params": "#2ca02c",
    "real_matched_flops": "#d62728",
}
FAMILY_LABEL = {
    "complex": "CVNN",
    "real_stacked": "Real (stacked)",
    "real_matched_params": "Real (≈params)",
    "real_matched_flops": "Real (≈FLOPs)",
}

RADIOML_CRELU_DIR = "radioml_modulation_sweep_crelu"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def fig_activation_tradeoff() -> Path:
    data = load_json(NOTEBOOKS / "activation_characterization" / "comparison.json")
    names = [d["name"] for d in data]
    grad_mean = np.array([d["gradient_norm_mean"] for d in data])
    grad_std = np.array([d["gradient_norm_std"] for d in data])
    cr_med = np.array([d["cr_median"] for d in data])
    cr_p95 = np.array([d["cr_p95"] for d in data])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, grad_mean, yerr=grad_std, color="#4c72b0", capsize=4, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Gradient norm at init (log scale)")
    ax.set_title("Gradient magnitude — mean ± std over grid")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)

    ax = axes[1]
    width = 0.38
    # Avoid zeros on log axis (zReLU has cr_median == 0 — strictly holomorphic
    # on its support).
    eps = 1e-16
    ax.bar(
        x - width / 2,
        np.maximum(cr_med, eps),
        width,
        label="median",
        color="#55a868",
    )
    ax.bar(
        x + width / 2,
        np.maximum(cr_p95, eps),
        width,
        label="p95",
        color="#c44e52",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Cauchy–Riemann residual (log scale)")
    ax.set_title("Holomorphy defect across the grid")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)

    fig.suptitle("Activation trade-off: stability vs. holomorphy", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "activation_tradeoff.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _family_bar(
    ax: object,
    families: list[str],
    means: list[float],
    ci_low: list[float],
    ci_high: list[float],
    params: list[float],
    madds: list[float] | None = None,
) -> None:
    x = np.arange(len(families))
    err_low = np.array(means) - np.array(ci_low)
    err_high = np.array(ci_high) - np.array(means)
    colors = [FAMILY_COLOR[f] for f in families]
    ax.bar(x, means, yerr=[err_low, err_high], color=colors, capsize=5, alpha=0.9)  # type: ignore[attr-defined]
    ax.set_xticks(x)  # type: ignore[attr-defined]
    ax.set_xticklabels(  # type: ignore[attr-defined]
        [FAMILY_LABEL[f] for f in families], rotation=15, ha="right"
    )
    ax.set_ylabel("Test accuracy (95% CI)")  # type: ignore[attr-defined]
    for i, (m, p) in enumerate(zip(means, params, strict=True)):
        label = f"p={int(p)}"
        if madds is not None:
            label += f"\nm={int(madds[i])}"
        ax.text(i, m + 0.005, label, ha="center", va="bottom", fontsize=8)  # type: ignore[attr-defined]


def fig_synthetic_phase() -> Path:
    summary = load_json(RESULTS / "synthetic_phase_classification" / "summary.json")
    rows = {r["model_family"]: r for r in summary["summaries"]}
    families = [f for f in FAMILY_ORDER if f in rows]
    means = [rows[f]["test_accuracy_mean"] for f in families]
    lo = [rows[f]["test_accuracy_ci_low"] for f in families]
    hi = [rows[f]["test_accuracy_ci_high"] for f in families]
    params = [rows[f]["parameter_count"] for f in families]
    madds = [rows[f]["estimated_forward_madds"] for f in families]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _family_bar(ax, families, means, lo, hi, params, madds)
    ymin = min(lo) - 0.01
    ymax = max(hi) + 0.03
    ax.set_ylim(ymin, ymax)
    ax.set_title("Synthetic phase classification — fixed config (seeds=0,1,2)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "synthetic_phase_classification.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_sweep_pareto() -> Path:
    sweep = load_json(RESULTS / "synthetic_phase_classification_sweep" / "trials.json")
    summary = load_json(
        RESULTS / "synthetic_phase_classification_sweep" / "summary.json"
    )
    selections = {s["family"]: s for s in summary["selections"]}
    trials = sweep["trials"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for fam in FAMILY_ORDER:
        fam_trials = [t for t in trials if t["family"] == fam]
        params = [t["extra"]["parameter_count_mean"] for t in fam_trials]
        accs = [t["test_accuracy_mean"] for t in fam_trials]
        ax.scatter(
            params,
            accs,
            color=FAMILY_COLOR[fam],
            alpha=0.55,
            s=42,
            label=FAMILY_LABEL[fam],
            edgecolor="white",
            linewidth=0.6,
        )
        sel = selections.get(fam)
        if sel is not None:
            ax.scatter(
                [sel["selected_extra"]["parameter_count_mean"]],
                [sel["selected_test_accuracy_mean"]],
                color=FAMILY_COLOR[fam],
                marker="*",
                s=240,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Test accuracy (mean over 3 seeds)")
    ax.set_title(
        "Sweep on synthetic phase classification — 16 trials × 4 families\n"
        "★ = trial chosen by validation accuracy"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "sweep_pareto.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_rf_accuracy_vs_snr() -> Path:
    summary = load_json(RESULTS / "rf_synthetic_modulation" / "summary.json")
    rows = {r["model_family"]: r for r in summary["summaries"]}
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for fam in FAMILY_ORDER:
        if fam not in rows:
            continue
        per_snr = rows[fam]["accuracy_by_snr_db_mean"]
        snrs = sorted(int(k) for k in per_snr.keys())
        accs = [per_snr[str(s)] for s in snrs]
        ax.plot(
            snrs,
            accs,
            marker="o",
            color=FAMILY_COLOR[fam],
            label=f"{FAMILY_LABEL[fam]} (p={rows[fam]['parameter_count']})",
            linewidth=1.8,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean test accuracy")
    ax.set_title(
        "RF modulation per-SNR — flatten-MLP snapshot (1×3, preliminary)\n"
        "Headline conv-architecture comparison: see rf_synthetic_modulation_swept.png"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "rf_accuracy_vs_snr.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _swept_bars(
    summary_path: Path,
    *,
    title: str,
    out_name: str,
) -> Path:
    """Bar chart of swept-selection test accuracies with 95% CI from std / √n."""

    summary = load_json(summary_path)
    selection_rows = summary.get("matched_selections") or summary["selections"]
    selections = {s["family"]: s for s in selection_rows}
    n_seeds = len(summary["config"]["seeds"])
    families = [f for f in FAMILY_ORDER if f in selections]
    means = [selections[f]["selected_test_accuracy_mean"] for f in families]
    stds = [selections[f]["selected_test_accuracy_std"] for f in families]
    half_ci = [1.96 * s / np.sqrt(n_seeds) for s in stds]
    lo = [m - h for m, h in zip(means, half_ci, strict=True)]
    hi = [m + h for m, h in zip(means, half_ci, strict=True)]
    params = [selections[f]["selected_extra"]["parameter_count_mean"] for f in families]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _family_bar(ax, families, means, lo, hi, params)
    ymin = min(lo) - 0.01
    ymax = max(hi) + 0.025
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_synthetic_phase_swept() -> Path:
    return _swept_bars(
        RESULTS / "synthetic_phase_classification_sweep" / "summary.json",
        title=(
            "Synthetic phase classification — matched shared-trial configs\n"
            "16 trials × 3 seeds  ·  error bars = 95% CI (std / √n)"
        ),
        out_name="synthetic_phase_classification_swept.png",
    )


def fig_rf_swept() -> Path:
    summary_path = RESULTS / "rf_synthetic_modulation_sweep" / "summary.json"
    summary = load_json(summary_path)
    n_seeds = len(summary["config"]["seeds"])
    return _swept_bars(
        summary_path,
        title=(
            f"RF synthetic modulation (conv) — matched shared-trial configs\n"
            f"16 trials × {n_seeds} seeds  ·  "
            "error bars = 95% CI (std / √n)\n"
            "Complex wins at comparable parameter count"
        ),
        out_name="rf_synthetic_modulation_swept.png",
    )


def fig_rf_sweep_pareto() -> Path:
    sweep = load_json(RESULTS / "rf_synthetic_modulation_sweep" / "trials.json")
    summary = load_json(RESULTS / "rf_synthetic_modulation_sweep" / "summary.json")
    selection_rows = summary.get("matched_selections") or summary["selections"]
    selections = {s["family"]: s for s in selection_rows}
    trials = sweep["trials"]
    n_seeds = len(sweep["config"]["seeds"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for fam in FAMILY_ORDER:
        fam_trials = [t for t in trials if t["family"] == fam]
        params = [t["extra"]["parameter_count_mean"] for t in fam_trials]
        accs = [t["test_accuracy_mean"] for t in fam_trials]
        ax.scatter(
            params,
            accs,
            color=FAMILY_COLOR[fam],
            alpha=0.55,
            s=42,
            label=FAMILY_LABEL[fam],
            edgecolor="white",
            linewidth=0.6,
        )
        sel = selections.get(fam)
        if sel is not None:
            ax.scatter(
                [sel["selected_extra"]["parameter_count_mean"]],
                [sel["selected_test_accuracy_mean"]],
                color=FAMILY_COLOR[fam],
                marker="*",
                s=240,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel(f"Test accuracy (mean over {n_seeds} seeds)")
    ax.set_title(
        "Sweep on RF synthetic modulation — 16 trials × 4 families\n"
        "★ = trial chosen by validation accuracy\n"
        "Primary stars use the matched shared-trial comparison"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "rf_sweep_pareto.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_radioml_swept() -> Path:
    summary_path = RESULTS / RADIOML_CRELU_DIR / "summary.json"
    summary = load_json(summary_path)
    n_seeds = len(summary["config"]["seeds"])
    return _swept_bars(
        summary_path,
        title=(
            f"RadioML 2018.01A subset — matched shared-trial configs\n"
            f"16 trials × {n_seeds} seeds  ·  "
            "error bars = 95% CI (std / √n)\n"
            "Matched selection exposes real-baseline instability"
        ),
        out_name="radioml_modulation_swept.png",
    )


def fig_radioml_per_snr() -> Path:
    summary = load_json(RESULTS / RADIOML_CRELU_DIR / "summary.json")
    selection_rows = summary["matched_selections"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    snr_keys: list[str] = []
    for sel in selection_rows:
        per_seed = sel["selected_extra"].get("test_accuracy_by_snr_db_per_seed", [])
        if isinstance(per_seed, list):
            for item in per_seed:
                if isinstance(item, dict):
                    for snr in item:
                        if snr not in snr_keys:
                            snr_keys.append(snr)
    snr_keys = sorted(snr_keys, key=int)

    for sel in selection_rows:
        family = sel["family"]
        if family not in FAMILY_ORDER:
            continue
        per_seed = sel["selected_extra"].get("test_accuracy_by_snr_db_per_seed", [])
        if not isinstance(per_seed, list):
            continue
        means: list[float] = []
        for snr in snr_keys:
            values = [
                float(item[snr])
                for item in per_seed
                if isinstance(item, dict) and snr in item
            ]
            means.append(sum(values) / len(values) if values else float("nan"))
        ax.plot(
            [int(snr) for snr in snr_keys],
            means,
            marker="o",
            color=FAMILY_COLOR[family],
            label=FAMILY_LABEL[family],
            linewidth=1.8,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean test accuracy")
    ax.set_title(
        "RadioML 2018.01A — accuracy vs SNR (matched shared-trial)\n"
        "CReLU run: complex is stable; real baselines collapse at selected trial"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "radioml_per_snr.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_radioml_sweep_pareto() -> Path:
    sweep = load_json(RESULTS / RADIOML_CRELU_DIR / "trials.json")
    summary = load_json(RESULTS / RADIOML_CRELU_DIR / "summary.json")
    selection_rows = summary["matched_selections"]
    selections = {s["family"]: s for s in selection_rows}
    trials = sweep["trials"]
    n_seeds = len(sweep["config"]["seeds"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for fam in FAMILY_ORDER:
        fam_trials = [t for t in trials if t["family"] == fam]
        if not fam_trials:
            continue
        params = [t["extra"]["parameter_count_mean"] for t in fam_trials]
        accs = [t["test_accuracy_mean"] for t in fam_trials]
        ax.scatter(
            params,
            accs,
            color=FAMILY_COLOR[fam],
            alpha=0.55,
            s=42,
            label=FAMILY_LABEL[fam],
            edgecolor="white",
            linewidth=0.6,
        )
        sel = selections.get(fam)
        if sel is not None:
            ax.scatter(
                [sel["selected_extra"]["parameter_count_mean"]],
                [sel["selected_test_accuracy_mean"]],
                color=FAMILY_COLOR[fam],
                marker="*",
                s=240,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel(f"Test accuracy (mean over {n_seeds} seeds)")
    ax.set_title(
        "Sweep on RadioML 2018.01A — 16 trials × 4 families\n"
        "★ = matched shared-trial selection.  "
        "Complex sits upper-left."
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "radioml_sweep_pareto.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


RADIOML_ABLATION_ACTIVATIONS: tuple[str, ...] = (
    "crelu",
    "modrelu",
    "cardioid",
    "siglog",
    "zrelu",
)
RADIOML_ABLATION_DIRS: dict[str, str] = {
    # The first CReLU sweep used an odd-SNR request list and is kept as a
    # historical artifact at `radioml_modulation_sweep`. The corrected run uses
    # the same eight even SNRs as the other activation runs.
    "crelu": RADIOML_CRELU_DIR,
    "modrelu": "radioml_modulation_sweep_modrelu",
    "cardioid": "radioml_modulation_sweep_cardioid",
    "siglog": "radioml_modulation_sweep_siglog",
    "zrelu": "radioml_modulation_sweep_zrelu",
}


def fig_radioml_activation_ablation() -> Path | None:
    """Lines: x = activation, y = test accuracy, one line per family.

    Skipped (returns None) until at least two activation runs are present,
    so the figures script doesn't break before the GPU sweeps land.
    """

    rows: dict[str, dict[str, tuple[float, float, int]]] = {}
    for activation in RADIOML_ABLATION_ACTIVATIONS:
        summary_path = RESULTS / RADIOML_ABLATION_DIRS[activation] / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        n_seeds = len(summary["config"]["seeds"])
        selection_rows = summary.get("matched_selections") or summary["selections"]
        for sel in selection_rows:
            family = sel["family"]
            if family not in FAMILY_ORDER:
                continue
            mean = sel["selected_test_accuracy_mean"]
            std = sel["selected_test_accuracy_std"]
            rows.setdefault(activation, {})[family] = (mean, std, n_seeds)

    if len(rows) < 2:
        print(
            "skipping radioml_activation_ablation.png: need >=2 activation runs, "
            f"have {len(rows)}"
        )
        return None

    activations = [a for a in RADIOML_ABLATION_ACTIVATIONS if a in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(activations))
    for family in FAMILY_ORDER:
        means: list[float] = []
        ci_half: list[float] = []
        for activation in activations:
            cell = rows[activation].get(family)
            if cell is None:
                means.append(float("nan"))
                ci_half.append(0.0)
                continue
            mean, std, n_seeds = cell
            means.append(mean)
            ci_half.append(1.96 * std / np.sqrt(n_seeds))
        ax.errorbar(
            x,
            means,
            yerr=ci_half,
            marker="o",
            color=FAMILY_COLOR[family],
            label=FAMILY_LABEL[family],
            linewidth=1.8,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(activations, rotation=10)
    ax.set_xlabel("Activation (complex side; real baselines use ReLU)")
    ax.set_ylabel("Test accuracy (matched shared-trial, 95% CI)")
    ax.set_title(
        "RadioML 2018.01A — activation ablation on complex side\n"
        "Lines parallel = robust to activation choice; lines crossing = "
        "activation-specific finding"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "radioml_activation_ablation.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


TELEMETRY_DIR = RESULTS / "radioml_telemetry"
SYNTHETIC_TELEMETRY_DIR = RESULTS / "synthetic_rf_telemetry"


def _load_telemetry_records(
    activation: str,
    family: str,
    seed: int,
    *,
    root: Path = TELEMETRY_DIR,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    path = root / activation / f"{family}_seed{seed}.jsonl"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    summary = json.loads(lines[0])["summary"]
    records = [json.loads(line) for line in lines[1:]]
    return summary, records


def _telemetry_present(
    activations: list[str], *, root: Path = TELEMETRY_DIR
) -> list[str]:
    """Return activations whose telemetry JSONL files exist for all families."""

    available: list[str] = []
    for activation in activations:
        ok = True
        for family in FAMILY_ORDER:
            for seed in (0, 1, 2):
                if _load_telemetry_records(activation, family, seed, root=root) is None:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            available.append(activation)
    return available


def _render_telemetry_panels(
    *,
    root: Path,
    metric_key: str,
    out_name: str,
    suptitle: str,
    ylabel: str,
    show_chance_line: bool,
) -> Path | None:
    activations = _telemetry_present(list(RADIOML_ABLATION_ACTIVATIONS), root=root)
    if len(activations) < 2:
        print(
            f"skipping {out_name}: telemetry for "
            f"{len(activations)} activations only at {root}"
        )
        return None

    n = len(activations)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, activation in enumerate(activations):
        ax = axes_flat[idx]
        for family in FAMILY_ORDER:
            for seed in (0, 1, 2):
                payload = _load_telemetry_records(activation, family, seed, root=root)
                if payload is None:
                    continue
                _, records = payload
                steps = [r["step"] for r in records]
                values = [max(r[metric_key], 1e-6) for r in records]
                ax.plot(
                    steps,
                    values,
                    color=FAMILY_COLOR[family],
                    alpha=0.55,
                    linewidth=1.0,
                    label=FAMILY_LABEL[family] if seed == 0 else None,
                )
        if show_chance_line:
            ax.axhline(np.log(3), linestyle=":", color="grey", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{activation}")
        ax.grid(True, which="both", linestyle=":", alpha=0.3)
        if idx == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=8)
    for idx in range(len(activations), len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def fig_telemetry_loss_curves() -> Path | None:
    return _render_telemetry_panels(
        root=TELEMETRY_DIR,
        metric_key="train_loss",
        out_name="radioml_telemetry_loss.png",
        suptitle=(
            "RadioML telemetry — train loss (3 seeds per family per "
            "activation)\nDotted line = ln(3) chance level; lines collapsing "
            "onto it are dead seeds"
        ),
        ylabel="train loss (log)",
        show_chance_line=True,
    )


def fig_telemetry_grad_norms() -> Path | None:
    return _render_telemetry_panels(
        root=TELEMETRY_DIR,
        metric_key="total_grad_norm",
        out_name="radioml_telemetry_grad.png",
        suptitle=(
            "RadioML telemetry — total gradient norm (post-backward)\n"
            "Real-baseline spikes at step 1 under crelu/cardioid/siglog mark "
            "the explosion-into-dead-region failure mode"
        ),
        ylabel="total grad norm (log)",
        show_chance_line=False,
    )


def fig_synthetic_telemetry_loss_curves() -> Path | None:
    return _render_telemetry_panels(
        root=SYNTHETIC_TELEMETRY_DIR,
        metric_key="train_loss",
        out_name="synthetic_rf_telemetry_loss.png",
        suptitle=(
            "Synthetic RF telemetry — train loss (same matched-shared-trial "
            "hp regimes as RadioML, applied to AWGN-only synthetic data)\n"
            "Same crelu/cardioid/siglog seeds collapse onto ln(3) chance — "
            "mechanism is hp-driven, not data-driven"
        ),
        ylabel="train loss (log)",
        show_chance_line=True,
    )


def fig_synthetic_telemetry_grad_norms() -> Path | None:
    return _render_telemetry_panels(
        root=SYNTHETIC_TELEMETRY_DIR,
        metric_key="total_grad_norm",
        out_name="synthetic_rf_telemetry_grad.png",
        suptitle=(
            "Synthetic RF telemetry — total gradient norm\n"
            "Step-1 explosions on synthetic data are if anything BIGGER than "
            "RadioML's: AWGN-only signals have higher per-sample variance"
        ),
        ylabel="total grad norm (log)",
        show_chance_line=False,
    )


def main() -> None:
    figures: list[Path | None] = [
        fig_activation_tradeoff(),
        fig_synthetic_phase(),
        fig_synthetic_phase_swept(),
        fig_sweep_pareto(),
        fig_rf_accuracy_vs_snr(),
        fig_rf_swept(),
        fig_rf_sweep_pareto(),
        fig_radioml_swept(),
        fig_radioml_per_snr(),
        fig_radioml_sweep_pareto(),
        fig_radioml_activation_ablation(),
        fig_telemetry_loss_curves(),
        fig_telemetry_grad_norms(),
        fig_synthetic_telemetry_loss_curves(),
        fig_synthetic_telemetry_grad_norms(),
    ]
    for path in figures:
        if path is None:
            continue
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
