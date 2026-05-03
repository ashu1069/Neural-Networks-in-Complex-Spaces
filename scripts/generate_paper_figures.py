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
    selections = {s["family"]: s for s in sweep["selections"]}
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
    selections = {s["family"]: s for s in summary["selections"]}
    n_seeds = len(summary["config"]["seeds"])
    families = [f for f in FAMILY_ORDER if f in selections]
    means = [selections[f]["selected_test_accuracy_mean"] for f in families]
    stds = [selections[f]["selected_test_accuracy_std"] for f in families]
    half_ci = [1.96 * s / np.sqrt(n_seeds) for s in stds]
    lo = [m - h for m, h in zip(means, half_ci, strict=True)]
    hi = [m + h for m, h in zip(means, half_ci, strict=True)]
    params = [
        selections[f]["selected_extra"]["parameter_count_mean"] for f in families
    ]

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
            "Synthetic phase classification — selected configs\n"
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
            f"RF synthetic modulation (conv) — selected configs\n"
            f"16 trials × {n_seeds} seeds  ·  "
            "error bars = 95% CI (std / √n)\n"
            "Complex wins at the smallest parameter count"
        ),
        out_name="rf_synthetic_modulation_swept.png",
    )


def fig_rf_sweep_pareto() -> Path:
    sweep = load_json(RESULTS / "rf_synthetic_modulation_sweep" / "trials.json")
    selections = {s["family"]: s for s in sweep["selections"]}
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
        "Complex sits upper-left: higher accuracy, fewer parameters"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "rf_sweep_pareto.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    figures = [
        fig_activation_tradeoff(),
        fig_synthetic_phase(),
        fig_synthetic_phase_swept(),
        fig_sweep_pareto(),
        fig_rf_accuracy_vs_snr(),
        fig_rf_swept(),
        fig_rf_sweep_pareto(),
    ]
    for path in figures:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
