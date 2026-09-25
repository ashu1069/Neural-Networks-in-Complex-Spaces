"""Build the paper's secondary figures from committed result directories.

Every figure reads only `summary.json` / `manifest`-adjacent files under
`results/`, so it never retrains and cannot drift from the runs it reports.
The one exception is the Adam first-step figure, which is a deterministic
numerical check of an identity and computes its own inputs.

Visual system is shared with `plot_crossover_figure.py`: the complex model is
always blue, the polar real baseline is always vermillion, every other
baseline sits on a neutral grey ramp. Identity is also carried by marker fill,
dash pattern, or a direct label, so no reading depends on colour alone.

    .venv/bin/python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

COMPLEX = "#0072B2"
POLAR = "#D55E00"
CONTEXT = ("#565656", "#838383", "#A5A5A5")
INK = "#1A1A1A"
MUTED = "#6E6E6E"
GRID = "#D8D8D8"

CARTESIAN = ("real_stacked", "real_matched_params", "real_matched_flops")
FAMILY_LABEL = {
    "complex": "complex",
    "real_equivariant": "real, $aI{+}bJ$ constrained",
    "real_matched_params": "real, param-matched",
    "real_matched_flops": "real, FLOP-matched",
    "real_stacked": "real, stacked",
    "real_polar": "real, polar",
    "real_phase": "real, phase-only",
    "real_magnitude": "real, magnitude-only",
}
# Activations ordered magnitude-blind first, then magnitude-aware.
ACTIVATIONS = ("zrelu", "crelu", "cardioid", "siglog", "modrelu")
MAGNITUDE_AWARE = {"cardioid", "siglog", "modrelu"}
ACT_LABEL = {
    "zrelu": "ZReLU",
    "crelu": "CReLU",
    "cardioid": "Cardioid",
    "siglog": "Siglog",
    "modrelu": "ModReLU",
}


def _style(ax: Any, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=8.0, loc="left", color=INK, pad=6)
    ax.set_xlabel(xlabel, fontsize=7.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=7.5, color=INK)
    ax.tick_params(labelsize=7.0, colors=MUTED, length=2.5, width=0.6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, linestyle=":", alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(GRID)


def _save(fig: Any, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _sweep(results: Path, name: str) -> dict[str, dict[str, dict[str, Any]]]:
    data = json.loads((results / name / "summary.json").read_text())
    return {
        rule: {s["family"]: s for s in data[f"{rule}_selections"]}
        for rule in ("matched", "independent")
    }


def _pilot(results: Path, run: str, condition: str) -> dict[str, dict[str, float]]:
    path = results / run / condition / "summary.json"
    data = json.loads(path.read_text())
    return {
        row["model_family"]: {
            "mean": row["test_accuracy_mean"],
            "std": row["test_accuracy_std"],
        }
        for row in data["summaries"]
    }


def _gap(sel: dict[str, dict[str, Any]], pool: tuple[str, ...]) -> float:
    c = sel["complex"]["selected_test_accuracy_mean"]
    best = max(sel[f]["selected_test_accuracy_mean"] for f in pool if f in sel)
    return float(100.0 * (c - best))


# ---------------------------------------------------------------------------
def fig_confounds(results: Path, out: Path) -> None:
    """Slope chart: the same gap under progressively fairer protocols."""
    configs = [
        ("radioml_clip_intervention_noclip", "3 PSK classes (original)", True),
        ("radioml_geom10_crelu", "10 classes", False),
        ("radioml_geom7_len256_crelu", "7 classes, length 256", False),
        ("radioml_geom7_crelu", "7 classes", False),
    ]
    stages = [
        "Cartesian baselines,\nshared trial",
        "+ polar baseline,\nshared trial",
        "all baselines,\nindependent tuning",
    ]
    fig, ax = plt.subplots(figsize=(3.1, 2.1))
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
    # The lines are all complex-minus-real gaps, so none is drawn in the
    # complex blue: the headline configuration takes ink, the rest grey.
    context = iter(CONTEXT)
    series = []
    for name, label, headline in configs:
        s_ = _sweep(results, name)
        reals = tuple(f for f in s_["matched"] if f.startswith("real_"))
        ys = [
            _gap(s_["matched"], CARTESIAN),
            _gap(s_["matched"], reals),
            _gap(s_["independent"], reals),
        ]
        color = INK if headline else next(context)
        series.append((label, ys, color, headline))
        ax.plot(
            range(3),
            ys,
            color=color,
            linewidth=1.9 if headline else 1.1,
            marker="o",
            markersize=4.0 if headline else 3.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4 if headline else 3,
        )
    for label, ys, color, headline in series:
        ax.plot(
            [],
            [],
            color=color,
            linewidth=1.9 if headline else 1.1,
            marker="o",
            markersize=3.2,
            label=f"{label}  ({ys[0]:+.1f} \u2192 {ys[2]:+.1f})",
        )
    head = next(t for t in series if t[3])
    ax.annotate(
        f"{head[1][2]:+.1f} pp",
        xy=(2, head[1][2]),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=6.4,
        color=INK,
    )
    ax.legend(
        fontsize=6.2,
        frameon=False,
        loc="upper right",
        handlelength=1.6,
        borderaxespad=0.1,
        labelspacing=0.3,
    )
    ax.set_xticks(range(3), stages)
    ax.set_xlim(-0.12, 2.25)
    _style(
        ax,
        title="Complex minus best real baseline",
        ylabel="gap (percentage points)",
    )
    ax.tick_params(axis="x", labelsize=6.2)
    _save(fig, out, "confounds")


# ---------------------------------------------------------------------------
def fig_activation(results: Path, out: Path) -> None:
    """Small multiples: complex accuracy by activation, four tasks, four domains."""
    panels = [
        (
            "neuro_eeg_activation_{a}",
            "amplitude_event",
            "EEG \u00b7 amplitude event",
            0.25,
        ),
        (
            "neuro_eeg_activation_{a}",
            "phase_amplitude_coupling",
            "EEG \u00b7 phase\u2013amp. coupl.",
            0.25,
        ),
        (
            "physics_quantum_activation_{a}",
            "potential_inverse",
            "Quantum \u00b7 potential inv.",
            0.20,
        ),
        ("rf_activation_{a}", "qam_representation", "RF \u00b7 QAM only", 1 / 3),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.1), sharey=False)
    for ax, (run, cond, title, chance) in zip(axes, panels, strict=True):
        means, stds, polars = [], [], []
        for act in ACTIVATIONS:
            r = _pilot(results, run.format(a=act), cond)
            means.append(r["complex"]["mean"])
            stds.append(r["complex"]["std"])
            if "real_polar" in r:
                polars.append(r["real_polar"]["mean"])
        x = list(range(len(ACTIVATIONS)))
        ax.axvspan(1.5, 4.5, color="#EEF3F7", zorder=0, linewidth=0)
        ax.axhline(chance, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)), zorder=1)
        if polars:
            ref = statistics.fmean(polars)
            ax.axhline(ref, color=POLAR, linewidth=1.1, linestyle=(0, (4, 2)), zorder=2)
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="none",
            ecolor=COMPLEX,
            elinewidth=0.8,
            capsize=0,
            zorder=3,
        )
        for xi, act, m in zip(x, ACTIVATIONS, means, strict=True):
            aware = act in MAGNITUDE_AWARE
            ax.plot(
                xi,
                m,
                marker="o",
                markersize=4.6,
                zorder=4,
                markerfacecolor=COMPLEX if aware else "white",
                markeredgecolor=COMPLEX,
                markeredgewidth=1.1,
            )
        ax.set_xticks(x, [ACT_LABEL[a] for a in ACTIVATIONS], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.05)
        _style(ax, title=title, ylabel="complex test accuracy" if ax is axes[0] else "")
        ax.tick_params(axis="x", labelsize=6.2)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            color=COMPLEX,
            marker="o",
            linestyle="none",
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="complex, magnitude-blind activation",
        ),
        Line2D(
            [],
            [],
            color=COMPLEX,
            marker="o",
            linestyle="none",
            markersize=4.6,
            label="complex, magnitude-aware activation",
        ),
        Line2D(
            [],
            [],
            color=POLAR,
            linestyle=(0, (4, 2)),
            linewidth=1.1,
            label="polar real baseline",
        ),
        Line2D(
            [], [], color=MUTED, linestyle=(0, (1, 2)), linewidth=0.6, label="chance"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=6.2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    _save(fig, out, "activation_gating")


# ---------------------------------------------------------------------------
def fig_dead_seeds(results: Path, out: Path) -> None:
    """Per-seed training loss at the shared trial: dead seeds sit at ln 3."""
    s = _sweep(results, "radioml_clip_intervention_noclip")["matched"]
    panels = [
        ("complex", COMPLEX),
        ("real_polar", POLAR),
        ("real_stacked", CONTEXT[0]),
        ("real_matched_params", CONTEXT[0]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.9), sharey=True)
    lr = s["complex"]["selected_hyperparameters"]["learning_rate"]
    for ax, (family, color) in zip(axes, panels, strict=True):
        curves = s[family]["selected_extra"]["train_loss_curve_per_seed"]
        dead = 0
        for c in curves:
            is_dead = abs(c[-1] - math.log(3)) < 0.01
            dead += is_dead
            ax.plot(
                range(1, len(c) + 1),
                c,
                color=color,
                linewidth=0.8,
                zorder=3,
                alpha=0.95 if is_dead else 0.55,
                linestyle=(0, (3, 1.5)) if is_dead else "-",
            )
        ax.axhline(math.log(3), color=INK, linewidth=0.6, linestyle=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 12)
        ax.set_yticks([0.5, math.log(3), 3, 10], ["0.5", "ln 3", "3", "10"])
        ax.minorticks_off()
        _style(
            ax,
            title=f"{FAMILY_LABEL[family]}   {dead}/{len(curves)} dead",
            xlabel="training step",
            ylabel="training loss" if ax is axes[0] else "",
        )
        ax.title.set_fontsize(7.2)
    fig.suptitle(
        f"Three PSK classes, shared trial selected by the complex model "
        f"(learning rate {lr:.4f})",
        fontsize=7.2,
        color=MUTED,
        x=0.01,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(w_pad=0.5)
    _save(fig, out, "dead_seeds")


# ---------------------------------------------------------------------------
def fig_instability(results: Path, out: Path) -> None:
    """Dumbbell: seed std at the shared trial, without and with clipping."""
    a = _sweep(results, "radioml_clip_intervention_noclip")["matched"]
    b = _sweep(results, "radioml_clip_intervention_clip1.0")["matched"]
    order = [
        "complex",
        "real_polar",
        "real_magnitude",
        "real_stacked",
        "real_matched_params",
        "real_matched_flops",
        "real_phase",
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for i, fam in enumerate(order):
        y = len(order) - 1 - i
        s0 = a[fam]["selected_test_accuracy_std"]
        s1 = b[fam]["selected_test_accuracy_std"]
        color = (
            COMPLEX
            if fam == "complex"
            else POLAR
            if fam == "real_polar"
            else CONTEXT[0]
        )
        ax.plot([s0, s1], [y, y], color=GRID, linewidth=2.2, zorder=1)
        ax.plot(
            s0,
            y,
            "o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=3,
        )
        ax.plot(
            s1,
            y,
            "o",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=color,
            zorder=3,
        )
    ax.set_yticks(range(len(order)), [FAMILY_LABEL[f] for f in reversed(order)])
    ax.set_xlim(0, 0.17)
    _style(
        ax,
        title="Seed-to-seed std. at the shared trial",
        xlabel="test-accuracy std. across 6 seeds",
    )
    ax.grid(True, axis="x", color=GRID, linewidth=0.5, linestyle=":")
    ax.grid(False, axis="y")
    ax.plot(
        [], [], "o", markerfacecolor="white", markeredgecolor=MUTED, label="no clipping"
    )
    ax.plot(
        [],
        [],
        "o",
        markerfacecolor=MUTED,
        markeredgecolor=MUTED,
        label="clip $\\|g\\|\\leq 1$",
    )
    ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    _save(fig, out, "instability_clipping")


# ---------------------------------------------------------------------------
def fig_adam_identity(out: Path) -> None:
    """First Adam step is eta*sign(g): step size is flat in gradient scale."""
    scales = [10.0**k for k in range(-3, 4)]
    eta = 0.02
    rows: dict[str, list[float]] = {"Adam": [], "Adam + clip 1.0": [], "SGD": []}
    gnorms = []
    for scale in scales:
        for name in rows:
            torch.manual_seed(0)
            m = torch.nn.Linear(16, 4)
            w0 = m.weight.detach().clone()
            opt: torch.optim.Optimizer
            if name == "SGD":
                opt = torch.optim.SGD(m.parameters(), lr=eta)
            else:
                opt = torch.optim.AdamW(m.parameters(), lr=eta, weight_decay=0.0)
            (m(torch.randn(32, 16)) * scale).sum().backward()
            grad = m.weight.grad
            assert grad is not None
            if name == "Adam":
                gnorms.append(float(grad.norm()))
            if name == "Adam + clip 1.0":
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            rows[name].append(float((m.weight.detach() - w0).abs().max()))
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    styles: dict[str, dict[str, Any]] = {
        "SGD": dict(color=CONTEXT[1], linestyle=(0, (3, 2)), marker="s"),
        "Adam + clip 1.0": dict(
            color=POLAR,
            linestyle="-",
            marker="^",
            markerfacecolor="white",
            markeredgecolor=POLAR,
        ),
        "Adam": dict(color=COMPLEX, linestyle="-", marker="o"),
    }
    for name in ("SGD", "Adam + clip 1.0", "Adam"):
        kw = dict(styles[name])
        big = name == "Adam + clip 1.0"
        ax.plot(
            gnorms,
            rows[name],
            linewidth=0 if big else 1.2,
            markersize=6.0 if big else 3.4,
            label=name,
            markeredgewidth=0.9 if big else 0.6,
            **({"markeredgecolor": "white"} | kw),
        )
    ax.axhline(eta, color=INK, linewidth=0.6, linestyle=(0, (1, 2)))
    ax.text(
        gnorms[-1],
        eta * 0.55,
        "Adam and Adam + clipping coincide at $\\eta$",
        fontsize=6.0,
        color=INK,
        ha="right",
        va="top",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(
        ax,
        title="Size of the first update vs. gradient norm",
        xlabel="gradient norm at step 1",
        ylabel="max $|\\Delta w|$ at step 1",
    )
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    _save(fig, out, "adam_first_step")


# ---------------------------------------------------------------------------
def fig_u1(results: Path, out: Path) -> None:
    """Per-SNR decomposition of the complex advantage into constraint and readout.

    Uses independent per-family tuning. Complex and constrained-real select
    the same trial (asserted below), so they share the trial, the seeds and
    forward- and gradient-equivalent convolutions; the
    only difference between them is the classifier readout (|z| of a complex
    linear head versus a real linear head). Differences are paired by seed,
    and bands are +/- one standard error of the paired difference.
    """
    # Independent tuning is the fair rule for the constraint contrast: under
    # the shared trial the unconstrained baseline is destabilised by the
    # learning rate the complex model selected, which would re-measure the
    # instability confound instead of the constraint.
    s_ = _sweep(results, "radioml_u1_ablation")["independent"]
    same = (
        s_["complex"]["selected_trial_index"]
        == s_["real_equivariant"]["selected_trial_index"]
    )
    if not same:
        msg = (
            "complex and constrained-real selected different trials; the "
            "readout contrast would confound hyperparameters"
        )
        raise SystemExit(msg)

    def seeds(fam: str) -> tuple[list[int], list[list[float]]]:
        rows = s_[fam]["selected_extra"]["test_accuracy_by_snr_db_per_seed"]
        snrs = sorted(int(k) for k in rows[0])
        return snrs, [[r[str(k)] for k in snrs] for r in rows]

    def mean(rows: list[list[float]]) -> list[float]:
        return [statistics.fmean(col) for col in zip(*rows, strict=True)]

    def paired(a: str, b: str) -> tuple[list[float], list[float]]:
        _, ra = seeds(a)
        _, rb = seeds(b)
        diffs = [
            [100 * (x - y) for x, y in zip(sa, sb, strict=True)]
            for sa, sb in zip(ra, rb, strict=True)
        ]
        cols = list(zip(*diffs, strict=True))
        m = [statistics.fmean(c) for c in cols]
        se = [statistics.stdev(c) / math.sqrt(len(c)) for c in cols]
        return m, se

    fams = [
        ("real_matched_params", CONTEXT[0], (0, (3, 1.5)), "s"),
        ("real_polar", POLAR, "-", "D"),
        ("real_equivariant", "#56B4E9", "-", "^"),
        ("complex", COMPLEX, "-", "o"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for fam, color, ls, mk in fams:
        x, rows = seeds(fam)
        ax1.plot(
            x,
            mean(rows),
            color=color,
            linestyle=ls,
            marker=mk,
            linewidth=1.5,
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=FAMILY_LABEL[fam],
        )
    _style(
        ax1,
        title="(a) Accuracy by SNR, independent tuning",
        xlabel="SNR (dB)",
        ylabel="test accuracy",
    )
    ax1.legend(
        fontsize=5.8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
    )

    x, _ = seeds("complex")
    ax2.axhline(0, color=INK, linewidth=0.8)
    for (a, b), color, mk, ls, label in [
        (
            ("real_equivariant", "real_matched_params"),
            "#56B4E9",
            "^",
            "-",
            "$U(1)$ constraint: constrained $-$ unconstrained,\nsame parameter count",
        ),
        (
            ("complex", "real_equivariant"),
            COMPLEX,
            "o",
            (0, (3, 1.5)),
            "readout: $|z|$ head $-$ linear head,\n"
            "same selected trial, seeds and convolutions",
        ),
    ]:
        m, se = paired(a, b)
        ax2.fill_between(
            x,
            [u - v for u, v in zip(m, se, strict=True)],
            [u + v for u, v in zip(m, se, strict=True)],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax2.plot(
            x,
            m,
            color=color,
            marker=mk,
            linestyle=ls,
            linewidth=1.6,
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
    _style(
        ax2,
        title="(b) What each ingredient is worth, by SNR",
        xlabel="SNR (dB)",
        ylabel="paired difference (pp)",
    )
    ax2.legend(
        fontsize=5.8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
    )
    fig.tight_layout(w_pad=1.2)
    _save(fig, out, "u1_ablation")


# ---------------------------------------------------------------------------
def fig_lr(results: Path, out: Path) -> None:
    """The EEG complex-model deficit is stable over a 100x learning-rate range."""
    lrs = ["0.0003", "0.001", "0.003", "0.01", "0.03"]
    conds = [
        ("amplitude_event", "(a) EEG amplitude event"),
        ("phase_amplitude_coupling", "(b) EEG phase-amplitude coupling"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)
    for ax, (cond, title) in zip(axes, conds, strict=True):
        for fam, color, mk in (("real_polar", POLAR, "D"), ("complex", COMPLEX, "o")):
            ys = [
                _pilot(results, f"neuro_eeg_lr_{lr}", cond)[fam]["mean"] for lr in lrs
            ]
            ax.plot(
                [float(v) for v in lrs],
                ys,
                color=color,
                marker=mk,
                linewidth=1.6,
                markersize=3.6,
                markeredgecolor="white",
                label=FAMILY_LABEL[fam],
            )
        ax.axhline(0.25, color=MUTED, linewidth=0.6, linestyle=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        _style(
            ax,
            title=title,
            xlabel="learning rate",
            ylabel="test accuracy" if ax is axes[0] else "",
        )
    axes[0].legend(fontsize=6.2, frameon=False, loc="center left")
    fig.tight_layout(w_pad=0.8)
    _save(fig, out, "lr_robustness")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    fig_confounds(args.results, args.out)
    fig_activation(args.results, args.out)
    fig_dead_seeds(args.results, args.out)
    fig_instability(args.results, args.out)
    fig_u1(args.results, args.out)
    fig_lr(args.results, args.out)
    fig_adam_identity(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
