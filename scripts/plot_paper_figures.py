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


def _save(fig: Any, out: Path, name: str, *, crop: Any = None) -> None:
    """Save PDF and PNG. `crop` is an explicit bbox in inches for canvas-style
    figures, whose full-figure axes defeat matplotlib's tight bounding box."""
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            out / f"{name}.{ext}",
            dpi=400,
            bbox_inches="tight" if crop is None else crop,
        )
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


# Short-budget PSK-only accuracies as published in the preprint's stress-test
# table (complex, stacked real). They are the one input to this figure not
# re-derived from results/: the short-budget runs predate the manifest harness.
PREPRINT_SHORT_BUDGET_PSK = {"complex": 0.821, "real_stacked": 0.728}


def fig_overview(results: Path, out: Path) -> None:
    """One-figure summary of the paper, drawn at printed width (5.5 in)."""
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(5.5, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 0.95], wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) Waterfall on the seven-class headline configuration.
    head = _sweep(results, "radioml_geom7_crelu")
    reals = tuple(f for f in head["matched"] if f.startswith("real_"))
    zrelu = _sweep(results, "radioml_geom7_zrelu")
    values = [
        _gap(head["matched"], CARTESIAN),
        _gap(head["matched"], reals),
        _gap(head["independent"], reals),
        _gap(zrelu["independent"], reals),
    ]
    ax_a.axhline(0, color=INK, linewidth=0.7, zorder=1)
    for i, v in enumerate(values):
        lo, hi = (0.0, v) if i == 0 else sorted((values[i - 1], v))
        ax_a.add_patch(
            Rectangle(
                (i - 0.32, lo),
                0.64,
                hi - lo,
                facecolor=INK if i == 0 else CONTEXT[2],
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
        )
        if i:
            ax_a.plot(
                [i - 1 + 0.32, i - 0.32],
                [values[i - 1]] * 2,
                color=MUTED,
                linewidth=0.5,
                linestyle=(0, (1, 1.5)),
                zorder=1,
            )
        # First bar: value above its top. A tall drop: value just inside the bar,
        # above its new end. A short drop: value below its new end.
        if i == 0 or v > values[i - 1]:
            y, va = v + 0.35, "bottom"
        elif values[i - 1] - v > 2.0:
            y, va = v + 0.3, "bottom"
        else:
            y, va = v - 0.35, "top"
        ax_a.text(i, y, f"{v:+.1f}", ha="center", va=va, fontsize=6.3, color=INK)
    ax_a.set_xlim(-0.6, 3.6)
    ax_a.set_ylim(min(values) - 1.6, values[0] + 1.6)
    ax_a.set_xticks(range(4), ["as reported", "+ polar", "own tuning", "ZReLU"])
    _style(ax_a, title="(a) Remove the confounds", ylabel="complex $-$ best real (pp)")
    ax_a.tick_params(axis="x", labelsize=6.0, length=0)
    plt.setp(ax_a.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # (b) What survives: complex minus polar real, per SNR.
    sel = head["matched"]

    def per_snr(fam: str) -> dict[int, float]:
        rows = sel[fam]["selected_extra"]["test_accuracy_by_snr_db_per_seed"]
        return {int(k): statistics.fmean(r[k] for r in rows) for k in rows[0]}

    cx, po = per_snr("complex"), per_snr("real_polar")
    snrs = sorted(cx)
    gaps = [100 * (cx[k] - po[k]) for k in snrs]
    top, bot = max(gaps), min(gaps)
    ax_b.axvspan(2, 6, color="#EEF3F7", linewidth=0, zorder=0)
    ax_b.axhline(0, color=INK, linewidth=0.7, zorder=1)
    ax_b.bar(
        snrs,
        gaps,
        width=2.9,
        color=[COMPLEX if g > 0 else POLAR for g in gaps],
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    ax_b.text(
        -15.5,
        top + 1.2,
        "complex ahead:\nphase is noisy",
        fontsize=5.7,
        color=INK,
        ha="left",
        va="bottom",
    )
    ax_b.text(
        12,
        bot - 1.2,
        "polar ahead:\nsignal is clean",
        fontsize=5.7,
        color=INK,
        ha="center",
        va="top",
    )
    ax_b.text(
        4,
        top + 1.2,
        "cross-\nover",
        fontsize=5.4,
        color=MUTED,
        ha="center",
        va="bottom",
    )
    ax_b.set_ylim(bot - 7.0, top + 7.0)
    ax_b.set_xticks([-10, 0, 10])
    _style(
        ax_b,
        title="(b) What survives",
        xlabel="SNR (dB)",
        ylabel="complex $-$ polar (pp)",
    )

    # (c) Effect sizes: every non-architectural knob is larger.
    budget = _pilot(
        results, "rf_representation_stress_tests_full", "psk_representation"
    )
    short = PREPRINT_SHORT_BUDGET_PSK
    budget_effect = 100 * (
        (short["complex"] - short["real_stacked"])
        - (budget["complex"]["mean"] - budget["real_stacked"]["mean"])
    )
    swing = 0.0
    for cond in ("amplitude_event", "phase_amplitude_coupling"):
        accs = [
            _pilot(results, f"neuro_eeg_activation_{a}", cond)["complex"]["mean"]
            for a in ACTIVATIONS
        ]
        swing = max(swing, 100 * (max(accs) - min(accs)))
    rows = [
        ("activation choice (EEG)", swing, CONTEXT[0]),
        ("training budget (synthetic PSK)", budget_effect, CONTEXT[0]),
        ("protocol (RadioML, panel a)", values[0] - values[2], CONTEXT[0]),
        ("architecture, fairly tuned", abs(values[2]), COMPLEX),
    ]
    for i, (name, v, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax_c.barh(y, v, height=0.34, color=color, edgecolor="none", zorder=2)
        ax_c.plot([v], [y], marker="|", markersize=7, color=color, zorder=3)
        ax_c.text(
            v + 1.2,
            y,
            f"{v:.1f}",
            va="center",
            fontsize=6.3,
            color=COMPLEX if color == COMPLEX else INK,
        )
        ax_c.text(0, y + 0.27, name, va="bottom", ha="left", fontsize=5.9, color=INK)
    ax_c.set_yticks([])
    ax_c.set_ylim(-0.5, len(rows) - 0.2)
    ax_c.set_xlim(0, 75)
    _style(ax_c, title="(c) Effect sizes", xlabel="percentage points")
    ax_c.grid(True, axis="x", color=GRID, linewidth=0.5, linestyle=":")
    ax_c.grid(False, axis="y")
    for ax in (ax_a, ax_b, ax_c):
        ax.title.set_fontsize(7.2)
    _save(fig, out, "overview")


def fig_concept(out: Path) -> None:
    """Concept diagram for the introduction and background (no data).

    Drawn at printed width (5.5 in) on an inch-coordinate canvas so that every
    element is placed, not nudged. Panels: (a) one complex sample and the four
    real coordinate views of it; (b) the complex layer as the aI+bJ subspace of
    real two-channel mixings; (c) the Liouville trilemma as a 2x2 grid, with
    each activation labelled by the quantity it gates on.
    """
    import numpy as np
    from matplotlib.patches import Arc, FancyBboxPatch, Rectangle

    W, H = 5.5, 2.0
    fig = plt.figure(figsize=(W, H))
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, W)
    canvas.set_ylim(0, H)
    canvas.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        edge: str,
        fill: str,
        lw: float = 0.7,
        r: float = 0.05,
    ) -> None:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
            )
        )

    def text(
        x: float,
        y: float,
        s: str,
        *,
        size: float = 5.8,
        color: str = INK,
        ha: str = "left",
        va: str = "center",
    ) -> None:
        canvas.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, linespacing=1.28)

    def matrix(
        x: float,
        y: float,
        rows: list[list[str]],
        *,
        dx: float = 0.2,
        dy: float = 0.13,
        color: str = INK,
    ) -> None:
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                text(x + j * dx, y - i * dy, cell, size=6.3, color=color, ha="center")
        top, bot = y + dy * 0.55, y - (len(rows) - 1) * dy - dy * 0.55
        left, right = x - dx * 0.55, x + (len(rows[0]) - 1) * dx + dx * 0.55
        for xb, sgn in ((left, 1), (right, -1)):
            canvas.plot(
                [xb + sgn * 0.035, xb, xb, xb + sgn * 0.035],
                [top, top, bot, bot],
                color=color,
                lw=0.6,
            )

    for xs in (1.86, 3.70):
        canvas.plot([xs, xs], [0.1, 1.86], color=GRID, lw=0.6)

    # (a) Where does the label live? ----------------------------------------
    text(0.05, 1.93, "(a) Where does the label live?", size=7.2, va="top")
    text(
        0.05,
        1.71,
        "RF IQ · Fourier · quantum $\\psi$ · EEG analytic",
        size=5.2,
        color=MUTED,
    )
    plane = fig.add_axes((0.04 / W, 0.84 / H, 0.9 / W, 0.76 / H))
    plane.set_xlim(-0.22, 1.25)
    plane.set_ylim(-0.2, 1.05)
    plane.axis("off")
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=5)
    plane.annotate("", xy=(1.2, 0), xytext=(-0.18, 0), arrowprops=arrow)
    plane.annotate("", xy=(0, 1.02), xytext=(0, -0.18), arrowprops=arrow)
    plane.text(1.2, -0.05, "Re", fontsize=5.4, color=MUTED, ha="right", va="top")
    plane.text(0.05, 1.02, "Im", fontsize=5.4, color=MUTED, ha="left", va="top")
    th = np.deg2rad(36)
    zx, zy = 0.95 * np.cos(th), 0.95 * np.sin(th)
    dots: dict[str, Any] = dict(color=MUTED, lw=0.5, ls=(0, (1.5, 1.5)))
    plane.plot([zx, zx], [0, zy], **dots)
    plane.plot([0, zx], [zy, zy], **dots)
    plane.plot([0, zx], [0, zy], color=COMPLEX, lw=1.3)
    plane.plot(zx, zy, "o", ms=3.4, color=COMPLEX)
    plane.add_patch(Arc((0, 0), 0.52, 0.52, theta1=0, theta2=36, color=POLAR, lw=1.0))
    plane.text(0.31, 0.06, r"$\theta$", fontsize=6.4, color=POLAR)
    plane.text(zx * 0.42 - 0.1, zy * 0.42 + 0.07, "$r$", fontsize=6.4, color=COMPLEX)
    plane.text(
        zx,
        zy + 0.09,
        r"$z = x+iy = re^{i\theta}$",
        fontsize=5.8,
        color=INK,
        ha="center",
        va="bottom",
    )
    plane.text(zx, -0.05, "$x$", fontsize=5.6, color=MUTED, ha="center", va="top")
    plane.text(-0.04, zy, "$y$", fontsize=5.6, color=MUTED, ha="right", va="center")
    text(1.02, 1.12, "a real model is\ngiven one view:", size=5.3, color=MUTED)
    views = [
        ("Cartesian  $(x, y)$", CONTEXT[0]),
        (r"polar  $(r, \cos\theta, \sin\theta)$", POLAR),
        (r"phase only  $(\cos\theta, \sin\theta)$", CONTEXT[0]),
        ("magnitude only  $r$", CONTEXT[0]),
    ]
    for i, (label, edge) in enumerate(views):
        y = 0.64 - i * 0.165
        box(0.08, y, 1.64, 0.135, edge=edge, fill="white")
        text(0.9, y + 0.0675, label, ha="center")

    # (b) A complex layer is a constrained real layer ------------------------
    x0 = 1.93
    text(x0, 1.93, "(b) A constrained real layer", size=7.2, va="top")
    box(x0 + 0.02, 0.36, 1.66, 1.36, edge=CONTEXT[1], fill="#F4F4F4", r=0.08)
    text(x0 + 0.1, 1.61, r"$\mathbb{R}^{2\times2}$: every two-channel mixing")
    matrix(x0 + 0.27, 1.41, [["$p$", "$q$"], ["$r$", "$s$"]])
    text(x0 + 0.66, 1.345, "4 parameters per tap", size=5.4, color=MUTED)
    box(x0 + 0.34, 0.44, 1.26, 0.74, edge=COMPLEX, fill="#E3EEF6", lw=0.9, r=0.06)
    text(x0 + 0.78, 1.0, "$aI+bJ\\,=$", size=6.3, ha="right")
    matrix(x0 + 0.95, 1.065, [["$a$", "$-b$"], ["$b$", "$a$"]], dx=0.24, color=COMPLEX)
    text(
        x0 + 0.97,
        0.66,
        "multiplication by $a+ib$\ncommutes with every $R_\\phi$\n2 parameters per tap",
        size=5.4,
        ha="center",
    )
    text(
        x0 + 0.85,
        0.2,
        "fewer degrees of freedom: pays when phase is\n"
        "noisy, costs once the signal is clean",
        size=5.2,
        color=MUTED,
        ha="center",
    )

    # (c) What an activation can be: the Liouville trilemma ------------------
    x0 = 3.77
    text(x0, 1.93, "(c) What an activation can be", size=7.2, va="top")
    gx, gy, cw, ch = x0 + 0.5, 0.36, 0.6, 0.52
    text(gx + 0.5 * cw, gy + 2 * ch + 0.06, "holomorphic", ha="center", va="bottom")
    text(
        gx + 1.5 * cw, gy + 2 * ch + 0.06, "not\nholomorphic", ha="center", va="bottom"
    )
    text(gx - 0.05, gy + 1.5 * ch, "bounded", ha="right")
    text(gx - 0.05, gy + 0.5 * ch, "unbounded", ha="right")
    cells = {
        (0, 1): ("none\n(Liouville)", True),
        (1, 1): ("Siglog  $|z|$", False),
        (0, 0): ("tanh\n(poles)", False),
        (1, 0): (
            "CReLU  Re, Im\nZReLU  quadrant\nModReLU  $|z|$\nCardioid  $\\theta$",
            False,
        ),
    }
    for (cx, cy), (label, forbidden) in cells.items():
        canvas.add_patch(
            Rectangle(
                (gx + cx * cw, gy + cy * ch),
                cw,
                ch,
                facecolor="#EFEFEF" if forbidden else "white",
                edgecolor=CONTEXT[1],
                linewidth=0.7,
                hatch="////" if forbidden else None,
            )
        )
        if forbidden:
            canvas.add_patch(
                Rectangle(
                    (gx + cx * cw + 0.07, gy + cy * ch + 0.13),
                    cw - 0.14,
                    ch - 0.26,
                    facecolor="#EFEFEF",
                    edgecolor="none",
                )
            )
        text(
            gx + (cx + 0.5) * cw,
            gy + (cy + 0.5) * ch,
            label,
            size=5.0 if (cx, cy) == (1, 0) else 5.6,
            color=MUTED if forbidden else INK,
            ha="center",
        )
    text(
        gx + cw,
        0.2,
        "each labelled by what it gates on",
        size=5.2,
        color=MUTED,
        ha="center",
    )
    _save(fig, out, "concept")


def fig_workflow(results: Path, out: Path) -> None:
    """Cover-style workflow diagram of the paper, for the introduction.

    Four tinted stages read left to right: (1) complex-valued data reduce to
    one sample z; (2) it is given to a complex network, whose layers are the
    U(1)-equivariant aI+bJ subspace, and to real networks given one coordinate
    view; (3) the comparison is controlled, each control annotated with its
    measured effect, beside a before/after gauge of the RadioML gap; (4) what
    survives is a crossover, drawn from committed per-SNR accuracies. Blue and
    vermillion are reserved for the complex and polar-real families.
    """
    import numpy as np
    from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle
    from matplotlib.transforms import Bbox
    from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

    W, H = 5.5, 2.0
    fig = plt.figure(figsize=(W, H))
    cv = fig.add_axes((0, 0, 1, 1))
    cv.set_xlim(0, W)
    cv.set_ylim(0, H)
    cv.axis("off")
    PANEL, PANEL_EDGE, HAIR = "#F3F6FA", "#DFE5EC", "#CBD3DC"
    BLUE_TINT, CHEV = "#E2EEF8", "#A9C3DA"

    def text(
        x: float,
        y: float,
        s: str,
        *,
        size: float = 5.3,
        color: str = INK,
        ha: str = "left",
        va: str = "center",
        weight: str = "normal",
        z: float = 8,
    ) -> None:
        cv.text(
            x,
            y,
            s,
            fontsize=size,
            color=color,
            ha=ha,
            va=va,
            fontweight=weight,
            linespacing=1.18,
            zorder=z,
        )

    def rbox(
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        edge: str,
        lw: float = 0.6,
        r: float = 0.05,
        ls: Any = "solid",
        z: float = 2,
    ) -> None:
        cv.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                linestyle=ls,
                zorder=z,
            )
        )

    def stage(x: float, w: float, n: int, title: str) -> None:
        rbox(x, 0.16, w, 1.66, fill=PANEL, edge=PANEL_EDGE, r=0.07, z=1)
        cv.add_patch(
            Circle(
                (x + 0.13, 1.68), 0.062, facecolor=COMPLEX, edgecolor="none", zorder=3
            )
        )
        text(
            x + 0.13, 1.68, str(n), size=5.4, color="white", ha="center", weight="bold"
        )
        text(x + 0.23, 1.68, title, size=6.1, weight="bold")

    def chevron(x: float) -> None:
        cv.add_patch(
            Polygon(
                [(x - 0.03, 1.04), (x + 0.03, 0.99), (x - 0.03, 0.94)],
                closed=True,
                facecolor=CHEV,
                edgecolor="none",
                zorder=2,
            )
        )

    def bracket_matrix(x: float, y: float, rows: list[list[str]], color: str) -> None:
        dx, dy = 0.15, 0.1
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                text(x + j * dx, y - i * dy, cell, size=5.3, color=color, ha="center")
        top, bot = y + dy * 0.6, y - dy * 1.6
        for xb, sgn in ((x - dx * 0.6, 1), (x + dx * 1.6, -1)):
            cv.plot(
                [xb + sgn * 0.025, xb, xb, xb + sgn * 0.025],
                [top, top, bot, bot],
                color=color,
                lw=0.55,
                zorder=8,
            )

    P1, P2, P3, P4 = (0.05, 1.02), (1.17, 1.3), (2.57, 1.72), (4.39, 1.06)
    stage(*P1, 1, "Complex data")
    stage(*P2, 2, "Two model families")
    stage(*P3, 3, "Control the comparison")
    stage(*P4, 4, "What survives")
    for x in (1.12, 2.52, 4.34):
        chevron(x)

    # ---- 1. data tiles and the one sample they share --------------------
    t = np.linspace(-1, 1, 200)
    tiles = [
        (0.11, 1.12, "RF IQ"),
        (0.59, 1.12, "quantum $\\psi$"),
        (0.11, 0.68, "EEG"),
        (0.59, 0.68, "Fourier"),
    ]
    for i, (x, y, label) in enumerate(tiles):
        rbox(x, y, 0.42, 0.33, fill="white", edge=PANEL_EDGE, r=0.04, z=3)
        ax = fig.add_axes(((x + 0.05) / W, (y + 0.05) / H, 0.32 / W, 0.23 / H))
        ax.axis("off")
        if i == 0:
            a = np.arange(8) * np.pi / 4 + np.pi / 8
            ax.plot(np.cos(a), np.sin(a), "o", ms=1.7, color=INK)
            ax.set_xlim(-1.9, 1.9)
            ax.set_ylim(-1.35, 1.35)
        elif i == 1:
            ax.plot(t, np.exp(-6 * t**2) * np.cos(18 * t), color=INK, lw=0.6)
            ax.set_ylim(-1.2, 1.2)
        elif i == 2:
            env = 0.35 + 0.65 * np.exp(-12 * (t - 0.15) ** 2)
            ax.plot(t, env * np.sin(30 * t), color=INK, lw=0.55)
            ax.set_ylim(-1.2, 1.2)
        else:
            ax.bar(
                np.arange(7),
                [0.3, 0.9, 0.55, 0.75, 0.25, 0.4, 0.15],
                width=0.62,
                color=INK,
            )
            ax.set_ylim(0, 1)
        text(x + 0.21, y - 0.06, label, size=4.9, color=MUTED, ha="center")
    zc = (0.26, 0.36)
    cv.add_patch(Circle(zc, 0.1, facecolor="white", edgecolor=INK, lw=0.7, zorder=4))
    th = np.deg2rad(38)
    cv.plot(
        [zc[0], zc[0] + 0.085 * np.cos(th)],
        [zc[1], zc[1] + 0.085 * np.sin(th)],
        color=INK,
        lw=0.9,
        zorder=5,
        solid_capstyle="round",
    )
    cv.plot(
        zc[0] + 0.085 * np.cos(th),
        zc[1] + 0.085 * np.sin(th),
        "o",
        ms=1.8,
        color=INK,
        zorder=5,
    )
    text(0.41, 0.41, "one sample", size=5.0, color=MUTED)
    text(0.41, 0.3, "$z = re^{i\\theta}$", size=5.9)

    # ---- 2. the two families --------------------------------------------
    x2 = P2[0] + 0.07
    rbox(x2, 0.98, 1.16, 0.56, fill=BLUE_TINT, edge=COMPLEX, lw=0.8, z=3)
    text(x2 + 0.07, 1.43, "complex network", size=5.9, weight="bold", color=COMPLEX)
    text(x2 + 0.07, 1.25, "layers are $aI{+}bJ$", size=5.1)
    text(x2 + 0.07, 1.11, "$U(1)$-equivariant", size=4.9, color=MUTED)
    bracket_matrix(x2 + 0.87, 1.27, [["$a$", "$-b$"], ["$b$", "$a$"]], COMPLEX)
    rbox(x2, 0.23, 1.16, 0.68, fill="white", edge=HAIR, z=3)
    text(x2 + 0.07, 0.8, "real network, one view", size=5.9, weight="bold")
    chips = [
        ("$(x,y)$", HAIR),
        ("$(r,\\cos\\theta,\\sin\\theta)$", POLAR),
        ("$(\\cos\\theta,\\sin\\theta)$", HAIR),
        ("$r$", HAIR),
    ]
    for i, (label, edge) in enumerate(chips):
        cx, cy = x2 + 0.07 + (i % 2) * 0.53, 0.56 - (i // 2) * 0.17
        rbox(
            cx,
            cy,
            0.49,
            0.13,
            fill="white",
            edge=edge,
            lw=0.9 if edge == POLAR else 0.6,
            r=0.035,
            z=4,
        )
        text(
            cx + 0.245,
            cy + 0.065,
            label,
            size=4.8,
            color=POLAR if edge == POLAR else INK,
            ha="center",
        )
    text(x2 + 0.07, 0.3, "+ parameter- & FLOP-matched", size=4.6, color=MUTED)

    # ---- 3. control the comparison ---------------------------------------
    head = _sweep(results, "radioml_geom7_crelu")
    reals = tuple(fam for fam in head["matched"] if fam.startswith("real_"))
    g_cart = _gap(head["matched"], CARTESIAN)
    g_all = _gap(head["matched"], reals)
    g_ind = _gap(head["independent"], reals)
    conv = _pilot(results, "rf_representation_stress_tests_full", "psk_representation")
    short = PREPRINT_SHORT_BUDGET_PSK
    d_budget = 100 * (
        (conv["complex"]["mean"] - conv["real_stacked"]["mean"])
        - (short["complex"] - short["real_stacked"])
    )
    swing = max(
        100 * (max(acc) - min(acc))
        for acc in (
            [
                _pilot(results, f"neuro_eeg_activation_{act}", cond)["complex"]["mean"]
                for act in ACTIVATIONS
            ]
            for cond in ("amplitude_event", "phase_amplitude_coupling")
        )
    )

    def icon_polar(cx: float, cy: float) -> None:
        cv.add_patch(
            Arc(
                (cx - 0.035, cy - 0.035),
                0.11,
                0.11,
                theta1=0,
                theta2=90,
                color=POLAR,
                lw=0.8,
                zorder=9,
            )
        )
        cv.plot(
            [cx - 0.035, cx + 0.03],
            [cy - 0.035, cy + 0.02],
            color=POLAR,
            lw=0.8,
            zorder=9,
            solid_capstyle="round",
        )

    def icon_tune(cx: float, cy: float) -> None:
        for dx, knob in ((-0.025, 0.02), (0.025, -0.022)):
            cv.plot(
                [cx + dx, cx + dx],
                [cy - 0.045, cy + 0.045],
                color=MUTED,
                lw=0.6,
                zorder=9,
            )
            cv.plot(cx + dx, cy + knob, "o", ms=2.4, color=INK, zorder=9)

    def icon_converge(cx: float, cy: float) -> None:
        u = np.linspace(0, 1, 40)
        cv.plot(
            cx - 0.05 + 0.1 * u,
            cy - 0.035 + 0.075 * np.exp(-4 * u),
            color=INK,
            lw=0.8,
            zorder=9,
        )

    def icon_act(cx: float, cy: float) -> None:
        cv.plot(
            [cx - 0.05, cx, cx + 0.045],
            [cy - 0.02, cy - 0.02, cy + 0.035],
            color=MUTED,
            lw=0.8,
            zorder=9,
        )

    rows = [
        (
            icon_polar,
            "Add a polar real baseline",
            f"{g_all - g_cart:+.1f} pp",
            "RadioML",
        ),
        (icon_tune, "Tune each family alone", f"{g_ind - g_all:+.1f} pp", "RadioML"),
        (icon_converge, "Train to convergence", f"{d_budget:+.1f} pp", "synthetic PSK"),
    ]
    x3 = P3[0] + 0.08
    text(
        x3 + 0.02,
        1.5,
        "$\\Delta$ = complex $-$ best real baseline",
        size=4.8,
        color=MUTED,
    )
    for k, (icon, name, eff, task) in enumerate(rows):
        y = 1.27 - k * 0.285
        cv.add_patch(
            Circle(
                (x3 + 0.08, y),
                0.075,
                facecolor="white",
                edgecolor=HAIR,
                lw=0.6,
                zorder=8,
            )
        )
        icon(x3 + 0.08, y)
        text(x3 + 0.2, y + 0.055, name, size=5.0)
        text(
            x3 + 0.2,
            y - 0.06,
            "Δ " + eff.replace("-", "−"),
            size=5.0,
            color=COMPLEX,
            weight="bold",
        )
        text(x3 + 0.64, y - 0.06, task, size=4.6, color=MUTED)
    ya = 0.4
    rbox(
        x3,
        ya - 0.16,
        1.04,
        0.3,
        fill="none",
        edge=MUTED,
        lw=0.55,
        r=0.04,
        ls=(0, (2, 1.4)),
        z=7,
    )
    cv.add_patch(
        Circle(
            (x3 + 0.08, ya), 0.075, facecolor="white", edgecolor=HAIR, lw=0.6, zorder=8
        )
    )
    icon_act(x3 + 0.08, ya)
    text(x3 + 0.2, ya + 0.055, "Activation choice", size=5.0)
    text(
        x3 + 0.2, ya - 0.06, f"design axis, up to {swing:.0f} pp", size=4.8, color=MUTED
    )

    # before/after gauge of the RadioML gap
    gx, zero, scale = x3 + 1.2, 0.62, 0.075
    text(gx + 0.2, 1.37, "$\\Delta$, RadioML", size=4.7, color=MUTED, ha="center")
    cv.plot([gx - 0.02, gx + 0.42], [zero, zero], color=INK, lw=0.6, zorder=8)
    for j, (val, lab, color) in enumerate(
        ((g_cart, "before", COMPLEX), (g_ind, "after", CONTEXT[1]))
    ):
        bx = gx + 0.01 + j * 0.23
        lo, hi = sorted((zero, zero + val * scale))
        cv.add_patch(
            Rectangle(
                (bx, lo),
                0.15,
                max(hi - lo, 0.012),
                facecolor=color,
                edgecolor="none",
                zorder=8,
            )
        )
        text(
            bx + 0.075,
            (hi if val > 0 else lo) + (0.04 if val > 0 else -0.04),
            f"{val:+.1f}".replace("-", "−"),
            size=5.2,
            weight="bold",
            color=INK,
            ha="center",
            va="bottom" if val > 0 else "top",
        )
        text(bx + 0.075, zero - 0.16, lab, size=4.6, color=MUTED, ha="center", va="top")

    # ---- 4. what survives: two accuracy curves that cross ----------------
    sel = head["matched"]

    def per_snr(fam: str) -> tuple[np.ndarray, np.ndarray]:
        rs = sel[fam]["selected_extra"]["test_accuracy_by_snr_db_per_seed"]
        ks = sorted(int(k) for k in rs[0])
        return np.array(ks, float), np.array(
            [statistics.fmean(r[str(k)] for r in rs) for k in ks]
        )

    snr, acc_c = per_snr("complex")
    _, acc_p = per_snr("real_polar")
    xx = np.linspace(snr[0], snr[-1], 300)
    yc_, yp_ = PchipInterpolator(snr, acc_c)(xx), PchipInterpolator(snr, acc_p)(xx)
    ax = fig.add_axes(((P4[0] + 0.14) / W, 0.52 / H, (P4[1] - 0.22) / W, 0.95 / H))
    ax.fill_between(
        xx, yc_, yp_, where=yc_ >= yp_, color=COMPLEX, alpha=0.2, lw=0, interpolate=True
    )
    ax.fill_between(
        xx, yc_, yp_, where=yc_ < yp_, color=POLAR, alpha=0.2, lw=0, interpolate=True
    )
    ax.plot(xx, yp_, color=POLAR, lw=1.3, solid_capstyle="round")
    ax.plot(xx, yc_, color=COMPLEX, lw=1.3, solid_capstyle="round")
    cross = xx[np.argmax((yc_ < yp_) & (xx > -5))]
    ax.plot(
        cross,
        PchipInterpolator(snr, acc_c)(cross),
        "o",
        ms=2.8,
        color="white",
        mec=INK,
        mew=0.7,
        zorder=5,
    )
    ax.text(-15, 0.52, "complex", fontsize=5.0, color=COMPLEX, fontweight="bold")
    ax.text(
        cross + 1.4,
        float(PchipInterpolator(snr, acc_c)(cross)) - 0.06,
        "crossover",
        fontsize=4.6,
        color=MUTED,
        ha="left",
        va="top",
    )
    ax.text(
        18.5,
        float(yp_[-1]) + 0.035,
        "polar real",
        fontsize=5.0,
        color=POLAR,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    ax.set_xlim(-15, 19)
    ax.set_ylim(0.08, 0.76)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIR)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks([])
    ax.set_xticks([-10, 0, 10])
    ax.tick_params(
        axis="x", labelsize=4.6, colors=MUTED, length=1.8, width=0.5, pad=1.5
    )
    ax.set_facecolor("none")
    text(P4[0] + P4[1] / 2 + 0.03, 0.36, "SNR (dB)", size=4.7, color=MUTED, ha="center")
    text(
        P4[0] + 0.1,
        0.29,
        "complex: low SNR",
        size=4.7,
        color=COMPLEX,
        weight="bold",
    )
    text(
        P4[0] + 0.1,
        0.21,
        "polar real: high SNR",
        size=4.7,
        color=POLAR,
        weight="bold",
    )
    _save(fig, out, "workflow", crop=Bbox([[0.02, 0.13], [W - 0.02, 1.85]]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    fig_workflow(args.results, args.out)
    fig_concept(args.out)
    fig_overview(args.results, args.out)
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
