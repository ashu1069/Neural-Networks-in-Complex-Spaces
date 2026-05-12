"""Headline narrative figures for the paper.

Four self-contained, publication-ready plots:

1. fig_selection_rule_contrast      — bar chart: matched-shared-trial vs.
                                       independent selection (22.94 pp -> 2.46 pp).
2. fig_lr_activation_disambiguation — 2x2 heatmap of dead-seed counts
                                       and step-1 head.weight gradients.
3. fig_dead_seed_mechanism          — per-step trajectories of head.weight
                                       gradient and train-loss for alive vs
                                       dead seeds.
4. fig_paper_at_a_glance            — composite 2x2 summary that maps every
                                       finding into a single image.

All figures are written to results/figures/ and paper/figures/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR_RESULTS = RESULTS / "figures"
FIG_DIR_PAPER = ROOT / "paper" / "figures"
for d in (FIG_DIR_RESULTS, FIG_DIR_PAPER):
    d.mkdir(parents=True, exist_ok=True)

CMAP_GAP = "RdYlGn_r"
COMPLEX_COLOR = "#1f77b4"
REAL_COLOR = "#d62728"
SAFE_COLOR = "#2ca02c"
UNSAFE_COLOR = "#c0392b"

DEAD_LOSS_THRESHOLD = math.log(3) - 0.05


def _save(fig: plt.Figure, name: str) -> Path:
    out_results = FIG_DIR_RESULTS / f"{name}.png"
    out_paper = FIG_DIR_PAPER / f"{name}.png"
    fig.savefig(out_results, dpi=200, bbox_inches="tight")
    fig.savefig(out_paper, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_paper


# ---------------------------------------------------------------- 1. selection rule
def fig_selection_rule_contrast() -> Path:
    matched = {"complex": 0.7293, "best_real": 0.4999}
    independent = {"complex": 0.7293, "best_real": 0.7047}
    gap_matched = (matched["complex"] - matched["best_real"]) * 100
    gap_indep = (independent["complex"] - independent["best_real"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.42})

    # Left: paired bars per selection rule.
    ax = axes[0]
    rules = ["matched-shared-trial", "independent per-family"]
    cvnn_acc = [matched["complex"], independent["complex"]]
    real_acc = [matched["best_real"], independent["best_real"]]
    x = np.arange(len(rules))
    w = 0.36
    ax.bar(x - w / 2, cvnn_acc, w, color=COMPLEX_COLOR, label="CVNN", edgecolor="white")
    ax.bar(
        x + w / 2, real_acc, w, color=REAL_COLOR, label="best real", edgecolor="white"
    )
    for i, (c, r) in enumerate(zip(cvnn_acc, real_acc, strict=False)):
        ax.text(i - w / 2, c + 0.012, f"{c:.3f}", ha="center", fontsize=10)
        ax.text(i + w / 2, r + 0.012, f"{r:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(rules)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Test accuracy (3-class RadioML subset)")
    ax.set_title("CVNN vs. best real, by hyperparameter selection rule")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Right: gap collapse arrow.
    ax = axes[1]
    bars = ax.bar(
        ["matched-shared-trial", "independent per-family"],
        [gap_matched, gap_indep],
        color=[UNSAFE_COLOR, SAFE_COLOR],
        edgecolor="white",
        width=0.55,
    )
    for bar, value in zip(bars, [gap_matched, gap_indep], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            f"+{value:.2f} pp",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.annotate(
        "9.3× collapse",
        xy=(1, gap_indep + 0.4),
        xytext=(0.5, (gap_matched + gap_indep) / 2 + 4),
        ha="center",
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="#444"),
    )
    ax.set_ylim(0, gap_matched * 1.18)
    ax.set_ylabel("CVNN − best-real gap (pp)")
    ax.set_title("Same data, same trials — selection rule changes the headline")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Methodological finding: the 23 pp gap on CReLU is selection-rule-dependent",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, "headline_selection_rule_contrast")


# ---------------------------------------------------------------- 2. lr-vs-activation
def _load_disambiguation():
    idx = json.loads(
        (RESULTS / "lr_activation_disambiguation" / "index.json").read_text()
    )
    cells: dict[str, dict] = {}
    for r in idx:
        cells.setdefault(r["cell"], {"runs": [], "lr": r["lr"]})
        cells[r["cell"]]["runs"].append(r)
    # Step-1 head.weight gradient (max across the 9 real runs).
    for _cell_name, info in cells.items():
        head_grads = []
        final_losses = []
        for r in info["runs"]:
            with open(ROOT / r["output_path"]) as f:
                lines = f.readlines()
            step1 = json.loads(lines[2])
            head_grads.append(step1["per_layer_grad_norm"]["head.weight"])
            final_losses.append(json.loads(lines[0])["summary"]["final_train_loss"])
        info["step1_head_max"] = max(head_grads)
        info["step1_head_mean"] = float(np.mean(head_grads))
        info["dead"] = sum(1 for fl in final_losses if fl >= DEAD_LOSS_THRESHOLD)
        info["n"] = len(final_losses)
    return cells


def fig_lr_activation_disambiguation() -> Path:
    cells = _load_disambiguation()
    # Lay out as 2x2: rows=activation, cols=lr.
    activations = ["crelu", "zrelu"]
    lrs = [0.0024, 0.0236]
    cell_lookup = {
        ("crelu", 0.0236): cells["crelu_highlr"],
        ("crelu", 0.0024): cells["crelu_lowlr"],
        ("zrelu", 0.0024): cells["zrelu_lowlr"],
        ("zrelu", 0.0236): cells["zrelu_highlr"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.32})

    # Left: dead-seed heatmap.
    ax = axes[0]
    grid_dead = np.array(
        [[cell_lookup[(a, lr)]["dead"] for lr in lrs] for a in activations],
        dtype=float,
    )
    im = ax.imshow(grid_dead, cmap=CMAP_GAP, vmin=0, vmax=9, aspect="auto")
    for i, a in enumerate(activations):
        for j, lr in enumerate(lrs):
            cell = cell_lookup[(a, lr)]
            color = "white" if grid_dead[i, j] >= 4 else "black"
            ax.text(
                j,
                i,
                f"{cell['dead']}/{cell['n']} dead",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"lr = {lr}" for lr in lrs])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels(
        [f"\\code{{{a}}}".replace("\\code{", "").replace("}", "") for a in activations]
    )
    ax.set_yticklabels(activations)
    ax.set_title("Dead seeds (out of 9 real runs)")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("activation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("dead seeds")

    # Annotate diagonals as "matched-shared-trial regimes".
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5), 1, 1, fill=False, edgecolor="black", lw=2.5, ls="--"
        )
    )
    ax.add_patch(
        plt.Rectangle((0.5, 0.5), 1, 1, fill=False, edgecolor="black", lw=2.5, ls="--")
    )
    # Smaller in-cell tag (top-left corner) so it doesn't run into axis labels.
    ax.text(
        -0.45,
        -0.42,
        "matched",
        ha="left",
        va="top",
        fontsize=8,
        color="black",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.5),
    )
    ax.text(
        1.45,
        1.42,
        "matched",
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.5),
    )

    # Right: step-1 head.weight grad heatmap (log).
    ax = axes[1]
    grid_grad = np.array(
        [[cell_lookup[(a, lr)]["step1_head_max"] for lr in lrs] for a in activations]
    )
    im = ax.imshow(np.log10(grid_grad), cmap="magma", aspect="auto", vmin=-1, vmax=1.6)
    for i, _a in enumerate(activations):
        for j, _lr in enumerate(lrs):
            v = grid_grad[i, j]
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if np.log10(v) > 0.4 else "black",
            )
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"lr = {lr}" for lr in lrs])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels(activations)
    ax.set_title("Step-1 head.weight gradient (max over 9 runs)")
    ax.set_xlabel("learning rate")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label(r"$\log_{10}\,\|\nabla_{\mathrm{head.weight}}L\|$")

    fig.suptitle(
        "Lr-vs-activation disambiguation: the threshold is primarily lr-driven",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, "headline_lr_activation_disambiguation")


# ---------------------------------------------------------------- 3. mechanism
def _load_jsonl(path: Path):
    with open(path) as f:
        lines = f.readlines()
    summary = json.loads(lines[0])["summary"]
    steps = [json.loads(line) for line in lines[1:]]
    return summary, steps


def fig_dead_seed_mechanism() -> Path:
    """Per-step head.weight gradient and loss for alive vs dead real-baseline seeds.

    Pull from the existing synthetic_rf_telemetry/crelu/ runs (where
    matched-shared-trial selection produces dead seeds) plus
    synthetic_rf_telemetry/zrelu/ as the stable reference.
    """

    def collect(act: str):
        traces = []
        for fam in ["real_stacked", "real_matched_params", "real_matched_flops"]:
            for seed in [0, 1, 2]:
                p = RESULTS / "synthetic_rf_telemetry" / act / f"{fam}_seed{seed}.jsonl"
                summary, steps = _load_jsonl(p)
                head = [s["per_layer_grad_norm"]["head.weight"] for s in steps]
                loss = [s["train_loss"] for s in steps]
                step_idx = [s["step"] for s in steps]
                dead = summary["final_train_loss"] >= DEAD_LOSS_THRESHOLD
                traces.append(
                    {
                        "step": step_idx,
                        "head": head,
                        "loss": loss,
                        "dead": dead,
                        "seed": seed,
                        "fam": fam,
                    }
                )
        return traces

    crelu = collect("crelu")
    zrelu = collect("zrelu")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0), sharex=True)

    for col, (act_name, traces, _lr_label) in enumerate(
        [
            ("crelu @ lr=0.0236 (matched)", crelu, "high lr"),
            ("zrelu @ lr=0.0024 (matched)", zrelu, "low lr"),
        ]
    ):
        ax_g, ax_l = axes[0, col], axes[1, col]
        for t in traces:
            color = UNSAFE_COLOR if t["dead"] else "#888"
            alpha = 0.95 if t["dead"] else 0.5
            lw = 2.0 if t["dead"] else 1.2
            ax_g.plot(t["step"][:30], t["head"][:30], color=color, alpha=alpha, lw=lw)
            ax_l.plot(t["step"][:30], t["loss"][:30], color=color, alpha=alpha, lw=lw)

        ax_g.set_yscale("log")
        ax_g.set_title(act_name, fontsize=12)
        ax_g.set_ylabel(r"$\|\nabla_{\mathrm{head.weight}} L\|$  (log)")
        ax_g.grid(True, which="both", linestyle=":", alpha=0.4)
        ax_g.axhline(1.0, color="#444", lw=0.5, ls="--")

        ax_l.axhline(
            DEAD_LOSS_THRESHOLD,
            color="#444",
            lw=0.8,
            ls=":",
            label=f"dead thr (ln 3 ≈ {math.log(3):.2f})",
        )
        ax_l.set_ylabel("train loss")
        ax_l.set_xlabel("step")
        ax_l.grid(True, linestyle=":", alpha=0.4)
        if col == 0:
            ax_l.legend(loc="upper right", fontsize=9, frameon=False)

    # Custom legend for line colors.
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=UNSAFE_COLOR, lw=2, label="dead seed"),
        Line2D([0], [0], color="#888", lw=1.5, label="alive seed"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        "Step-1 head-gradient explosion → dead-ReLU region (real baselines, synthetic AWGN)",  # noqa: E501
        fontsize=13,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout()
    return _save(fig, "headline_dead_seed_mechanism")


# ---------------------------------------------------------------- 4. paper-at-a-glance
def fig_paper_at_a_glance() -> Path:
    """Single 2x2 image summarizing the paper's four headline findings."""
    fig = plt.figure(figsize=(13.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # ---- (A) representation conditionality ---------------------------
    ax = fig.add_subplot(gs[0, 0])
    conditions = [
        "PSK",
        "QAM",
        "mixed",
        "low SNR\nPSK",
        "high SNR\nPSK",
        "fixed-rot.\nPSK",
        "rot-aug.\nPSK",
    ]
    complex_acc = [0.821, 0.509, 0.507, 0.526, 0.949, 0.252, 0.654]
    real_best = [0.735, 0.524, 0.487, 0.537, 0.952, 0.328, 0.580]
    x = np.arange(len(conditions))
    w = 0.4
    ax.bar(x - w / 2, complex_acc, w, color=COMPLEX_COLOR, label="complex")
    ax.bar(x + w / 2, real_best, w, color=REAL_COLOR, label="best real")
    for i, (c, r) in enumerate(zip(complex_acc, real_best, strict=False)):
        winner = "C" if c > r else "R"
        col = COMPLEX_COLOR if winner == "C" else REAL_COLOR
        ax.text(
            i,
            max(c, r) + 0.025,
            winner,
            ha="center",
            color=col,
            fontweight="bold",
            fontsize=11,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("test accuracy")
    ax.set_title(
        "(A) Complex helps when phase/amplitude geometry matches",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # ---- (B) selection rule contrast --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    rules = ["matched-shared-trial", "independent per-family"]
    gap = [22.94, 2.46]
    colors = [UNSAFE_COLOR, SAFE_COLOR]
    bars = ax.bar(rules, gap, color=colors, width=0.55)
    for b, g in zip(bars, gap, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            g + 0.5,
            f"+{g:.2f} pp",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_ylabel("CVNN − best-real gap (pp)")
    ax.set_ylim(0, max(gap) * 1.2)
    ax.set_title(
        "(B) Same trials, two rules — gap collapses 9.3×",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # ---- (C) lr-vs-activation 2x2 dead seeds -------------------------
    ax = fig.add_subplot(gs[1, 0])
    cells = _load_disambiguation()
    activations = ["crelu", "zrelu"]
    lrs = [0.0024, 0.0236]
    lookup = {
        ("crelu", 0.0236): cells["crelu_highlr"],
        ("crelu", 0.0024): cells["crelu_lowlr"],
        ("zrelu", 0.0024): cells["zrelu_lowlr"],
        ("zrelu", 0.0236): cells["zrelu_highlr"],
    }
    grid = np.array(
        [[lookup[(a, lr)]["dead"] for lr in lrs] for a in activations], dtype=float
    )
    ax.imshow(grid, cmap=CMAP_GAP, vmin=0, vmax=9, aspect="auto")
    for i, a in enumerate(activations):
        for j, lr in enumerate(lrs):
            d = lookup[(a, lr)]["dead"]
            g = lookup[(a, lr)]["step1_head_max"]
            ax.text(
                j,
                i - 0.12,
                f"{d}/9 dead",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if d >= 4 else "black",
            )
            ax.text(
                j,
                i + 0.22,
                f"step-1 grad {g:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if d >= 4 else "#333",
            )
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"lr={lr}" for lr in lrs])
    ax.set_yticks(range(len(activations)))
    ax.set_yticklabels(activations)
    ax.set_title(
        "(C) Dead seeds track lr, not activation", fontsize=11, fontweight="bold"
    )

    # ---- (D) phase-equivariance / activation map (qualitative) -------
    ax = fig.add_subplot(gs[1, 1])
    acts = ["modrelu", "siglog", "crelu", "cardioid", "zrelu"]
    phase_eq = [True, True, False, False, False]
    stable_at_matched = [True, False, False, False, True]
    # 2D scatter: x = phase-equivariant?, y = stable @ matched-shared-trial?
    # crelu and cardioid both occupy (PE=no, stable=no) so split them.
    offsets = {
        "modrelu": (1.0, 1.0),
        "siglog": (1.0, 0.0),
        "zrelu": (0.0, 1.0),
        "crelu": (-0.12, -0.05),
        "cardioid": (0.12, 0.08),
    }
    label_off = {
        "modrelu": (12, 6),
        "siglog": (12, 6),
        "zrelu": (12, 6),
        "crelu": (-50, -6),
        "cardioid": (12, 6),
    }
    for a, _pe, st in zip(acts, phase_eq, stable_at_matched, strict=False):
        x_base, y_base = offsets[a]
        x = x_base
        y = y_base
        col = SAFE_COLOR if st else UNSAFE_COLOR
        ax.scatter(x, y, s=320, color=col, edgecolor="black", zorder=3, lw=1.5)
        dx, dy = label_off[a]
        ax.annotate(
            a,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["not phase-\nequivariant", "phase-\nequivariant"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["unstable\n(dead seeds)", "stable\n(0 dead)"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_title(
        "(D) Phase-equivariance ≠ optimization stability",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(
        0.5,
        -0.3,
        "stability splits orthogonally to equivariance →\noptimization-level mechanism, not symmetry-level",  # noqa: E501
        ha="center",
        fontsize=9,
        style="italic",
        color="#444",
        transform=ax.transAxes,
    )

    fig.suptitle(
        "Complex-valued networks for IQ data — when, why, and the methodology that magnifies the gap",  # noqa: E501
        fontsize=14,
        fontweight="bold",
    )
    return _save(fig, "headline_paper_at_a_glance")


def main() -> None:
    paths = [
        fig_selection_rule_contrast(),
        fig_lr_activation_disambiguation(),
        fig_dead_seed_mechanism(),
        fig_paper_at_a_glance(),
    ]
    for p in paths:
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
