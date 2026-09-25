"""Build the SNR-crossover figure from finished RadioML sweep directories.

Panel (a) plots per-SNR test accuracy by representation for the headline
configuration, with the complex and polar-real families emphasised because the
crossover between them is the paper's claim. Panel (b) plots the complex-minus-
polar gap for every configuration, which turns the crossover from an
intersection of two lines into a zero crossing and lets the class-composition
and sequence-length controls sit in the same axes.

Reads only `summary.json`, so it never retrains. Run after the sweeps:

    .venv/bin/python scripts/plot_crossover_figure.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

# Categorical palette. The accent pair carries the claim and is validated for
# colour-vision deficiency: protan dE 21.9, deutan 24.6, tritan 30.9, both
# above 3:1 against white. The three context baselines sit on a single-hue
# neutral ramp and are separated by dash pattern and direct label as well as
# lightness, so identity never rests on colour alone.
COMPLEX_COLOR = "#0072B2"
POLAR_COLOR = "#D55E00"
CONTEXT_COLORS = ("#565656", "#838383", "#A5A5A5")
INK = "#1A1A1A"
MUTED = "#6E6E6E"
GRID = "#D8D8D8"
BAND = "#E9EEF2"

# Families drawn in panel (a), in fixed order: the accent pair first, then the
# context baselines. Order is fixed so a configuration with a missing family
# never repaints the others.
ACCENT_FAMILIES = ("complex", "real_polar")
CONTEXT_FAMILIES = ("real_magnitude", "real_stacked", "real_phase")
CONTEXT_DASHES = ((0, (5, 2)), (0, (1, 1.6)), (0, (3, 1.4, 1, 1.4)))

FAMILY_LABEL = {
    "complex": "complex",
    "real_polar": "polar real",
    "real_magnitude": "magnitude real",
    "real_stacked": "stacked real",
    "real_phase": "phase real",
}

# The crossover band. Every configuration changes sign between these two
# levels, so the band is the tightest interval containing all three crossings.
CROSSOVER_LO_DB = 2.0
CROSSOVER_HI_DB = 6.0


class Configuration:
    """One finished sweep directory plus how to label it in panel (b)."""

    def __init__(self, run_dir: Path, label: str, headline: bool) -> None:
        self.run_dir = run_dir
        self.label = label
        self.headline = headline


def _as_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    return cast(dict[str, Any], value)


def _as_array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON array, got {type(value).__name__}")
    return value


def _accuracy_by_snr(
    run_dir: Path, selection: str
) -> tuple[dict[str, dict[int, list[float]]], int]:
    """Return {family: {snr_db: [accuracy per seed]}} and the class count."""

    summary = _as_object(json.loads((run_dir / "summary.json").read_text()))
    key = f"{selection}_selections"
    if key not in summary:
        raise KeyError(f"{run_dir / 'summary.json'} has no '{key}'")
    n_classes = len(_as_array(_as_object(summary["config"])["modulations"]))

    per_family: dict[str, dict[int, list[float]]] = {}
    for entry_value in _as_array(summary[key]):
        entry = _as_object(entry_value)
        family = str(entry["family"])
        extra = _as_object(entry["selected_extra"])
        by_seed = _as_array(extra["test_accuracy_by_snr_db_per_seed"])
        by_snr: dict[int, list[float]] = {}
        for seed_value in by_seed:
            for snr_text, accuracy in _as_object(seed_value).items():
                by_snr.setdefault(int(snr_text), []).append(float(accuracy))
        per_family[family] = by_snr
    return per_family, n_classes


def _means(by_snr: dict[int, list[float]]) -> tuple[list[int], list[float]]:
    levels = sorted(by_snr)
    return levels, [statistics.fmean(by_snr[level]) for level in levels]


def _stdevs(by_snr: dict[int, list[float]]) -> list[float]:
    return [
        statistics.stdev(by_snr[level]) if len(by_snr[level]) > 1 else 0.0
        for level in sorted(by_snr)
    ]


def _style_axes(ax: Any, *, xlabel: str, ylabel: str, title: str) -> None:
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


def _plot_representation_map(
    ax: Any, per_family: dict[str, dict[int, list[float]]], n_classes: int
) -> None:
    chance = 1.0 / n_classes
    ax.axhline(
        chance, color=MUTED, linewidth=0.6, linestyle=(0, (4, 3)), alpha=0.8, zorder=1
    )

    for family, color, dashes in zip(
        CONTEXT_FAMILIES, CONTEXT_COLORS, CONTEXT_DASHES, strict=True
    ):
        if family not in per_family:
            continue
        levels, means = _means(per_family[family])
        ax.plot(
            levels,
            means,
            color=color,
            linewidth=1.0,
            linestyle=dashes,
            zorder=2,
            label=FAMILY_LABEL[family],
        )

    for family, color in zip(
        ACCENT_FAMILIES, (COMPLEX_COLOR, POLAR_COLOR), strict=True
    ):
        if family not in per_family:
            continue
        levels, means = _means(per_family[family])
        spread = _stdevs(per_family[family])
        ax.fill_between(
            levels,
            [mean - sd for mean, sd in zip(means, spread, strict=True)],
            [mean + sd for mean, sd in zip(means, spread, strict=True)],
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=3,
        )
        ax.plot(
            levels,
            means,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.4,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.7,
            zorder=4,
            label=FAMILY_LABEL[family],
        )

    ax.annotate(
        f"chance {chance:.3f}",
        xy=(7.0, chance),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=6.4,
        color=MUTED,
    )
    _style_axes(
        ax,
        xlabel="SNR (dB)",
        ylabel="test accuracy",
        title="(a) Representation map, 7 classes, length 128",
    )
    ax.set_ylim(0.10, 0.74)
    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index(FAMILY_LABEL[f])
        for f in ACCENT_FAMILIES
        if FAMILY_LABEL[f] in labels
    ]
    order += [i for i in range(len(labels)) if i not in order]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=6.6,
        frameon=False,
        loc="upper left",
        ncols=2,
        handlelength=2.2,
        columnspacing=1.1,
        labelspacing=0.35,
        borderpad=0.0,
    )


def _plot_gap(ax: Any, configurations: list[Configuration], selection: str) -> None:
    ax.axvspan(CROSSOVER_LO_DB, CROSSOVER_HI_DB, color=BAND, zorder=0)
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)

    context_index = 0
    for configuration in configurations:
        per_family, _ = _accuracy_by_snr(configuration.run_dir, selection)
        if "complex" not in per_family or "real_polar" not in per_family:
            continue
        levels, complex_means = _means(per_family["complex"])
        _, polar_means = _means(per_family["real_polar"])
        gap = [100.0 * (c - p) for c, p in zip(complex_means, polar_means, strict=True)]
        if configuration.headline:
            ax.plot(
                levels,
                gap,
                color=INK,
                linewidth=2.0,
                marker="o",
                markersize=3.4,
                markeredgecolor="#FFFFFF",
                markeredgewidth=0.7,
                zorder=4,
                label=configuration.label,
            )
        else:
            ax.plot(
                levels,
                gap,
                color=CONTEXT_COLORS[context_index % len(CONTEXT_COLORS)],
                linewidth=1.1,
                linestyle=CONTEXT_DASHES[context_index % len(CONTEXT_DASHES)],
                marker="^" if context_index == 0 else "s",
                markersize=2.9,
                zorder=3,
                label=configuration.label,
            )
            context_index += 1

    ax.set_ylim(-12.5, 15.5)
    ax.annotate(
        "complex ahead",
        xy=(-14.6, 13.0),
        fontsize=6.4,
        color=MUTED,
    )
    ax.annotate(
        "polar ahead",
        xy=(-14.6, -11.6),
        fontsize=6.4,
        color=MUTED,
    )
    ax.annotate(
        f"crossover {CROSSOVER_LO_DB:.0f}\u2013{CROSSOVER_HI_DB:.0f} dB",
        xy=(4.0, 13.0),
        ha="center",
        fontsize=6.4,
        color=MUTED,
    )
    _style_axes(
        ax,
        xlabel="SNR (dB)",
        ylabel="complex $-$ polar real (pp)",
        title="(b) The crossover, and both controls",
    )
    # Anchored into the upper-right quadrant, which every configuration
    # leaves empty: the gap is negative at every level from 6 dB up.
    ax.legend(
        fontsize=6.6,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.72),
        handlelength=2.2,
        labelspacing=0.35,
        borderpad=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--stem", default="crossover")
    parser.add_argument(
        "--selection",
        choices=["matched", "independent"],
        default="matched",
        help=(
            "which selection rule to plot. The paper's primary comparison is "
            "matched-shared-trial."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configurations = [
        Configuration(
            args.results_dir / "radioml_geom7_crelu", "7 classes, length 128", True
        ),
        Configuration(
            args.results_dir / "radioml_geom10_crelu", "10 classes, length 128", False
        ),
        Configuration(
            args.results_dir / "radioml_geom7_len256_crelu",
            "7 classes, length 256",
            False,
        ),
    ]
    missing = [
        c.run_dir for c in configurations if not (c.run_dir / "summary.json").is_file()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        print(f"missing summary.json in: {joined}", file=sys.stderr)
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headline = next(c for c in configurations if c.headline)
    per_family, n_classes = _accuracy_by_snr(headline.run_dir, args.selection)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    _plot_representation_map(axes[0], per_family, n_classes)
    _plot_gap(axes[1], configurations, args.selection)
    fig.tight_layout(pad=0.45, w_pad=1.6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".pdf", ".png"):
        path = (args.output_dir / args.stem).with_suffix(suffix)
        fig.savefig(path, dpi=400, bbox_inches="tight")
        written.append(path)
    plt.close(fig)

    levels, complex_means = _means(per_family["complex"])
    _, polar_means = _means(per_family["real_polar"])
    print(f"selection: {args.selection}-shared-trial, chance {1.0 / n_classes:.3f}")
    print("snr_db  complex  polar   gap_pp")
    for level, c, p in zip(levels, complex_means, polar_means, strict=True):
        print(f"{level:>6}  {c:.3f}    {p:.3f}  {100.0 * (c - p):+6.2f}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
