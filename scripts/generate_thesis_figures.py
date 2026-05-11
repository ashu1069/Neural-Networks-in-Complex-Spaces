"""Thesis-anchor figures: visual story for the CVNN paper.

Three publication-grade figures designed for narrative impact rather than
exhaustive statistical detail:

  A. fig_iq_anatomy           - constellations + per-coordinate-view atlas
                                that explains why phase or amplitude wins
                                on each task (the lead figure).
  B. fig_liouville_trilemma   - triangle diagram with each activation
                                placed in its trade-off region.
  C. fig_activation_atlas     - 6 activations x {|sigma|, arg sigma,
                                |d_zbar sigma|} heatmaps on the complex
                                plane.

Outputs land in `paper/figures/` and `results/figures/` so the paper
build picks them up automatically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

import torch

from cvnn.activations.functions import (
    complex_cardioid,
    complex_tanh,
    crelu,
    modrelu,
    siglog,
    zrelu,
)

# --------------------------------------------------------------------------
# Aesthetic configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [ROOT / "paper" / "figures", ROOT / "results" / "figures"]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
PAPER = "#fdfdfd"
COMPLEX_C = "#2E5266"  # deep teal
REAL_C = "#c44536"     # warm brick
PHASE_C = "#6b9080"    # sage
MAG_C = "#d4a373"      # warm sand
MUTED = "#9aa0a6"
RULE = "#dde2e6"

PHASE_CMAP = "twilight"  # cyclic, perfect for arg(z)
MAG_CMAP = LinearSegmentedColormap.from_list(
    "warmsand", ["#fdfdfd", MAG_C, "#7a4f24"]
)
RESID_CMAP = LinearSegmentedColormap.from_list(
    "residual", ["#fdfdfd", "#f4a261", "#7a1a1a"]
)


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
        "axes.titlesize": 11,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _save(fig, name: str) -> None:
    for d in OUT_DIRS:
        fig.savefig(d / f"{name}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Synthetic IQ generators
# --------------------------------------------------------------------------
def _psk_symbols(order: int = 8, n: int = 256, snr_db: float = 20.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    k = rng.integers(0, order, size=n)
    sym = np.exp(2j * np.pi * k / order)
    return _add_awgn(sym, snr_db, rng)


def _qam_symbols(order: int = 16, n: int = 256, snr_db: float = 20.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(1)
    side = int(np.sqrt(order))
    levels = np.linspace(-1, 1, side)
    grid = np.array([(x + 1j * y) for x in levels for y in levels])
    grid /= np.sqrt(np.mean(np.abs(grid) ** 2))
    idx = rng.integers(0, len(grid), size=n)
    sym = grid[idx]
    return _add_awgn(sym, snr_db, rng)


def _add_awgn(sym: np.ndarray, snr_db: float,
              rng: np.random.Generator) -> np.ndarray:
    sig_p = np.mean(np.abs(sym) ** 2)
    n_p = sig_p / (10 ** (snr_db / 10))
    noise = (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape))
    noise *= np.sqrt(n_p / 2)
    return sym + noise


# --------------------------------------------------------------------------
# Figure A: Anatomy of an IQ sample
# --------------------------------------------------------------------------
def fig_iq_anatomy() -> None:
    """Lead figure: PSK vs QAM constellations + what each coordinate view sees.

    Layout: 2 rows (PSK / QAM), 4 cols
        col 1: IQ constellation (the data)
        col 2: |z| histogram         (what magnitude-only sees)
        col 3: arg(z) histogram      (what phase-only sees)
        col 4: a one-line verdict per row

    Story: PSK lives on a circle (no info in |z|), QAM on a grid
    (info in both |z| and arg(z)). The reader sees in 5 seconds why
    magnitude-only wins QAM and phase-only wins PSK.
    """
    _set_style()
    fig = plt.figure(figsize=(11, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 0.85, 0.85, 1.05],
                          hspace=0.35, wspace=0.25)

    rng = np.random.default_rng(20260601)
    psk = _psk_symbols(order=8, n=600, snr_db=18, rng=rng)
    qam = _qam_symbols(order=16, n=600, snr_db=22, rng=rng)

    rows = [("PSK-8", psk, PHASE_C, "phase"),
            ("QAM-16", qam, MAG_C, "magnitude")]

    for r, (name, sym, accent, winner) in enumerate(rows):
        # ---- col 1: constellation -----------------------------------------
        ax = fig.add_subplot(gs[r, 0])
        ax.set_aspect("equal")
        ax.scatter(sym.real, sym.imag, s=10, color=accent, alpha=0.55,
                   edgecolors="none")
        # noise-free reference
        if "PSK" in name:
            theta = np.linspace(0, 2 * np.pi, 9)[:-1]
            ax.scatter(np.cos(theta), np.sin(theta), s=70, marker="o",
                       facecolor="none", edgecolor=INK, linewidth=1.2)
            circ = patches.Circle((0, 0), 1.0, fill=False,
                                  edgecolor=MUTED, linewidth=0.6, ls="--")
            ax.add_patch(circ)
        else:
            side = 4
            levels = np.linspace(-1, 1, side)
            grid_r = np.array([(x, y) for x in levels for y in levels])
            grid_r /= np.sqrt(np.mean(grid_r[:, 0] ** 2 + grid_r[:, 1] ** 2))
            ax.scatter(grid_r[:, 0], grid_r[:, 1], s=70, marker="s",
                       facecolor="none", edgecolor=INK, linewidth=1.2)
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("I (Re z)"); ax.set_ylabel("Q (Im z)")
        ax.set_title(f"{name}: where the symbols live", loc="left",
                     color=INK, pad=4)
        ax.axhline(0, color=RULE, lw=0.6, zorder=0)
        ax.axvline(0, color=RULE, lw=0.6, zorder=0)

        # ---- col 2: magnitude histogram -----------------------------------
        ax = fig.add_subplot(gs[r, 1])
        mag = np.abs(sym)
        ax.hist(mag, bins=40, color=MAG_C, alpha=0.85, edgecolor="none")
        ax.set_xlim(0, 1.6)
        ax.set_yticks([])
        ax.set_xlabel("|z|")
        if "PSK" in name:
            ax.set_title("magnitude-only sees:", loc="left", pad=4)
            ax.text(0.97, 0.92, "one peak\n(no class info)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e8",
                              ec=MAG_C, lw=0.8))
        else:
            ax.set_title("magnitude-only sees:", loc="left", pad=4)
            ax.text(0.97, 0.92, "three rings\n(class info!)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e8",
                              ec=MAG_C, lw=0.8))

        # ---- col 3: phase histogram ---------------------------------------
        ax = fig.add_subplot(gs[r, 2])
        ang = np.angle(sym)
        ax.hist(ang, bins=64, range=(-np.pi, np.pi),
                color=PHASE_C, alpha=0.85, edgecolor="none")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(["-π", "0", "π"])
        ax.set_yticks([])
        ax.set_xlabel("arg z")
        if "PSK" in name:
            ax.set_title("phase-only sees:", loc="left", pad=4)
            ax.text(0.97, 0.92, "8 spikes\n(class info!)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#eaf2ec",
                              ec=PHASE_C, lw=0.8))
        else:
            ax.set_title("phase-only sees:", loc="left", pad=4)
            ax.text(0.97, 0.92, "broad spread\n(weak class info)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#eaf2ec",
                              ec=PHASE_C, lw=0.8))

        # ---- col 4: verdict + measured accuracy ---------------------------
        ax = fig.add_subplot(gs[r, 3])
        ax.axis("off")
        if "PSK" in name:
            verdict = "phase wins"
            mag_acc, phase_acc, complex_acc = 0.333, 0.735, 0.821
        else:
            verdict = "magnitude wins"
            mag_acc, phase_acc, complex_acc = 0.524, 0.487, 0.509
        ax.text(0.0, 0.92, verdict, transform=ax.transAxes,
                fontsize=15, weight="bold", color=accent)
        ax.text(0.0, 0.78, "measured (RF stress test):",
                transform=ax.transAxes, fontsize=8.5, color=MUTED)
        bar_y = [0.60, 0.42, 0.24]
        bar_labels = ["complex", "phase-only", "magnitude-only"]
        bar_vals = [complex_acc, phase_acc, mag_acc]
        bar_colors = [COMPLEX_C, PHASE_C, MAG_C]
        for y, lbl, v, c in zip(bar_y, bar_labels, bar_vals, bar_colors):
            w = 0.55 * v
            rect = patches.FancyBboxPatch(
                (0.0, y - 0.04), w, 0.07,
                boxstyle="round,pad=0.005,rounding_size=0.01",
                fc=c, ec="none", transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(w + 0.015, y, f"{v:.2f}", transform=ax.transAxes,
                    fontsize=9, color=INK, va="center")
            ax.text(0.0, y + 0.055, lbl, transform=ax.transAxes,
                    fontsize=8, color=MUTED)
        ax.text(0.0, 0.04, "chance = 0.33 (3-class)" if "PSK" in name
                else "chance = 0.33 (3-class)",
                transform=ax.transAxes, fontsize=8, color=MUTED, style="italic")

    fig.suptitle(
        "What's actually in an IQ sample — and why it picks the winning model",
        x=0.01, ha="left", fontsize=13.5, weight="bold", color=INK, y=1.02)

    _save(fig, "thesis_iq_anatomy")


# --------------------------------------------------------------------------
# Figure B: The Liouville trilemma
# --------------------------------------------------------------------------
def fig_liouville_trilemma() -> None:
    """Triangle with vertices = three desiderata; each activation lives
    in the cell defined by the two it satisfies (and the one it drops).

    Layout: small triangle in centre, three labelled cells far outside,
    arrows from each cell to the vertex it "drops". The forbidden
    interior is shaded with the Liouville verdict.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.set_aspect("equal")
    ax.axis("off")

    # Smaller triangle so cells have breathing room.
    s = 0.85
    A = np.array([0.0, s * np.sqrt(3.0) / 2])      # top: BOUNDED
    B = np.array([-s * 0.5, -s * np.sqrt(3.0) / 6])  # bottom-left: HOLOMORPHIC
    C = np.array([+s * 0.5, -s * np.sqrt(3.0) / 6])  # bottom-right: NONCONSTANT
    centroid = (A + B + C) / 3.0

    # ----- forbidden interior -----
    ax.add_patch(patches.Polygon(
        [A, B, C], closed=True, facecolor="#f8e8e8",
        edgecolor=REAL_C, linewidth=1.6, zorder=1))
    ax.text(centroid[0], centroid[1] + 0.04, "Liouville says:",
            ha="center", va="center", fontsize=8.5,
            color=REAL_C, style="italic")
    ax.text(centroid[0], centroid[1] - 0.10, "no σ here",
            ha="center", va="center", fontsize=11.5,
            color=REAL_C, weight="bold")
    ax.text(centroid[0], centroid[1] - 0.22, "(constant only)",
            ha="center", va="center", fontsize=8, color=REAL_C)

    # ----- vertex dots + labels — placed outward from centroid -----
    def outward(V, dist):
        d = V - centroid; d /= np.linalg.norm(d)
        return V + d * dist

    vertex_specs = [
        (A, "BOUNDED",     "|σ(z)| ≤ M",                                 "center"),
        (B, "HOLOMORPHIC", r"$\partial_{\bar z}\sigma \equiv 0$",        "right"),
        (C, "NONCONSTANT", "σ does something",                           "left"),
    ]
    for V, name, sub, ha in vertex_specs:
        L = outward(V, 0.18)
        ax.plot(*V, "o", color=INK, ms=6, zorder=4)
        ax.text(L[0], L[1] + 0.05, name, ha=ha, va="center",
                fontsize=11.5, weight="bold", color=INK, zorder=4)
        ax.text(L[0], L[1] - 0.07, sub, ha=ha, va="center",
                fontsize=8.5, color=MUTED, style="italic", zorder=4)

    # ----- cells: each cell sits on the side of the triangle OPPOSITE
    # to the vertex it drops (south for dropping the top vertex, etc.) -----
    def opposite(V, dist=2.0):
        d = centroid - V; d /= np.linalg.norm(d)
        return centroid + d * dist

    cell_drop_bounded   = opposite(A, 2.85)   # cell south of triangle
    cell_drop_holo      = opposite(B, 2.85)   # cell NE of triangle
    cell_drop_nonconst  = opposite(C, 2.85)   # cell NW of triangle

    cells = [
        # (anchor, vertex_to_drop, drop-text, kept text, activations, fc, ec)
        (cell_drop_bounded, A,
         "DROP BOUNDEDNESS",
         "holomorphic · nonconstant\n· blows up at poles",
         ["ComplexTanh"],
         "#fff7e8", MAG_C),
        (cell_drop_holo, B,
         "DROP HOLOMORPHY",
         "bounded · nonconstant\n· non-holomorphic",
         ["Siglog", "ModReLU", "ComplexCardioid"],
         "#eaf2ec", PHASE_C),
        (cell_drop_nonconst, C,
         "DROP NONCONSTANCY",
         "bounded · holomorphic\n· constant",
         ["(degenerate — useless)"],
         "#f0f0f0", MUTED),
    ]

    for anchor, vertex, drop_text, kept, names, fc, ec in cells:
        w, h = 2.20, 1.30
        x, y = anchor[0] - w / 2, anchor[1] - h / 2
        # cell box
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.07",
            fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(anchor[0], anchor[1] + 0.46, drop_text,
                ha="center", va="center", fontsize=10.5,
                weight="bold", color=ec, zorder=4)
        ax.text(anchor[0], anchor[1] + 0.20, kept,
                ha="center", va="center", fontsize=8.2,
                color=MUTED, style="italic", zorder=4)
        for i, nm in enumerate(names):
            ax.text(anchor[0], anchor[1] - 0.16 - 0.16 * i, "• " + nm,
                    ha="center", va="center", fontsize=9.5,
                    color=INK, zorder=4)
        # leader from cell edge to dropped vertex (curved gently)
        direction = vertex - anchor
        nd = np.linalg.norm(direction); ud = direction / nd
        # exit cell perimeter
        t = min((w / 2) / max(abs(ud[0]), 1e-6),
                (h / 2) / max(abs(ud[1]), 1e-6))
        start = anchor + ud * (t + 0.04)
        end = vertex - ud * 0.12
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color=ec,
                                    lw=1.4, shrinkA=0, shrinkB=0))

    # Footer
    ax.text(0.0, -4.30,
            "CReLU and ZReLU live in the 'drop holomorphy' cell, "
            "but additionally drop phase equivariance\n"
            "(a 4th axis orthogonal to the trilemma — see §1.3).",
            ha="center", fontsize=9, color=MUTED, style="italic")

    ax.set_xlim(-5.2, 5.2); ax.set_ylim(-4.6, 4.0)

    fig.suptitle(
        "The Liouville trilemma — every useful complex activation drops one",
        fontsize=13.5, weight="bold", color=INK, y=0.97, x=0.5)

    _save(fig, "thesis_liouville_trilemma")


# --------------------------------------------------------------------------
# Figure C: Activation atlas on the complex plane
# --------------------------------------------------------------------------
def _grid(extent: float = 2.5, n: int = 240) -> torch.Tensor:
    xs = torch.linspace(-extent, extent, n)
    ys = torch.linspace(-extent, extent, n)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    return torch.complex(X, Y)


def _cr_residual(fn, z: torch.Tensor) -> np.ndarray:
    """Finite-difference estimate of |partial_{bar z} sigma|.

    For sigma = u + i v with z = x + i y,
        partial_{bar z} sigma = 1/2 [(d_x u - d_y v) + i (d_x v + d_y u)]
    Holomorphic <=> |partial_{bar z} sigma| == 0.
    """
    h = 1e-3
    sx = fn(z + h); sx_ = fn(z - h)
    sy = fn(z + 1j * h); sy_ = fn(z - 1j * h)
    du_dx = (sx.real - sx_.real) / (2 * h)
    dv_dx = (sx.imag - sx_.imag) / (2 * h)
    du_dy = (sy.real - sy_.real) / (2 * h)
    dv_dy = (sy.imag - sy_.imag) / (2 * h)
    re = 0.5 * (du_dx - dv_dy)
    im = 0.5 * (dv_dx + du_dy)
    r = (re ** 2 + im ** 2).sqrt()
    return r.detach().numpy()


def fig_activation_atlas() -> None:
    """6 activations x 3 views (|sigma|, arg sigma, |d_zbar sigma|)."""
    _set_style()
    z = _grid(extent=2.5, n=220)
    extent = (-2.5, 2.5, -2.5, 2.5)

    activations = [
        ("CReLU",            crelu),
        ("ZReLU",            zrelu),
        ("ModReLU",          lambda zz: modrelu(zz, bias=torch.tensor(-0.5))),
        ("ComplexCardioid",  complex_cardioid),
        ("Siglog",           siglog),
        ("ComplexTanh",      complex_tanh),
    ]

    fig, axes = plt.subplots(6, 3, figsize=(11.5, 19.0),
                             constrained_layout=True)

    for r, (name, fn) in enumerate(activations):
        with torch.no_grad():
            out = fn(z)
        mag = out.abs().detach().numpy()
        ang = out.angle().detach().numpy()
        cr = _cr_residual(fn, z)

        # cap dynamic range for tanh (poles cause huge values)
        if name == "ComplexTanh":
            mag = np.clip(mag, 0, 6.0)
            cr = np.clip(cr, 0, 6.0)

        # --- col 0: magnitude
        ax = axes[r, 0]
        im = ax.imshow(mag, cmap=MAG_CMAP, extent=extent, origin="lower")
        ax.set_title("|σ(z)|", loc="left",
                     fontsize=10.5, color=INK, pad=2)
        # row label outside, well-clear of plot
        ax.text(-0.42, 0.5, name, transform=ax.transAxes,
                rotation=90, ha="center", va="center",
                fontsize=13, weight="bold", color=INK)
        _atlas_axes(ax)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

        # --- col 1: phase
        ax = axes[r, 1]
        im = ax.imshow(ang, cmap=PHASE_CMAP, extent=extent, origin="lower",
                       vmin=-np.pi, vmax=np.pi)
        ax.set_title("arg σ(z)", loc="left", fontsize=10, color=INK, pad=2)
        _atlas_axes(ax)
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                          ticks=[-np.pi, 0, np.pi])
        cb.ax.set_yticklabels(["-π", "0", "π"])

        # --- col 2: CR residual
        ax = axes[r, 2]
        im = ax.imshow(cr, cmap=RESID_CMAP, extent=extent, origin="lower")
        ax.set_title("|∂_z̄ σ|  (0 ⇔ holomorphic)", loc="left",
                     fontsize=10, color=INK, pad=2)
        _atlas_axes(ax)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(
        "Atlas of complex activations — magnitude, phase, holomorphy defect",
        x=0.01, ha="left", fontsize=13.5, weight="bold", color=INK, y=1.005)

    _save(fig, "thesis_activation_atlas")


def _atlas_axes(ax) -> None:
    ax.axhline(0, color=RULE, lw=0.5)
    ax.axvline(0, color=RULE, lw=0.5)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------
def main() -> None:
    fig_iq_anatomy()
    print("wrote thesis_iq_anatomy")
    fig_liouville_trilemma()
    print("wrote thesis_liouville_trilemma")
    fig_activation_atlas()
    print("wrote thesis_activation_atlas")


if __name__ == "__main__":
    main()
