"""High-quality narrative animations for the paper.

Three artifacts:

1. anim_dead_seed_mechanism — replays the first 30 training steps of nine
   real-baseline seeds under crelu@high-lr (which dies) and zrelu@low-lr
   (which trains). Step-1 explosion appears as a dramatic spike, dead
   seeds turn red as they cross the chance-level loss threshold.

2. anim_selection_rule_morph — bars morph smoothly between
   matched-shared-trial accuracies (gap = 22.94 pp) and independent
   per-family accuracies (gap = 2.46 pp). The gap arrow shrinks live.

3. anim_conditional_advantage — RF stress test conditions reveal one by
   one. Winner stamp ("Complex" / "Real") slides in for each, and a
   running tally accumulates at the top.

All animations are rendered in a modern dark theme and saved as both
MP4 (high-quality, ffmpeg required) and GIF (universally embeddable).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "animations"
PAPER_OUT_DIR = ROOT / "paper" / "figures" / "animations"
for d in (OUT_DIR, PAPER_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- modern dark palette ----------------------------------------------------
BG = "#0b1220"
PANEL_BG = "#111a2f"
GRID = "#1e2a44"
FG = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT_BLUE = "#60a5fa"      # complex / cool
ACCENT_TEAL = "#2dd4bf"      # alive
ACCENT_RED = "#f43f5e"       # dead / unstable
ACCENT_AMBER = "#fbbf24"     # caution
ACCENT_LIME = "#a3e635"      # safe / stable
ACCENT_PURPLE = "#a78bfa"    # callouts
DEAD_LOSS_THRESHOLD = math.log(3) - 0.05


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": ":",
        "text.color": FG,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": BG,
        "savefig.edgecolor": "none",
    })


def _save_anim(anim_obj: animation.FuncAnimation, name: str, fps: int = 24) -> list[Path]:
    """Save as MP4 (ffmpeg) and GIF (Pillow). Return list of written paths."""
    paths = []
    try:
        mp4_path = OUT_DIR / f"{name}.mp4"
        anim_obj.save(
            str(mp4_path),
            writer=animation.FFMpegWriter(fps=fps, bitrate=4500, codec="libx264"),
            dpi=150,
        )
        paths.append(mp4_path)
        # also drop a copy under paper/figures/animations
        (PAPER_OUT_DIR / f"{name}.mp4").write_bytes(mp4_path.read_bytes())
    except Exception as e:  # pragma: no cover
        print(f"  mp4 save failed for {name}: {e}")

    try:
        gif_path = OUT_DIR / f"{name}.gif"
        anim_obj.save(
            str(gif_path),
            writer=animation.PillowWriter(fps=fps),
            dpi=110,
        )
        paths.append(gif_path)
        (PAPER_OUT_DIR / f"{name}.gif").write_bytes(gif_path.read_bytes())
    except Exception as e:  # pragma: no cover
        print(f"  gif save failed for {name}: {e}")

    return paths


# ============================================================================
# 1. dead-seed mechanism animation
# ============================================================================
def _load_jsonl(path: Path):
    with open(path) as f:
        lines = f.readlines()
    summary = json.loads(lines[0])["summary"]
    steps = [json.loads(l) for l in lines[1:]]
    return summary, steps


def _collect(act_dir: Path):
    traces = []
    for fam in ["real_stacked", "real_matched_params", "real_matched_flops"]:
        for seed in range(3):
            p = act_dir / f"{fam}_seed{seed}.jsonl"
            summary, steps = _load_jsonl(p)
            head = np.array([s["per_layer_grad_norm"]["head.weight"] for s in steps])
            loss = np.array([s["train_loss"] for s in steps])
            step = np.array([s["step"] for s in steps])
            dead = summary["final_train_loss"] >= DEAD_LOSS_THRESHOLD
            traces.append({"step": step, "head": head, "loss": loss,
                            "dead": dead, "fam": fam, "seed": seed})
    return traces


def anim_dead_seed_mechanism():
    _style()
    crelu = _collect(RESULTS / "synthetic_rf_telemetry" / "crelu")
    zrelu = _collect(RESULTS / "synthetic_rf_telemetry" / "zrelu")

    n_steps = 30
    fps = 12
    pad_frames = 12  # hold the final frame so viewers can read it

    fig = plt.figure(figsize=(13, 7.2))
    fig.suptitle(
        "Dead-seed mechanism — same model, same data, two learning rates",
        fontsize=15, fontweight="bold", color=FG, y=0.97,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.20,
                           left=0.07, right=0.97, top=0.88, bottom=0.10)
    ax_g_crelu, ax_g_zrelu = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_l_crelu, ax_l_zrelu = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for ax in (ax_g_crelu, ax_g_zrelu):
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1e2)
        ax.set_xlim(-0.5, n_steps - 0.5)
        ax.set_ylabel(r"$\|\nabla_{\mathrm{head.weight}} L\|$", color=FG)
        ax.grid(True, which="both")
        ax.axhline(1.0, color=MUTED, lw=0.6, ls="--", alpha=0.6)

    for ax in (ax_l_crelu, ax_l_zrelu):
        ax.set_xlim(-0.5, n_steps - 0.5)
        ax.set_xlabel("training step", color=FG)
        ax.set_ylabel("train loss", color=FG)
        ax.grid(True)
        ax.axhline(DEAD_LOSS_THRESHOLD, color=ACCENT_RED, lw=1.0, ls=":", alpha=0.8)

    ax_l_crelu.set_ylim(0, 6.0)
    ax_l_zrelu.set_ylim(0.3, 1.3)

    ax_g_crelu.set_title("CReLU @ lr = 0.0236  (matched-shared-trial)", color=ACCENT_AMBER)
    ax_g_zrelu.set_title("ZReLU @ lr = 0.0024  (matched-shared-trial)", color=ACCENT_LIME)

    # Banner labels
    ax_l_crelu.text(0.5, 1.18, "the unstable regime", transform=ax_l_crelu.transAxes,
                    ha="center", fontsize=10, color=ACCENT_AMBER, alpha=0.85)
    ax_l_zrelu.text(0.5, 1.18, "the stable regime", transform=ax_l_zrelu.transAxes,
                    ha="center", fontsize=10, color=ACCENT_LIME, alpha=0.85)

    # Dead-threshold annotation on crelu loss panel
    ax_l_crelu.text(n_steps - 1, DEAD_LOSS_THRESHOLD + 0.06, "dead-loss threshold",
                    color=ACCENT_RED, fontsize=9, ha="right", alpha=0.85)

    # Pre-create empty line objects per trace
    def make_lines(ax_g, ax_l, traces):
        gs_lines = []
        ls_lines = []
        for _ in traces:
            (lg,) = ax_g.plot([], [], color=ACCENT_TEAL, lw=1.3, alpha=0.8)
            (ll,) = ax_l.plot([], [], color=ACCENT_TEAL, lw=1.3, alpha=0.8)
            gs_lines.append(lg)
            ls_lines.append(ll)
        return gs_lines, ls_lines

    crelu_g, crelu_l = make_lines(ax_g_crelu, ax_l_crelu, crelu)
    zrelu_g, zrelu_l = make_lines(ax_g_zrelu, ax_l_zrelu, zrelu)

    # Step counter (huge text top-right of figure)
    step_text = fig.text(
        0.97, 0.93, "step  0", fontsize=18, color=ACCENT_PURPLE,
        ha="right", va="center", fontweight="bold", family="monospace",
    )
    counter_text = fig.text(
        0.5, 0.02, "", fontsize=12, color=FG, ha="center",
    )

    n_frames = n_steps + pad_frames

    def update(frame):
        s = min(frame, n_steps - 1)
        step_text.set_text(f"step {s:>2}")

        def draw(traces, lines_g, lines_l):
            n_dead_so_far = 0
            for t, lg, ll in zip(traces, lines_g, lines_l):
                lg.set_data(t["step"][: s + 1], t["head"][: s + 1])
                ll.set_data(t["step"][: s + 1], t["loss"][: s + 1])
                # Decide color: dead if (1) trace ends dead and (2) we've passed step 2
                will_die = t["dead"] and s >= 2
                if will_die:
                    lg.set_color(ACCENT_RED)
                    lg.set_alpha(0.95)
                    lg.set_linewidth(2.2)
                    ll.set_color(ACCENT_RED)
                    ll.set_alpha(0.95)
                    ll.set_linewidth(2.2)
                    n_dead_so_far += 1
                else:
                    lg.set_color(ACCENT_TEAL)
                    lg.set_alpha(0.7)
                    lg.set_linewidth(1.4)
                    ll.set_color(ACCENT_TEAL)
                    ll.set_alpha(0.7)
                    ll.set_linewidth(1.4)
            return n_dead_so_far

        n_d_crelu = draw(crelu, crelu_g, crelu_l)
        n_d_zrelu = draw(zrelu, zrelu_g, zrelu_l)
        if frame >= n_steps - 1:
            counter_text.set_text(
                f"end of replay  ▸  CReLU: {n_d_crelu}/9 dead seeds   ZReLU: {n_d_zrelu}/9 dead seeds"
            )
        else:
            counter_text.set_text("")
        return crelu_g + crelu_l + zrelu_g + zrelu_l + [step_text, counter_text]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )
    paths = _save_anim(anim, "anim_dead_seed_mechanism", fps=fps)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    return paths


# ============================================================================
# 2. selection-rule morph animation
# ============================================================================
def anim_selection_rule_morph():
    _style()
    matched = (0.7293, 0.4999)   # (cvnn, best real)
    indep = (0.7293, 0.7047)
    fps = 24
    morph_frames = 60
    hold_frames = 18
    n_frames = hold_frames + morph_frames + hold_frames * 2

    fig = plt.figure(figsize=(11, 6.0))
    fig.suptitle(
        "Same trials. Same data. Different selection rule.",
        fontsize=16, fontweight="bold", color=FG, y=0.96,
    )
    fig.text(
        0.5, 0.905,
        "matched-shared-trial  ➜  independent per-family",
        ha="center", color=MUTED, fontsize=11, alpha=0.9,
    )

    ax = fig.add_axes([0.13, 0.16, 0.55, 0.7])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("test accuracy (3-class RadioML)")
    ax.grid(True, axis="y")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CVNN", "best real"])
    ax.set_title("model accuracy", color=FG)

    bar_c = ax.bar(0, matched[0], width=0.5, color=ACCENT_BLUE,
                    edgecolor=ACCENT_BLUE, linewidth=0)
    bar_r = ax.bar(1, matched[1], width=0.5, color=ACCENT_RED,
                    edgecolor=ACCENT_RED, linewidth=0)
    txt_c = ax.text(0, matched[0] + 0.025, f"{matched[0]:.3f}",
                     ha="center", color=FG, fontsize=12, fontweight="bold")
    txt_r = ax.text(1, matched[1] + 0.025, f"{matched[1]:.3f}",
                     ha="center", color=FG, fontsize=12, fontweight="bold")

    # Right-side gap pane
    ax2 = fig.add_axes([0.74, 0.16, 0.22, 0.7])
    ax2.set_ylim(0, 26)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_xticks([])
    ax2.set_ylabel("gap (pp)")
    ax2.grid(True, axis="y")
    ax2.set_title("CVNN − best-real", color=FG)
    gap_initial = (matched[0] - matched[1]) * 100
    bar_gap = ax2.bar(0, gap_initial, width=0.7, color=ACCENT_AMBER, linewidth=0)
    txt_gap = ax2.text(0, gap_initial + 0.7, f"+{gap_initial:.2f} pp",
                        ha="center", color=FG, fontsize=13, fontweight="bold")

    rule_label = fig.text(
        0.40, 0.05, "rule:  matched-shared-trial",
        fontsize=12, color=ACCENT_AMBER, fontweight="bold",
    )

    def smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def update(frame):
        # phases:
        if frame < hold_frames:
            t = 0.0
        elif frame < hold_frames + morph_frames:
            t = smoothstep((frame - hold_frames) / morph_frames)
        else:
            t = 1.0

        c = matched[0] * (1 - t) + indep[0] * t
        r = matched[1] * (1 - t) + indep[1] * t
        gap = (c - r) * 100

        bar_c[0].set_height(c)
        bar_r[0].set_height(r)
        txt_c.set_position((0, c + 0.025))
        txt_c.set_text(f"{c:.3f}")
        txt_r.set_position((1, r + 0.025))
        txt_r.set_text(f"{r:.3f}")

        bar_gap[0].set_height(gap)
        # Animate color from amber to lime as gap shrinks
        gap_color = (1 - t) * np.array([0.984, 0.749, 0.141]) + t * np.array([0.639, 0.902, 0.208])
        bar_gap[0].set_color(gap_color)
        txt_gap.set_position((0, gap + 0.6))
        txt_gap.set_text(f"+{gap:.2f} pp")
        txt_gap.set_color(gap_color)

        if t < 0.5:
            rule_label.set_text("rule:  matched-shared-trial")
            rule_label.set_color(ACCENT_AMBER)
        else:
            rule_label.set_text("rule:  independent per-family")
            rule_label.set_color(ACCENT_LIME)

        return [*bar_c, *bar_r, txt_c, txt_r, *bar_gap, txt_gap, rule_label]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000 / fps, blit=False)
    paths = _save_anim(anim, "anim_selection_rule_morph", fps=fps)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    return paths


# ============================================================================
# 3. conditional advantage reveal animation
# ============================================================================
def anim_conditional_advantage():
    _style()
    conditions = [
        # (label, complex acc, best-real acc, winner)
        ("PSK-only",            0.821, 0.735, "complex"),
        ("QAM-only",            0.509, 0.524, "real"),
        ("mixed PSK+QAM",       0.507, 0.487, "complex"),
        ("low-SNR PSK",         0.526, 0.537, "real"),
        ("high-SNR PSK",        0.949, 0.952, "real"),
        ("unit-magnitude mix",  0.476, 0.491, "real"),
        ("fixed-rotation PSK",  0.252, 0.328, "real"),
        ("rot-augmented PSK",   0.654, 0.580, "complex"),
    ]
    fps = 12
    reveal_frames_per = 8       # frames to reveal each condition
    hold_after_reveal = 4
    final_hold = 24
    n_frames = len(conditions) * (reveal_frames_per + hold_after_reveal) + final_hold

    fig = plt.figure(figsize=(13, 7.2))
    fig.suptitle(
        "When does complex-valued help?  Stress tests on synthetic IQ data",
        fontsize=15, fontweight="bold", color=FG, y=0.96,
    )

    ax = fig.add_axes([0.10, 0.13, 0.86, 0.74])
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c[0] for c in conditions], rotation=18, ha="right",
                       fontsize=10, color=FG)
    ax.set_ylabel("test accuracy")
    ax.grid(True, axis="y")
    ax.axhline(1 / 3, color=MUTED, lw=0.6, ls="--", alpha=0.7)
    ax.text(len(conditions) - 0.55, 1/3 + 0.018, "3-class chance",
            color=MUTED, fontsize=9, ha="right", alpha=0.8)

    # Pre-create invisible bars
    bars_c = []
    bars_r = []
    crowns = []
    for i, (label, c, r, w) in enumerate(conditions):
        bc = ax.bar(i - 0.18, 0, width=0.34, color=ACCENT_BLUE, alpha=0.0,
                     edgecolor=ACCENT_BLUE, linewidth=0)[0]
        br = ax.bar(i + 0.18, 0, width=0.34, color=ACCENT_RED, alpha=0.0,
                     edgecolor=ACCENT_RED, linewidth=0)[0]
        bars_c.append(bc)
        bars_r.append(br)
        # winner stamp text (created invisible)
        txt = ax.text(i, 0, "", ha="center", color=FG, fontsize=13,
                      fontweight="bold", alpha=0.0)
        crowns.append(txt)

    # Tally HUD top right
    tally_complex = fig.text(
        0.92, 0.91, "complex 0", color=ACCENT_BLUE, fontsize=14,
        ha="right", fontweight="bold",
    )
    tally_real = fig.text(
        0.92, 0.875, "real    0", color=ACCENT_RED, fontsize=14,
        ha="right", fontweight="bold", family="monospace",
    )
    fig.text(
        0.07, 0.91, "complex", color=ACCENT_BLUE, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.07, 0.876, "best real", color=ACCENT_RED, fontsize=12, fontweight="bold",
    )

    summary_text = fig.text(
        0.5, 0.04, "", color=FG, fontsize=12, ha="center", alpha=0.0,
    )

    def smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def update(frame):
        cumulative_c = 0
        cumulative_r = 0
        for i, (_, c, r, w) in enumerate(conditions):
            start = i * (reveal_frames_per + hold_after_reveal)
            local = frame - start
            if local <= 0:
                progress = 0.0
            elif local >= reveal_frames_per:
                progress = 1.0
            else:
                progress = smoothstep(local / reveal_frames_per)

            bars_c[i].set_height(c * progress)
            bars_r[i].set_height(r * progress)
            bars_c[i].set_alpha(min(0.95, progress * 1.1))
            bars_r[i].set_alpha(min(0.95, progress * 1.1))

            if progress >= 1.0:
                if w == "complex":
                    crowns[i].set_text("✦ complex")
                    crowns[i].set_color(ACCENT_BLUE)
                    cumulative_c += 1
                else:
                    crowns[i].set_text("● real")
                    crowns[i].set_color(ACCENT_RED)
                    cumulative_r += 1
                crowns[i].set_position((i, max(c, r) + 0.04))
                crowns[i].set_alpha(0.95)
            else:
                crowns[i].set_alpha(0.0)

        tally_complex.set_text(f"complex {cumulative_c}")
        tally_real.set_text(f"real    {cumulative_r}")

        if frame >= len(conditions) * (reveal_frames_per + hold_after_reveal):
            summary_text.set_text(
                f"complex wins {cumulative_c}/{len(conditions)}  —  ranking inverts on QAM, low-SNR, fixed rotation"
            )
            summary_text.set_alpha(0.95)
        else:
            summary_text.set_alpha(0.0)
        return bars_c + bars_r + crowns + [tally_complex, tally_real, summary_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000 / fps, blit=False)
    paths = _save_anim(anim, "anim_conditional_advantage", fps=fps)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    return paths


# ============================================================================
# 4. five-activation sweep animation
# ============================================================================
def anim_activation_sweep():
    """Replay first 30 steps for all 5 activations × 9 real-baseline seeds.

    Each column = one activation. Top row = head.weight gradient (log scale),
    bottom row = train loss with the dead-loss threshold overlaid. Stable
    activations (modrelu, zrelu) are highlighted in lime; unstable
    (crelu, cardioid, siglog) in amber. Per-column dead-seed counter
    increments as seeds cross the threshold.
    """
    _style()
    activations = [
        # name, lr, regime ("unstable"/"stable")
        ("crelu",    0.0236, "unstable"),
        ("cardioid", 0.0236, "unstable"),
        ("siglog",   0.0398, "unstable"),
        ("modrelu",  0.0079, "stable"),
        ("zrelu",    0.0024, "stable"),
    ]
    data = {act: _collect(RESULTS / "synthetic_rf_telemetry" / act)
            for act, *_ in activations}

    n_steps = 30
    fps = 12
    pad_frames = 18

    fig = plt.figure(figsize=(17, 8.0))
    fig.suptitle(
        "Five-activation sweep — same task, same seeds, same step budget",
        fontsize=16, fontweight="bold", color=FG, y=0.965,
    )
    fig.text(
        0.5, 0.92,
        "real-baseline trajectories (9 seeds per panel) at each activation's "
        "matched-shared-trial selected hyperparameters",
        ha="center", color=MUTED, fontsize=11,
    )

    gs = fig.add_gridspec(
        2, 5, hspace=0.30, wspace=0.22,
        left=0.05, right=0.985, top=0.86, bottom=0.10,
    )

    grad_axes, loss_axes = [], []
    dead_counters = []  # text artists per column
    for j, (act, lr, regime) in enumerate(activations):
        ax_g = fig.add_subplot(gs[0, j])
        ax_l = fig.add_subplot(gs[1, j])
        regime_color = ACCENT_AMBER if regime == "unstable" else ACCENT_LIME

        ax_g.set_yscale("log")
        ax_g.set_ylim(1e-7, 1e2)
        ax_g.set_xlim(-0.5, n_steps - 0.5)
        ax_g.grid(True, which="both")
        ax_g.axhline(1.0, color=MUTED, lw=0.6, ls="--", alpha=0.5)
        title = f"{act}\n" + r"lr = " + f"{lr:.4f}"
        ax_g.set_title(title, color=regime_color, fontsize=12, pad=8)
        if j == 0:
            ax_g.set_ylabel(r"$\|\nabla_{\mathrm{head.weight}} L\|$", color=FG)
        else:
            ax_g.set_yticklabels([])

        ax_l.set_xlim(-0.5, n_steps - 0.5)
        ax_l.set_ylim(0, 6.0 if regime == "unstable" else 1.3)
        ax_l.grid(True)
        ax_l.axhline(DEAD_LOSS_THRESHOLD, color=ACCENT_RED, lw=1.0, ls=":", alpha=0.85)
        ax_l.set_xlabel("step", color=FG)
        if j == 0:
            ax_l.set_ylabel("train loss", color=FG)

        # In-panel regime tag
        ax_g.text(
            0.04, 0.94, regime, transform=ax_g.transAxes,
            color=regime_color, fontsize=9, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc=PANEL_BG,
                      ec=regime_color, lw=0.8, alpha=0.85),
        )

        # Per-column live dead-seed counter, shown at top of loss panel
        counter = ax_l.text(
            0.04, 0.94, "0/9 dead", transform=ax_l.transAxes,
            color=MUTED, fontsize=10, fontweight="bold", va="top",
        )
        dead_counters.append(counter)

        grad_axes.append(ax_g)
        loss_axes.append(ax_l)

    # Pre-create line artists per (activation, trace).
    grad_lines: dict[str, list] = {}
    loss_lines: dict[str, list] = {}
    for j, (act, *_unused) in enumerate(activations):
        gls, lls = [], []
        for _ in data[act]:
            (lg,) = grad_axes[j].plot([], [], color=ACCENT_TEAL, lw=1.2, alpha=0.7)
            (ll,) = loss_axes[j].plot([], [], color=ACCENT_TEAL, lw=1.2, alpha=0.7)
            gls.append(lg)
            lls.append(ll)
        grad_lines[act] = gls
        loss_lines[act] = lls

    # Step clock + global tally HUD
    step_clock = fig.text(
        0.985, 0.92, "step  0", fontsize=18, color=ACCENT_PURPLE,
        ha="right", va="center", fontweight="bold", family="monospace",
    )
    global_summary = fig.text(
        0.5, 0.025, "", color=FG, fontsize=12, ha="center", alpha=0.0,
    )

    n_frames = n_steps + pad_frames

    def update(frame):
        s = min(frame, n_steps - 1)
        step_clock.set_text(f"step {s:>2}")

        total_dead = 0
        for j, (act, _lr, _regime) in enumerate(activations):
            n_dead = 0
            for t, lg, ll in zip(data[act], grad_lines[act], loss_lines[act]):
                lg.set_data(t["step"][: s + 1], t["head"][: s + 1])
                ll.set_data(t["step"][: s + 1], t["loss"][: s + 1])
                will_die = t["dead"] and s >= 2
                if will_die:
                    lg.set_color(ACCENT_RED)
                    lg.set_alpha(0.95)
                    lg.set_linewidth(2.0)
                    ll.set_color(ACCENT_RED)
                    ll.set_alpha(0.95)
                    ll.set_linewidth(2.0)
                    n_dead += 1
                else:
                    lg.set_color(ACCENT_TEAL)
                    lg.set_alpha(0.7)
                    lg.set_linewidth(1.3)
                    ll.set_color(ACCENT_TEAL)
                    ll.set_alpha(0.7)
                    ll.set_linewidth(1.3)
            if n_dead > 0:
                dead_counters[j].set_color(ACCENT_RED)
                dead_counters[j].set_text(f"{n_dead}/9 dead")
            else:
                dead_counters[j].set_color(ACCENT_LIME)
                dead_counters[j].set_text("0/9 dead")
            total_dead += n_dead

        if frame >= n_steps - 1:
            unstable_dead = sum(
                1
                for act, _, regime in activations
                if regime == "unstable"
                for t in data[act] if t["dead"]
            )
            stable_dead = sum(
                1
                for act, _, regime in activations
                if regime == "stable"
                for t in data[act] if t["dead"]
            )
            global_summary.set_text(
                f"end of replay  ▸  unstable activations: {unstable_dead}/27 dead seeds   "
                f"·   stable activations: {stable_dead}/18 dead seeds"
            )
            global_summary.set_alpha(0.95)
        else:
            global_summary.set_alpha(0.0)

        all_artists = []
        for act, *_ in activations:
            all_artists.extend(grad_lines[act])
            all_artists.extend(loss_lines[act])
        return all_artists + dead_counters + [step_clock, global_summary]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )
    paths = _save_anim(anim, "anim_activation_sweep", fps=fps)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    return paths


# ============================================================================
# 4b. five-activation sweep — full training budget per activation
# ============================================================================
def anim_activation_sweep_full(
    *,
    figsize=(17, 8.0),
    out_name="anim_activation_sweep_full",
    title_size=16,
    subtitle_size=11,
    panel_title_size=12,
):
    """Like anim_activation_sweep, but uses each activation's full
    matched-shared-trial step budget (200 / 400 / 800) instead of the
    200-step cap. Animation paces slowly through the first 30 steps
    (where the dead-seed explosion happens) and then sweeps through
    the rest of training to convergence.

    Pass `figsize=(16, 9)` and `out_name="..._16x9"` for the
    Twitter-friendly aspect ratio.
    """
    _style()
    activations = [
        ("crelu",    0.0236, "unstable"),
        ("cardioid", 0.0236, "unstable"),
        ("siglog",   0.0398, "unstable"),
        ("modrelu",  0.0079, "stable"),
        ("zrelu",    0.0024, "stable"),
    ]
    # Each trace is the FULL run (steps from sweep selection).
    data = {act: _collect(RESULTS / "synthetic_rf_telemetry_full" / act)
            for act, *_ in activations}
    # Per-activation full step counts.
    panel_steps = {act: int(data[act][0]["step"][-1]) + 1 for act, *_ in activations}
    max_panel_steps = max(panel_steps.values())

    fps = 24
    early_frames = 60   # ~2.5s slow pass over steps 0..30
    late_frames = 144   # ~6s sweep through the rest
    hold_frames = 30    # ~1.2s final hold
    total_frames = early_frames + late_frames + hold_frames

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        "Five-activation sweep — full training trajectory per activation",
        fontsize=title_size, fontweight="bold", color=FG, y=0.965,
    )
    fig.text(
        0.5, 0.92,
        "real-baseline trajectories (9 seeds per panel) trained to each "
        "activation's matched-shared-trial selected step budget",
        ha="center", color=MUTED, fontsize=subtitle_size,
    )

    gs = fig.add_gridspec(
        2, 5, hspace=0.30, wspace=0.22,
        left=0.05, right=0.985, top=0.86, bottom=0.10,
    )

    grad_axes, loss_axes = [], []
    dead_counters = []
    for j, (act, lr, regime) in enumerate(activations):
        n = panel_steps[act]
        ax_g = fig.add_subplot(gs[0, j])
        ax_l = fig.add_subplot(gs[1, j])
        regime_color = ACCENT_AMBER if regime == "unstable" else ACCENT_LIME

        ax_g.set_yscale("log")
        ax_g.set_ylim(1e-7, 1e2)
        ax_g.set_xlim(-n * 0.02, n * 1.02)
        ax_g.grid(True, which="both")
        ax_g.axhline(1.0, color=MUTED, lw=0.6, ls="--", alpha=0.5)
        title = f"{act}\nlr = {lr:.4f}   ·   {n} steps"
        ax_g.set_title(title, color=regime_color, fontsize=panel_title_size, pad=8)
        if j == 0:
            ax_g.set_ylabel(r"$\|\nabla_{\mathrm{head.weight}} L\|$", color=FG)
        else:
            ax_g.set_yticklabels([])

        ax_l.set_xlim(-n * 0.02, n * 1.02)
        ax_l.set_ylim(0, 6.0 if regime == "unstable" else 1.3)
        ax_l.grid(True)
        ax_l.axhline(DEAD_LOSS_THRESHOLD, color=ACCENT_RED, lw=1.0, ls=":", alpha=0.85)
        ax_l.set_xlabel("step", color=FG)
        if j == 0:
            ax_l.set_ylabel("train loss", color=FG)

        ax_g.text(
            0.04, 0.94, regime, transform=ax_g.transAxes,
            color=regime_color, fontsize=9, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc=PANEL_BG,
                      ec=regime_color, lw=0.8, alpha=0.85),
        )
        counter = ax_l.text(
            0.05, 0.94, "0/9 dead", transform=ax_l.transAxes,
            color=MUTED, fontsize=10, fontweight="bold", va="top",
        )
        dead_counters.append(counter)

        grad_axes.append(ax_g)
        loss_axes.append(ax_l)

    grad_lines: dict[str, list] = {}
    loss_lines: dict[str, list] = {}
    for j, (act, *_unused) in enumerate(activations):
        gls, lls = [], []
        for _ in data[act]:
            (lg,) = grad_axes[j].plot([], [], color=ACCENT_TEAL, lw=1.2, alpha=0.7)
            (ll,) = loss_axes[j].plot([], [], color=ACCENT_TEAL, lw=1.2, alpha=0.7)
            gls.append(lg)
            lls.append(ll)
        grad_lines[act] = gls
        loss_lines[act] = lls

    # Two clocks: panel-relative step in big purple, fraction-of-training in small mute.
    step_clock = fig.text(
        0.985, 0.93, "max step  0",
        fontsize=16, color=ACCENT_PURPLE, ha="right", va="center",
        fontweight="bold", family="monospace",
    )
    progress_label = fig.text(
        0.985, 0.895, "0% of training",
        fontsize=10, color=MUTED, ha="right", va="center", family="monospace",
    )

    global_summary = fig.text(
        0.5, 0.025, "", color=FG, fontsize=12, ha="center", alpha=0.0,
    )

    def update(frame):
        # Map global frame -> per-panel step index. Slow pass through 0..30
        # for all panels, then linear pass to each panel's n_steps.
        if frame < early_frames:
            # frame 0 -> step 0; frame early_frames-1 -> step ~30 (or n if smaller)
            early_target_step = int(round((frame / max(1, early_frames - 1)) * 30))

            def step_for(act):
                return min(early_target_step, panel_steps[act] - 1)
            global_progress = early_target_step / max_panel_steps
        elif frame < early_frames + late_frames:
            local = frame - early_frames
            t = local / max(1, late_frames - 1)

            def step_for(act):
                n = panel_steps[act]
                start = min(30, n - 1)
                return min(int(round(start + t * (n - 1 - start))), n - 1)
            global_progress = (30 + t * (max_panel_steps - 30)) / max_panel_steps
        else:
            def step_for(act):
                return panel_steps[act] - 1
            global_progress = 1.0

        max_step_shown = max(step_for(act) for act, *_ in activations)
        step_clock.set_text(f"max step {max_step_shown:>4}")
        progress_label.set_text(f"{int(round(global_progress * 100)):>3}% of training")

        for j, (act, _lr, _regime) in enumerate(activations):
            s = step_for(act)
            n_dead = 0
            for tr, lg, ll in zip(data[act], grad_lines[act], loss_lines[act]):
                lg.set_data(tr["step"][: s + 1], tr["head"][: s + 1])
                ll.set_data(tr["step"][: s + 1], tr["loss"][: s + 1])
                will_die = tr["dead"] and s >= 2
                if will_die:
                    lg.set_color(ACCENT_RED); lg.set_alpha(0.95); lg.set_linewidth(2.0)
                    ll.set_color(ACCENT_RED); ll.set_alpha(0.95); ll.set_linewidth(2.0)
                    n_dead += 1
                else:
                    lg.set_color(ACCENT_TEAL); lg.set_alpha(0.7); lg.set_linewidth(1.3)
                    ll.set_color(ACCENT_TEAL); ll.set_alpha(0.7); ll.set_linewidth(1.3)
            if n_dead > 0:
                dead_counters[j].set_color(ACCENT_RED)
                dead_counters[j].set_text(f"{n_dead}/9 dead")
            else:
                dead_counters[j].set_color(ACCENT_LIME)
                dead_counters[j].set_text("0/9 dead")

        if frame >= early_frames + late_frames:
            unstable_dead = sum(
                1
                for act, _, regime in activations
                if regime == "unstable"
                for tr in data[act] if tr["dead"]
            )
            stable_dead = sum(
                1
                for act, _, regime in activations
                if regime == "stable"
                for tr in data[act] if tr["dead"]
            )
            global_summary.set_text(
                f"end of training  ▸  unstable activations: {unstable_dead}/27 dead seeds   "
                f"·   stable activations: {stable_dead}/18 dead seeds"
            )
            global_summary.set_alpha(0.95)
        else:
            global_summary.set_alpha(0.0)

        all_artists = []
        for act, *_ in activations:
            all_artists.extend(grad_lines[act])
            all_artists.extend(loss_lines[act])
        return all_artists + dead_counters + [step_clock, progress_label, global_summary]

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False
    )
    paths = _save_anim(anim, out_name, fps=fps)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    return paths


# ============================================================================
# 5. activation sweep — Twitter cut (intro + sweep + outro)
# ============================================================================
def anim_activation_sweep_twitter():
    """Self-contained 12-second Twitter cut.

    Phase 1 (intro, 3.0 s): title card with the setup question.
    Phase 2 (replay, 5.0 s): the 5×2 sweep replay.
    Phase 3 (outro, 4.0 s): takeaway / punchline card.
    """
    _style()
    activations = [
        ("crelu",    0.0236, "unstable"),
        ("cardioid", 0.0236, "unstable"),
        ("siglog",   0.0398, "unstable"),
        ("modrelu",  0.0079, "stable"),
        ("zrelu",    0.0024, "stable"),
    ]
    data = {act: _collect(RESULTS / "synthetic_rf_telemetry" / act)
            for act, *_ in activations}

    fps = 12
    n_steps = 30
    intro_frames = 36
    sweep_frames = n_steps + 18  # replay + hold
    outro_frames = 48
    total_frames = intro_frames + sweep_frames + outro_frames

    fig = plt.figure(figsize=(17, 8.0))

    # --- panels (always created; we'll just show/hide with alpha)
    fig.suptitle("", color=FG)
    main_title = fig.text(
        0.5, 0.965, "Five-activation sweep — same task, same seeds, same step budget",
        fontsize=16, fontweight="bold", color=FG, ha="center", alpha=0.0,
    )
    main_subtitle = fig.text(
        0.5, 0.92,
        "real-baseline trajectories (9 seeds per panel) at each activation's "
        "matched-shared-trial selected hyperparameters",
        ha="center", color=MUTED, fontsize=11, alpha=0.0,
    )

    gs = fig.add_gridspec(
        2, 5, hspace=0.30, wspace=0.22,
        left=0.05, right=0.985, top=0.86, bottom=0.10,
    )

    grad_axes, loss_axes, dead_counters = [], [], []
    panel_artists = []  # everything we'll fade in/out

    for j, (act, lr, regime) in enumerate(activations):
        ax_g = fig.add_subplot(gs[0, j])
        ax_l = fig.add_subplot(gs[1, j])
        regime_color = ACCENT_AMBER if regime == "unstable" else ACCENT_LIME

        ax_g.set_yscale("log")
        ax_g.set_ylim(1e-7, 1e2)
        ax_g.set_xlim(-0.5, n_steps - 0.5)
        ax_g.grid(True, which="both")
        ax_g.axhline(1.0, color=MUTED, lw=0.6, ls="--", alpha=0.5)
        ax_g.set_title(f"{act}\nlr = {lr:.4f}", color=regime_color, fontsize=12, pad=8)
        if j == 0:
            ax_g.set_ylabel(r"$\|\nabla_{\mathrm{head.weight}} L\|$", color=FG)
        else:
            ax_g.set_yticklabels([])

        ax_l.set_xlim(-0.5, n_steps - 0.5)
        ax_l.set_ylim(0, 6.0 if regime == "unstable" else 1.3)
        ax_l.grid(True)
        ax_l.axhline(DEAD_LOSS_THRESHOLD, color=ACCENT_RED, lw=1.0, ls=":", alpha=0.85)
        ax_l.set_xlabel("step", color=FG)
        if j == 0:
            ax_l.set_ylabel("train loss", color=FG)

        ax_g.text(
            0.04, 0.94, regime, transform=ax_g.transAxes,
            color=regime_color, fontsize=9, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc=PANEL_BG, ec=regime_color, lw=0.8, alpha=0.85),
        )
        counter = ax_l.text(
            0.05, 0.94, "0/9 dead", transform=ax_l.transAxes,
            color=MUTED, fontsize=10, fontweight="bold", va="top",
        )
        dead_counters.append(counter)
        grad_axes.append(ax_g)
        loss_axes.append(ax_l)

    # Pre-create line artists per (activation, trace).
    grad_lines, loss_lines = {}, {}
    for j, (act, *_) in enumerate(activations):
        gls, lls = [], []
        for _ in data[act]:
            (lg,) = grad_axes[j].plot([], [], color=ACCENT_TEAL, lw=1.2, alpha=0.0)
            (ll,) = loss_axes[j].plot([], [], color=ACCENT_TEAL, lw=1.2, alpha=0.0)
            gls.append(lg)
            lls.append(ll)
        grad_lines[act] = gls
        loss_lines[act] = lls

    step_clock = fig.text(
        0.985, 0.92, "step  0", fontsize=18, color=ACCENT_PURPLE,
        ha="right", va="center", fontweight="bold", family="monospace", alpha=0.0,
    )

    # --- intro overlay -------------------------------------------------------
    intro_ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    intro_ax.set_xticks([]); intro_ax.set_yticks([])
    intro_ax.set_xlim(0, 1); intro_ax.set_ylim(0, 1)
    intro_ax.set_facecolor(BG)
    intro_ax.patch.set_alpha(1.0)
    intro_q = intro_ax.text(
        0.5, 0.66, "“Why does my CVNN beat real baselines by 23 pp?”",
        ha="center", va="center", fontsize=26, color=FG, fontweight="bold", style="italic",
    )
    intro_a = intro_ax.text(
        0.5, 0.55,
        "Maybe it's the activation. Let's sweep five of them.",
        ha="center", va="center", fontsize=18, color=ACCENT_BLUE, fontweight="bold",
    )
    intro_setup = intro_ax.text(
        0.5, 0.40,
        "Same architecture. Same hyperparameter search. Same step budget.\n"
        "Real baselines use ReLU; CVNN varies the complex activation.\n"
        "Watching the first 30 training steps for 9 real-baseline seeds per activation.",
        ha="center", va="center", fontsize=14, color=MUTED, linespacing=1.55,
    )
    intro_hint = intro_ax.text(
        0.5, 0.18,
        "▶  press play",
        ha="center", va="center", fontsize=12, color=ACCENT_PURPLE, fontweight="bold",
    )
    intro_artists = [intro_q, intro_a, intro_setup, intro_hint]

    # --- outro overlay -------------------------------------------------------
    outro_ax = fig.add_axes([0, 0, 1, 1], frameon=False, zorder=10)
    outro_ax.set_xticks([]); outro_ax.set_yticks([])
    outro_ax.set_xlim(0, 1); outro_ax.set_ylim(0, 1)
    outro_ax.set_facecolor(BG)
    outro_ax.patch.set_alpha(0.0)
    outro_title = outro_ax.text(
        0.5, 0.78, "What just happened?",
        ha="center", va="center", fontsize=22, color=FG, fontweight="bold", alpha=0.0,
    )
    outro_lines = [
        ("3 activations  →  real baselines die at step 1   (8 / 27 dead seeds)", ACCENT_AMBER, 0.62),
        ("2 activations  →  real baselines train fine       (0 / 18 dead seeds)", ACCENT_LIME,  0.54),
    ]
    outro_textboxes = []
    for txt, col, y in outro_lines:
        t = outro_ax.text(0.5, y, txt, ha="center", va="center",
                          fontsize=18, color=col, fontweight="bold",
                          family="monospace", alpha=0.0)
        outro_textboxes.append(t)
    outro_punch = outro_ax.text(
        0.5, 0.36,
        "Same data. Same code. Same task.",
        ha="center", va="center", fontsize=18, color=FG, alpha=0.0, fontweight="bold",
    )
    outro_punch2 = outro_ax.text(
        0.5, 0.28,
        "The 23 pp “complex advantage” was hiding an optimization-side quirk.",
        ha="center", va="center", fontsize=18, color=ACCENT_BLUE, alpha=0.0, fontweight="bold",
    )
    outro_signature = outro_ax.text(
        0.5, 0.12,
        "Independent per-family hp selection collapses the gap to +2.46 pp.",
        ha="center", va="center", fontsize=14, color=MUTED, alpha=0.0, style="italic",
    )
    outro_artists = [outro_title, *outro_textboxes, outro_punch, outro_punch2, outro_signature]

    def _set_intro_alpha(a):
        intro_ax.patch.set_alpha(a)
        for art in intro_artists:
            art.set_alpha(a)

    def _set_outro_alpha(a, stagger_progress=1.0):
        # Fade in title first, then bullets, then punchlines.
        outro_ax.patch.set_alpha(min(a, 0.97))
        outro_title.set_alpha(min(1.0, stagger_progress * 1.5))
        for k, art in enumerate(outro_textboxes):
            t = max(0.0, stagger_progress - 0.18 - k * 0.12) * 2.0
            art.set_alpha(min(1.0, t))
        outro_punch.set_alpha(min(1.0, max(0.0, stagger_progress - 0.5) * 2.5))
        outro_punch2.set_alpha(min(1.0, max(0.0, stagger_progress - 0.6) * 2.5))
        outro_signature.set_alpha(min(1.0, max(0.0, stagger_progress - 0.75) * 4.0))

    def _set_panels_alpha(a):
        # Toggle the panel axes wholesale so the intro/outro overlays read clean.
        for ax in (*grad_axes, *loss_axes):
            ax.set_visible(a > 0.02)
        main_title.set_alpha(a)
        main_subtitle.set_alpha(a)
        step_clock.set_alpha(a)
        for j in range(len(activations)):
            dead_counters[j].set_alpha(a)

    def update(frame):
        if frame < intro_frames:
            # intro: full-screen title card. Panels hidden until last 6 frames.
            fade_out = max(0.0, (frame - (intro_frames - 6)) / 6.0)
            _set_intro_alpha(1.0 - fade_out)
            _set_outro_alpha(0.0, stagger_progress=0.0)
            _set_panels_alpha(fade_out)  # 0 during card, ramps up at end
            for act, *_ in activations:
                for lg, ll in zip(grad_lines[act], loss_lines[act]):
                    lg.set_alpha(0.0); ll.set_alpha(0.0)
            step_clock.set_text("step  0")
            return intro_artists

        elif frame < intro_frames + sweep_frames:
            local = frame - intro_frames
            s = min(local, n_steps - 1)
            _set_intro_alpha(0.0)
            _set_outro_alpha(0.0, stagger_progress=0.0)
            _set_panels_alpha(1.0)
            step_clock.set_text(f"step {s:>2}")

            for j, (act, _lr, _regime) in enumerate(activations):
                n_dead = 0
                for t, lg, ll in zip(data[act], grad_lines[act], loss_lines[act]):
                    lg.set_data(t["step"][: s + 1], t["head"][: s + 1])
                    ll.set_data(t["step"][: s + 1], t["loss"][: s + 1])
                    will_die = t["dead"] and s >= 2
                    if will_die:
                        lg.set_color(ACCENT_RED); lg.set_alpha(0.95); lg.set_linewidth(2.0)
                        ll.set_color(ACCENT_RED); ll.set_alpha(0.95); ll.set_linewidth(2.0)
                        n_dead += 1
                    else:
                        lg.set_color(ACCENT_TEAL); lg.set_alpha(0.7); lg.set_linewidth(1.3)
                        ll.set_color(ACCENT_TEAL); ll.set_alpha(0.7); ll.set_linewidth(1.3)
                if n_dead > 0:
                    dead_counters[j].set_color(ACCENT_RED)
                    dead_counters[j].set_text(f"{n_dead}/9 dead")
                else:
                    dead_counters[j].set_color(ACCENT_LIME)
                    dead_counters[j].set_text("0/9 dead")
            return []

        else:
            # outro phase: clean takeaway card — hide panels entirely.
            local = frame - intro_frames - sweep_frames
            t = min(1.0, local / (outro_frames * 0.55))
            _set_intro_alpha(0.0)
            _set_outro_alpha(t, stagger_progress=t)
            # Hide panels once outro patch is opaque enough.
            panels_visible = max(0.0, 1.0 - t * 4.0)
            _set_panels_alpha(panels_visible)
            for act, *_ in activations:
                for lg, ll in zip(grad_lines[act], loss_lines[act]):
                    lg.set_alpha(max(0.0, lg.get_alpha() - t * 0.6) if lg.get_alpha() else 0.0)
                    ll.set_alpha(max(0.0, ll.get_alpha() - t * 0.6) if ll.get_alpha() else 0.0)
            return outro_artists

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False
    )
    paths = _save_anim(anim, "anim_activation_sweep_twitter", fps=fps)
    plt.close(fig)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    return paths


def main() -> None:
    print("Generating animations…")
    print("[1/5] dead-seed mechanism")
    anim_dead_seed_mechanism()
    print("[2/5] selection-rule morph")
    anim_selection_rule_morph()
    print("[3/5] conditional advantage")
    anim_conditional_advantage()
    print("[4/6] five-activation sweep (200-step cap)")
    anim_activation_sweep()
    print("[5/7] five-activation sweep (full step budget)")
    anim_activation_sweep_full()
    print("[6/7] five-activation sweep (full step budget, 16:9 for Twitter)")
    anim_activation_sweep_full(
        figsize=(16, 9), out_name="anim_activation_sweep_full_16x9",
        title_size=18, subtitle_size=12, panel_title_size=13,
    )
    print("[7/7] activation sweep — Twitter cut (with intro + outro)")
    anim_activation_sweep_twitter()
    print("done.")


if __name__ == "__main__":
    main()
