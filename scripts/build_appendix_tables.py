"""Emit the paper's data-driven appendix tables from finished sweep directories.

Reads only `summary.json`, `index.json` and `manifest.json`, so it never
retrains, and writes a single LaTeX fragment that `paper/main.tex` includes.
Regenerate after any sweep is re-run so the appendix cannot drift from the
evidence:

    .venv/bin/python scripts/build_appendix_tables.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

FAMILY_LABEL = {
    "complex": "complex",
    "real_equivariant": "real, $U(1)$-constrained",
    "real_stacked": "stacked real",
    "real_matched_params": "param-matched real",
    "real_matched_flops": "FLOP-matched real",
    "real_polar": "polar real",
    "real_phase": "phase-only real",
    "real_magnitude": "magnitude-only real",
}
FAMILY_ORDER = tuple(FAMILY_LABEL)

PER_SNR_RUNS = (
    ("radioml_geom7_crelu", "Seven classes, length 128, CReLU", "snr:geom7"),
    ("radioml_geom7_zrelu", "Seven classes, length 128, ZReLU", "snr:geom7z"),
    ("radioml_geom10_crelu", "Ten classes, length 128, CReLU", "snr:geom10"),
    ("radioml_geom7_len256_crelu", "Seven classes, length 256, CReLU", "snr:len256"),
)

EEG_LR_RUNS = (
    ("neuro_eeg_lr_0.0003", "0.0003"),
    ("neuro_eeg_lr_0.001", "0.001"),
    ("neuro_eeg_lr_0.003", "0.003"),
    ("neuro_eeg_lr_0.01", "0.01"),
    ("neuro_eeg_lr_0.03", "0.03"),
)

PILOT_FAMILIES = (
    "complex",
    "real_stacked",
    "real_phase",
    "real_polar",
    "real_magnitude",
)
# Short column heads: five families of full labels overrun the 5.5in ICLR
# column, so the caption carries the long names instead.
PILOT_SHORT = {
    "complex": "complex",
    "real_stacked": "stacked",
    "real_phase": "phase",
    "real_polar": "polar",
    "real_magnitude": "magnitude",
}


def _obj(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    return value


def _arr(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"expected a JSON array, got {type(value).__name__}")
    return value


def _read(path: Path) -> dict[str, Any]:
    return _obj(json.loads(path.read_text()))


def _per_snr(run_dir: Path) -> tuple[dict[str, dict[int, float]], list[int]]:
    summary = _read(run_dir / "summary.json")
    out: dict[str, dict[int, float]] = {}
    levels: list[int] = []
    for entry_value in _arr(summary["matched_selections"]):
        entry = _obj(entry_value)
        extra = _obj(entry["selected_extra"])
        by_snr: dict[int, list[float]] = {}
        for seed_value in _arr(extra["test_accuracy_by_snr_db_per_seed"]):
            for snr_text, accuracy in _obj(seed_value).items():
                by_snr.setdefault(int(snr_text), []).append(float(accuracy))
        out[str(entry["family"])] = {k: statistics.fmean(v) for k, v in by_snr.items()}
        levels = sorted(by_snr)
    return out, levels


def _pilot_rows(run_dir: Path) -> list[tuple[str, dict[str, float]]]:
    """Per-condition accuracy by family, from the suite's `results` list.

    Each result records one `<family>_accuracy` key per family it ran, so the
    families present vary by condition and by suite.
    """

    index = _read(run_dir / "index.json")
    rows: list[tuple[str, dict[str, float]]] = []
    for entry_value in _arr(index["results"]):
        entry = _obj(entry_value)
        accuracies = {
            key[: -len("_accuracy")]: float(value)
            for key, value in entry.items()
            if key.endswith("_accuracy") and isinstance(value, (int, float))
        }
        rows.append((str(entry["condition_id"]), accuracies))
    return rows


def _tex_name(text: str) -> str:
    return text.replace("_", r"\_")


def _per_snr_table(run_dir: Path, caption: str, label: str) -> list[str]:
    families, levels = _per_snr(run_dir)
    cols = "l" + "r" * len(levels)
    head = " & ".join(f"${lvl}$" for lvl in levels)
    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        rf"\caption{{{caption} Matched-shared-trial selection, mean over seeds. "
        rf"Source: \texttt{{results/{_tex_name(run_dir.name)}}}.}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        rf"Family & \multicolumn{{{len(levels)}}}{{c}}{{SNR (dB)}} \\",
        rf"\cmidrule(l){{2-{len(levels) + 1}}}",
        rf" & {head} \\",
        r"\midrule",
    ]
    for family in FAMILY_ORDER:
        if family not in families:
            continue
        cells = " & ".join(f"${families[family][lvl]:.3f}$" for lvl in levels)
        lines.append(rf"{FAMILY_LABEL[family]} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def _pilot_table(run_dir: Path, caption: str, label: str) -> list[str]:
    rows = _pilot_rows(run_dir)
    present = [f for f in PILOT_FAMILIES if any(f in acc for _, acc in rows)]
    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        rf"\caption{{{caption} Columns are the complex model and the "
        rf"stacked-real, phase-only, polar and magnitude-only real baselines. "
        rf"Source: \texttt{{results/{_tex_name(run_dir.name)}}}.}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{l{'r' * len(present)}}}",
        r"\toprule",
        "Condition & " + " & ".join(PILOT_SHORT[f] for f in present) + r" \\",
        r"\midrule",
    ]
    for condition, acc in rows:
        cells = " & ".join(f"${acc[f]:.4f}$" if f in acc else "---" for f in present)
        lines.append(rf"\texttt{{{_tex_name(condition)}}} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def _eeg_lr_table(results_dir: Path) -> list[str]:
    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{EEG learning-rate control at the short budget. The complex "
        r"model's deficit on both amplitude-structured tasks is stable across a "
        r"$100\times$ range, so it is not a tuning artifact. Sources: "
        r"\texttt{results/neuro\_eeg\_lr\_*}.}",
        r"\label{tab:eeglr}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Amplitude event}"
        r" & \multicolumn{2}{c}{Phase--amp.\ coupling} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Learning rate & complex & polar real & complex & polar real \\",
        r"\midrule",
    ]
    for name, label in EEG_LR_RUNS:
        rows = dict(_pilot_rows(results_dir / name))
        amp, pac = rows["amplitude_event"], rows["phase_amplitude_coupling"]
        lines.append(
            rf"${label}$ & ${amp['complex']:.4f}$ & ${amp['real_polar']:.4f}$ "
            rf"& ${pac['complex']:.4f}$ & ${pac['real_polar']:.4f}$ \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def _repro_section(results_dir: Path) -> list[str]:
    runs = sorted(p for p in results_dir.iterdir() if (p / "manifest.json").is_file())
    first = _read(runs[0] / "manifest.json")
    env = _obj(first["environment"])
    lines = [
        r"\section{Reproducibility: manifests, seeds, environment}",
        "",
        r"Every run writes a manifest recording the git commit, environment, "
        r"seeds and configuration alongside its summary and plots. All runs "
        r"reported here were produced on one machine: "
        rf"\texttt{{{_tex_name(str(env['torch']))}}} on an "
        rf"{_tex_name(str(env['cuda_device']))} "
        rf"(CUDA {_tex_name(str(env['cuda_version']))}), "
        rf"Python {_tex_name(str(env['python']))}, "
        rf"\texttt{{{_tex_name(str(env['platform']))}}}, "
        rf"dtype \texttt{{{_tex_name(str(env['dtype']))}}}. "
        r"Table~\ref{tab:runs} lists the runs.",
        "",
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Committed runs, named by the suite that produced them "
        r"(\texttt{radioml\_}, \texttt{rf\_}, \texttt{physics\_quantum\_}, "
        r"\texttt{neuro\_eeg\_}). Seeds are as recorded in each manifest. The "
        r"figure and every table in this paper is generated from these "
        r"directories by \texttt{scripts/plot\_crossover\_figure.py} and "
        r"\texttt{scripts/build\_appendix\_tables.py}.}",
        r"\label{tab:runs}",
        r"\footnotesize",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Run directory & Seeds \\",
        r"\midrule",
    ]
    for run in runs:
        manifest = _read(run / "manifest.json")
        seeds = manifest.get("seeds")
        n_seeds = len(_arr(seeds)) if isinstance(seeds, list) else 0
        lines.append(rf"\texttt{{{_tex_name(run.name)}}} & ${n_seeds}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("paper/appendix_tables.tex")
    )
    args = parser.parse_args()

    results: Path = args.results_dir
    missing = [
        name
        for name, _, _ in PER_SNR_RUNS
        if not (results / name / "summary.json").is_file()
    ]
    if missing:
        print(f"missing sweeps: {', '.join(missing)}", file=sys.stderr)
        return 1

    out: list[str] = [
        "% Generated by scripts/build_appendix_tables.py -- do not edit by hand.",
        "",
        r"\section{Quantum wavefunction pilot}",
        "",
        r"The quantum pilot probes whether the representation logic transfers "
        r"outside RF. Momentum is carried by the phase gradient and is invisible "
        r"to $|\psi|$, so every phase-aware view solves it and the "
        r"magnitude-only view stays at chance; the potential-inverse and "
        r"global-phase tasks mix amplitude and phase structure, and there the "
        r"explicit polar view wins. These are representation probes, not "
        r"domain benchmarks.",
        "",
    ]
    out += _pilot_table(
        results / "physics_quantum_wavefunction_full",
        "Quantum wavefunction pilot at the converged budget. Chance is $0.25$ "
        "for momentum and $0.20$ for the potential tasks.",
        "quantumfull",
    )
    out += [
        r"\section{EEG analytic-signal pilot, full tables}",
        "",
        r"The EEG pilot separates phase locking, amplitude events and "
        r"phase--amplitude coupling on synthetic analytic signals. Phase-dominant "
        r"conditions are solved by every phase-aware view; the amplitude and "
        r"coupling conditions are the ones that discriminate, and they are where "
        r"the complex model's activation choice dominates "
        r"(Table~\ref{tab:activation}).",
        "",
    ]
    out += _pilot_table(
        results / "neuro_eeg_analytic_signal_full",
        "EEG analytic-signal pilot at the converged budget, CReLU. Chance is "
        "$0.25$ throughout.",
        "eegfull",
    )
    out += _eeg_lr_table(results)
    out += [
        r"\section{Per-SNR tables, all configurations}",
        "",
        r"Figure~\ref{fig:crossover} plots the first and fourth of these. All "
        r"four are given here so the crossover band can be read off directly: "
        r"in every configuration the complex row leads the polar row below "
        r"$2$\,dB and trails it above $6$\,dB.",
        "",
    ]
    for name, caption, label in PER_SNR_RUNS:
        out += _per_snr_table(results / name, caption + ".", label)
    out += _repro_section(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n")
    print(f"wrote {args.output} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
