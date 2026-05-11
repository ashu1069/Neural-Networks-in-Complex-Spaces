# Paper build

LaTeX source for the NeurIPS Concept & Feasibility submission.

## Build

```bash
make pdf      # one-shot build (latexmk)
make watch    # continuous rebuild
make clean    # remove build artifacts
```

If `latexmk` is unavailable, run the full pass sequence directly:

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

## Figures

`figures/` contains symlinks into `../results/figures/`. After regenerating
artifacts via `scripts/generate_paper_figures.py`, refresh the symlinks:

```bash
make links
```

The symlinks store paths of the form `../../results/figures/<name>.png`,
which resolve correctly from inside `paper/figures/`.

## NeurIPS style

`paper.tex` loads `neurips_2025.sty` (the 2026 style file is not yet
released as of writing). The build uses the `[preprint]` option so the
PDF is non-anonymous and line-number-free for working drafts. Before
submission:

1. Replace `\usepackage[preprint]{neurips_2025}` with
   `\usepackage{neurips_2026}` (or whatever the 2026 file is named).
2. Add the appropriate track option once the 2026 template defines a
   Concept & Feasibility option (the 2025 file does not).

The body is drop-in compatible: section structure, figure / table
layout, and citation style match the NeurIPS template's conventions.

## Notes

- `microtype` is loaded with `[disable]` because TeX Live 2026 basic ships
  bitmap fonts on this system, which break auto font expansion. NeurIPS
  style ships scalable Type 1 fonts via the `times` package, so the build
  is fine; the `[disable]` is a local workaround.
- `cleveref` is replaced by lightweight `\providecommand{\Cref}` /
  `\providecommand{\cref}` shims to avoid the dependency.
- The bundled `neurips_2025.sty` has been patched to remove the
  `\usepackage{environ}` dependency (TeX Live 2026 basic does not
  ship `environ.sty` / `trimspaces.sty`); the patched version uses a
  plain `\newenvironment{ack}` that is behaviourally identical for our
  use. Diff vs. upstream: lines 100--106. Reverse before submission if
  the conference build farm has `environ` installed (most do).
- One BibTeX entry (`krzyston2020complex`) emits an empty-journal warning
  pending venue verification; flagged in `references.bib`.

## Appendices

The paper includes 9 appendix sections (A--I) covering the phase
equivariance proof in full, hyperparameter search spaces, architecture
details, per-activation results across all four baseline families,
gradient telemetry per-layer per-step traces, the synthetic-data
replication of the threshold mechanism, RadioML loader edge cases,
the manifest schema and CI guards, and the full-archive scale-up plan.
The appendix is unbounded under NeurIPS rules; after the RF
representation stress-test and quantum pilot updates, the working draft
builds to 22 pages total.
