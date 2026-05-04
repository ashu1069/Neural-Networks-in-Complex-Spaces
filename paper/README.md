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

`paper.tex` currently uses `\documentclass{article}` as a stand-in. At
submission time, drop in `neurips_2026.sty` from the conference site and
replace the documentclass / preamble accordingly. Section structure,
citations, and figure layout are written to be drop-in compatible.

## Notes

- `microtype` is loaded with `[disable]` because TeX Live 2026 basic ships
  bitmap fonts on this system, which break auto font expansion. NeurIPS
  style ships scalable Type 1 fonts, so this can be removed at submission.
- `cleveref` is replaced by lightweight `\providecommand{\Cref}` /
  `\providecommand{\cref}` shims to avoid the dependency.
- One BibTeX entry (`krzyston2020complex`) emits an empty-journal warning
  pending venue verification; flagged in `references.bib`.
