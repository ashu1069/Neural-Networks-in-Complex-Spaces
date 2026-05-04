.PHONY: pdf clean watch links

PAPER ?= paper

pdf:
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error $(PAPER).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(PAPER).tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml *.synctex.gz

# Refresh symlinks from results/figures/ into paper/figures/.
# Run after a `scripts/generate_paper_figures.py` regen if the figure list
# changed. The included figures are referenced as `figures/<name>.png` from
# paper.tex.
links:
	rm -f figures/*.png
	for f in ../results/figures/*.png; do \
	  ln -sf "../$$f" "figures/$$(basename $$f)"; \
	done
