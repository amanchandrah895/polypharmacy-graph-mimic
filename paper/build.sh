#!/usr/bin/env bash
# Regenerate all figures/tables from results/ (local, gitignored) and
# recompile the paper. Run from the repo root: bash paper/build.sh
set -e
cd "$(dirname "$0")/.."

echo "Generating figures and tables from results/..."
python3 paper/generate_paper_assets.py

echo "Compiling LaTeX (pdflatex -> bibtex -> pdflatex -> pdflatex)..."
cd latex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
rm -f main.aux main.bbl main.blg main.log main.out

echo "Done. See latex/main.pdf"
