#!/usr/bin/env bash
# Compila o artigo do CEC (pdflatex + bibtex) e mostra paginas, erros e referencias indefinidas.
# Uso: bash cec/build.sh   (MiKTeX no PATH do Git Bash)
set -u
export PATH="/c/Users/vinic/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
python make_bib.py
pdflatex -interaction=nonstopmode -halt-on-error cec2027_paper.tex > cec.build.log 2>&1
bibtex cec2027_paper >> cec.build.log 2>&1
pdflatex -interaction=nonstopmode cec2027_paper.tex >> cec.build.log 2>&1
pdflatex -interaction=nonstopmode cec2027_paper.tex >> cec.build.log 2>&1
echo "== cec2027_paper: $(pdfinfo cec2027_paper.pdf 2>/dev/null | awk '/Pages/{print $2}') paginas"
grep -E "^! " cec2027_paper.log | head -5
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" cec2027_paper.log | sort | uniq | head -10
grep -c "Overfull" cec2027_paper.log | sed 's/^/   overfull hbox: /'
