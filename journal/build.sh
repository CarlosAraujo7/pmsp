#!/usr/bin/env bash
# Compila a versao completa (sem limite de paginas): pdflatex -> bibtex -> pdflatex x2; mostra paginas, erros e referencias indefinidas.
# Uso (Git Bash): bash journal/build.sh      (MiKTeX no PATH; elsarticle.cls e elsarticle-harv.bst vem de paper/)
set -u
export PATH="/c/Users/vinic/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
python make_bib.py
cp -n ../paper/elsarticle.cls ../paper/elsarticle-harv.bst . 2>/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > main.build.log 2>&1
bibtex main >> main.build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> main.build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> main.build.log 2>&1
echo "== journal/main: $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') paginas"
grep -E "^! " main.log | head -8
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" main.log | sort | uniq | head -10
grep -c "Overfull" main.log | sed 's/^/   overfull hbox: /'
