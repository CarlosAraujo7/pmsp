#!/usr/bin/env bash
# Compila a versao CONDENSADA (main_short.tex): pdflatex -> bibtex -> pdflatex x2; mostra paginas, erros e referencias indefinidas.
# Uso (Git Bash): bash journal/build_short.sh      (MiKTeX no PATH; a versao completa, o material suplementar, e' journal/build.sh)
set -u
export PATH="/c/Users/vinic/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
python make_bib.py
cp -n ../paper/elsarticle.cls ../paper/elsarticle-harv.bst . 2>/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main_short.tex > main_short.build.log 2>&1
bibtex main_short >> main_short.build.log 2>&1
pdflatex -interaction=nonstopmode main_short.tex >> main_short.build.log 2>&1
pdflatex -interaction=nonstopmode main_short.tex >> main_short.build.log 2>&1
echo "== journal/main_short: $(pdfinfo main_short.pdf 2>/dev/null | awk '/Pages/{print $2}') paginas"
grep -E "^! " main_short.log | head -8
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" main_short.log | sort | uniq | head -10
grep -c "Overfull" main_short.log | sed 's/^/   overfull hbox: /'
