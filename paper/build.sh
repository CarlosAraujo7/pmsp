#!/usr/bin/env bash
# Compila o artigo principal e o suplemento (ordem: main -> supl -> main), mostra paginas, erros e referencias indefinidas.
# Uso (de qualquer pasta, no Git Bash): bash paper/build.sh
set -u
export PATH="/c/Users/vinic/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
run() {  # $1 = nome sem extensao
  xelatex -interaction=nonstopmode -halt-on-error "$1.tex" > "$1.build.log" 2>&1
  bibtex "$1" >> "$1.build.log" 2>&1
  xelatex -interaction=nonstopmode "$1.tex" >> "$1.build.log" 2>&1
  xelatex -interaction=nonstopmode "$1.tex" >> "$1.build.log" 2>&1
}
run learnheuristic_pmsp
run supplementary_material
run learnheuristic_pmsp
for f in learnheuristic_pmsp supplementary_material; do
  pages=$(pdfinfo "$f.pdf" 2>/dev/null | grep Pages | awk '{print $2}')
  echo "== $f: ${pages:-?} paginas"
  grep -E "^! " "$f.log" | head -5
  grep -E "Citation .* undefined|Reference .* undefined|There were undefined" "$f.log" | sort | uniq | head -10
  grep -c "Overfull" "$f.log" | sed 's/^/   overfull hbox: /'
done
