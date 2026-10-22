#!/usr/bin/env bash
# Compila o artigo do CEC (pdflatex + bibtex) e mostra paginas, erros e referencias indefinidas.
# Uso: bash cec/build.sh         (rascunho com nomes e notas vermelhas -> cec2027_paper.pdf)
#      bash cec/build.sh anon    (versao DUPLO-ANONIMA para submissao: sem nomes, e-mail, link do repositorio e notas -> cec2027_anon.pdf)
# MiKTeX no PATH do Git Bash.
set -u
export PATH="/c/Users/vinic/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
python make_bib.py
NAME=cec2027_paper
if [ "${1:-}" = "anon" ]; then
  NAME=cec2027_anon
  sed -e 's/\\anonfalse/\\anontrue/' -e 's/\\draftmarkstrue/\\draftmarksfalse/' cec2027_paper.tex > $NAME.tex
fi
pdflatex -interaction=nonstopmode -halt-on-error $NAME.tex > cec.build.log 2>&1
bibtex $NAME >> cec.build.log 2>&1
pdflatex -interaction=nonstopmode $NAME.tex >> cec.build.log 2>&1
pdflatex -interaction=nonstopmode $NAME.tex >> cec.build.log 2>&1
echo "== $NAME: $(pdfinfo $NAME.pdf 2>/dev/null | awk '/Pages/{print $2}') paginas"
grep -E "^! " $NAME.log | head -5
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" $NAME.log | sort | uniq | head -10
grep -c "Overfull" $NAME.log | sed 's/^/   overfull hbox: /'
if [ "${1:-}" = "anon" ]; then
  echo "texto de identificacao no PDF anonimo (deve ficar vazio):"
  pdftotext $NAME.pdf - | grep -n -i -E "Ara.jo|carlos|uece|CarlosAraujo7|aluno\.uece" | head
fi
