#!/usr/bin/env bash
# Compila o rascunho v8 do artigo do CEC (re-enquadramento da segunda revisao cega; o v7 auditado fica intacto em cec2027_paper.tex).
# Uso: bash cec/build_v8.sh         (rascunho com notas vermelhas -> cec2027_v8.pdf)
#      bash cec/build_v8.sh anon    (versao DUPLO-ANONIMA -> cec2027_v8_anon.pdf)
set -u
export PATH="/c/Users/vinic/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")"
python make_bib.py
NAME=cec2027_v8
if [ "${1:-}" = "anon" ]; then
  NAME=cec2027_v8_anon
  sed -e 's/\\anonfalse/\\anontrue/' -e 's/\\draftmarkstrue/\\draftmarksfalse/' cec2027_v8.tex > $NAME.tex
fi
pdflatex -interaction=nonstopmode -halt-on-error $NAME.tex > cec_v8.build.log 2>&1
bibtex $NAME >> cec_v8.build.log 2>&1
pdflatex -interaction=nonstopmode $NAME.tex >> cec_v8.build.log 2>&1
pdflatex -interaction=nonstopmode $NAME.tex >> cec_v8.build.log 2>&1
echo "== $NAME: $(pdfinfo $NAME.pdf 2>/dev/null | awk '/Pages/{print $2}') paginas"
grep -E "^! " $NAME.log | head -5
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" $NAME.log | sort | uniq | head -10
grep -c "Overfull" $NAME.log | sed 's/^/   overfull hbox: /'
if [ "${1:-}" = "anon" ]; then
  echo "texto de identificacao no PDF anonimo (deve ficar vazio):"
  pdftotext $NAME.pdf - | grep -n -i -E "Ara.jo|carlos|uece|CarlosAraujo7|aluno\.uece" | head
fi
