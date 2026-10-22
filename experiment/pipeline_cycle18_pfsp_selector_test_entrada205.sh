#!/usr/bin/env bash
# Entrada 205 (Ciclo 18, frente D'): o seletor offline do flowshop (modelos ja' ajustados e congelados) e a estatica das 540 instancias nas 360 sinteticas e 90 de Taillard, F5 e F15, 6 fatias por espaco.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_pfsp_selector_entrada205"
LOG=results/_run205_test.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in F5 F15; do for sh in 0 1 2 3 4 5; do $PY --run $sp --shard $sh --nshards 6 >> results/_run205_test_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] seletor concluido" >> $LOG
