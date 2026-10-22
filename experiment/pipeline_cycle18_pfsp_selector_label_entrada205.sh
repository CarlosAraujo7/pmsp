#!/usr/bin/env bash
# Entrada 205 (Ciclo 18, frente D'): rotulos do seletor offline do flowshop (cada acao fixa, 3 sementes, 540 instancias de treino), 6 fatias por espaco, e ajuste do modelo.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_pfsp_selector_entrada205"
LOG=results/_run205_label.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in F5 F15; do for sh in 0 1 2 3 4 5; do $PY --label $sp --shard $sh --nshards 6 >> results/_run205_label_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] rotulos concluidos" >> $LOG
$PY --fit >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] ajuste FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] modelos ajustados" >> $LOG
