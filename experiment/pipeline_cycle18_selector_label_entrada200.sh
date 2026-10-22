#!/usr/bin/env bash
# Entrada 200 (Ciclo 18, frente B): rotulos do seletor offline (cada acao fixa, 3 sementes, nas instancias de treino de manifest.csv B, C, E), 5 fatias por espaco.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_selector_entrada200"
LOG=results/_run200_label.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for sh in 0 1 2 3 4; do $PY --label $sp --shard $sh --nshards 5 >> results/_run200_label_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] rotulos concluidos" >> $LOG
