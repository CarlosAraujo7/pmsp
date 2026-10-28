#!/usr/bin/env bash
# Entrada 217 (Ciclo 19, frente E): horizonte longo (T = 5.000 e 20.000) na familia C publicada held-out, tercis 1-2, S5, 340 instancias, 6 fatias por horizonte.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_longc_entrada217"
LOG=results/_run217_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for T in 5000 20000; do for sh in 0 1 2 3 4 5; do $PY --run $T --shard $sh --nshards 6 >> results/_run217_T${T}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
