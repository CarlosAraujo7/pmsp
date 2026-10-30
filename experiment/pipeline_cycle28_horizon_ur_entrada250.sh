#!/usr/bin/env bash
# Entrada 250 (Ciclo 28): horizonte longo no R||Cmax. Tres etapas (T = 5.000, 20.000, 100.000) de 6 processos (2 espacos x 3 fatias).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle28_horizon_ur_entrada250"
LOG=results/_run250_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for T in 5000 20000 100000; do
  for sp in S5 S15; do for sh in 0 1 2; do $PY --run $sp $T --shard $sh --nshards 3 >> results/_run250_${sp}_T${T}_$sh.log 2>&1 & done; done
  wait
  echo "[$(date +%H:%M:%S)] T=$T concluido" >> $LOG
done
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
