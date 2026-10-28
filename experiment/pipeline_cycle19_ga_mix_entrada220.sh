#!/usr/bin/env bash
# Entrada 220 (Ciclo 19, frente F, estagio 2): a mistura estatica validada: calibracao do K (SOZINHA) e teste (6 fatias por espaco).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_ga_mix_entrada220"
LOG=results/_run220_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --calibrate >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] calibracao FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in G9 G18; do for sh in $(seq 0 5); do $PY --run $sp --shard $sh --nshards 6 >> results/_run220_test_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
