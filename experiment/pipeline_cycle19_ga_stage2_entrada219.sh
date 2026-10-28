#!/usr/bin/env bash
# Entrada 219 (Ciclo 19, frente F), etapa 2: calibracao dos K (SOZINHA e sequencial) e teste (6 fatias por espaco).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_ga_entrada219"
LOG=results/_run219_stage2.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --calibrate >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] calibracao FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in G9 G18; do for sh in $(seq 0 5); do $PY --run $sp --shard $sh --nshards 6 >> results/_run219_test_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
