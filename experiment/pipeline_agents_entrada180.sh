#!/usr/bin/env bash
# Entrada 180: verificacao, ajuste (4 espacos em paralelo), calibracao de CPU (SOZINHA, sequencial), teste (4 espacos em paralelo).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_agents_numba_entrada180"
LOG=results/_run180_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --verify >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] verificacao concluida" >> $LOG
for sp in S3 S5 S15 S30; do $PY --tune $sp >> results/_run180_tune_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --calibrate >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in S3 S5 S15 S30; do $PY --run $sp >> results/_run180_run_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
