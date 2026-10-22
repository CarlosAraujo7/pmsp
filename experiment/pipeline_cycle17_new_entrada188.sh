#!/usr/bin/env bash
# Entrada 188 (Ciclo 17, frente B'): verificacao (regra de parada), ajuste (paralelo), calibracao de CPU (SOZINHA e sequencial), teste (paralelo).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle17_new_entrada188"
LOG=results/_run188_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --verify >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] verificacao FALHOU: parada" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] verificacao concluida" >> $LOG
for sp in S5 S15; do $PY --tune $sp >> results/_run188_tune_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --calibrate >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in S5 S15; do $PY --run $sp >> results/_run188_run_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
