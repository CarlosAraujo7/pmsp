#!/usr/bin/env bash
# Entrada 185 (Ciclo 16): verificacao (regra de parada), ajuste (tudo em paralelo; independentes), calibracao de CPU (SOZINHA e sequencial, maquina em repouso), teste (em paralelo).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle16_entrada185"
LOG=results/_run185_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --verify >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] verificacao FALHOU: parada" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] verificacao concluida" >> $LOG
for sp in S5 S15; do $PY --aos-tune $sp >> results/_run185_aostune_$sp.log 2>&1 & done
for T in 5000 20000 100000; do $PY --long-tune $T >> results/_run185_longtune_$T.log 2>&1 & done
for sp in S5 S15; do $PY --fine-tune $sp >> results/_run185_finetune_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --aos-calibrate >> $LOG 2>&1
$PY --fine-calibrate >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] calibracoes concluidas" >> $LOG
for sp in S5 S15; do $PY --aos-run $sp >> results/_run185_aosrun_$sp.log 2>&1 & done
for sp in S5 S15; do $PY --fine-run $sp >> results/_run185_finerun_$sp.log 2>&1 & done
$PY --plant >> results/_run185_plant.log 2>&1 &
for T in 5000 20000 100000; do $PY --long-run $T >> results/_run185_longrun_$T.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
