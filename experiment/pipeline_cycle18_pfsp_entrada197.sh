#!/usr/bin/env bash
# Entrada 197 (Ciclo 18, frente D): flowshop de permutacao: ajuste (12 fatias por espaco), escolha, calibracao do K (SOZINHA e sequencial) e teste (6 fatias por espaco).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_pfsp_entrada197"
LOG=results/_run197_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in F5 F15; do for sh in $(seq 0 11); do $PY --tune $sp --shard $sh --nshards 12 >> results/_run197_tune_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --choose >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] escolha FALHOU" >> $LOG; exit 1; }
$PY --calibrate >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] calibracao FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in F5 F15; do for sh in $(seq 0 5); do $PY --run $sp --shard $sh --nshards 6 >> results/_run197_test_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
