#!/usr/bin/env bash
# Entrada 213 (Ciclo 19, frente C): calibracao dos K do seletor e do Q aquecido (SOZINHA e sequencial) e teste a CPU igual (descoberta e held-out; S5 e S15; 3 fatias por par).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_equalcpu_entrada213"
LOG=results/_run213_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --calibrate >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] calibracao FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in S5 S15; do for cj in discovery heldout; do for sh in 0 1 2; do $PY --run $cj $sp --shard $sh --nshards 3 >> results/_run213_test_${sp}_${cj}_$sh.log 2>&1 & done; done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
