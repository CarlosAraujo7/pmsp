#!/usr/bin/env bash
# Entrada 228 (Ciclo 21, frente A): tres sementes novas por acao fixa nas instancias de tempos finos (U[1,10^4]) para o oraculo com validacao cruzada.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle21_cffine_entrada228"
LOG=results/_run228_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do $PY --space $sp >> results/_run228_fine_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
