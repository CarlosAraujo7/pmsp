#!/usr/bin/env bash
# Entrada 219 (Ciclo 19, frente F), etapa 1: ajuste (12 fatias por espaco) e escolha (acao validada e hiperparametros por classe).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_ga_entrada219"
LOG=results/_run219_stage1.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in G9 G18; do for sh in $(seq 0 11); do $PY --tune $sp --shard $sh --nshards 12 >> results/_run219_tune_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --choose >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] escolha FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] etapa 1 concluida" >> $LOG
