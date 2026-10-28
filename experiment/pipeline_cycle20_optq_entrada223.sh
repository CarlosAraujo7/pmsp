#!/usr/bin/env bash
# Entrada 223 (Ciclo 20, frente D): Q-Learning otimista: ajuste (6 fatias por espaco), escolha por familia e teste (descoberta e held-out; 3 fatias por par).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle20_optq_entrada223"
LOG=results/_run223_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for sh in 0 1 2 3 4 5; do $PY --tune $sp --shard $sh --nshards 6 >> results/_run223_tune_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --choose >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] escolha FALHOU" >> $LOG; exit 1; }
for sp in S5 S15; do for cj in discovery heldout; do for sh in 0 1 2; do $PY --run $cj $sp --shard $sh --nshards 3 >> results/_run223_test_${sp}_${cj}_$sh.log 2>&1 & done; done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
