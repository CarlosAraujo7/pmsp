#!/usr/bin/env bash
# Entrada 211 (Ciclo 19, frente B): pre-treino da tabela Q (S5 e S15) nas 5.720 instancias de treino e teste do Q aquecido (wq, wq0) na descoberta e no held-out (3 fatias por par).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_warmq_entrada211"
LOG=results/_run211_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do $PY --pretrain $sp >> results/_run211_pretrain_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] pre-treino concluido" >> $LOG
for sp in S5 S15; do for cj in discovery heldout; do for sh in 0 1 2; do $PY --run $cj $sp --shard $sh --nshards 3 >> results/_run211_test_${sp}_${cj}_$sh.log 2>&1 & done; done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
