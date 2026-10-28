#!/usr/bin/env bash
# Entrada 212 (Ciclo 19, frente B, estagio 2): o Q aquecido com desempate aleatorio (wqn, wqn0) na descoberta e no held-out; S5 e S15; 3 fatias por par.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_warmq_tb_entrada212"
LOG=results/_run212_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for cj in discovery heldout; do for sh in 0 1 2; do $PY --run $cj $sp --shard $sh --nshards 3 >> results/_run212_test_${sp}_${cj}_$sh.log 2>&1 & done; done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
