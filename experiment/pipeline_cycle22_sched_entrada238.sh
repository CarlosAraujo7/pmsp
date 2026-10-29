#!/usr/bin/env bash
# Entrada 238 (Ciclo 22, frente B): teste do cronograma de acoes (oraculo dinamico no tempo). O ajuste (--tune, instancias de AJUSTE) ja' foi feito; as escolhas estao em results/sched_choice_*_entrada238.json (incluidas nos hashes).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle22_sched_entrada238"
LOG=results/_run238_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for h in pc ur; do for sh in 0 1 2; do $PY --run $h --shard $sh --nshards 3 >> results/_run238_${h}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
