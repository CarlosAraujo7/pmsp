#!/usr/bin/env bash
# Entrada 234 (Ciclo 22, frente A): teste do hospedeiro de maquinas nao relacionadas (R||Cmax). O ajuste (--tune e --tuneqo, instancias de AJUSTE) ja' foi feito e suas escolhas estao em results/ur_choice_*_entrada234.json (incluidas nos hashes).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle22_ur_entrada234"
LOG=results/_run234_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for sh in 0 1 2; do $PY --run $sp --shard $sh --nshards 3 >> results/_run234_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
