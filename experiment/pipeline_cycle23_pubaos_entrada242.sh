#!/usr/bin/env bash
# Entrada 242 (Ciclo 23, frente A): teste dos dois projetos de AOS da literatura. O ajuste (--tune, instancias de AJUSTE) ja' foi feito; as escolhas estao em results/pubaos_choice_*_entrada242.json (incluidas nos hashes).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle23_pubaos_entrada242"
LOG=results/_run242_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for h in pc ur; do for sp in S5 S15; do $PY --run $h $sp >> results/_run242_${h}_$sp.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
