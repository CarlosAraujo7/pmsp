#!/usr/bin/env bash
# Entrada 248 (Ciclo 26): CPU igual no R||Cmax. A calibracao dos K (--calibrate, SOZINHA na maquina) ja' foi feita (results/ur_eqcpu_k_entrada248.json, incluida nos hashes); aqui so' o teste, em 6 processos (2 espacos x 3 fatias).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle26_eqcpu_ur_entrada248"
LOG=results/_run248_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for sh in 0 1 2; do $PY --run $sp --shard $sh --nshards 3 >> results/_run248_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
