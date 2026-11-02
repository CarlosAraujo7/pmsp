#!/usr/bin/env bash
# Entrada 254 (Ciclo 30): o protocolo no R||Cmax com as familias publicadas de instancias. O ajuste da acao validada (--tune, instancias de AJUSTE) ja' foi feito e suas escolhas estao em results/ur_ext_choice_*_entrada254.json (incluidas nos hashes); aqui so' o teste, em 6 processos (2 espacos x 3 fatias).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle30_ur_external_entrada254"
LOG=results/_run254_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for sh in 0 1 2; do $PY --run $sp --shard $sh --nshards 3 >> results/_run254_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
