#!/usr/bin/env bash
# Entrada 190 (Ciclo 17, estagio 3): replicacao confirmatoria nas instancias publicadas held-out (nada e' reajustado): preparo e teste em 10 fatias independentes (2 espacos x 5).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle17_replication_entrada190"
LOG=results/_run190_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --prepare >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] preparo FALHOU: parada" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] preparo concluido" >> $LOG
for sp in S5 S15; do for sh in 0 1 2 3 4; do $PY --run $sp --shard $sh --nshards 5 >> results/_run190_run_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
