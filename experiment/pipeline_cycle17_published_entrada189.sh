#!/usr/bin/env bash
# Entrada 189 (Ciclo 17, frente A'): preparo (limiares e grupos), ajuste (paralelo), calibracao de CPU (SOZINHA e sequencial), teste em 10 fatias independentes (2 espacos x 5).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle17_published_entrada189"
LOG=results/_run189_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --prepare >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] preparo FALHOU: parada" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] preparo concluido" >> $LOG
for sp in S5 S15; do $PY --tune $sp >> results/_run189_tune_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
$PY --calibrate >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in S5 S15; do for sh in 0 1 2 3 4; do $PY --run $sp --shard $sh --nshards 5 >> results/_run189_run_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
