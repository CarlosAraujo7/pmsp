#!/usr/bin/env bash
# Entrada 249 (Ciclo 27): oraculos de malha fechada (miope e de rollout). Duas etapas de 6 processos (dois hospedeiro-espacos x 3 fatias).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle27_closedloop_entrada249"
LOG=results/_run249_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for hs in "ur S15" "pc S15"; do set -- $hs; for sh in 0 1 2; do $PY --run $1 $2 --shard $sh --nshards 3 >> results/_run249_$1_$2_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] etapa 1 (S15) concluida" >> $LOG
for hs in "ur S5" "pc S5"; do set -- $hs; for sh in 0 1 2; do $PY --run $1 $2 --shard $sh --nshards 3 >> results/_run249_$1_$2_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
