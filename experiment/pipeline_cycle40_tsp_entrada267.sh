#!/usr/bin/env bash
# Entrada 267: espera a Entrada 266 terminar (CPU), ajusta as 15 acoes nas instancias de ajuste, escolhe a validada, congela os hashes e roda o teste em 12 fatias.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle40_tsp_entrada267"
until [ -f results/_run266_done.txt ]; do sleep 20; done
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --tune --shard $i --nshards 12 > results/_run267_tune_$i.log 2>&1 & done
wait
$PY --choose > results/_run267_choose.log 2>&1
md5sum experiment/run_cycle40_tsp_entrada267.py analysis/report_entrada267_tsp.py pmsp/tsp_numba.py pmsp/instance_tsp.py results/tsp_choice_entrada267.json results/ur_choice_qo_S15_entrada234.json > results/entrada267_prereg_hashes.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run267_test_$i.log 2>&1 & done
wait
echo concluido > results/_run267_done.txt
