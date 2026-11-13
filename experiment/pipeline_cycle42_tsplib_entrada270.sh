#!/usr/bin/env bash
# Entrada 270: espera a Entrada 268 terminar (CPU), escolhe as acoes estaticas nas instancias SINTETICAS de ajuste, congela os hashes e roda o teste na TSPLIB em 12 fatias.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle42_tsplib_entrada270"
until [ -f results/_run268_done.txt ]; do sleep 20; done
$PY --choose > results/_run270_choose.log 2>&1
md5sum experiment/run_cycle42_tsplib_entrada270.py analysis/report_entrada270_tsplib.py experiment/run_cycle40_tsp_entrada267.py pmsp/tsp_numba.py pmsp/instance_tsp.py pmsp/instance_tsplib.py results/tsplib_choice_entrada270.json external_benchmarks/tsplib/solutions_mastqe.txt external_benchmarks/tsplib/manifest_tsplib.json > results/entrada270_prereg_hashes.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run270_test_$i.log 2>&1 & done
wait
echo concluido > results/_run270_done.txt
