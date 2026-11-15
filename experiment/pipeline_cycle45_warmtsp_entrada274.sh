#!/usr/bin/env bash
# Entradas 274 e 275: a replica, no TSP, do achado da Entrada 271. Congela os hashes (prioris, corredores, relatorios, nucleo, testes) ANTES de rodar o teste; roda a Entrada 274 (400 instancias sinteticas) e depois a 275 (37 da TSPLIB), 12 fatias cada.
cd "$(dirname "$0")/.."
PY4="python -m experiment.run_cycle45_warmtsp_entrada274"
PY5="python -m experiment.run_cycle46_warmtsplib_entrada275"
md5sum experiment/run_cycle45_warmtsp_entrada274.py analysis/report_entrada274_warmtsp.py pmsp/tsp_warm.py pmsp/tsp_numba.py pmsp/instance_tsp.py tests/test_tsp_warm_entrada274.py results/tspwarm_prior_entrada274.json results/tsp_choice_entrada267.json results/ur_choice_qo_S15_entrada234.json > results/entrada274_prereg_hashes.txt
md5sum experiment/run_cycle46_warmtsplib_entrada275.py analysis/report_entrada275_warmtsplib.py experiment/run_cycle45_warmtsp_entrada274.py pmsp/tsp_warm.py pmsp/tsp_numba.py pmsp/instance_tsplib.py tests/test_tsp_warm_entrada274.py results/tsplibwarm_prior_entrada275.json results/tsplib_choice_entrada270.json external_benchmarks/tsplib/manifest_tsplib.json results/ur_choice_qo_S15_entrada234.json > results/entrada275_prereg_hashes.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY4 --run --shard $i --nshards 12 > results/_run274_$i.log 2>&1 & done
wait
echo concluido > results/_run274_done.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY5 --run --shard $i --nshards 12 > results/_run275_$i.log 2>&1 & done
wait
echo concluido > results/_run275_done.txt
