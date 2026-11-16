#!/usr/bin/env bash
# Entrada 280: o Q-Learning com desempate aleatorio (ql_rt) em todas as colecoes das 85 celulas (B5, familias de R||Cmax, P||Cmax sintetico, P||Cmax publicado held-out, TSP), 12 fatias cada. Congela os hashes (executor, relatorio, nucleos com desempate, testes, escolhas) ANTES de rodar.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle48_tiesall_entrada280"
md5sum experiment/run_cycle48_tiesall_entrada280.py analysis/report_entrada280_tiesall.py pmsp/ur_ties.py pmsp/ig_ties.py pmsp/tsp_ties.py pmsp/ig_numba_agents5.py pmsp/ur_numba_agents5.py pmsp/tsp_numba.py tests/test_ties_more_entrada280.py tests/test_ur_ties_entrada279.py results/entrada279_ties.txt > results/entrada280_prereg_hashes.txt
run12() {  # $1 = argumentos do executor
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY $1 --shard $i --nshards 12 > results/_run280_$(echo "$1" | tr -d ' -')_$i.log 2>&1 & done
  wait
}
run12 "--bench b5"
run12 "--bench families --space S5"
run12 "--bench families --space S15"
run12 "--bench pcmax --space S5"
run12 "--bench pcmax --space S15"
run12 "--bench pub --space S5"
run12 "--bench pub --space S15"
echo concluido > results/_run280_part1_done.txt
run12 "--bench tsp"
echo concluido > results/_run280_done.txt
