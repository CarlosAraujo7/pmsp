#!/usr/bin/env bash
# Entrada 279: aprendizes sem travamento na primeira acao e a ordem das acoes (B30 e depois B15, 12 fatias cada). Congela os hashes (executor, relatorio, nucleo com desempate, nucleos originais, testes, escolhas congeladas) ANTES de rodar o teste.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle47_ties_entrada279"
md5sum experiment/run_cycle47_ties_entrada279.py analysis/report_entrada279_ties.py pmsp/ur_ties.py pmsp/ur_numba_agents.py pmsp/ur_numba_agents5.py pmsp/ur_numba_agents6.py tests/test_ur_ties_entrada279.py experiment/run_cycle32_fanjul_entrada256.py results/fr_choice_B30_entrada256.json results/fr_choice_B15_entrada256.json results/ur_choice_qo_S15_entrada234.json results/pubaos_ext_choice_S15_entrada255.json > results/entrada279_prereg_hashes.txt
for sp in B30 B15; do
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --space $sp --shard $i --nshards 12 > results/_run279_${sp}_$i.log 2>&1 & done
  wait
  echo concluido > results/_run279_${sp}_done.txt
done
