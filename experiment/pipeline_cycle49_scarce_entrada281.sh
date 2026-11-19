#!/usr/bin/env bash
# Entrada 281: o regime de poucos dados de ajuste (aquecido, congelado e estatica com k = 1, 2, 3, 5 instancias de ajuste por classe; 4 sorteios por k). Congela os hashes ANTES de rodar o teste; 12 fatias.
cd "$(dirname "$0")/.."
md5sum experiment/run_cycle49_scarce_entrada281.py analysis/report_entrada281_scarce.py tests/test_scarce_entrada281.py pmsp/ur_numba_agents5.py experiment/run_cycle32_fanjul_entrada256.py experiment/run_cycle43_warmqo_entrada271.py experiment/run_cycle44_frozenqo_entrada272.py results/scarce_prior_entrada281.json results/ur_choice_qo_S15_entrada234.json > results/entrada281_prereg_hashes.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m experiment.run_cycle49_scarce_entrada281 --run --shard $i --nshards 12 > results/_run281_$i.log 2>&1 & done
wait
echo concluido > results/_run281_done.txt
