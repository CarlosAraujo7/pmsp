#!/usr/bin/env bash
# Entrada 268: espera a Entrada 267 terminar (CPU quieta para medir K), calibra K em sequencia, congela os hashes e roda o teste a CPU igual em 12 fatias.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle41_eqcpu_fanjul_entrada268"
until [ -f results/_run267_done.txt ]; do sleep 20; done
sleep 30
$PY --calibrate > results/_run268_calibrate.log 2>&1
md5sum experiment/run_cycle41_eqcpu_fanjul_entrada268.py analysis/report_entrada268_eqcpu_fanjul.py experiment/run_cycle32_fanjul_entrada256.py pmsp/ur_numba_agents.py pmsp/ur_numba_agents5.py pmsp/ur_numba_agents6.py results/eqcpu_K_entrada268.json results/fr_choice_B30_entrada256.json results/ur_choice_qo_S15_entrada234.json results/pubaos_ext_choice_S15_entrada255.json > results/entrada268_prereg_hashes.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run268_test_$i.log 2>&1 & done
wait
echo concluido > results/_run268_done.txt
