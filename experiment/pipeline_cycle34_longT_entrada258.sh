#!/usr/bin/env bash
# Entrada 258 (frente 1): espera o ajuste a T = 10.000 terminar, escolhe a validada, congela os hashes e roda o teste em 12 fatias.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle34_longT_entrada258 --T 10000"
until [ "$(ls results/frL_tune_B30_T10000_s*of12_entrada258.csv 2>/dev/null | wc -l)" -ge 12 ]; do sleep 20; done
$PY --choose > results/_run258L_choose.log 2>&1
md5sum experiment/run_cycle34_longT_entrada258.py analysis/report_entrada258L_longT.py experiment/run_cycle32_fanjul_entrada256.py pmsp/ur_numba_agents.py pmsp/ur_numba_agents5.py pmsp/ur_numba_agents6.py results/frL_choice_B30_T10000_entrada258.json results/ur_choice_qo_S15_entrada234.json results/pubaos_ext_choice_S15_entrada255.json > results/entrada258L_prereg_hashes.txt
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run258L_test_$i.log 2>&1 & done
wait
echo concluido > results/_run258L_done.txt
