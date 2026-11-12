#!/usr/bin/env bash
# Entradas 259/262 (frente 4): treina as politicas neurais restantes, congela os hashes e roda o teste (B15 e B30; 4 fatias cada).
cd "$(dirname "$0")/.."
TR="python -m experiment.train_neural_entrada259"
until [ -f results/nn_theta_B30_full_entrada259.npy ]; do sleep 30; done
$TR --space B30 --variant const --gens 60 --workers 6 > results/_run259_B30_const.log 2>&1
$TR --space B15 --variant full --gens 100 --workers 6 > results/_run259_B15_full.log 2>&1
$TR --space B15 --variant const --gens 60 --workers 6 > results/_run259_B15_const.log 2>&1
md5sum experiment/run_cycle36_neural_test_entrada262.py analysis/report_entrada262_neural.py experiment/train_neural_entrada259.py pmsp/ur_numba_neural.py experiment/run_cycle32_fanjul_entrada256.py results/nn_theta_B30_full_entrada259.npy results/nn_theta_B30_const_entrada259.npy results/nn_theta_B15_full_entrada259.npy results/nn_theta_B15_const_entrada259.npy results/rcmax_manifest_entrada137.csv > results/entrada262_prereg_hashes.txt
for sp in B15 B30; do for i in 0 1 2 3; do python -m experiment.run_cycle36_neural_test_entrada262 --space $sp --shard $i --nshards 4 > results/_run262_${sp}_$i.log 2>&1 & done; done
wait
echo concluido > results/_run262_done.txt
