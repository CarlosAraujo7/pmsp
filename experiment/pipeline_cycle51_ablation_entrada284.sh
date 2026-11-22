#!/usr/bin/env bash
# Entrada 284: ablacao dos componentes da estrutura do IG. Os hashes (executor, relatorio, nucleo, testes, razoes de CPU calibradas) foram congelados em results/entrada284_prereg_hashes.txt ANTES de rodar o teste.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle51_ablation_entrada284"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run284_test_$i.log 2>&1 & done
wait
echo concluido > results/_run284_done.txt
