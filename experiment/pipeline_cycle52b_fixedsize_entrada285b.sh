#!/usr/bin/env bash
# Entrada 285b: o ILS classico com o tamanho da perturbacao fixo e ajustado. Os hashes (executor, relatorio, nucleo, entradas das Entradas 284 e 285, razoes de CPU calibradas) foram congelados em results/entrada285b_prereg_hashes.txt ANTES de rodar o teste.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle52b_fixedsize_entrada285b"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run285b_test_$i.log 2>&1 & done
wait
echo concluido > results/_run285b_done.txt
