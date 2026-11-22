#!/usr/bin/env bash
# Entrada 285: a ablacao do IG com a descida completa (o ILS classico). Os hashes (executor, relatorio, nucleo, testes, entradas da Entrada 284, razoes de CPU calibradas) foram congelados em results/entrada285_prereg_hashes.txt ANTES de rodar o teste.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle52_fulldescent_entrada285"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run285_test_$i.log 2>&1 & done
wait
echo concluido > results/_run285_done.txt
