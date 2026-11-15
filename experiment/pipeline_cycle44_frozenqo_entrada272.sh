#!/usr/bin/env bash
# Entrada 272: controle da Entrada 271 (a priori offline sem aprendizagem; B30; 700 instancias de teste), 12 fatias.
cd "$(dirname "$0")/.."
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m experiment.run_cycle44_frozenqo_entrada272 --run --shard $i --nshards 12 > results/_run272_$i.log 2>&1 & done
wait
echo concluido > results/_run272_done.txt
