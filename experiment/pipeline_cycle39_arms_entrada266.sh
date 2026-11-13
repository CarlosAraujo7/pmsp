#!/usr/bin/env bash
# Entrada 266: todas as acoes fixas de B30 no teste do benchmark de Fanjul-Peyro e Ruiz, 12 fatias.
cd "$(dirname "$0")/.."
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m experiment.run_cycle39_arms_entrada266 --shard $i --nshards 12 > results/_run266_$i.log 2>&1 & done
wait
echo concluido > results/_run266_done.txt
