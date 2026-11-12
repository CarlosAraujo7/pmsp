#!/usr/bin/env bash
# Entrada 263: ql_perm a T = 10.000 (B30, 700 instancias de teste) em 12 fatias.
cd "$(dirname "$0")/.."
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m experiment.run_cycle37_qperm_longT_entrada263 --shard $i --nshards 12 > results/_run263_$i.log 2>&1 & done
wait
echo concluido > results/_run263_done.txt
