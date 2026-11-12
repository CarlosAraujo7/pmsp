#!/usr/bin/env bash
# Entrada 264: ql_perm nas instancias publicadas held-out (S5 e S15), 12 fatias.
cd "$(dirname "$0")/.."
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  ( python -m experiment.run_cycle38_qperm_pub_entrada264 --space S5 --shard $i --nshards 12 > results/_run264_S5_$i.log 2>&1
    python -m experiment.run_cycle38_qperm_pub_entrada264 --space S15 --shard $i --nshards 12 > results/_run264_S15_$i.log 2>&1 ) &
done
wait
echo concluido > results/_run264_done.txt
