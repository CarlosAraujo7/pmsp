#!/usr/bin/env bash
# Entrada 271: Q-Learning otimista aquecido por uma priori offline (B30; 700 instancias de teste), 12 fatias.
cd "$(dirname "$0")/.."
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do python -m experiment.run_cycle43_warmqo_entrada271 --run --shard $i --nshards 12 > results/_run271_$i.log 2>&1 & done
wait
echo concluido > results/_run271_done.txt
