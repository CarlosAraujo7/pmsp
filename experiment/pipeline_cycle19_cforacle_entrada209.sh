#!/usr/bin/env bash
# Entrada 209 (Ciclo 19, frente A): tres sementes novas por acao fixa nas instancias de teste (syn, descoberta, held-out, flowshop) para o oraculo com validacao cruzada.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle19_cforacle_entrada209"
LOG=results/_run209_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do $PY --syn $sp >> results/_run209_syn_$sp.log 2>&1 & done
for cj in discovery heldout; do for sp in S5 S15; do for sh in 0 1 2; do $PY --pub $cj $sp --shard $sh --nshards 3 >> results/_run209_pub_${cj}_${sp}_$sh.log 2>&1 & done; done; done
for sp in F5 F15; do for sh in 0 1 2 3 4 5; do $PY --pfsp $sp --shard $sh --nshards 6 >> results/_run209_pfsp_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
