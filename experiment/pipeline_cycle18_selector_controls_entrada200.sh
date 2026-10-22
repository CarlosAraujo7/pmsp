#!/usr/bin/env bash
# Entrada 200, estagio 2: controles do seletor offline (valbig, valclass e as versoes a avaliacoes iguais) nas instancias publicadas de teste (discovery 2.550, heldout 3.950; S5 e S15; 3 fatias por par).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_selector_controls_entrada200"
LOG=results/_run200_controls.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
for sp in S5 S15; do for cj in discovery heldout; do for sh in 0 1 2; do $PY --run $sp $cj --shard $sh --nshards 3 >> results/_run200_controls_${sp}_${cj}_$sh.log 2>&1 & done; done; done
wait
echo "[$(date +%H:%M:%S)] controles concluidos" >> $LOG
