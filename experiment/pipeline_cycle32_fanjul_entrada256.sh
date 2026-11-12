#!/usr/bin/env bash
# Entrada 256 (Ciclo 32): o protocolo no benchmark padrao de R||Cmax de Fanjul-Peyro e Ruiz.
# Fase 1 (--tune): escolha da acao validada por classe (tipo, n) nas 280 instancias de AJUSTE; fase 2 (--choose); fase 3 (--run): 700 instancias de TESTE (hashes da escolha congelados antes: ver results/entrada256_prereg_hashes.txt).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle32_fanjul_entrada256"
LOG=results/_run256_pipeline.log
PHASE="${1:-all}"
if [ "$PHASE" = "tune" ] || [ "$PHASE" = "all" ]; then
  echo "[$(date +%H:%M:%S)] ajuste" > $LOG
  for sp in B5 B15 B30; do for sh in 0 1 2 3; do $PY --tune $sp --shard $sh --nshards 4 >> results/_run256_tune_${sp}_$sh.log 2>&1 & done; done
  wait
  for sp in B5 B15 B30; do $PY --choose $sp >> $LOG 2>&1; done
  echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
fi
if [ "$PHASE" = "run" ] || [ "$PHASE" = "all" ]; then
  echo "[$(date +%H:%M:%S)] teste" >> $LOG
  for sp in B5 B15 B30; do for sh in 0 1 2 3; do $PY --run $sp --shard $sh --nshards 4 >> results/_run256_test_${sp}_$sh.log 2>&1 & done; done
  wait
  echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
fi
