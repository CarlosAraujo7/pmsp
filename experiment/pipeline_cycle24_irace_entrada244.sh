#!/usr/bin/env bash
# Entrada 244 (Ciclo 24): o irace como configurador. Fase 1 (--tune, so' instancias de AJUSTE) -> hashes das escolhas -> fase 2 (--run, instancias de teste) -> relatorio.
# Requer R 4.6 e o pacote irace 4.5 (variaveis IRACE_RSCRIPT e R_LIBS_USER; veja cec/REPRODUCE.md).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle24_irace_entrada244"
LOG=results/_run244_pipeline.log
echo "[$(date +%H:%M:%S)] inicio do ajuste" > $LOG
$PY --tune pc >> results/_run244_tune_pc.log 2>&1
$PY --tune ur >> results/_run244_tune_ur.log 2>&1
echo "[$(date +%H:%M:%S)] ajuste concluido" >> $LOG
