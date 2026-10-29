#!/usr/bin/env bash
# Entrada 244, fase 2: avalia as configuracoes do irace nas instancias de TESTE (depois de gravar os hashes das escolhas).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle24_irace_entrada244"
LOG=results/_run244_pipeline.log
echo "[$(date +%H:%M:%S)] inicio do teste" >> $LOG
$PY --run pc >> results/_run244_test_pc.log 2>&1
$PY --run ur >> results/_run244_test_ur.log 2>&1
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
