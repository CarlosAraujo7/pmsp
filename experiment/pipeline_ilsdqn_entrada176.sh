#!/usr/bin/env bash
# Frente I2 (Entrada 176): ajuste do lr e dos operadores unicos, escolha, teste e calibracao de CPU. Quatro processos independentes por fase.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_ils_dqn_entrada176"
LOG=results/_run176_pipeline.log
NS=4
phase() {  # $1 = flag da fase, $2 = nome do CSV a fundir
  for s in $(seq 0 $((NS-1))); do $PY --$1 --shard $s --nshards $NS >> results/_run176_$1_s$s.log 2>&1 & done
  wait
  $PY --merge $2 >> $LOG 2>&1
  echo "[$(date +%H:%M:%S)] fase $1 concluida" >> $LOG
}
echo "[$(date +%H:%M:%S)] inicio" > $LOG
phase tune tune
phase arms-tune arms_tune
$PY --choose >> $LOG 2>&1
phase run run
$PY --calibrate >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
