#!/usr/bin/env bash
# Frente I (Entrada 173): ajuste do lr, escolhas, teste. Quatro processos independentes por fase (a maquina e' compartilhada); sem ProcessPoolExecutor.
cd "$(dirname "$0")/.."
PY="python -m experiment.run_dqn_entrada173"
LOG=results/_run173_pipeline.log
NS=4
phase() {  # $1 = flag da fase, $2 = nome do CSV a fundir
  for s in $(seq 0 $((NS-1))); do $PY --$1 --shard $s --nshards $NS >> results/_run173_$1_s$s.log 2>&1 & done
  wait
  $PY --merge $2 >> $LOG 2>&1
  echo "[$(date +%H:%M:%S)] fase $1 concluida" >> $LOG
}
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --arms15-tune >> $LOG 2>&1
$PY --arms15-test >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] bracos compilados concluidos" >> $LOG
phase tune-p5 tune_p5
phase tune-p15 tune_p15
phase tune-r tune_r
$PY --choose >> $LOG 2>&1
phase run-p5 test_p5
phase run-p15 test_p15
phase run-r test_r
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
