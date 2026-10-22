#!/usr/bin/env bash
# Entrada 195 (Ciclo 18, frente A): calibracao do K da acao validada (SOZINHA e sequencial) e execucao da validada a CPU igual do aprendiz (publicadas descoberta e held-out em 5 fatias por espaco; sinteticas S5 e S15).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_valeq_entrada195"
LOG=results/_run195_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --calibrate >> $LOG 2>&1 || { echo "[$(date +%H:%M:%S)] calibracao FALHOU" >> $LOG; exit 1; }
echo "[$(date +%H:%M:%S)] calibracao concluida" >> $LOG
for sp in S5 S15; do $PY --run-syn $sp >> results/_run195_syn_$sp.log 2>&1 & done
for st in discovery heldout; do for sp in S5 S15; do for sh in 0 1; do $PY --run-pub $st $sp --shard $sh --nshards 2 >> results/_run195_pub_${st}_${sp}_$sh.log 2>&1 & done; done; done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
