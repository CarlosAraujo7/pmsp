#!/usr/bin/env bash
# Entrada 199 (Ciclo 18, frente C): familia F publicada, fora da amostra do pre-teste: (as sondas ja' rodaram) escolha da acao validada por celula e execucao dos testes (S5 e S15, 5 fatias cada).
cd "$(dirname "$0")/.."
PY="python -m experiment.run_cycle18_famF_entrada199"
LOG=results/_run199_test.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
$PY --choose >> $LOG 2>&1 || { echo "choose FALHOU" >> $LOG; exit 1; }
for sp in S5 S15; do for sh in 0 1 2 3 4; do $PY --run $sp --shard $sh --nshards 5 >> results/_run199_test_${sp}_$sh.log 2>&1 & done; done
wait
echo "[$(date +%H:%M:%S)] testes concluidos" >> $LOG
