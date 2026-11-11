#!/usr/bin/env bash
# Entrada 255 (Ciclo 31): irace e desenhos publicados de AOS nas familias publicadas de R||Cmax. O ajuste (irace --tune e AOS --tune, instancias de AJUSTE) ja' foi feito e suas escolhas estao em results/irace_tune_ext_entrada255.jsonl e results/pubaos_ext_choice_*_entrada255.json (incluidos nos hashes); aqui so' os testes (instancias de TESTE).
cd "$(dirname "$0")/.."
LOG=results/_run255_pipeline.log
echo "[$(date +%H:%M:%S)] inicio" > $LOG
python -m experiment.run_cycle31_irace_ext_entrada255 --run >> results/_run255_irace_test.log 2>&1 &
for sp in S5 S15; do python -m experiment.run_cycle31_pubaos_ext_entrada255 --run $sp >> results/_run255_pubaos_test_$sp.log 2>&1 & done
wait
echo "[$(date +%H:%M:%S)] pipeline concluido" >> $LOG
