#!/usr/bin/env bash
# Entrada 282: o protocolo numa implementacao de terceiros (ALNS para o CVRP com a biblioteca `alns`). Congela os hashes (executor, relatorio, hospedeiro, testes, versoes das bibliotecas, arquivos X baixados) ANTES de rodar o ajuste e o teste.
cd "$(dirname "$0")/.."
python -m pip freeze 2>/dev/null | grep -i -E "^(alns|mabwiser|vrplib|numba|numpy|scipy|pandas)==" > results/cvrp_env_entrada282.txt
md5sum experiment/run_cycle50_cvrp_entrada282.py analysis/report_entrada282_cvrp.py pmsp/cvrp_alns.py tests/test_cvrp_entrada282.py results/cvrp_env_entrada282.txt external_benchmarks/cvrp_x/raw/x_set.bin external_benchmarks/cvrp_x/raw/x_bks.bin > results/entrada282_prereg_hashes.txt
PY="python -m experiment.run_cycle50_cvrp_entrada282"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --tune --shard $i --nshards 12 > results/_run282_tune_$i.log 2>&1 & done
wait
$PY --choose > results/_run282_choose.log 2>&1
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --run --shard $i --nshards 12 > results/_run282_test_$i.log 2>&1 & done
wait
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do $PY --runx --shard $i --nshards 12 > results/_run282_x_$i.log 2>&1 & done
wait
echo concluido > results/_run282_done.txt
