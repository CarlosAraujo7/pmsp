"""
experiment/extend_sota_n100_entrada37.py
==========================================
Estende multifit e sa_li_perkins de N=30 para N=100 nas 5 classes
nao-triviais (mesma convencao ja usada para ig_ql/ils_rand_hi/ils_ql_hi
nas Entradas 25 e 30 -- rand_20j_4m fica em N=30, nunca motivo de
preocupacao). A pedido de Carlos (Entrada 37): mais poder estatistico
exatamente onde mais importa -- o empate em rand_50j_10m (N=30, p=0.10)
e a virada em rand_75j_15m/rand_100j_20m contra o MULTIFIT.

Mesmo padrao seguro contra crash (flush+fsync por linha, idempotente
via existing_keys) dos scripts de extensao anteriores.
"""
import csv, os, sys, time
sys.path.insert(0, '.')
from pmsp.instance import PMSPInstance
from experiment.runner import run_single
from experiment import config as CFG

csv_path = 'results/results.csv'
fieldnames = ["instance", "method", "rep", "n", "m", "lb",
              "makespan", "gap_pct", "cpu_time", "n_improvements"]

existing_keys = set()
with open(csv_path, newline='') as f_exist:
    for row in csv.DictReader(f_exist):
        existing_keys.add((row["instance"], row["method"], row["rep"]))

targets = [
    (20, 6, "rand_20j_6m"),
    (30, 8, "rand_30j_8m"),
    (50, 10, "rand_50j_10m"),
    (75, 15, "rand_75j_15m"),
    (100, 20, "rand_100j_20m"),
]
methods = ["multifit", "sa_li_perkins"]

n_new = 0
t0 = time.perf_counter()
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for n, m, inst_name in targets:
        for method in methods:
            for inst_id in range(30, 100):
                key = (inst_name, method, str(inst_id))
                if key in existing_keys:
                    continue
                seed = CFG.BASE_SEED + inst_id
                inst = PMSPInstance.random(n=n, m=m, seed=seed, name=inst_name)
                r = run_single(inst, method, inst_id, max_iter=CFG.MAX_ITER, q_init=None)
                writer.writerow({k: r[k] for k in fieldnames})
                f.flush()
                os.fsync(f.fileno())
                n_new += 1
print(f"Novo: {n_new} linhas (esperado 700 = 70 x 5 classes x 2 metodos), "
      f"{time.perf_counter()-t0:.1f}s")
