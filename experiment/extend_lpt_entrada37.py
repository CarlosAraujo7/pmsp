"""
experiment/extend_lpt_entrada37.py
===================================
Fecha uma lacuna remanescente encontrada na Entrada 37 (diario_pesquisa.md):
a Entrada 30 estendeu o LPT para N=100 em rand_20j_6m e rand_75j_15m (as 2
classes que ELA estendeu), mas as 3 classes estendidas ANTES, na Entrada 25
(rand_30j_8m, rand_50j_10m, rand_100j_20m -- via
experiment/extend_replicas_n100_entrada25.py, que nunca incluiu 'lpt' na
sua lista de metodos), continuavam com LPT so ate rep=29. Isso deixava o
teste de sanidade "nunca pior que o LPT" e as tabelas de Delta-sobre-LPT
vaziamente fracos (poucos pares) para reps>=30 nessas 3 classes.

Mesmo padrao seguro contra crash (flush+fsync por linha, idempotente via
existing_keys) do script da Entrada 30 -- inclui as 2 classes ja
resolvidas (serao puladas via existing_keys) + as 3 que faltavam, para
este script ficar auto-suficiente e completo.
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
    (20, 6, "rand_20j_6m"),     # ja resolvido na Entrada 30 (sera pulado)
    (75, 15, "rand_75j_15m"),   # ja resolvido na Entrada 30 (sera pulado)
    (30, 8, "rand_30j_8m"),     # faltava desde a Entrada 25
    (50, 10, "rand_50j_10m"),   # faltava desde a Entrada 25
    (100, 20, "rand_100j_20m"), # faltava desde a Entrada 25
]
n_new = 0
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for n, m, inst_name in targets:
        for inst_id in range(30, 100):
            key = (inst_name, "lpt", str(inst_id))
            if key in existing_keys:
                continue
            seed = CFG.BASE_SEED + inst_id
            inst = PMSPInstance.random(n=n, m=m, seed=seed, name=inst_name)
            r = run_single(inst, "lpt", inst_id, max_iter=CFG.MAX_ITER, q_init=None)
            writer.writerow({k: r[k] for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
            n_new += 1
print(f"LPT novo: {n_new} linhas (esperado 210 = 70 x 3 classes faltantes)")
