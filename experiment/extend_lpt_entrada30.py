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

targets = [(20, 6, "rand_20j_6m"), (75, 15, "rand_75j_15m")]
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
print(f"LPT novo: {n_new} linhas (esperado 140 = 70 x 2 classes)")
