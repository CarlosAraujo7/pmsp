"""
Driver crash-seguro para estender resultados de N=30 para N=100 replicas,
apenas para os metodos/classes da validacao N=100 (Entrada 25).
Replica EXATAMENTE a logica de experiment.runner.run_all() (mesma geracao
de instancia, mesmo dispatch via run_single, mesmo fieldnames), mas com
flush()+fsync() apos CADA linha -- necessario porque, na classe rand_100j_20m,
uma unica linha (ils_ql_hi) pode levar ~6s, e o buffer de I/O padrao do
Python so grava no disco a cada ~8KB (~80-100 linhas) ou ao fechar o arquivo;
sem flush explicito, matar o processo no meio (timeout do shell remoto)
perderia TODO o progresso do chunk, nao so a ultima linha.
"""
import csv
import os
import sys
import time

sys.path.insert(0, '.')
from pmsp.instance import PMSPInstance
from experiment.runner import run_single
from experiment import config as CFG

n = int(sys.argv[1])
m = int(sys.argv[2])
rep_start = int(sys.argv[3]) if len(sys.argv) > 3 else 30
rep_end = int(sys.argv[4]) if len(sys.argv) > 4 else 100  # exclusivo
methods = ['ig_ql', 'ils_ql_hi', 'ils_rand_hi']
max_iter_default = CFG.MAX_ITER

inst_name = f"rand_{n}j_{m}m"
csv_path = 'results/results.csv'
qtables_dir = 'results/qtables'
os.makedirs(qtables_dir, exist_ok=True)

fieldnames = ["instance", "method", "rep", "n", "m", "lb",
              "makespan", "gap_pct", "cpu_time", "n_improvements"]

existing_keys = set()
with open(csv_path, newline='') as f_exist:
    for row in csv.DictReader(f_exist):
        existing_keys.add((row["instance"], row["method"], row["rep"]))

print(f"=== {inst_name}, reps [{rep_start},{rep_end}), metodos={methods} "
      f"-- {len(existing_keys)} chaves ja existentes no csv ===", flush=True)

t0 = time.time()
n_done_this_run = 0
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for inst_id in range(rep_start, rep_end):
        seed = CFG.BASE_SEED + inst_id
        inst = None  # instancia so' e' gerada se necessario (lazy)
        for method in methods:
            key = (inst_name, method, str(inst_id))
            if key in existing_keys:
                continue
            if inst is None:
                inst = PMSPInstance.random(n=n, m=m, seed=seed, name=inst_name)

            if method in ("ils_ql_hi", "ils_rand_hi"):
                mi = CFG.MAX_ITER_BY_SIZE.get((n, m), max_iter_default)
            else:
                mi = max_iter_default

            r = run_single(inst, method, inst_id, max_iter=mi, q_init=None)

            if method.startswith("ils_ql"):
                sel = r.get("selector")
                if sel is not None:
                    fname = f"{inst_name}_{method}_rep{inst_id:02d}"
                    sel.save(os.path.join(qtables_dir, fname))

            writer.writerow({k: r[k] for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
            n_done_this_run += 1
            elapsed = time.time() - t0
            print(f"  [ok] {inst_name} / {method:12s} / rep{inst_id:02d} "
                  f"gap={r['gap_pct']:.4f}% cpu={r['cpu_time']:.3f}s "
                  f"(total decorrido: {elapsed:.1f}s)", flush=True)

print(f"=== CHUNK CONCLUIDO: {n_done_this_run} linhas novas em "
      f"{time.time()-t0:.1f}s ===", flush=True)
