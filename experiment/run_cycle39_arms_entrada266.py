"""
experiment/run_cycle39_arms_entrada266.py (Entrada 266, Ciclo 39)
TODAS as acoes fixas do espaco B30 (30 acoes; as 15 primeiras sao as de B15 e as 5 primeiras as de B5) nas 700 instancias de TESTE do benchmark de Fanjul-Peyro e Ruiz, 3 sementes, T = 1.000 (as mesmas condicoes da Entrada 256).
Motivo: permitir, sem nova execucao, a avaliacao de politicas estaticas escolhidas offline com qualquer GRANULARIDADE de conhecimento (uma acao global; por tamanho n; por tipo; por classe = a validada da Entrada 256) e em regimes de
TRANSFERENCIA (acao escolhida sem o tipo, ou sem o tamanho, da instancia testada), para compara-las com os aprendizes online e a politica neural (analysis/report_entrada266_granularity.py).
Uso: python -m experiment.run_cycle39_arms_entrada266 --shard i --nshards n     Saida: results/arms_test_B30_s{i}of{n}_entrada266.csv (type, instance, m, n, idx, arm, seed_i, makespan, bks, evals)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import run_cycle32_fanjul_entrada256 as F  # noqa: E402

R = F.R
SPACE = "B30"
FIELDS = ["type", "instance", "m", "n", "idx", "arm", "seed_i", "makespan", "bks", "evals"]


def run(shard, nshards):
    man = F.manifest()
    man = man[man.role == "test"]
    out = R / f"arms_test_{SPACE}_s{shard}of{nshards}_entrada266.csv"
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    F.warm()
    A = len(F.arm_table_b(SPACE, 100)[0])
    new = not out.exists()
    t0, cnt = time.perf_counter(), 0
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard or (r["type"], str(int(r.instance))) in done:
            continue
        P, a0, od, lb = F.load(r)
        rows = []
        for a in range(A):
            for s in range(F.N_SEED):
                mk, ev = F.run_fixed_or_agent(SPACE, f"arm{a}", P, a0, int(r.m), od, lb, int(r.n), F.seed_of(SPACE, int(r.ri), f"arm{a}", s, False))
                rows.append(dict(type=r["type"], instance=int(r.instance), m=int(r.m), n=int(r.n), idx=int(r.idx), arm=a, seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
        with open(out, "w" if new else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        new = False
        cnt += 1
        if cnt % 20 == 0:
            print(f"[arms s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[arms s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    run(a.shard, a.nshards)
