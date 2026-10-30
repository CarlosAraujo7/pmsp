"""
experiment/run_cycle28_horizon_ur_entrada250.py
================================================
Entrada 250 (Ciclo 28): HORIZONTE LONGO no R||Cmax (o hospedeiro com folga real). As revisoes cegas das Entradas 246 e 249 dizem que 'aprendizado online nao compensa' pode ser so' 'nada e' aprendivel em 1.000 passos' (a tabela do Q-Learning tem 81 estados x 5-15 acoes; o epsilon cai a 0,05 em ~600 passos) e que o estudo de horizonte do artigo so' cobre o P||Cmax sintetico com cinco tamanhos. Aqui: T = 5.000, 20.000 e 100.000 iteracoes no R||Cmax (S5 e S15), 12 classes x 20 instancias (as primeiras de cada classe), 2 sementes por metodo.
Metodos: rand (controle aleatorio), ql (Q-Learning com os hiperparametros padrao: epsilon cai por 0,995 por passo), qls (Q-Learning com o epsilon ESCALADO a T: decaimento 0,995^(1000/T), de modo que o epsilon chega ao piso na mesma fracao da execucao que em T = 1.000), qo (Q-Learning otimista da Entrada 234, hiperparametros congelados), alns (roleta ALNS, hiperparametros congelados) e val (a acao fixa validada por classe em T = 1.000, congelada da Entrada 234). Nada e' reajustado ao horizonte (a validada e os hiperparametros dos aprendizes vem de T = 1.000; ver os limites no diario).
Fases (retomaveis): --run ESPACO T --shard i --nshards n [--pilot: 3 instancias de AJUSTE por classe]. Saidas: results/ur_hz_ESPACO_T<T>_s*of*_entrada250.csv. Pre-registro: analysis/report_entrada250_ur_horizon.py (docstring; hashes em results/entrada250_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle22_ur_entrada234 as U  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402

R = ROOT / "results"
SPACES = U.SPACES
HORIZONS = (5000, 20000, 100000)
METHODS = ["rand", "ql", "qls", "qo", "alns", "val"]
N_RUN, N_SEED = 20, 2
FIELDS = ["space", "T", "cls", "rep", "method", "seed_i", "makespan", "lb", "evals"]


def seed_of(space, T, ci, rep, method, s):
    return 600_000_000 + 50_000_000 * SPACES.index(space) + 10_000_000 * HORIZONS.index(T) + 1_000_000 * METHODS.index(method) + 20_000 * ci + 5 * rep + s


def hp_scaled(T):
    return np.array([0.1, 0.9, 0.995 ** (1000.0 / T), 0.05, 0.0], dtype=np.float64)


def run_method(space, method, P, a0, m, od, lb, n, T, seed, val_arm, qo_cfg):
    if method == "rand":
        return U.run(space, P, a0, m, od, lb, n, UA.AGENT_RANDOM, 0, 0.1, 0.1, seed, T=T)
    if method == "ql":
        return U.run(space, P, a0, m, od, lb, n, UA.AGENT_Q, 0, 0.1, 0.1, seed, T=T)
    if method == "qls":
        st, d, ls = arm_table(space, n)
        best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, T, UA.AGENT_Q, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp_scaled(T), 0)
        return int(best), float(ev)
    if method == "qo":
        return U.run_qo(space, P, a0, m, od, lb, n, qo_cfg, seed, T=T)
    if method == "alns":
        return U.run(space, P, a0, m, od, lb, n, UA.AGENT_ALNS, 0, U.ALNS_HP[0], U.ALNS_HP[1], seed, T=T)
    return U.run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, val_arm, 0.1, 0.1, seed, T=T)


def run_space(space, T, shard, nshards, pilot):
    import json
    val = json.loads(U.choice_path(space).read_text())["val"]
    qo_cfg = tuple(json.loads(U.choice_qo_path(space).read_text())["cfg"])
    out = R / (f"ur_hz_pilot_{space}_T{T}_entrada250.csv" if pilot else f"ur_hz_{space}_T{T}_s{shard}of{nshards}_entrada250.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    U.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    cnt = 0
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(3 if pilot else N_RUN):
            if (not pilot) and (ci * N_RUN + rep) % nshards != shard:
                continue
            if (cls, str(rep)) in done:
                continue
            P, a0, od, lb = U.make(n, m, rep, ci, bool(pilot))
            rows = []
            for me in METHODS:
                for s in range(N_SEED):
                    mk, ev = run_method(space, me, P, a0, m, od, lb, n, T, seed_of(space, T, ci, rep, me, s), val[cls], qo_cfg)
                    rows.append(dict(space=space, T=T, cls=cls, rep=rep, method=me, seed_i=s, makespan=mk, lb=lb, evals=ev))
            with open(out, "w" if new else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            new = False
            cnt += 1
            if cnt % 20 == 0:
                print(f"[hz {space} T{T} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[hz {space} T{T} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, metavar=("SPACE", "T"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.run:
        run_space(a.run[0], int(a.run[1]), a.shard, a.nshards, a.pilot)
