"""
experiment/run_cycle22_sched_entrada238.py
============================================
Entrada 238 (Ciclo 22, frente B): o ORACULO/COMPETIDOR DINAMICO NO TEMPO que as revisoes cegas #6 e #7 pediram: a melhor SEQUENCIA de acoes por classe (um cronograma de 3 fases, 125 combinacoes das 5 acoes do espaco S5; pmsp/sched_numba.py), escolhida nas instancias de AJUSTE e avaliada nas de TESTE,
contra a acao estatica validada (as 5 acoes fixas sao os cronogramas constantes). Se o cronograma nao supera a estatica validada, nao ha' estrutura no TEMPO a explorar, ao nivel da classe, por um aprendiz que se adapta durante a execucao (o aprendiz tabular tem a fracao do orcamento no estado).
Hospedeiros: 'pc' = P||Cmax sintetico (12 classes, 100 instancias de teste, 30 de ajuste; as mesmas das Entradas 180/209) e 'ur' = R||Cmax (Entrada 234). Espaco S5 (tamanho d/n em {0,05; 0,10; 0,20; 0,30; 0,40}, destruicao aleatoria), T = 1000 em tres fases de 333, 333 e 334 iteracoes.
Ajuste: por classe, 30 instancias x 125 cronogramas x 2 sementes; escolhe-se o cronograma de menor makespan medio. Teste: 5 sementes por instancia, metodo 'sch'; a acao validada ('val') e o aleatorio ('rand') vem dos arquivos existentes (agents_S5_entrada180.csv; ur_test_S5_*_entrada234.csv).
Fases (retomaveis): --tune HOSPEDEIRO | --run HOSPEDEIRO [--shard i --nshards n]. Saidas: results/sched_choice_HOSPEDEIRO_entrada238.json, results/sched_test_HOSPEDEIRO_s*of*_entrada238.csv. Pre-registro: analysis/report_entrada238_sched.py (docstring; hashes em results/entrada238_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle22_ur_entrada234 as UR  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from experiment.run_agents_numba_entrada180 import make as make_pc  # noqa: E402
from pmsp import sched_numba as S  # noqa: E402

R = ROOT / "results"
T, PH, A = 1000, 3, 5
SCHEDS = list(itertools.product(range(A), repeat=PH))
FIELDS = ["host", "cls", "rep", "seed_i", "makespan", "sched"]
HOSTS = {"pc": 0, "ur": 1}


def seq_of(sch):
    bounds = [0, T // PH, 2 * (T // PH), T]
    return np.concatenate([np.full(bounds[i + 1] - bounds[i], sch[i], dtype=np.int64) for i in range(PH)])


def instance(host, n, m, rep, ci, tune):
    if host == "pc":
        inst, p, a0, od = make_pc(n, m, rep, tune)
        return p, a0, od
    P, a0, od, lb = UR.make(n, m, rep, ci, tune)
    return P, a0, od


def run1(host, X, a0, m, n, od, seq, seed):
    st, d, ls = arm_table("S5", n)
    core = S.ig_sched_core if host == "pc" else S.ur_sched_core
    best, ev = core(X, a0, m, st, d, ls, seq, seed, od)
    return int(best)


def seed_of(host, kind, ci, rep, s, k=0):
    return 100_000_000 * (1 + HOSTS[host]) + (0 if kind == "test" else 50_000_000) + 1000 * ci + rep + 50_000 * s + 7_919 * k


def warm():
    for host in HOSTS:
        X, a0, od = instance(host, 20, 4, 0, 0, True)
        run1(host, X, a0, 4, 20, od, seq_of((0, 1, 2)), 1)


def tune(host):
    warm()
    out = {}
    rows = []
    t0 = time.perf_counter()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        tot = np.zeros(len(SCHEDS))
        for rep in range(UR.N_TUNE):
            X, a0, od = instance(host, n, m, rep, ci, True)
            for k, sch in enumerate(SCHEDS):
                seq = seq_of(sch)
                for s in range(2):
                    tot[k] += run1(host, X, a0, m, n, od, seq, seed_of(host, "tune", ci, rep, s, k))
        mean = tot / (UR.N_TUNE * 2)
        k = int(np.argmin(mean))
        stat = {a: float(mean[SCHEDS.index((a, a, a))]) for a in range(A)}
        a_best = min(stat, key=stat.get)
        out[cls] = dict(sched=list(SCHEDS[k]), mean=float(mean[k]), static_best=a_best, static_mean=stat[a_best], rel_gain_vs_static_pct=100 * (stat[a_best] - float(mean[k])) / stat[a_best])
        rows.append(f"{cls} {SCHEDS[k]} {mean[k]:.1f} static({a_best}) {stat[a_best]:.1f}")
        print(f"[sched tune {host}] {cls} -> {SCHEDS[k]} ({time.perf_counter() - t0:.0f}s)", flush=True)
    (R / f"sched_choice_{host}_entrada238.json").write_text(json.dumps(out, indent=1))
    print("\n".join(rows))


def run_test(host, shard, nshards):
    ch = json.loads((R / f"sched_choice_{host}_entrada238.json").read_text())
    out = R / f"sched_test_{host}_s{shard}of{nshards}_entrada238.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    new = not out.exists()
    t0 = time.perf_counter()
    k = 0
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        seq = seq_of(tuple(ch[cls]["sched"]))
        for rep in range(UR.N_TEST):
            if (ci * UR.N_TEST + rep) % nshards != shard or (cls, str(rep)) in done:
                continue
            X, a0, od = instance(host, n, m, rep, ci, False)
            rows = [dict(host=host, cls=cls, rep=rep, seed_i=s, makespan=run1(host, X, a0, m, n, od, seq, seed_of(host, "test", ci, rep, s)), sched="".join(map(str, ch[cls]["sched"]))) for s in range(5)]
            with open(out, "w" if new else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            new = False
            k += 1
    print(f"[sched {host} s{shard}] concluido ({k} instancias, {time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.tune:
        tune(a.tune)
    if a.run:
        run_test(a.run, a.shard, a.nshards)
