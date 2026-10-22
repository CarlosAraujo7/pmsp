"""
experiment/run_cycle18_famF_entrada199.py
===========================================
Entrada 199 (Ciclo 18, frente C): VALIDACAO FORA DA AMOSTRA do pre-teste de folga/ruido (VNR) na familia F do benchmark publicado de Ostojic et al. (2025) para P||Cmax (FFU e FFNU; m em {5, 10, 25}; n em {10, 50, 100, 500, 1000}; 26 classes (subpasta, m, n) de 30 instancias),
que NENHUM aprendiz compilado, hiperparametro, acao validada ou limiar do pre-teste viu (a familia F so' entrou no ILS-semente da Entrada 117 e em exploracoes anteriores do IG sem aprendizes). O limiar tau = 0,877 foi congelado em results/pretest_tau_entrada198.json.
Por classe (celula): as 12 primeiras instancias (por inst_id) sao de SONDA (3 sementes do aleatorio e 3 de cada acao fixa; insumo do VNR e da acao validada da celula) e as 18 restantes de TESTE (5 sementes: aleatorio, Q-Learning, UCB1, roleta ALNS, acao validada e, com 1 semente, cada acao fixa para o oraculo). T = 1000; espacos S5 e S15 (partida LPT, aceita se nao piora).
Fases (retomaveis): --probe ESPACO | --choose | --run ESPACO [--shard i --nshards n]. Saidas: results/famF_probe_ESPACO_entrada199.csv, famF_choice_entrada199.json, famF_test_ESPACO_s*of*_entrada199.csv.
Pre-registro: analysis/report_entrada199_pretest_outofsample.py (docstring, congelado antes de rodar; hashes em results/entrada199_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXT))

from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402
from pmsp.constructive import lpt  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

R = ROOT / "results"
N_PROBE, N_SEED_PROBE, N_SEED_TEST = 12, 3, 5
LEARNERS = {"ql": (H.AGENT_Q, 0.1, 0.1), "ucb": (H.AGENT_UCB, 0.1, 0.1), "alns": (H.AGENT_ALNS, 0.3, 0.0)}


def seed_of(space, kind, member, method, s):
    return (zlib.crc32(member.encode()) % 1000003 * 131 + s * 7919 + zlib.crc32(method.encode()) % 100003 * 104729 + ["S5", "S15"].index(space) * 32452843 + {"probe": 1, "test": 2}[kind] * 49979687) % (2 ** 31 - 1)


def load_f():
    """Instancias da familia F de manifest.csv, agrupadas por classe (subpasta|m|n) e ordenadas por inst_id: (probe, test) por classe."""
    from loader import load_many
    rows = [r for r in csv.DictReader(open(EXT / "manifest.csv", newline="")) if r["family"] == "F"]
    data = load_many("F", [r["member"] for r in rows])
    by = {}
    for r in rows:
        m, n, p = data[r["member"]]
        assert sum(p) < 2 ** 62
        by.setdefault(f"{r['subfolder']}|{m}|{n}", []).append(dict(member=r["member"], inst_id=int(r["inst_id"]), m=m, n=n, p=np.asarray(p, dtype=np.int64)))
    out = {}
    for cell, lst in by.items():
        lst.sort(key=lambda x: x["inst_id"])
        assert len(lst) == 30, (cell, len(lst))
        out[cell] = (lst[:N_PROBE], lst[N_PROBE:])
    return out


def prepare_inst(rec):
    inst = PMSPInstance(p=rec["p"], m=rec["m"], name=rec["member"])
    return inst.lower_bound, lpt(inst).assignment.astype(np.int64), np.argsort(rec["p"])[::-1].astype(np.int64)


def run(space, rec, lb, a0, od, agent, fixed, lr, lr2, seed, T=1000):
    st, d, ls = arm_table(space, rec["n"])
    best, _, ev, used, _ = H.ig_agent_core3(rec["p"], a0, rec["m"], st, d, ls, T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0)
    return int(best), float(ev)


def warm():
    rec = dict(p=np.arange(1, 21, dtype=np.int64), m=4, n=20, member="warm")
    lb, a0, od = prepare_inst(rec)
    for sp in ("S5", "S15"):
        for ag in range(11):
            run(sp, rec, lb, a0, od, ag, 0, 0.5, 0.1, 1, T=30)


def probe(space):
    warm()
    A = len(arm_table(space, 100)[0])
    rows = []
    t0 = time.perf_counter()
    for cell, (pr, _) in load_f().items():
        for rec in pr:
            lb, a0, od = prepare_inst(rec)
            base = dict(space=space, cls=cell, rep=rec["inst_id"], lb=lb, T=1000)
            for s in range(N_SEED_PROBE):
                mk, ev = run(space, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "probe", rec["member"], "rand", s))
                rows.append(dict(base, method="rand", seed_i=s, makespan=mk, evals=ev))
                for a in range(A):
                    mk, ev = run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "probe", rec["member"], f"arm{a}", s))
                    rows.append(dict(base, method=f"arm{a}", seed_i=s, makespan=mk, evals=ev))
        print(f"[famF probe {space}] {cell} ok ({time.perf_counter() - t0:.0f}s)", flush=True)
    _write(R / f"famF_probe_{space}_entrada199.csv", rows, True, ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"])


def choose():
    import pandas as pd
    ch = {}
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"famF_probe_{sp}_entrada199.csv")
        d["gap"] = 100 * (d.makespan - d.lb) / d.lb
        ch[sp] = {}
        for cell, x in d.groupby("cls"):
            arm = x[x.method.str.startswith("arm")].groupby("method").gap.mean()
            ch[sp][cell] = int(arm.idxmin()[3:])
    (R / "famF_choice_entrada199.json").write_text(json.dumps(ch, indent=1))
    print("escolhas:", ch)


def run_space(space, shard, nshards):
    ch = json.loads((R / "famF_choice_entrada199.json").read_text())[space]
    out = R / f"famF_test_{space}_s{shard}of{nshards}_entrada199.csv"
    done = set()
    if out.exists():
        done = {r["rep_key"] for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arm_table(space, 100)[0])
    items = [(cell, rec) for cell, (_, te) in load_f().items() for rec in te][shard::nshards]
    t00 = time.perf_counter()
    new = not out.exists()
    fields = ["space", "cls", "rep", "rep_key", "method", "seed_i", "makespan", "lb", "T", "evals"]
    for i, (cell, rec) in enumerate(items):
        key = rec["member"]
        if key in done:
            continue
        lb, a0, od = prepare_inst(rec)
        rows = []

        def add(method, s, mk, ev):
            rows.append(dict(space=space, cls=cell, rep=rec["inst_id"], rep_key=key, method=method, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
        for s in range(N_SEED_TEST):
            mk, ev = run(space, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "test", key, "rand", s))
            add("rand", s, mk, ev)
            for k, (ag, x, y) in LEARNERS.items():
                mk, ev = run(space, rec, lb, a0, od, ag, 0, x, y, seed_of(space, "test", key, k, s))
                add(k, s, mk, ev)
            mk, ev = run(space, rec, lb, a0, od, H.AGENT_FIXED, ch[cell], 0.1, 0.1, seed_of(space, "test", key, "val", s))
            add("val", s, mk, ev)
        for a in range(A):
            mk, ev = run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "test", key, f"arm{a}", 0))
            add(f"arm{a}", 0, mk, ev)
        _write(out, rows, new, fields)
        new = False
        if i % 50 == 0:
            print(f"[famF {space} shard {shard}] {i}/{len(items)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[famF {space} shard {shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


def _write(out, rows, new, fields):
    import os
    with open(out, "a" if not new else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--choose", action="store_true")
    for f in ("probe", "run"):
        ap.add_argument("--" + f, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.probe:
        probe(a.probe)
    if a.choose:
        choose()
    if a.run:
        run_space(a.run, a.shard, a.nshards)
