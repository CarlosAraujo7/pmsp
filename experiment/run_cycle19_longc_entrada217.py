"""
experiment/run_cycle19_longc_entrada217.py
============================================
Entrada 217 (Ciclo 19, frente E): HORIZONTE LONGO onde ha' folga. O estudo de horizonte das Entradas 185-187 cobriu so' celulas sinteticas de folga baixa (a critica 4/5 da revisao cega #2). Aqui, na familia C publicada, tercis 1 e 2 (a folga e' 0,3-0,4% a T = 1000), T = 5.000 e 20.000, espaco S5 (cinco tamanhos):
as instancias do HELD-OUT (3.950) da familia C com tercil >= 1, ordenadas por nome e tomadas de tres em tres (340 de 1.020; deterministico). Metodos (3 sementes cada, T em {5000, 20000}): rand, ql (Q-Learning frio padrao), wq (Q aquecido da Entrada 211), alns (roleta; hiperparametros da familia), val (acao validada do grupo), sel (acao do seletor offline) e cada acao fixa (4 sementes: o oraculo com validacao cruzada
escolhe nas sementes {0, 1} e avalia nas {2, 3}). Fases (retomaveis): --run T --shard i --nshards n. Saidas: results/longc_T<T>_s*of*_entrada217.csv. Pre-registro: analysis/report_entrada217_longc.py (docstring; hashes em results/entrada217_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment import run_cycle19_equalcpu_entrada213 as EQ  # noqa: E402
from experiment import run_cycle19_warmq_entrada211 as W  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402

R = ROOT / "results"
SPACE = "S5"
N_SEED, N_ARM_SEED = 3, 4
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]


def subset():
    recs, info = S.test_records("heldout")
    sel = [r for r in recs if info[r["member"]][0] == "C" and info[r["member"]][1] >= 1]
    sel.sort(key=lambda r: r["member"])
    return sel[::3], info


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_T(T, shard, nshards):
    import joblib
    model = joblib.load(R / f"selector_model_{SPACE}_entrada200.joblib")
    Q0 = np.load(W.q_path(SPACE))
    ch = json.loads(P.choice_path(SPACE).read_text())
    recs, info = subset()
    recs = recs[shard::nshards]
    out = R / f"longc_T{T}_s{shard}of{nshards}_entrada217.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        a_sel = EQ.sel_arm(model, rec, lb)
        a_val = ch[fam]["val"][str(g)]
        xa, ya = ch[fam]["hp"]["alns"]
        rows = []

        def add(meth, s, mk, ev):
            rows.append(dict(space=SPACE, cls=cls, rep=rec["inst_id"], method=meth, seed_i=s, makespan=mk, lb=lb, T=T, evals=ev))
        for s in range(N_SEED):
            sd = lambda m: P.seed_of(SPACE, "test", fam, rec["inst_id"], f"{m}_T{T}", s)
            mk, ev, _ = P.run(SPACE, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, sd("rand"), T=T)
            add("rand", s, mk, ev)
            mk, ev, _ = P.run(SPACE, rec, lb, a0, od, H.AGENT_Q, 0, 0.1, 0.1, sd("ql"), T=T)
            add("ql", s, mk, ev)
            mk, ev, _ = P.run(SPACE, rec, lb, a0, od, H.AGENT_ALNS, 0, xa, ya, sd("alns"), T=T)
            add("alns", s, mk, ev)
            mk, ev, _ = P.run(SPACE, rec, lb, a0, od, H.AGENT_FIXED, a_val, 0.1, 0.1, sd("val"), T=T)
            add("val", s, mk, ev)
            mk, ev, _ = P.run(SPACE, rec, lb, a0, od, H.AGENT_FIXED, a_sel, 0.1, 0.1, sd("sel"), T=T)
            add("sel", s, mk, ev)
            mk, ev, _ = W.run5(SPACE, rec, lb, a0, od, sd("wq"), W.HP_WQ, Q0, 0.05, T=T)
            add("wq", s, mk, ev)
        for a in range(5):
            for s in range(N_ARM_SEED):
                mk, ev, _ = P.run(SPACE, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, P.seed_of(SPACE, "test", fam, rec["inst_id"], f"arm{a}_T{T}", s), T=T)
                add(f"arm{a}", s, mk, ev)
        _append(out, rows, new)
        new = False
        if i % 20 == 0:
            print(f"[longc T={T} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[longc T={T} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.run:
        run_T(a.run, a.shard, a.nshards)
