"""
experiment/run_cycle19_cforacle_entrada209.py
===============================================
Entrada 209 (Ciclo 19, frente A): o ORACULO COM VALIDACAO CRUZADA DE SEMENTES. A folga do oraculo das Entradas 180-197 e' o melhor makespan de UMA semente por acao em cada instancia, logo enviesada para cima pelo ruido entre execucoes (critica 3 da revisao cega #2; e o ruido e' justamente o que explica o flowshop).
Aqui cada acao fixa recebe 3 sementes NOVAS (1, 2, 3) nas mesmas instancias de teste; com a semente 0 ja' existente nos arquivos das Entradas 180, 189, 190 e 197, a ACAO do oraculo e' escolhida nas sementes {0, 1} e AVALIADA nas sementes {2, 3} (analysis/report_entrada209_cforacle.py).
Conjuntos: syn (12 classes x 100 instancias sinteticas do P||Cmax; S5 e S15), discovery (2.550 publicadas; S5 e S15), heldout (3.950; S5 e S15) e pfsp (360 sinteticas + 90 de Taillard do flowshop; F5 e F15). T = 1000.
Fases (retomaveis): --syn ESPACO | --pub CONJUNTO ESPACO --shard i --nshards n | --pfsp ESPACO --shard i --nshards n. Saidas: results/cfo_<conjunto>_ESPACO[_s*of*]_entrada209.csv (espaco, cls, rep, arm, seed_i, makespan). Pre-registro: analysis/report_entrada209_cforacle.py (docstring; hashes em results/entrada209_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_pfsp_entrada197 as F  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from experiment.run_cycle16_entrada185 import make  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402
from pmsp import pfsp_numba as N  # noqa: E402

R = ROOT / "results"
SEEDS = (1, 2, 3)
FIELDS = ["space", "cls", "rep", "arm", "seed_i", "makespan"]


def seed_cf(space, key, arm, s):
    return (zlib.crc32(key.encode()) % 1000003 * 131 + s * 7919 + arm * 104729 + ["S5", "S15", "F5", "F15"].index(space) * 32452843 + 6 * 49979687) % (2 ** 31 - 1)


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def syn(space):
    P.warm()
    A = len(arm_table(space, 100)[0])
    out = R / f"cfo_syn_{space}_entrada209.csv"
    rows = []
    t0 = time.perf_counter()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        st, d, ls = arm_table(space, n)
        for rep in range(100):
            inst, p, a0, od = make(n, m, rep, False)
            lb = inst.lower_bound
            for a in range(A):
                for s in SEEDS:
                    best, _, ev, used, _ = H.ig_agent_core3(p, a0, m, st, d, ls, 1000, H.AGENT_FIXED, a, 0.1, 0.1, seed_cf(space, f"{cls}#{rep}", a, s), od, lb, 0, 0, 0.0)
                    rows.append(dict(space=space, cls=cls, rep=rep, arm=a, seed_i=s, makespan=int(best)))
        print(f"[cfo syn {space}] {cls} ({time.perf_counter() - t0:.0f}s)", flush=True)
    _append(out, rows, True)


def pub(conj, space, shard, nshards):
    P.warm()
    recs, info = S.test_records(conj)
    recs = recs[shard::nshards]
    A = len(arm_table(space, 100)[0])
    out = R / f"cfo_{conj}_{space}_s{shard}of{nshards}_entrada209.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        st, d, ls = arm_table(space, rec["n"])
        rows = []
        for a in range(A):
            for s in SEEDS:
                best, _, ev, used, _ = H.ig_agent_core3(rec["p"], a0, rec["m"], st, d, ls, 1000, H.AGENT_FIXED, a, 0.1, 0.1, seed_cf(space, rec["member"], a, s), od, lb, 0, 0, 0.0)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], arm=a, seed_i=s, makespan=int(best)))
        _append(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[cfo {conj} {space} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[cfo {conj} {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


def pfsp(space, shard, nshards):
    F.warm()
    A = len(F.arms(space)[0])
    out = R / f"cfo_pfsp_{space}_s{shard}of{nshards}_entrada209.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    new = not out.exists()
    t0 = time.perf_counter()
    items = F.test_items()[shard::nshards]
    for i, (kind, ci, n, m, rep, cls_id) in enumerate(items):
        if cls_id in done:
            continue
        if kind == "syn":
            p, lb = F.synth(n, m, ci, rep, False)
        else:
            p, lb, _, _ = F.taillard(rep)
        rows = []
        for a in range(A):
            for s in SEEDS:
                mk, ev = F.run(space, p, lb, N.AGENT_FIXED, a, 0.1, 0.1, seed_cf(space, cls_id, a, s))
                rows.append(dict(space=space, cls=cls_id, rep=rep, arm=a, seed_i=s, makespan=mk))
        _append(out, rows, new)
        new = False
        if i % 20 == 0:
            print(f"[cfo pfsp {space} s{shard}] {i}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[cfo pfsp {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", default=None)
    ap.add_argument("--pub", nargs=2, default=None, metavar=("CONJUNTO", "ESPACO"))
    ap.add_argument("--pfsp", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.syn:
        syn(a.syn)
    if a.pub:
        pub(a.pub[0], a.pub[1], a.shard, a.nshards)
    if a.pfsp:
        pfsp(a.pfsp, a.shard, a.nshards)
