"""
experiment/run_cycle48_tiesall_entrada280.py (Entrada 280, Ciclo 48)
O Q-Learning padrao com DESEMPATE ALEATORIO A CADA DECISAO (ql_rt) em TODAS as colecoes em que o Q-Learning 'invariante a ordem' (ql_perm, permutacao por execucao) foi repetido na Entrada 260-267: o espaco B5 do benchmark padrao (Entrada 279 fez B15 e B30),
as familias publicadas de R||Cmax (Entrada 260), a bateria sintetica de P||Cmax (Entrada 261), as instancias publicadas held-out de P||Cmax (Entrada 264) e o ILS do TSP (Entrada 267). Mesmas instancias e mesmos hiperparametros do Q-Learning padrao de cada bateria
(alfa 0,1, gama 0,9, epsilon 1 -> 0,05 por 0,995, recompensa nao negativa, tabela inicial zero); so' o desempate muda (pmsp/ur_ties.py, pmsp/ig_ties.py, pmsp/tsp_ties.py: tie_mode = 1); sementes proprias.
Uso: python -m experiment.run_cycle48_tiesall_entrada280 --bench {b5,families,pcmax,pub,tsp} [--space S] --shard i --nshards n [--pilot]
Saidas: results/ties280_{bench}_{espaco}_s{i}of{n}_entrada280.csv
"""
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
from pmsp import ig_ties as IT  # noqa: E402
from pmsp import tsp_ties as TT  # noqa: E402
from pmsp import ur_ties as UT  # noqa: E402

R = ROOT / "results"
T = 1000
HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
OFFSET = 51_000_000


def write(out, fields, rows, new):
    with open(out, "w" if new else "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def ur_ql_rt(P, a0, m, od, lb, st, d, ls, seed):
    out = UT.ur_agent_core_ties(P, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, len(st))), 1.0, 1)
    return int(out[0]), float(out[2])


def ig_ql_rt(p, a0, m, od, lb, st, d, ls, seed):
    out = IT.ig_agent_core_ties(p, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, len(st))), 1.0, 1)
    return int(out[0]), float(out[2])


# ---------------------------------------------------------------- B5 (benchmark padrao)
def run_b5(shard, nshards, pilot):
    from experiment import run_cycle32_fanjul_entrada256 as F
    space = "B5"
    man = F.manifest()
    man = man[man.role == ("tune" if pilot else "test")]
    out = R / f"ties280_b5_{space}_s{shard}of{nshards}_entrada280.csv"
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"]) for r in csv.DictReader(open(out, newline=""))}
    P0 = np.random.default_rng(1).integers(1, 50, size=(30, 4)).astype(np.int64)
    a00, od0 = F.constructive_ur(P0)
    st0, d0, ls0 = F.arm_table_b(space, 30)
    ur_ql_rt(P0, a00, 4, od0, float(F.lower_bound_ur(P0)), st0, d0, ls0, 1)
    new, t0, cnt = not out.exists(), time.perf_counter(), 0
    fields = ["type", "instance", "m", "n", "idx", "method", "seed_i", "makespan", "bks", "evals"]
    for _, r in man.iterrows():
        if int(r.ri) % nshards != shard or (r["type"], str(int(r.instance))) in done:
            continue
        P, a0, od, lb = F.load(r)
        st, d, ls = F.arm_table_b(space, int(r.n))
        rows = []
        for s in range(F.N_SEED):
            seed = F.seed_of(space, int(r.ri), "ql", s, False) + OFFSET
            mk, ev = ur_ql_rt(P, a0, int(r.m), od, lb, st, d, ls, seed)
            rows.append(dict(type=r["type"], instance=int(r.instance), m=int(r.m), n=int(r.n), idx=int(r.idx), method="ql_rt", seed_i=s, makespan=mk, bks=int(r.best_known), evals=ev))
        write(out, fields, rows, new)
        new = False
        cnt += 1
        if pilot and cnt >= 2:
            break
    print(f"[ties280 b5 s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- familias publicadas de R||Cmax
def run_families(space, shard, nshards, pilot):
    from experiment import run_cycle30_ur_external_entrada254 as E
    from experiment.run_agents_numba_entrada180 import arm_table
    E.U.warm()
    out = R / f"ties280_families_{space}_s{shard}of{nshards}_entrada280.csv"
    fields = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
    rows, t0 = [], time.perf_counter()
    for ci, (f, n, m) in enumerate(E.CLASSES):
        cls = E.cls_name(f, n, m)
        st, d, ls = arm_table(space, n)
        for rep in range(E.N_TEST):
            if (ci * E.N_TEST + rep) % nshards != shard:
                continue
            if pilot and rep >= 1:
                break
            P, a0, od, lb = E.make(ci, rep, pilot)
            for s in range(5):
                seed = E.seed_of(space, ci, rep, "ql", s) + OFFSET
                mk, ev = ur_ql_rt(P, a0, m, od, lb, st, d, ls, seed)
                rows.append(dict(space=space, cls=cls, rep=rep, method="ql_rt", seed_i=s, makespan=mk, lb=lb, T=T, evals=ev))
    write(out, fields, rows, True)
    print(f"[ties280 families {space} s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- P||Cmax sintetico
def run_pcmax(space, shard, nshards, pilot):
    from experiment import run_agents_numba_entrada180 as E
    E.warm()
    out = R / f"ties280_pcmax_{space}_s{shard}of{nshards}_entrada280.csv"
    fields = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
    rows, t0 = [], time.perf_counter()
    for ci, (n, m) in enumerate(E.CLASSES):
        cls = f"rand_{n}j_{m}m"
        st, d, ls = E.arm_table(space, n)
        for rep in range(E.N_TEST):
            if (ci * E.N_TEST + rep) % nshards != shard:
                continue
            if pilot and rep >= 1:
                break
            inst, p, a0, od = E.make(n, m, rep, pilot)
            for s in range(E.N_SEED):
                seed = E.seed_of(space, ci, rep, "ql", s) + OFFSET
                mk, ev = ig_ql_rt(p, a0, m, od, inst.lower_bound, st, d, ls, seed)
                rows.append(dict(space=space, cls=cls, rep=rep, method="ql_rt", seed_i=s, makespan=mk, lb=inst.lower_bound, T=T, evals=ev))
    write(out, fields, rows, True)
    print(f"[ties280 pcmax {space} s{shard}] {len(rows)} execucoes ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- P||Cmax publicado held-out
def run_pub(space, shard, nshards, pilot):
    from experiment import run_cycle17_published_entrada189 as P
    from experiment import run_cycle17_replication_entrada190 as RP
    from experiment.run_agents_numba_entrada180 import arm_table
    info = json.loads(RP.groups_path().read_text())
    test = P.load_set(RP.M_HOLD)[shard::nshards]
    out = R / f"ties280_pub_{space}_s{shard}of{nshards}_entrada280.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    fields = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
    new, t0, cnt = not out.exists(), time.perf_counter(), 0
    for i, rec in enumerate(test):
        fam, g = info["test"][rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        st, d, ls = arm_table(space, rec["n"])
        rows = []
        for s in range(P.N_SEED):
            seed = P.seed_of(space, "test", fam, rec["inst_id"], "ql", s) + OFFSET
            mk, ev = ig_ql_rt(rec["p"], a0, rec["m"], od, lb, st, d, ls, seed % (2 ** 31 - 1))
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method="ql_rt", seed_i=s, makespan=mk, lb=lb, T=T, evals=ev))
        write(out, fields, rows, new)
        new = False
        cnt += 1
        if pilot and cnt >= 2:
            break
    print(f"[ties280 pub {space} s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- ILS do TSP
def run_tsp(shard, nshards, pilot):
    from experiment import run_cycle40_tsp_entrada267 as X
    from pmsp import tsp_numba as TS
    from pmsp.instance_tsp import build
    D0, NL0 = build("unif", 40, 1)
    t00, _ = TS.construct(D0, NL0)
    for sp in X.SPACES:
        k0, s0 = TS.arm_table(sp)
        TT.tsp_agent_core_ties(D0, NL0, t00, k0, s0, X.T, TS.AGENT_Q, 0, 0.1, 0.1, 1, X.HP_Q, 0.0, 1)
    out = R / f"ties280_tsp_s{shard}of{nshards}_entrada280.csv"
    done = set()
    if out.exists():
        done = {int(r["inst"]) for r in csv.DictReader(open(out, newline=""))}
    fields = ["space", "kind", "n", "rep", "inst", "method", "seed_i", "length", "len0", "evals", "share0"]
    new, t0, cnt = not out.exists(), time.perf_counter(), 0
    for it in X.instances("tune" if pilot else "test"):
        if it["inst"] % nshards != shard or it["inst"] in done:
            continue
        D, NL = build(it["kind"], it["n"], it["seed"])
        tour0, _ = TS.construct(D, NL)
        len0 = float(TS.tour_len(tour0, D))
        rows = []
        for sp in X.SPACES:
            kinds, sizes = TS.arm_table(sp)
            for s in range(X.N_SEED):
                seed = X.seed_of(sp, it["inst"], "ql", s) + OFFSET
                r = TT.tsp_agent_core_ties(D, NL, tour0, kinds, sizes, X.T, TS.AGENT_Q, 0, 0.1, 0.1, seed, X.HP_Q, 0.0, 1)
                rows.append(dict(space=sp, kind=it["kind"], n=it["n"], rep=it["rep"], inst=it["inst"], method="ql_rt", seed_i=s, length=float(r[0]), len0=len0, evals=float(r[2]), share0=float(r[3][0]) / X.T))
        write(out, fields, rows, new)
        new = False
        cnt += 1
        if pilot and cnt >= 2:
            break
    print(f"[ties280 tsp s{shard}] concluido, {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--space", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.bench == "b5":
        run_b5(a.shard, a.nshards, a.pilot)
    elif a.bench == "families":
        run_families(a.space, a.shard, a.nshards, a.pilot)
    elif a.bench == "pcmax":
        run_pcmax(a.space, a.shard, a.nshards, a.pilot)
    elif a.bench == "pub":
        run_pub(a.space, a.shard, a.nshards, a.pilot)
    else:
        run_tsp(a.shard, a.nshards, a.pilot)
