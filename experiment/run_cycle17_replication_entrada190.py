"""
experiment/run_cycle17_replication_entrada190.py
==================================================
Entrada 190 (Ciclo 17, estagio 3): REPLICACAO CONFIRMATORIA do estagio 2 (Entrada 189) nas instancias publicadas do conjunto HELD-OUT da Entrada 90 (external_benchmarks/pcmax_symmetry2025/manifest_holdout_entrada90.csv: B 1500, C 1500, E 1160; disjuntas das de ajuste e das de teste do estagio 2).
NADA e' reajustado: os hiperparametros dos aprendizes, as acoes validadas por (familia, tercil de n), os limiares de n e a razao de CPU K por grupo sao os do estagio 2 (results/pub_choice_S5|S15_entrada189.json, pub_groups_entrada189.json, pub_cpu_ratio_entrada189.json). Mesmos nove aprendizes, S5 e S15, T = 1000, 5 sementes,
rand, val, rand_eq_<aprendiz> e todas as acoes fixas (oraculo, 1 semente). Instancias de soma de tempos >= 2^62 (familia B, ~10^18) excluidas como no estagio 2 (nao cabem no int64 do nucleo compilado).
Fases (retomaveis): --prepare | --run ESPACO --shard i --nshards n. Pre-registro: analysis/report_entrada190_replication.py (docstring, congelado antes de rodar; hashes em results/entrada190_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from experiment.run_cycle16_entrada185 import write  # noqa: E402

R = ROOT / "results"
M_HOLD = P.EXT / "manifest_holdout_entrada90.csv"


def groups_path():
    return R / "pubh_groups_entrada190.json"


def prepare():
    thr = json.loads(P.groups_path().read_text())["thresholds"]
    hold = P.load_set(M_HOLD)
    info = dict(thresholds=thr, test={r["member"]: [r["family"], P.group_of(r["family"], r["n"], thr)] for r in hold}, excluded=P.n_excluded(M_HOLD))
    groups_path().write_text(json.dumps(info, indent=1))
    cnt = {f: [sum(1 for r in hold if r["family"] == f and P.group_of(f, r["n"], thr) == g) for g in range(3)] for f in P.FAMS}
    print("instancias held-out por (familia, grupo):", cnt, "| excluidas (soma >= 2^62):", info["excluded"], "| total:", len(hold))


def run_space(space, shard, nshards):
    info = json.loads(groups_path().read_text())
    ch = json.loads(P.choice_path(space).read_text())
    K = json.loads(P.k_path().read_text())[space]
    test = P.load_set(M_HOLD)[shard::nshards]
    out = R / f"pubh_{space}_s{shard}of{nshards}_entrada190.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    A = len(arm_table(space, 100)[0])
    t00 = time.perf_counter()
    new = not out.exists()
    for i, rec in enumerate(test):
        fam, g = info["test"][rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        rows = []

        def add(method, s, mk, ev, TT):
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=method, seed_i=s, makespan=mk, lb=lb, T=TT, evals=ev))
        for s in range(P.N_SEED):
            mk, ev, _ = P.run(space, rec, lb, a0, od, P.H.AGENT_RANDOM, 0, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], "rand", s))
            add("rand", s, mk, ev, 1000)
            for k in P.LEARNERS:
                x, y = ch[fam]["hp"].get(k, [0.1, 0.1])
                mk, ev, _ = P.run(space, rec, lb, a0, od, P.AGENT[k], 0, x, y, P.seed_of(space, "test", fam, rec["inst_id"], k, s))
                add(k, s, mk, ev, 1000)
                Teq = max(50, int(round(1000 * K[f"{fam}|{g}"][k])))
                mk, ev, _ = P.run(space, rec, lb, a0, od, P.H.AGENT_RANDOM, 0, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], "rand_eq" + k, s), T=Teq)
                add(f"rand_eq_{k}", s, mk, ev, Teq)
            mk, ev, _ = P.run(space, rec, lb, a0, od, P.H.AGENT_FIXED, ch[fam]["val"][str(g)], 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], "val", s))
            add("val", s, mk, ev, 1000)
        for a in range(A):
            mk, ev, _ = P.run(space, rec, lb, a0, od, P.H.AGENT_FIXED, a, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], f"arm{a}", 0))
            add(f"arm{a}", 0, mk, ev, 1000)
        write(out, rows, new)
        new = False
        if i % 150 == 0:
            print(f"[pubh {space} shard {shard}] {i}/{len(test)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[pubh {space} shard {shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.run:
        run_space(a.run, a.shard, a.nshards)
