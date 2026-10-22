"""
experiment/run_cycle17_published_entrada189.py
================================================
Entrada 189 (Ciclo 17, frente A'): os aprendizes COMPILADOS e os concorrentes (nove aprendizes: ql, ucb, pg, pm, ap, alns, ts, dmab, race) nas instancias PUBLICADAS do benchmark de Ostojic et al. (2025) para P||Cmax (familias B, C e E), nos espacos S5 (5 tamanhos de destruicao)
e S15 (3 estrategias x 5 tamanhos), T = 1000, partida LPT, aceita se nao piora. Instancias de TESTE: as de external_benchmarks/pcmax_symmetry2025/manifest_confirm2_entrada115.csv (B 980, C 1050, E 660; 157 classes (subpasta, m, n)); elas ja' foram usadas na Entrada 115 para outra pergunta
(Q-Learning original vs aleatorio com o IG em Python); nenhum aprendiz, hiperparametro ou acao validada de agora foi ajustado nelas. Instancias de AJUSTE: as 553 de manifest_calib_entrada115.csv (B 196, C 225, E 132), disjuntas das de teste.
Grupos: (familia, tercil de n), com limiares = percentis 33,3 e 66,7 do n das instancias de ajuste da familia; as acoes validadas (a melhor acao fixa pela lacuna media ao limite inferior classico) e a razao de CPU K de cada aprendiz sobre o aleatorio sao por grupo; os hiperparametros (lr do PG, alfa do PM,
(alfa, beta) do AP, rho da roleta, gama do TS, (lambda, delta) do DMAB, rho da corrida) sao por familia, pela menor lacuna media nas instancias de ajuste da familia. Metodos por instancia: rand, os nove aprendizes, val, rand_eq_<aprendiz> (T_eq = round(1000 K do grupo)) -- 5 sementes cada -- e todas as acoes fixas (oraculo, 1 semente).
Fases (retomaveis): --prepare | --tune ESPACO | --calibrate | --run ESPACO [--shard i --nshards n]. Pre-registro: analysis/report_entrada189_published.py (docstring, congelado antes de rodar; hashes em results/entrada189_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
from experiment.run_cycle16_entrada185 import FIELDS, write  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402
from pmsp.constructive import lpt  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

R = ROOT / "results"
FAMS = ["B", "C", "E"]
M_CAL = EXT / "manifest_calib_entrada115.csv"
M_TEST = EXT / "manifest_confirm2_entrada115.csv"
LEARNERS = ["ql", "ucb", "pg", "pm", "ap", "alns", "ts", "dmab", "race"]
AGENT = {"ql": H.AGENT_Q, "ucb": H.AGENT_UCB, "pg": H.AGENT_PG, "pm": H.AGENT_PM, "ap": H.AGENT_AP, "alns": H.AGENT_ALNS, "ts": H.AGENT_TS, "dmab": H.AGENT_DMAB, "race": H.AGENT_RACE}
GRID = {"pg": [(r, 0.1) for r in (0.03, 0.1, 0.3)], "pm": [(a, 0.0) for a in (0.1, 0.3, 0.8)], "ap": [(a, b) for a in (0.1, 0.3, 0.8) for b in (0.1, 0.3, 0.8)], "alns": [(r, 0.0) for r in (0.1, 0.3, 0.5)],
        "ts": [(g, 0.0) for g in (1.0, 0.99, 0.95)], "dmab": [(2.0, 0.0), (8.0, 0.0), (8.0, 0.02), (32.0, 0.02)], "race": [(r, 0.0) for r in (0.25, 0.5)]}
N_SEED = 5


def seed_of(space, kind, fam, inst_id, method, s, T=1000):
    return (inst_id * 131 + s * 7919 + zlib.crc32(method.encode()) % 100003 * 104729 + FAMS.index(fam) * 15485863 + ["S5", "S15"].index(space) * 32452843 + {"tune": 1, "test": 2, "calib": 3}[kind] * 49979687) % (2 ** 31 - 1)


def load_set(path):
    from loader import load_many
    rows = list(csv.DictReader(open(path, newline="")))
    out = []
    for fam in FAMS:
        rs = [r for r in rows if r["family"] == fam]
        data = load_many(fam, [r["member"] for r in rs])
        for r in rs:
            m, n, p = data[r["member"]]
            if sum(p) >= 2 ** 62:                 # tempos de ~10^18 (familia B, subpasta 13 e outras): a soma nao cabe no int64 do nucleo compilado; excluidas e contadas em prepare()
                continue
            out.append(dict(family=fam, member=r["member"], inst_id=int(r["inst_id"]), m=m, n=n, p=np.asarray(p, dtype=np.int64)))
    return out


def n_excluded(path):
    from loader import load_many
    rows = list(csv.DictReader(open(path, newline="")))
    cnt = {}
    for fam in FAMS:
        rs = [r["member"] for r in rows if r["family"] == fam]
        data = load_many(fam, rs)
        cnt[fam] = sum(1 for v in data.values() if sum(v[2]) >= 2 ** 62)
    return cnt


def prepare_inst(rec):
    inst = PMSPInstance(p=rec["p"], m=rec["m"], name=rec["member"])
    return inst.lower_bound, lpt(inst).assignment.astype(np.int64), np.argsort(rec["p"])[::-1].astype(np.int64)


def run(space, rec, lb, a0, od, agent, fixed, lr, lr2, seed, T=1000):
    st, d, ls = arm_table(space, rec["n"])
    best, _, ev, used, _ = H.ig_agent_core3(rec["p"], a0, rec["m"], st, d, ls, T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0)
    return int(best), float(ev), used


def groups_path():
    return R / "pub_groups_entrada189.json"


def choice_path(space):
    return R / f"pub_choice_{space}_entrada189.json"


def k_path():
    return R / "pub_cpu_ratio_entrada189.json"


def group_of(fam, n, thr):
    q1, q2 = thr[fam]
    return 0 if n <= q1 else (1 if n <= q2 else 2)


def warm():
    rec = dict(p=np.arange(1, 21, dtype=np.int64), m=4, n=20, member="warm", family="B", inst_id=0)
    lb, a0, od = prepare_inst(rec)
    for sp in ("S5", "S15"):
        for ag in range(11):
            run(sp, rec, lb, a0, od, ag, 0, 0.5, 0.1, 1, T=30)


def prepare():
    """Limiares de n por familia (percentis 33,3 e 66,7 do ajuste) e grupo de cada instancia de ajuste e de teste."""
    cal, test = load_set(M_CAL), load_set(M_TEST)
    thr = {fam: [float(x) for x in np.percentile([r["n"] for r in cal if r["family"] == fam], [100 / 3, 200 / 3])] for fam in FAMS}
    info = dict(thresholds=thr, calib={r["member"]: [r["family"], group_of(r["family"], r["n"], thr)] for r in cal}, test={r["member"]: [r["family"], group_of(r["family"], r["n"], thr)] for r in test})
    groups_path().write_text(json.dumps(info, indent=1))
    cnt = {f: [sum(1 for r in test if r["family"] == f and group_of(f, r["n"], thr) == g) for g in range(3)] for f in FAMS}
    info["excluded"] = dict(calib=n_excluded(M_CAL), test=n_excluded(M_TEST))
    groups_path().write_text(json.dumps(info, indent=1))
    print("limiares de n:", thr, "| instancias de teste por (familia, grupo):", cnt, "| excluidas (soma >= 2^62):", info["excluded"], "| total de teste:", len(test))


def tune(space):
    warm()
    info = json.loads(groups_path().read_text())
    cal = load_set(M_CAL)
    A = len(arm_table(space, 100)[0])
    rows = []
    for rec in cal:
        lb, a0, od = prepare_inst(rec)
        fam, g = info["calib"][rec["member"]]
        base = dict(space=space, cls=f"{fam}|{g}|{rec['member']}", rep=rec["inst_id"], seed_i=0, lb=lb, T=1000)
        for a in range(A):
            mk, ev, _ = run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "tune", fam, rec["inst_id"], f"arm{a}", 0))
            rows.append(dict(base, method=f"arm{a}", makespan=mk, evals=ev))
        for k, grid in GRID.items():
            for (x, y) in grid:
                mk, ev, _ = run(space, rec, lb, a0, od, AGENT[k], 0, x, y, seed_of(space, "tune", fam, rec["inst_id"], f"{k}@{x},{y}", 0))
                rows.append(dict(base, method=f"{k}@{x},{y}", makespan=mk, evals=ev))
    write(R / f"pub_tune_{space}_entrada189.csv", rows, True)
    import pandas as pd
    d = pd.DataFrame(rows)
    d["fam"] = d.cls.str.split("|").str[0]
    d["grp"] = d.cls.str.split("|").str[1].astype(int)
    d["gap"] = 100 * (d.makespan - d.lb) / d.lb
    ch = {}
    for fam in FAMS:
        df = d[d.fam == fam]
        hp = {}
        for k in GRID:
            sub = df[df.method.str.startswith(k + "@")].groupby("method").gap.mean()
            hp[k] = [float(x) for x in sub.idxmin().split("@")[1].split(",")]
        val = {}
        for g in range(3):
            sub = df[(df.grp == g) & df.method.str.startswith("arm")].groupby("method").gap.mean()
            val[str(g)] = int(sub.idxmin()[3:])
        ch[fam] = dict(hp=hp, val=val)
    choice_path(space).write_text(json.dumps(ch, indent=1))
    print(space, "ajuste concluido:", json.dumps({f: ch[f]["val"] for f in FAMS}))


def calibrate():
    info = json.loads(groups_path().read_text())
    cal = load_set(M_CAL)
    chs = {sp: json.loads(choice_path(sp).read_text()) for sp in ("S5", "S15")}
    warm()
    res = {}
    for sp in ("S5", "S15"):
        res[sp] = {}
        for fam in FAMS:
            for g in range(3):
                mem = [r for r in cal if r["family"] == fam and info["calib"][r["member"]][1] == g]
                med = float(np.median([r["n"] for r in mem]))
                pick = sorted(mem, key=lambda r: (abs(r["n"] - med), r["inst_id"]))[:2]
                ratios = {k: [] for k in LEARNERS}
                for rec in pick:
                    lb, a0, od = prepare_inst(rec)

                    def cpu(agent, x=0.1, y=0.1):
                        reps, t0 = 0, time.process_time()
                        while time.process_time() - t0 < 1.0:
                            run(sp, rec, lb, a0, od, agent, 0, x, y, reps + 7)
                            reps += 1
                        return (time.process_time() - t0) / reps
                    base = cpu(H.AGENT_RANDOM)
                    for k in LEARNERS:
                        x, y = chs[sp][fam]["hp"].get(k, [0.1, 0.1])
                        ratios[k].append(cpu(AGENT[k], x, y) / base)
                res[sp][f"{fam}|{g}"] = {k: float(np.median(v)) for k, v in ratios.items()}
                print(f"[pub-calib {sp}] {fam} grupo {g} (n ~ {med:.0f}): " + ", ".join(f"{k} {res[sp][f'{fam}|{g}'][k]:.2f}" for k in LEARNERS), flush=True)
    k_path().write_text(json.dumps(res, indent=1))


def run_space(space, shard, nshards):
    info = json.loads(groups_path().read_text())
    ch = json.loads(choice_path(space).read_text())
    K = json.loads(k_path().read_text())[space]
    test = load_set(M_TEST)[shard::nshards]
    out = R / f"pub_{space}_s{shard}of{nshards}_entrada189.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arm_table(space, 100)[0])
    t00 = time.perf_counter()
    new = not out.exists()
    for i, rec in enumerate(test):
        fam, g = info["test"][rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = prepare_inst(rec)
        rows = []

        def add(method, s, mk, ev, TT):
            rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=method, seed_i=s, makespan=mk, lb=lb, T=TT, evals=ev))
        for s in range(N_SEED):
            mk, ev, _ = run(space, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "test", fam, rec["inst_id"], "rand", s))
            add("rand", s, mk, ev, 1000)
            for j, k in enumerate(LEARNERS):
                x, y = ch[fam]["hp"].get(k, [0.1, 0.1])
                mk, ev, _ = run(space, rec, lb, a0, od, AGENT[k], 0, x, y, seed_of(space, "test", fam, rec["inst_id"], k, s))
                add(k, s, mk, ev, 1000)
                Teq = max(50, int(round(1000 * K[f"{fam}|{g}"][k])))
                mk, ev, _ = run(space, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, "test", fam, rec["inst_id"], "rand_eq" + k, s), T=Teq)
                add(f"rand_eq_{k}", s, mk, ev, Teq)
            mk, ev, _ = run(space, rec, lb, a0, od, H.AGENT_FIXED, ch[fam]["val"][str(g)], 0.1, 0.1, seed_of(space, "test", fam, rec["inst_id"], "val", s))
            add("val", s, mk, ev, 1000)
        for a in range(A):
            mk, ev, _ = run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, "test", fam, rec["inst_id"], f"arm{a}", 0))
            add(f"arm{a}", 0, mk, ev, 1000)
        write(out, rows, new)
        new = False
        if i % 100 == 0:
            print(f"[pub {space} shard {shard}] {i}/{len(test)} ({time.perf_counter() - t00:.0f}s)", flush=True)
    print(f"[pub {space} shard {shard}] concluido ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("prepare", "calibrate"):
        ap.add_argument("--" + f, action="store_true")
    for f in ("tune", "run"):
        ap.add_argument("--" + f, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.tune:
        tune(a.tune)
    if a.calibrate:
        calibrate()
    if a.run:
        run_space(a.run, a.shard, a.nshards)
