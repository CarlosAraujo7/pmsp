"""
experiment/run_cycle16_entrada185.py
======================================
Entrada 185 (Ciclo 16, frentes A2, B, C e D do plano da Entrada 183), todas com os agentes COMPILADOS de pmsp/ig_numba_agents2.py (P||Cmax, partida LPT, aceita se nao piora, estado de 81 bins, recompensa max(melhora, 0)).
  aos    (B)  seletores classicos de AOS sem estado -- probability matching (PM), adaptive pursuit (AP) e roleta tipo ALNS -- nos espacos S5 (5 tamanhos de destruicao) e S15 (3 estrategias x 5 tamanhos), T = 1000, 12 classes x 100 instancias de teste
              (as MESMAS de results/agents_S5|S15_entrada180.csv, cujas linhas rand, val, ql, ucb e pg sao reaproveitadas na analise) x 5 sementes; hiperparametros ajustados em 30 instancias por classe (grade: PM alfa {0,1; 0,3; 0,8}; AP (alfa, beta) em {0,1; 0,3; 0,8}^2;
              ALNS rho {0,1; 0,3; 0,5}; escolhe o de menor lacuna media ao limite inferior); K_CPU de cada um sobre o aleatorio calibrado em sequencia, maquina em repouso, para a comparacao a CPU igual.
  plant  (A2) CONTROLE POSITIVO: um bonus de recompensa plantado numa acao (mode 1: a mesma acao em todo estado; mode 2: depende do estado, s mod A) e bonus em {0,5; 1; 2; 5} unidades de makespan; mede a fracao de escolhas da acao plantada na segunda metade de T = 1000
              (aleatorio: 1/A) para os seis aprendizes (ql, ucb, pg, pm, ap, alns), S5 e S15, 12 classes x 10 instancias x 3 sementes.
  long   (C)  horizonte longo: S5, T em {5000, 20000, 100000}, 12 classes x 100 instancias de teste x 5 sementes (3 em T = 100000); aleatorio, ql, ucb, pg (lr ajustado em T), acao validada por classe (ajustada em T, 30 instancias) e todas as acoes fixas (oraculo, 1 semente).
  fine   (D)  tempos finos U[1, 10^4] (mais folga que U[1, 100]): S5 e S15, T = 1000, 12 classes x 100 instancias de teste (sementes de instancia BASE_SEED + rep) x 5 sementes; ql, ucb, pg, validada, oraculo; K_CPU calibrado nos tempos finos para a CPU igual.
Fases (retomaveis): --aos-tune S | --aos-calibrate | --aos-run S | --plant | --long-tune T | --long-run T | --fine-tune S | --fine-calibrate | --fine-run S | --verify
Pre-registro: analysis/report_entrada185_cycle16.py (docstring, congelado antes de rodar; hashes em results/entrada185_prereg_hashes.txt).
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

from experiment import config as CFG  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from pmsp import ig_numba_agents2 as H  # noqa: E402
from pmsp.constructive import lpt  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

R = ROOT / "results"
N_TUNE, N_TEST, TUNE_SEED0 = 30, 100, 90000
AOS = {"pm": H.AGENT_PM, "ap": H.AGENT_AP, "alns": H.AGENT_ALNS}
OLD = {"ql": H.AGENT_Q, "ucb": H.AGENT_UCB, "pg": H.AGENT_PG}
GRID = {"pm": [(a, 0.0) for a in (0.1, 0.3, 0.8)], "ap": [(a, b) for a in (0.1, 0.3, 0.8) for b in (0.1, 0.3, 0.8)], "alns": [(r, 0.0) for r in (0.1, 0.3, 0.5)]}
LRS = [0.03, 0.1, 0.3]
MID = {"rand": 1, "ql": 2, "ucb": 3, "pg": 4, "val": 5, "rand_eq": 6, "arm": 9, "tune": 10, "pm": 11, "ap": 12, "alns": 13, "plant": 14}
BATT = {"aos": 1, "plant": 2, "long": 3, "fine": 4}
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]


def make(n, m, rep, tune, high=100):
    inst = PMSPInstance.random(n=n, m=m, low=1, high=high, seed=(TUNE_SEED0 + rep) if tune else (CFG.BASE_SEED + rep), name=f"rand_{n}j_{m}m")
    return inst, np.asarray(inst.p, dtype=np.int64), lpt(inst).assignment.astype(np.int64), np.argsort(inst.p)[::-1].astype(np.int64)


def seed_of(batt, space, ci, rep, method, s, T=1000):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * MID[method] + 20_000_000 * (BATT[batt] * 4 + ["S3", "S5", "S15", "S30"].index(space)) + (T // 1000) * 7


def run(space, p, a0, m, od, lb, n, agent, fixed, lr, lr2, seed, T=1000, plant=(0, 0, 0.0)):
    st, d, ls = arm_table(space, n)
    best, _, ev, used, hit2 = H.ig_agent_core2(p, a0, m, st, d, ls, T, agent, fixed, lr, lr2, seed, od, lb, plant[0], plant[1], plant[2])
    return int(best), float(ev), used, int(hit2)


def warm():
    inst, p, a0, od = make(20, 4, 0, False)
    for sp in ("S5", "S15"):
        for ag in range(8):
            run(sp, p, a0, 4, od, inst.lower_bound, 20, ag, 0, 0.1, 0.1, 1, T=20, plant=(1, 0, 1.0))


def write(out, rows, new):
    with open(out, "a" if not new else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def mean_gap(rows):
    g = {}
    for r in rows:
        g.setdefault(r["method"], []).append(100.0 * (r["makespan"] - r["lb"]) / r["lb"])
    return {k: float(np.mean(v)) for k, v in g.items()}


# ----------------------------------------------------------------------------------------------------- verificacao
def verify():
    """As instancias de teste de agents_S5_entrada180.csv sao reproduzidas (mesmas semente e distribuicao): confere o lb de todas as linhas e a equivalencia exata dos agentes 0-4 com ig_agent_core."""
    import pandas as pd

    from pmsp import ig_numba_agents as G
    d = pd.read_csv(R / "agents_S5_entrada180.csv")
    d = d[d.method == "rand"].drop_duplicates(["cls", "rep"])
    bad = 0
    for _, r in d.iterrows():
        n, m = int(r.cls.split("_")[1][:-1]), int(r.cls.split("_")[2][:-1])
        inst, p, a0, od = make(n, m, int(r.rep), False)
        bad += int(inst.lower_bound != r.lb)
    warm()
    inst, p, a0, od = make(60, 6, 3, False)
    st, dd, ls = arm_table("S5", 60)
    for ag in (0, 1, 2, 3, 4):
        a = G.ig_agent_core(p, a0, 6, st, dd, ls, 500, ag, 1, 0.1, 5, od, inst.lower_bound)
        b = H.ig_agent_core2(p, a0, 6, st, dd, ls, 500, ag, 1, 0.1, 0.1, 5, od, inst.lower_bound, 0, 0, 0.0)
        bad += int(a[0] != b[0] or a[2] != b[2] or not np.array_equal(a[3], b[3]))
    print("verify: instancias e agentes 0-4 conferem" if bad == 0 else f"verify: {bad} divergencias")
    return bad


# ----------------------------------------------------------------------------------------------------- B: AOS classicos
def aos_choice_path(space):
    return R / f"aos_choice_{space}_entrada185.json"


def aos_k_path():
    return R / "aos_cpu_ratio_entrada185.json"


def aos_tune(space):
    warm()
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            inst, p, a0, od = make(n, m, rep, True)
            for k, ag in AOS.items():
                for (a, b) in GRID[k]:
                    mk, ev, _, _ = run(space, p, a0, m, od, inst.lower_bound, n, ag, 0, a, b, seed_of("aos", space, ci, rep, "tune", 0))
                    rows.append(dict(space=space, cls=f"rand_{n}j_{m}m", rep=rep, method=f"{k}@{a},{b}", seed_i=0, makespan=mk, lb=inst.lower_bound, T=1000, evals=ev))
    write(R / f"aos_tune_{space}_entrada185.csv", rows, True)
    g = mean_gap(rows)
    ch = {}
    for k in AOS:
        best = min((key for key in g if key.startswith(k + "@")), key=lambda key: g[key])
        a, b = (float(x) for x in best.split("@")[1].split(","))
        ch[k] = [a, b]
    ch["gaps"] = g
    aos_choice_path(space).write_text(json.dumps(ch, indent=1))
    print(space, "ajuste AOS:", {k: ch[k] for k in AOS})


def aos_calibrate():
    ch = {sp: json.loads(aos_choice_path(sp).read_text()) for sp in ("S5", "S15")}
    warm()
    res = {}
    for sp in ("S5", "S15"):
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            ratios = {k: [] for k in AOS}
            for rep in range(2):
                inst, p, a0, od = make(n, m, rep, False)

                def cpu(agent, a=0.1, b=0.1):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run(sp, p, a0, m, od, inst.lower_bound, n, agent, 0, a, b, reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(H.AGENT_RANDOM)
                for k, ag in AOS.items():
                    ratios[k].append(cpu(ag, *ch[sp][k]) / base)
            res[sp][f"{n}x{m}"] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[aos-calib {sp}] {n}x{m}: {res[sp][f'{n}x{m}']}", flush=True)
    aos_k_path().write_text(json.dumps(res, indent=1))


def aos_run(space):
    ch = json.loads(aos_choice_path(space).read_text())
    K = json.loads(aos_k_path().read_text())[space]
    out = R / f"aos_{space}_entrada185.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    t00 = time.perf_counter()
    new = not out.exists()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(N_TEST):
            if (cls, str(rep)) in done:
                continue
            inst, p, a0, od = make(n, m, rep, False)
            lb = inst.lower_bound
            rows = []
            for s in range(5):
                for k, ag in AOS.items():
                    mk, ev, _, _ = run(space, p, a0, m, od, lb, n, ag, 0, ch[k][0], ch[k][1], seed_of("aos", space, ci, rep, k, s))
                    rows.append(dict(space=space, cls=cls, rep=rep, method=k, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
                    Teq = max(50, int(round(1000 * K[f"{n}x{m}"][k])))
                    mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of("aos", space, ci, rep, "rand_eq", s) + 3 * list(AOS).index(k), T=Teq)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"rand_eq_{k}", seed_i=s, makespan=mk, lb=lb, T=Teq, evals=ev))
            write(out, rows, new)
            new = False
        print(f"[aos {space}] {n}x{m} ok ({time.perf_counter() - t00:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------------------------- A2: controle positivo
def plant():
    chs = {sp: json.loads(aos_choice_path(sp).read_text()) for sp in ("S5", "S15")}
    pg_lr = {sp: json.loads((R / f"agents_choice_{sp}_entrada180.json").read_text())["lr"] for sp in ("S5", "S15")}
    out = R / "plant_entrada185.csv"
    warm()
    rows = []
    learners = {"ql": (H.AGENT_Q, None), "ucb": (H.AGENT_UCB, None), "pg": (H.AGENT_PG, "pg"), "pm": (H.AGENT_PM, "pm"), "ap": (H.AGENT_AP, "ap"), "alns": (H.AGENT_ALNS, "alns")}
    T = 1000
    for sp in ("S5", "S15"):
        A = len(arm_table(sp, 100)[0])
        for name, (ag, key) in learners.items():
            a, b = (0.1, 0.1)
            if name == "pg":
                a = pg_lr[sp]
            elif key in AOS:
                a, b = chs[sp][key]
            for mode in (1, 2):
                for bonus in (0.5, 1.0, 2.0, 5.0):
                    for ci, (n, m) in enumerate(CLASSES):
                        for rep in range(10):
                            inst, p, a0, od = make(n, m, rep, False)
                            for s in range(3):
                                _, _, _, h2 = run(sp, p, a0, m, od, inst.lower_bound, n, ag, 0, a, b, seed_of("plant", sp, ci, rep, "plant", s) + 11 * mode + 101 * int(bonus * 2), T=T, plant=(mode, A // 2, bonus))
                                rows.append(dict(space=sp, learner=name, mode=mode, bonus=bonus, cls=f"rand_{n}j_{m}m", rep=rep, seed_i=s, hit2=h2, T=T, A=A))
            print(f"[plant {sp}] {name} ok", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["space", "learner", "mode", "bonus", "cls", "rep", "seed_i", "hit2", "T", "A"])
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------------------------------- C e D: genericos (horizonte longo; tempos finos)
def tag_of(batt, space, T):
    return f"{batt}_{space}" + (f"_T{T}" if batt == "long" else "")


def generic_choice_path(batt, space, T):
    return R / f"{tag_of(batt, space, T)}_choice_entrada185.json"


def generic_tune(batt, space, T, high):
    warm()
    A = len(arm_table(space, 100)[0])
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            inst, p, a0, od = make(n, m, rep, True, high)
            lb = inst.lower_bound
            for a in range(A):
                mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_FIXED, a, 0.1, 0.1, seed_of(batt, space, ci, rep, "arm", 0, T), T=T)
                rows.append(dict(space=space, cls=f"rand_{n}j_{m}m", rep=rep, method=f"arm{a}", seed_i=0, makespan=mk, lb=lb, T=T, evals=ev))
            for lr in LRS:
                mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_PG, 0, lr, 0.1, seed_of(batt, space, ci, rep, "tune", 0, T), T=T)
                rows.append(dict(space=space, cls=f"rand_{n}j_{m}m", rep=rep, method=f"pg@{lr}", seed_i=0, makespan=mk, lb=lb, T=T, evals=ev))
        print(f"[{tag_of(batt, space, T)} tune] {n}x{m} ok", flush=True)
    write(R / f"{tag_of(batt, space, T)}_tune_entrada185.csv", rows, True)
    arms = {}
    for r in rows:
        if r["method"].startswith("arm"):
            arms.setdefault((r["cls"], int(r["method"][3:])), []).append(r["makespan"])
    val = {}
    for (c, a), v in arms.items():
        val.setdefault(c, {})[a] = float(np.mean(v))
    pg = {k: v for k, v in mean_gap([r for r in rows if r["method"].startswith("pg@")]).items()}
    ch = dict(val={c: int(min(d, key=d.get)) for c, d in val.items()}, lr=float(min(pg, key=pg.get).split("@")[1]), lr_gap=pg)
    generic_choice_path(batt, space, T).write_text(json.dumps(ch, indent=1))
    print(tag_of(batt, space, T), "ajuste concluido:", ch["lr_gap"])


def fine_calibrate(high=10000):
    ch = {sp: json.loads(generic_choice_path("fine", sp, 1000).read_text()) for sp in ("S5", "S15")}
    warm()
    res = {}
    for sp in ("S5", "S15"):
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            ratios = {k: [] for k in OLD}
            for rep in range(2):
                inst, p, a0, od = make(n, m, rep, False, high)

                def cpu(agent, lr=0.1):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run(sp, p, a0, m, od, inst.lower_bound, n, agent, 0, lr, 0.1, reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(H.AGENT_RANDOM)
                for k, ag in OLD.items():
                    ratios[k].append(cpu(ag, ch[sp]["lr"]) / base)
            res[sp][f"{n}x{m}"] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[fine-calib {sp}] {n}x{m}: {res[sp][f'{n}x{m}']}", flush=True)
    (R / "fine_cpu_ratio_entrada185.json").write_text(json.dumps(res, indent=1))


def generic_run(batt, space, T, high, nseed):
    ch = json.loads(generic_choice_path(batt, space, T).read_text())
    K = json.loads((R / "fine_cpu_ratio_entrada185.json").read_text())[space] if batt == "fine" else None
    out = R / f"{tag_of(batt, space, T)}_entrada185.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arm_table(space, 100)[0])
    t00 = time.perf_counter()
    new = not out.exists()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(N_TEST):
            if (cls, str(rep)) in done:
                continue
            inst, p, a0, od = make(n, m, rep, False, high)
            lb = inst.lower_bound
            rows = []

            def add(method, s, mk, ev, TT):
                rows.append(dict(space=space, cls=cls, rep=rep, method=method, seed_i=s, makespan=mk, lb=lb, T=TT, evals=ev))
            for s in range(nseed):
                mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(batt, space, ci, rep, "rand", s, T), T=T)
                add("rand", s, mk, ev, T)
                for k, ag in OLD.items():
                    mk, ev, _, _ = run(space, p, a0, m, od, lb, n, ag, 0, ch["lr"], 0.1, seed_of(batt, space, ci, rep, k, s, T), T=T)
                    add(k, s, mk, ev, T)
                    if K is not None:
                        Teq = max(50, int(round(T * K[f"{n}x{m}"][k])))
                        mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(batt, space, ci, rep, "rand_eq", s, T) + 3 * list(OLD).index(k), T=Teq)
                        add(f"rand_eq_{k}", s, mk, ev, Teq)
                mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_FIXED, ch["val"][cls], 0.1, 0.1, seed_of(batt, space, ci, rep, "val", s, T), T=T)
                add("val", s, mk, ev, T)
            for a in range(A):
                mk, ev, _, _ = run(space, p, a0, m, od, lb, n, H.AGENT_FIXED, a, 0.1, 0.1, seed_of(batt, space, ci, rep, "arm", 0, T), T=T)
                add(f"arm{a}", 0, mk, ev, T)
            write(out, rows, new)
            new = False
        print(f"[{tag_of(batt, space, T)}] {n}x{m} ok ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("verify", "aos-calibrate", "plant", "fine-calibrate"):
        ap.add_argument("--" + f, action="store_true")
    for f in ("aos-tune", "aos-run", "long-tune", "long-run", "fine-tune", "fine-run"):
        ap.add_argument("--" + f, default=None)
    a = vars(ap.parse_args())
    if a["verify"]:
        sys.exit(1 if verify() else 0)
    if a["aos_tune"]:
        aos_tune(a["aos_tune"])
    if a["aos_calibrate"]:
        aos_calibrate()
    if a["aos_run"]:
        aos_run(a["aos_run"])
    if a["plant"]:
        plant()
    if a["long_tune"]:
        generic_tune("long", "S5", int(a["long_tune"]), 100)
    if a["long_run"]:
        T = int(a["long_run"])
        generic_run("long", "S5", T, 100, 3 if T >= 100000 else 5)
    if a["fine_tune"]:
        generic_tune("fine", a["fine_tune"], 1000, 10000)
    if a["fine_calibrate"]:
        fine_calibrate()
    if a["fine_run"]:
        generic_run("fine", a["fine_run"], 1000, 10000, 5)
