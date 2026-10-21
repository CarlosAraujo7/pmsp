"""
experiment/run_agents_numba_entrada180.py
===========================================
Entrada 180 (Ciclo 15, frentes L e M): AGENTES COMPILADOS e REPETICOES EM MASSA. Limites do artigo: (a) o custo do agente (1,4 a 1,7 vezes o do aleatorio em maquinas nao relacionadas; 3,3 vezes o de um DQN no ILS eficiente) foi medido com agentes Python;
(b) os espacos de acao testados tem no maximo 15 acoes e as repeticoes por instancia sao poucas para efeitos de centesimos de ponto; (c) falta um aprendiz de gradiente de politica. pmsp/ig_numba_agents.py compila o IG com os agentes aleatorio, Q-Learning tabular,
UCB1 normalizado e gradiente de politica (softmax por estado, linha de base), e uma tabela de acoes (estrategia, tamanho, com ou sem busca local).
Espacos (P||Cmax, 12 classes x 100 instancias de teste; ajuste: 30 por classe, seeds 90000+; partida LPT, T = 1000, aceita se nao piora):
  S3  3 estrategias de destruicao, d = 0,3 n                                  S5  tamanho d/n in {0,05; 0,10; 0,20; 0,30; 0,40}, estrategia aleatoria uniforme (0)
  S15 estrategia x tamanho (3 x 5), indice = estrategia*5 + tamanho            S30 S15 x {sem, com busca local} (as acoes com busca local custam mais CPU), indice = ls*15 + estrategia*5 + tamanho
Metodos por instancia (5 sementes cada, exceto o oraculo): rand (aleatorio uniforme), ql, ucb, pg (lr in {0,03; 0,1; 0,3} ajustado nas instancias de ajuste), val (melhor acao FIXA por classe no ajuste), rand_eq_<aprendiz> (aleatorio com T_eq = round(1000 K), K = razao
de CPU do aprendiz sobre o aleatorio, calibrada em sequencia por classe, maquina em repouso) e o oraculo (todas as acoes fixas, 1 semente).
Fases: --verify (equivalencia com os resultados Python armazenados) ; --tune ESPACO ; --calibrate ; --run ESPACO. Retomavel. Uso: python -m experiment.run_agents_numba_entrada180 --tune S5 ; --calibrate ; --run S5 ; --verify
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
from pmsp import ig_numba_agents as G  # noqa: E402
from pmsp.constructive import lpt  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

R = ROOT / "results"
CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
FR = [0.05, 0.10, 0.20, 0.30, 0.40]
SPACES = ["S3", "S5", "S15", "S30"]
LRS = [0.03, 0.1, 0.3]
N_TUNE, N_TEST, N_SEED, TUNE_SEED0 = 30, 100, 5, 90000
LEARNERS = {"ql": G.AGENT_Q, "ucb": G.AGENT_UCB, "pg": G.AGENT_PG}
MID = {"rand": 1, "ql": 2, "ucb": 3, "pg": 4, "val": 5, "rand_eq_ql": 6, "rand_eq_ucb": 7, "rand_eq_pg": 8, "arm": 9, "tune_pg": 10}
def choice_path(space):
    return R / f"agents_choice_{space}_entrada180.json"


KFILE = R / "agents_cpu_ratio_entrada180.json"
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals", "frac_ls"]


def arm_table(space, n):
    d = lambda f: max(2, round(f * n))
    if space == "S3":
        return np.array([0, 1, 2], dtype=np.int64), np.full(3, d(0.3), dtype=np.int64), np.zeros(3, dtype=np.int64)
    if space == "S5":
        return np.zeros(5, dtype=np.int64), np.array([d(f) for f in FR], dtype=np.int64), np.zeros(5, dtype=np.int64)
    s15 = np.array([s for s in range(3) for _ in FR], dtype=np.int64)
    d15 = np.array([d(f) for _ in range(3) for f in FR], dtype=np.int64)
    if space == "S15":
        return s15, d15, np.zeros(15, dtype=np.int64)
    return np.concatenate([s15, s15]), np.concatenate([d15, d15]), np.concatenate([np.zeros(15, dtype=np.int64), np.ones(15, dtype=np.int64)])


def make(n, m, rep, tune):
    inst = PMSPInstance.random(n=n, m=m, seed=(TUNE_SEED0 + rep) if tune else (CFG.BASE_SEED + rep), name=f"rand_{n}j_{m}m")
    p = np.asarray(inst.p, dtype=np.int64)
    return inst, p, lpt(inst).assignment.astype(np.int64), np.argsort(inst.p)[::-1].astype(np.int64)


def seed_of(space, ci, rep, method, s):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * MID[method] + 20_000_000 * SPACES.index(space)


def run(space, p, a0, m, od, lb, n, agent, fixed, lr, seed, T=1000):
    st, d, ls = arm_table(space, n)
    best, _, ev, used = G.ig_agent_core(p, a0, m, st, d, ls, T, agent, fixed, lr, seed, od, lb)
    frac_ls = float(used[ls == 1].sum() / used.sum()) if ls.any() else 0.0
    return int(best), float(ev), frac_ls


def warm():
    inst, p, a0, od = make(20, 4, 0, False)
    for sp in SPACES:
        for ag in (0, 1, 2, 3, 4):
            run(sp, p, a0, 4, od, inst.lower_bound, 20, ag, 0, 0.1, 1, T=20)


# ------------------------------------------------------------------------------------------- verificacao
def verify():
    out = R / "agents_verify_entrada180.csv"
    warm()
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TEST):
            inst, p, a0, od = make(n, m, rep, False)
            lb = inst.lower_bound
            for sp, meths in (("S5", ("rand", "ql", "ucb")), ("S3", ("rand", "ql"))):
                for me in meths:
                    ag = {"rand": 1, "ql": 2, "ucb": 3}[me]
                    mk, ev, _ = run(sp, p, a0, m, od, lb, n, ag, 0, 0.1, seed_of(sp, ci, rep, me if me != "ucb" else "ucb", 0))
                    rows.append(dict(space=sp, cls=f"rand_{n}j_{m}m", rep=rep, method=me, seed_i=0, makespan=mk, lb=lb, T=1000, evals=ev, frac_ls=0.0))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("verify ->", out, len(rows))


# ------------------------------------------------------------------------------------------- ajuste
def tune(space):
    out = R / f"agents_tune_{space}_entrada180.csv"
    warm()
    A = len(arm_table(space, 100)[0])
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            inst, p, a0, od = make(n, m, rep, True)
            lb = inst.lower_bound
            for a in range(A):
                mk, ev, _ = run(space, p, a0, m, od, lb, n, G.AGENT_FIXED, a, 0.1, seed_of(space, ci, rep, "arm", 0))
                rows.append(dict(space=space, cls=f"rand_{n}j_{m}m", rep=rep, method=f"arm{a}", seed_i=0, makespan=mk, lb=lb, T=1000, evals=ev, frac_ls=0.0))
            for lr in LRS:
                mk, ev, fl = run(space, p, a0, m, od, lb, n, G.AGENT_PG, 0, lr, seed_of(space, ci, rep, "tune_pg", 0))
                rows.append(dict(space=space, cls=f"rand_{n}j_{m}m", rep=rep, method=f"pg@{lr}", seed_i=0, makespan=mk, lb=lb, T=1000, evals=ev, frac_ls=fl))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    d = pd.read_csv(out)
    ch = {}
    arms = d[d.method.str.startswith("arm")].copy()
    arms["a"] = arms.method.str[3:].astype(int)
    pgd = d[d.method.str.startswith("pg@")].copy()
    pgd["gap"] = 100 * (pgd.makespan - pgd.lb) / pgd.lb
    ch[space] = dict(val={c: int(g.groupby("a").makespan.mean().idxmin()) for c, g in arms.groupby("cls")},
                     lr=float(pgd.groupby("method").gap.mean().idxmin().split("@")[1]), lr_gap={k: float(v) for k, v in pgd.groupby("method").gap.mean().items()})
    choice_path(space).write_text(json.dumps(ch[space], indent=1))
    print(space, "ajuste concluido:", json.dumps(ch[space]["lr_gap"]))


# ------------------------------------------------------------------------------------------- calibracao (sequencial, maquina em repouso)
def calibrate():
    ch = {sp: json.loads(choice_path(sp).read_text()) for sp in SPACES}
    warm()
    res = {}
    for sp in SPACES:
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            ratios = {k: [] for k in LEARNERS}
            for rep in range(2):
                inst, p, a0, od = make(n, m, rep, False)
                lb = inst.lower_bound

                def cpu(agent, lr=0.1):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run(sp, p, a0, m, od, lb, n, agent, 0, lr, reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(G.AGENT_RANDOM)
                for k, ag in LEARNERS.items():
                    ratios[k].append(cpu(ag, ch[sp]["lr"]) / base)
            res[sp][f"{n}x{m}"] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[calib {sp}] {n}x{m}: {res[sp][f'{n}x{m}']}", flush=True)
    KFILE.write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------- teste
def run_space(space):
    ch = json.loads(choice_path(space).read_text())
    K = json.loads(KFILE.read_text())[space]
    out = R / f"agents_{space}_entrada180.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arm_table(space, 100)[0])
    new = not out.exists()
    t00 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for ci, (n, m) in enumerate(CLASSES):
            cls = f"rand_{n}j_{m}m"
            for rep in range(N_TEST):
                if (cls, str(rep)) in done:
                    continue
                inst, p, a0, od = make(n, m, rep, False)
                lb = inst.lower_bound
                rows = []

                def add(method, s, mk, ev, T, fl):
                    rows.append(dict(space=space, cls=cls, rep=rep, method=method, seed_i=s, makespan=mk, lb=lb, T=T, evals=ev, frac_ls=fl))
                for s in range(N_SEED):
                    mk, ev, fl = run(space, p, a0, m, od, lb, n, G.AGENT_RANDOM, 0, 0.1, seed_of(space, ci, rep, "rand", s))
                    add("rand", s, mk, ev, 1000, fl)
                    for k, ag in LEARNERS.items():
                        mk, ev, fl = run(space, p, a0, m, od, lb, n, ag, 0, ch["lr"], seed_of(space, ci, rep, k, s))
                        add(k, s, mk, ev, 1000, fl)
                        Teq = max(50, int(round(1000 * K[f"{n}x{m}"][k])))
                        mk, ev, fl = run(space, p, a0, m, od, lb, n, G.AGENT_RANDOM, 0, 0.1, seed_of(space, ci, rep, f"rand_eq_{k}", s), T=Teq)
                        add(f"rand_eq_{k}", s, mk, ev, Teq, fl)
                    mk, ev, fl = run(space, p, a0, m, od, lb, n, G.AGENT_FIXED, ch["val"][cls], 0.1, seed_of(space, ci, rep, "val", s))
                    add("val", s, mk, ev, 1000, fl)
                for a in range(A):
                    mk, ev, fl = run(space, p, a0, m, od, lb, n, G.AGENT_FIXED, a, 0.1, seed_of(space, ci, rep, "arm", 0))
                    add(f"arm{a}", 0, mk, ev, 1000, fl)
                w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
            print(f"[{space}] {n}x{m} ok ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--tune", default=None)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.verify:
        verify()
    if a.tune:
        tune(a.tune)
    if a.calibrate:
        calibrate()
    if a.run:
        run_space(a.run)
