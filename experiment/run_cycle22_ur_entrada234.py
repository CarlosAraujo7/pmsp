"""
experiment/run_cycle22_ur_entrada234.py
=========================================
Entrada 234 (Ciclo 22, frente A): um HOSPEDEIRO NOVO, o IG de MAQUINAS NAO RELACIONADAS (R||Cmax; pmsp/ur_numba.py, pmsp/ur_numba_agents.py, pmsp/instance_ur.py), para testar FORA DA AMOSTRA a regra da folga (h* = 0,0475 pontos, congelado nas Entradas 209/210) e o resto das conclusoes do artigo em
uma estrutura de decisao diferente (o custo de por uma tarefa depende da maquina) e, possivelmente, com mais folga que o P||Cmax sintetico (a critica dos revisores cegos #4 a #7: o teste fica perto do teto).
Desenho (o mesmo dos P||Cmax sinteticos das Entradas 180/209, sem calibracao de CPU: so' iteracoes iguais): 12 classes (n, m) = as do P||Cmax (20x4 ... 300x20), tempos P[j, k] ~ U[1, 100] independentes; ajuste: 30 instancias por classe (sementes 30.000.000 + 1000 ci + rep), teste: 100 por classe (sementes 20.000.000 + 1000 ci + rep);
partida gulosa (pmsp.instance_ur.constructive_ur), T = 1000, aceita se nao piora. Espacos S5 (tamanho d/n em {0,05; 0,10; 0,20; 0,30; 0,40}, destruicao aleatoria) e S15 (estrategia x tamanho), como no P||Cmax (experiment.run_agents_numba_entrada180.arm_table).
Metodos por instancia de teste: rand, ql, ucb, pg, alns (5 sementes cada; hiperparametros CONGELADOS do P||Cmax sintetico: lr do pg = 0,03 em S5 e 0,3 em S15 (results/agents_choice_*), roleta ALNS (0,1; 0,0) (results/aos_choice_*), Q-Learning padrao (alfa 0,1, gama 0,9, epsilon 1 -> 0,05 por 0,995)),
val (a acao fixa validada por classe nas instancias de ajuste, 5 sementes) e cada acao fixa (4 sementes: 0, 1, 2, 3, para o oraculo com validacao cruzada de sementes: a acao e' escolhida nas {0, 1} e avaliada nas {2, 3}).
Mais o Q-Learning OTIMISTA (qo; Entrada 223: tabela inicial Q0 = c, epsilon constante 0,05, recompensa invariante a escala; (c, alfa) em {0,02; 0,1; 0,5} x {0,1; 0,3} ajustado nas instancias de ajuste, uma escolha por espaco), a reparacao mais favoravel ao aprendiz online.
Fases (retomaveis): --tune ESPACO | --tuneqo ESPACO | --run ESPACO --shard i --nshards n [--pilot: so' 3 instancias de AJUSTE por classe, saida results/ur_pilot_*, para depurar o codigo SEM tocar nas instancias de teste].
Saidas: results/ur_choice_ESPACO_entrada234.json, results/ur_test_ESPACO_s*of*_entrada234.csv. Pre-registro: analysis/report_entrada234_ur.py (docstring; hashes em results/entrada234_prereg_hashes.txt).
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

from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402
from pmsp import ur_numba_agents5 as UA5  # noqa: E402
from pmsp.instance_ur import constructive_ur, lower_bound_ur, random_ur  # noqa: E402

R = ROOT / "results"
N_TUNE, N_TEST, N_SEED_ARM = 30, 100, 4
SPACES = ["S5", "S15"]
MID = {"rand": 1, "ql": 2, "ucb": 3, "pg": 4, "alns": 5, "val": 6, "arm": 7, "tune": 8, "qo": 9, "tuneqo": 10}
GRID_QO = [(c, a) for c in (0.02, 0.1, 0.5) for a in (0.1, 0.3)]
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
PG_LR = {"S5": 0.03, "S15": 0.3}
ALNS_HP = (0.1, 0.0)
AGENTS = {"rand": (UA.AGENT_RANDOM, None), "ql": (UA.AGENT_Q, None), "ucb": (UA.AGENT_UCB, None), "pg": (UA.AGENT_PG, "pg"), "alns": (UA.AGENT_ALNS, "alns")}


def choice_path(space):
    return R / f"ur_choice_{space}_entrada234.json"


def make(n, m, rep, ci, tune):
    seed = (30_000_000 if tune else 20_000_000) + 1000 * ci + rep
    P = random_ur(n, m, seed)
    a0, od = constructive_ur(P)
    return P, a0, od, lower_bound_ur(P)


def seed_of(space, ci, rep, method, s):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * MID[method] + 20_000_000 * (SPACES.index(space) + 1)


def run(space, P, a0, m, od, lb, n, agent, fixed, lr, lr2, seed, T=1000):
    st, d, ls = arm_table(space, n)
    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
    return int(best), float(ev)


def choice_qo_path(space):
    return R / f"ur_choice_qo_{space}_entrada234.json"


def run_qo(space, P, a0, m, od, lb, n, cfg, seed, T=1000):
    """Q-Learning OTIMISTA (Entrada 223): tabela inicial Q0 = c, epsilon constante 0,05, recompensa invariante a escala; cfg = (c, alfa)"""
    c, a = cfg
    st, d, ls = arm_table(space, n)
    A = len(st)
    hp = np.array([a, 0.9, 1.0, 0.05, 0.0])
    out = UA5.ur_agent_core5(P, a0, m, st, d, ls, T, 2, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, hp, 2, np.full((81, A), c), 0.05)
    return int(out[0]), float(out[2])


def tune_qo(space):
    warm()
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            P, a0, od, lb = make(n, m, rep, ci, True)
            for gi, cfg in enumerate(GRID_QO):
                for s in range(2):
                    mk, ev = run_qo(space, P, a0, m, od, lb, n, cfg, seed_of(space, ci, rep, "tuneqo", s) + 11 * gi)
                    rows.append((gi, 100.0 * (mk - lb) / lb))
    import pandas as pd
    d = pd.DataFrame(rows, columns=["g", "gap"])
    g = d.groupby("g").gap.mean()
    best = int(g.idxmin())
    choice_qo_path(space).write_text(json.dumps(dict(cfg=list(GRID_QO[best]), index=best, gaps={str(GRID_QO[int(k)]): float(v) for k, v in g.items()}), indent=1))
    print(space, "ajuste do Q otimista concluido:", GRID_QO[best], {str(GRID_QO[int(k)]): round(float(v), 3) for k, v in g.items()})


def warm():
    P, a0, od, lb = make(20, 4, 0, 0, True)
    for sp in SPACES:
        for ag in (0, 1, 2, 3, 4, 7):
            run(sp, P, a0, 4, od, lb, 20, ag, 0, 0.1, 0.1, 1, T=20)


def tune(space):
    warm()
    A = len(arm_table(space, 100)[0])
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            P, a0, od, lb = make(n, m, rep, ci, True)
            for a in range(A):
                for s in range(2):
                    mk, ev = run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, ci, rep, "tune", s) + 7 * a)
                    rows.append((f"rand_{n}j_{m}m", a, mk))
    import pandas as pd
    d = pd.DataFrame(rows, columns=["cls", "arm", "makespan"])
    val = {c: int(g.groupby("arm").makespan.mean().idxmin()) for c, g in d.groupby("cls")}
    choice_path(space).write_text(json.dumps(dict(val=val), indent=1))
    print(space, "ajuste concluido; acoes validadas:", val)


def run_space(space, shard, nshards, pilot):
    ch = json.loads(choice_path(space).read_text())["val"]
    qo_cfg = tuple(json.loads(choice_qo_path(space).read_text())["cfg"])
    out = R / (f"ur_pilot_{space}_entrada234.csv" if pilot else f"ur_test_{space}_s{shard}of{nshards}_entrada234.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    warm()
    A = len(arm_table(space, 100)[0])
    new = not out.exists()
    t0 = time.perf_counter()
    k = 0
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(3 if pilot else N_TEST):
            if (not pilot) and (ci * N_TEST + rep) % nshards != shard:
                continue
            if (cls, str(rep)) in done:
                continue
            P, a0, od, lb = make(n, m, rep, ci, bool(pilot))
            rows = []

            def add(method, s, mk, ev):
                rows.append(dict(space=space, cls=cls, rep=rep, method=method, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
            for name, (ag, key) in AGENTS.items():
                lr, lr2 = (PG_LR[space], 0.1) if key == "pg" else (ALNS_HP if key == "alns" else (0.1, 0.1))
                for s in range(5):
                    mk, ev = run(space, P, a0, m, od, lb, n, ag, 0, lr, lr2, seed_of(space, ci, rep, name, s))
                    add(name, s, mk, ev)
            for s in range(5):
                mk, ev = run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, ch[cls], 0.1, 0.1, seed_of(space, ci, rep, "val", s))
                add("val", s, mk, ev)
            for s in range(5):
                mk, ev = run_qo(space, P, a0, m, od, lb, n, qo_cfg, seed_of(space, ci, rep, "qo", s))
                add("qo", s, mk, ev)
            for a in range(A):
                for s in range(N_SEED_ARM):
                    mk, ev = run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, a, 0.1, 0.1, seed_of(space, ci, rep, "arm", s) + 7 * a)
                    add(f"arm{a}", s, mk, ev)
            with open(out, "w" if new else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            new = False
            k += 1
            if k % 100 == 0:
                print(f"[ur {space} s{shard}] {k} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[ur {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default=None)
    ap.add_argument("--tuneqo", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.tune:
        tune(a.tune)
    if a.tuneqo:
        tune_qo(a.tuneqo)
    if a.run:
        run_space(a.run, a.shard, a.nshards, a.pilot)
