"""
experiment/run_cycle23_pubaos_entrada242.py
=============================================
Entrada 242 (Ciclo 23, frente A): DOIS PROJETOS DE AOS DA LITERATURA no protocolo (as revisoes cegas #4-#8: 'nenhum AOS/hiper-heuristica estabelecido'): o Q-Learning EPISODICO de estado BINARIO do IG para o flowshop (Karimi-Mamaghan et al. 2023, e a extensao arXiv 2408.14864) -- agente 'iq' -- e o FRRMAB (Fialho, Schoenauer, Sebag, 2010) -- agente 'fr'
(pmsp/ig_numba_agents6.py e pmsp/ur_numba_agents6.py; 17 testes). Os hiperparametros dos artigos nao estao disponiveis por inteiro (o do EJOR 2023 esta' atras de uma barreira de acesso; a extensao nao traz os numeros): sao AJUSTADOS nas instancias de ajuste, por hospedeiro, espaco e agente, numa grade pequena:
iq: tamanho do episodio E em {10, 50, 100} x decaimento do epsilon por episodio beta em {0,8; 0,95} (alfa 0,1; gama 0,9; eta 0,5; piso do epsilon 0,05); fr: constante C em {0,5; 2; 5} x decaimento D em {0,5; 1,0} (janela de 50).
Hospedeiros: 'pc' (P||Cmax sintetico: as 12 classes e as instancias de teste das Entradas 180/209) e 'ur' (R||Cmax: Entrada 234); espacos S5 e S15; T = 1000; 5 sementes no teste (100 instancias por classe); o aleatorio, a acao validada e os demais aprendizes vem dos arquivos existentes.
Fases (retomaveis): --tune HOSPEDEIRO ESPACO | --run HOSPEDEIRO ESPACO. Saidas: results/pubaos_choice_HOSPEDEIRO_ESPACO_entrada242.json, results/pubaos_test_HOSPEDEIRO_ESPACO_entrada242.csv. Pre-registro: analysis/report_entrada242_pubaos.py (docstring; hashes em results/entrada242_prereg_hashes.txt).
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
from pmsp import ig_numba_agents6 as G6  # noqa: E402
from pmsp import ur_numba_agents6 as U6  # noqa: E402

R = ROOT / "results"
GRID = {"iq": [(E, b) for E in (10, 50, 100) for b in (0.8, 0.95)], "fr": [(c, dd) for c in (0.5, 2.0, 5.0) for dd in (0.5, 1.0)]}
AGENT = {"iq": 11, "fr": 12}
FIELDS = ["host", "space", "cls", "rep", "method", "seed_i", "makespan"]
HOSTS = {"pc": 0, "ur": 1}
SPACES = {"S5": 0, "S15": 1}


def instance(host, n, m, rep, ci, tune):
    if host == "pc":
        inst, p, a0, od = make_pc(n, m, rep, tune)
        return p, a0, od, inst.lower_bound
    P, a0, od, lb = UR.make(n, m, rep, ci, tune)
    return P, a0, od, lb


def run1(host, space, X, a0, m, n, od, lb, name, cfg, seed):
    st, d, ls = arm_table(space, n)
    if name == "iq":
        E, b = cfg
        lr, lr2, hp = 0.1, float(E), np.array([0.1, 0.9, b, 0.05, 0.5])
    else:
        lr, lr2, hp = cfg[0], cfg[1], G6.HP_DEFAULT
    core = G6.ig_agent_core6 if host == "pc" else U6.ur_agent_core6
    best, _, ev, used, _ = core(X, a0, m, st, d, ls, 1000, AGENT[name], 0, lr, lr2, seed, od, lb, 0, 0, 0.0, hp, 0)
    return int(best)


def seed_of(host, space, kind, ci, rep, s, name, k=0):
    return 600_000_000 + 100_000_000 * HOSTS[host] + 20_000_000 * SPACES[space] + (0 if kind == "test" else 7_000_000) + 3_000_000 * (name == "fr") + 1000 * ci + rep + 50_000 * s + 7_919 * k


def warm():
    for host in HOSTS:
        X, a0, od, lb = instance(host, 20, 4, 0, 0, True)
        for name in AGENT:
            run1(host, "S5", X, a0, 4, 20, od, lb, name, GRID[name][0], 1)


def tune(host, space):
    warm()
    out = {}
    t0 = time.perf_counter()
    tot = {name: np.zeros(len(GRID[name])) for name in AGENT}
    cnt = 0
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(UR.N_TUNE):
            X, a0, od, lb = instance(host, n, m, rep, ci, True)
            for name in AGENT:
                for k, cfg in enumerate(GRID[name]):
                    for s in range(2):
                        mk = run1(host, space, X, a0, m, n, od, lb, name, cfg, seed_of(host, space, "tune", ci, rep, s, name, k))
                        tot[name][k] += 100.0 * (mk - lb) / lb
            cnt += 2
    for name in AGENT:
        mean = tot[name] / cnt
        k = int(np.argmin(mean))
        out[name] = dict(cfg=list(GRID[name][k]), index=k, gaps={str(c): float(v) for c, v in zip(GRID[name], mean)})
        print(host, space, name, "->", GRID[name][k], {str(c): round(float(v), 3) for c, v in zip(GRID[name], mean)}, f"({time.perf_counter() - t0:.0f}s)")
    (R / f"pubaos_choice_{host}_{space}_entrada242.json").write_text(json.dumps(out, indent=1))


def run_test(host, space):
    ch = json.loads((R / f"pubaos_choice_{host}_{space}_entrada242.json").read_text())
    out = R / f"pubaos_test_{host}_{space}_entrada242.csv"
    warm()
    rows = []
    t0 = time.perf_counter()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(UR.N_TEST):
            X, a0, od, lb = instance(host, n, m, rep, ci, False)
            for name in AGENT:
                cfg = tuple(ch[name]["cfg"])
                for s in range(5):
                    rows.append(dict(host=host, space=space, cls=cls, rep=rep, method=name, seed_i=s, makespan=run1(host, space, X, a0, m, n, od, lb, name, cfg, seed_of(host, space, "test", ci, rep, s, name))))
        print(f"[pubaos {host} {space}] {cls} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", nargs=2, default=None, metavar=("HOST", "SPACE"))
    ap.add_argument("--run", nargs=2, default=None, metavar=("HOST", "SPACE"))
    a = ap.parse_args()
    if a.tune:
        tune(*a.tune)
    if a.run:
        run_test(*a.run)
