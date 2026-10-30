"""
experiment/run_cycle26_eqcpu_ur_entrada248.py
==============================================
Entrada 248 (Ciclo 26): CPU IGUAL no hospedeiro de MAQUINAS NAO RELACIONADAS (R||Cmax), o unico com folga real (as tres revisoes cegas da Entrada 246 apontam as celulas '--' da Tabela IV). Ate' aqui o R||Cmax so' foi comparado a ITERACOES iguais (Entrada 234); a ressalva (i) do artigo diz que o CPU igual nao foi rodado ali.

Desenho (o mesmo do P||Cmax, Entradas 180 e 195): o custo K de um aprendiz e' a CPU de uma execucao de T = 1000 iteracoes dele sobre a do controle aleatorio (T = 1000), medida em sequencia, com a maquina em repouso, em duas instancias de AJUSTE por classe e espaco (mediana das duas razoes; processo de CPU >= 1 s por medida); o mesmo para a acao fixa validada (K_val). Cada aprendiz (ql, ucb, pg, alns e o Q-Learning otimista qo, com os hiperparametros CONGELADOS da Entrada 234) roda T = 1000 (resultados ja' gravados em results/ur_test_*_entrada234.csv, mesmas instancias e sementes); o controle aleatorio recebe T = max(50, round(1000 K_aprendiz)) iteracoes (`rand_eq_<aprendiz>`) e a acao validada, T = max(50, round(1000 K_aprendiz / K_val)) (`val_eq_<aprendiz>`), de modo que cada braco gasta a CPU do aprendiz. Ganho de CPU igual = ganho do aprendiz sobre `rand_eq`; efeito de decisao = ganho a iteracoes iguais (Entrada 234); efeito de custo = a diferenca. 5 sementes por metodo; 12 classes x 100 instancias x 2 espacos (S5, S15); nada e' reajustado.

Fases (retomaveis): --calibrate (SOZINHA na maquina) | --run ESPACO --shard i --nshards n [--pilot: 3 instancias de AJUSTE por classe, saida results/ur_eqcpu_pilot_*, para depurar o codigo SEM tocar nas instancias de teste].
Saidas: results/ur_eqcpu_k_entrada248.json, results/ur_eqcpu_ESPACO_s*of*_entrada248.csv. Pre-registro: analysis/report_entrada248_ur_eqcpu.py (docstring; hashes em results/entrada248_prereg_hashes.txt).
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

from experiment import run_cycle22_ur_entrada234 as U  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402

R = ROOT / "results"
SPACES = U.SPACES
LEARNERS = ["ql", "ucb", "pg", "alns", "qo"]
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]
FLOOR = 50


def k_path():
    return R / "ur_eqcpu_k_entrada248.json"


def seed_of(space, ci, rep, method, s):
    """Sementes disjuntas das da Entrada 234 (todas abaixo de 100 000 000)"""
    j = METHODS.index(method)
    return rep + 1000 * ci + 50000 * s + 1_000_000 * (11 + j) + 100_000_000 * (SPACES.index(space) + 1)


METHODS = [f"rand_eq_{k}" for k in LEARNERS] + [f"val_eq_{k}" for k in LEARNERS]


def t_rand(K, k):
    return max(FLOOR, int(round(1000 * K[k])))


def t_val(K, k):
    return max(FLOOR, int(round(1000 * K[k] / K["val"])))


def run_learner(space, name, P, a0, m, od, lb, n, seed, qo_cfg, T=1000):
    """um aprendiz (ou o controle aleatorio com T diferente) com os hiperparametros congelados da Entrada 234"""
    if name == "qo":
        return U.run_qo(space, P, a0, m, od, lb, n, qo_cfg, seed, T=T)
    ag, key = U.AGENTS[name]
    lr, lr2 = (U.PG_LR[space], 0.1) if key == "pg" else (U.ALNS_HP if key == "alns" else (0.1, 0.1))
    return U.run(space, P, a0, m, od, lb, n, ag, 0, lr, lr2, seed, T=T)


def load_choices(space):
    val = json.loads(U.choice_path(space).read_text())["val"]
    qo_cfg = tuple(json.loads(U.choice_qo_path(space).read_text())["cfg"])
    return val, qo_cfg


def calibrate():
    U.warm()
    res = {}
    for sp in SPACES:
        val, qo_cfg = load_choices(sp)
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            cls = f"rand_{n}j_{m}m"
            ratios = {k: [] for k in LEARNERS + ["val"]}
            for rep in range(2):
                P, a0, od, lb = U.make(n, m, rep, ci, True)

                def cpu(fn):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        fn(reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(lambda sd: U.run(sp, P, a0, m, od, lb, n, UA.AGENT_RANDOM, 0, 0.1, 0.1, sd))
                for k in LEARNERS:
                    ratios[k].append(cpu(lambda sd, k=k: run_learner(sp, k, P, a0, m, od, lb, n, sd, qo_cfg)) / base)
                ratios["val"].append(cpu(lambda sd: U.run(sp, P, a0, m, od, lb, n, UA.AGENT_FIXED, val[cls], 0.1, 0.1, sd)) / base)
            res[sp][cls] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[eqcpu-calib {sp}] {cls}: {({k: round(v, 3) for k, v in res[sp][cls].items()})}", flush=True)
    k_path().write_text(json.dumps(res, indent=1))


def run_space(space, shard, nshards, pilot):
    val, qo_cfg = load_choices(space)
    K = json.loads(k_path().read_text())[space]
    out = R / (f"ur_eqcpu_pilot_{space}_entrada248.csv" if pilot else f"ur_eqcpu_{space}_s{shard}of{nshards}_entrada248.csv")
    done = set()
    if out.exists():
        done = {(r["cls"], r["rep"]) for r in csv.DictReader(open(out, newline=""))}
    U.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    cnt = 0
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(3 if pilot else U.N_TEST):
            if (not pilot) and (ci * U.N_TEST + rep) % nshards != shard:
                continue
            if (cls, str(rep)) in done:
                continue
            P, a0, od, lb = U.make(n, m, rep, ci, bool(pilot))
            rows = []
            for k in LEARNERS:
                Tr, Tv = t_rand(K[cls], k), t_val(K[cls], k)
                for s in range(5):
                    mk, ev = U.run(space, P, a0, m, od, lb, n, UA.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, ci, rep, f"rand_eq_{k}", s), T=Tr)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"rand_eq_{k}", seed_i=s, makespan=mk, lb=lb, T=Tr, evals=ev))
                    mk, ev = U.run(space, P, a0, m, od, lb, n, UA.AGENT_FIXED, val[cls], 0.1, 0.1, seed_of(space, ci, rep, f"val_eq_{k}", s), T=Tv)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"val_eq_{k}", seed_i=s, makespan=mk, lb=lb, T=Tv, evals=ev))
            with open(out, "w" if new else "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            new = False
            cnt += 1
            if cnt % 100 == 0:
                print(f"[eqcpu {space} s{shard}] {cnt} instancias ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[eqcpu {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        run_space(a.run, a.shard, a.nshards, a.pilot)
