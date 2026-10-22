"""
experiment/run_cycle17_new_entrada188.py
==========================================
Entrada 188 (Ciclo 17, frente B'): tres concorrentes online que faltavam -- Thompson com desconto (ts), DMAB (dmab) e corrida por reducao pela metade (race) -- nos espacos S5 e S15 do P||Cmax sintetico (12 classes x 100 instancias de teste, T = 1000, 5 sementes; as MESMAS instancias
e as linhas rand, val, ql, ucb, pg, pm, ap e alns de results/agents_S5|S15_entrada180.csv e results/aos_S5|S15_entrada185.csv sao reaproveitadas na analise). pmsp/ig_numba_agents3.py (59 testes; os agentes 0-7 reproduzem ig_agent_core2 exatamente).
Hiperparametros ajustados em 30 instancias por classe (mesma regra das Entradas 180 e 185: menor lacuna media ao limite inferior): TS gama em {1; 0,99; 0,95}; DMAB (lambda, delta) em {(2; 0), (8; 0), (8; 0,02), (32; 0,02)}; corrida rho em {0,25; 0,5}.
K_CPU de cada um sobre o aleatorio calibrado em sequencia, maquina em repouso, para a comparacao a CPU igual (T_eq = round(1000 K)).
Fases (retomaveis): --verify | --tune S | --calibrate | --run S. Pre-registro: analysis/report_entrada188_new.py (docstring, congelado antes de rodar; hashes em results/entrada188_prereg_hashes.txt).
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
from experiment.run_cycle16_entrada185 import FIELDS, N_TEST, N_TUNE, make, mean_gap, write  # noqa: E402
from pmsp import ig_numba_agents2 as H2  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402

R = ROOT / "results"
NEW = {"ts": H.AGENT_TS, "dmab": H.AGENT_DMAB, "race": H.AGENT_RACE}
GRID = {"ts": [(g, 0.0) for g in (1.0, 0.99, 0.95)], "dmab": [(2.0, 0.0), (8.0, 0.0), (8.0, 0.02), (32.0, 0.02)], "race": [(r, 0.0) for r in (0.25, 0.5)]}
MID = {"rand_eq": 24, "tune": 25, "ts": 21, "dmab": 22, "race": 23}


def seed_of(space, ci, rep, method, s):
    return rep + 1000 * ci + 50000 * s + 1_000_000 * MID[method] + 20_000_000 * (40 + ["S5", "S15"].index(space))


def run(space, p, a0, m, od, lb, n, agent, fixed, lr, lr2, seed, T=1000):
    st, d, ls = arm_table(space, n)
    best, _, ev, used, _ = H.ig_agent_core3(p, a0, m, st, d, ls, T, agent, fixed, lr, lr2, seed, od, lb, 0, 0, 0.0)
    return int(best), float(ev), used


def warm():
    inst, p, a0, od = make(20, 4, 0, False)
    for sp in ("S5", "S15"):
        for ag in range(11):
            run(sp, p, a0, 4, od, inst.lower_bound, 20, ag, 0, 0.5, 0.1, 1, T=30)


def choice_path(space):
    return R / f"new_choice_{space}_entrada188.json"


def k_path():
    return R / "new_cpu_ratio_entrada188.json"


def verify():
    """Agentes 0-4 do nucleo 3 reproduzem o nucleo 2 (e, por ele, o de Entrada 180) em instancias de teste reais; regra de parada."""
    warm()
    bad = 0
    for sp in ("S5", "S15"):
        for ci, (n, m) in enumerate(CLASSES[:6]):
            inst, p, a0, od = make(n, m, 0, False)
            st, d, ls = arm_table(sp, n)
            for ag in range(8):
                a = H2.ig_agent_core2(p, a0, m, st, d, ls, 400, ag, 1, 0.1, 0.1, 5 + ci, od, inst.lower_bound, 0, 0, 0.0)
                b = H.ig_agent_core3(p, a0, m, st, d, ls, 400, ag, 1, 0.1, 0.1, 5 + ci, od, inst.lower_bound, 0, 0, 0.0)
                bad += int(a[0] != b[0] or a[2] != b[2] or not np.array_equal(a[3], b[3]))
    print("verify: agentes 0-7 conferem" if bad == 0 else f"verify: {bad} divergencias")
    return bad


def tune(space):
    warm()
    rows = []
    for ci, (n, m) in enumerate(CLASSES):
        for rep in range(N_TUNE):
            inst, p, a0, od = make(n, m, rep, True)
            for k, ag in NEW.items():
                for (a, b) in GRID[k]:
                    mk, ev, _ = run(space, p, a0, m, od, inst.lower_bound, n, ag, 0, a, b, seed_of(space, ci, rep, "tune", 0))
                    rows.append(dict(space=space, cls=f"rand_{n}j_{m}m", rep=rep, method=f"{k}@{a},{b}", seed_i=0, makespan=mk, lb=inst.lower_bound, T=1000, evals=ev))
    write(R / f"new_tune_{space}_entrada188.csv", rows, True)
    g = mean_gap(rows)
    ch = {}
    for k in NEW:
        best = min((key for key in g if key.startswith(k + "@")), key=lambda key: g[key])
        a, b = (float(x) for x in best.split("@")[1].split(","))
        ch[k] = [a, b]
    ch["gaps"] = g
    choice_path(space).write_text(json.dumps(ch, indent=1))
    print(space, "ajuste:", {k: ch[k] for k in NEW})


def calibrate():
    ch = {sp: json.loads(choice_path(sp).read_text()) for sp in ("S5", "S15")}
    warm()
    res = {}
    for sp in ("S5", "S15"):
        res[sp] = {}
        for ci, (n, m) in enumerate(CLASSES):
            ratios = {k: [] for k in NEW}
            for rep in range(2):
                inst, p, a0, od = make(n, m, rep, False)

                def cpu(agent, a=0.1, b=0.1):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 1.0:
                        run(sp, p, a0, m, od, inst.lower_bound, n, agent, 0, a, b, reps + 7)
                        reps += 1
                    return (time.process_time() - t0) / reps
                base = cpu(H.AGENT_RANDOM)
                for k, ag in NEW.items():
                    ratios[k].append(cpu(ag, *ch[sp][k]) / base)
            res[sp][f"{n}x{m}"] = {k: float(np.median(v)) for k, v in ratios.items()}
            print(f"[new-calib {sp}] {n}x{m}: {res[sp][f'{n}x{m}']}", flush=True)
    k_path().write_text(json.dumps(res, indent=1))


def run_space(space):
    ch = json.loads(choice_path(space).read_text())
    K = json.loads(k_path().read_text())[space]
    out = R / f"new_{space}_entrada188.csv"
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
                for j, (k, ag) in enumerate(NEW.items()):
                    mk, ev, _ = run(space, p, a0, m, od, lb, n, ag, 0, ch[k][0], ch[k][1], seed_of(space, ci, rep, k, s))
                    rows.append(dict(space=space, cls=cls, rep=rep, method=k, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev))
                    Teq = max(50, int(round(1000 * K[f"{n}x{m}"][k])))
                    mk, ev, _ = run(space, p, a0, m, od, lb, n, H.AGENT_RANDOM, 0, 0.1, 0.1, seed_of(space, ci, rep, "rand_eq", s) + 3 * j, T=Teq)
                    rows.append(dict(space=space, cls=cls, rep=rep, method=f"rand_eq_{k}", seed_i=s, makespan=mk, lb=lb, T=Teq, evals=ev))
            write(out, rows, new)
            new = False
        print(f"[new {space}] {n}x{m} ok ({time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("verify", "calibrate"):
        ap.add_argument("--" + f, action="store_true")
    for f in ("tune", "run"):
        ap.add_argument("--" + f, default=None)
    a = vars(ap.parse_args())
    if a["verify"]:
        sys.exit(1 if verify() else 0)
    if a["tune"]:
        tune(a["tune"])
    if a["calibrate"]:
        calibrate()
    if a["run"]:
        run_space(a["run"])
