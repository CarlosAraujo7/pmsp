"""
experiment/run_numba_entrada170.py
====================================
Entrada 170 (Ciclo 14, frente H): O ORCAMENTO 'A CPU IGUAL' COM UMA IMPLEMENTACAO COMPILADA E COM UMA UNIDADE DE ORCAMENTO INDEPENDENTE DE IMPLEMENTACAO. IG e IG com busca local de P||Cmax compilados com numba
(pmsp/ig_numba.py; busca local e reconstrucao exatamente equivalentes a' referencia, testes em tests/test_ig_numba.py) e com contador de avaliacoes.
Cenarios (os mesmos de Entrada 129/127/162; 12 classes x 100 reps, mesmas instancias):
  H: ig_random (3 estrategias sorteadas, d = max(2, round(0,3 n)), partida LPT)            R: receita (estrategia 0, d = max(2, round(frac_classe n)) com frac validada offline, partida pela melhor construtiva)
Metodos por instancia: igls (T = 1000, com busca local) | ig_T1000 (sem busca local, iteracoes iguais) | ig_eqcpu (T_eq = max(1000, round(1000 K_cpu)), K_cpu calibrado) | ig_eqev_w1 / _w0.25 / _w4
(sem busca local, parado quando as avaliacoes atingem as do igls da mesma instancia; w = peso de uma avaliacao da reconstrucao em relacao a uma avaliacao da busca local: 1 e' a definicao principal, 0,25 e 4 sao sensibilidade).
Fases:
  --calibrate : K_cpu por classe e cenario (sequencial, maquina em repouso, 3 instancias, CPU de processo acumulada por >= 2 s por metodo, mediana das tres razoes) -> results/numba_cpu_ratio_entrada170.json
  --run       : 12 classes x 100 reps x 2 cenarios x 6 metodos -> results/numba_entrada170.csv (retomavel)
Uso: python -m experiment.run_numba_entrada170 --calibrate ; --run
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

from experiment import config as CFG  # noqa: E402
from experiment.run_igcvls_entrada127 import d_choice  # noqa: E402
from experiment.run_dsel_entrada114 import FRACS  # noqa: E402
from pmsp import ig_numba as N  # noqa: E402
from pmsp.constructive import lpt  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402
from pmsp.seeding import best_constructive  # noqa: E402

CLASSES = [(20, 4), (20, 6), (30, 8), (40, 5), (50, 10), (60, 20), (75, 15), (100, 20), (150, 15), (150, 30), (250, 25), (300, 20)]
RATIO = ROOT / "results" / "numba_cpu_ratio_entrada170.json"
OUT = ROOT / "results" / "numba_entrada170.csv"
FIELDS = ["scenario", "instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "n_improvements", "evals", "iters", "wall_s", "T_req"]
WEIGHTS = {"ig_eqev_w1": 1.0, "ig_eqev_w0.25": 0.25, "ig_eqev_w4": 4.0}
METHOD_ID = {"igls": 1, "ig_T1000": 2, "ig_eqcpu": 3, "ig_eqev_w1": 4, "ig_eqev_w0.25": 5, "ig_eqev_w4": 6}


def setup(scenario, n, m, rep, ch):
    """Instancia, atribuicao inicial, d, estrategia e p/ordem decrescente (as pecas que o nucleo compilado recebe)."""
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    p = np.asarray(inst.p, dtype=np.int64)
    od = np.argsort(inst.p)[::-1].astype(np.int64)
    if scenario == "H":
        a0, d, strat = lpt(inst).assignment.astype(np.int64), max(2, round(0.3 * n)), -1
    else:
        a0, d, strat = best_constructive(inst).assignment.astype(np.int64), max(2, round(ch[f"rand_{n}j_{m}m"] * n)), 0
    return inst, p, od, a0, d, strat


def seed_of(scenario, ci, rep, method):
    return CFG.BASE_SEED + rep + 1000 * METHOD_ID[method] + 100000 * ci + (10 ** 7 if scenario == "R" else 0)


def call(p, a0, m, d, strat, T, budget, ls, seed, od, w=1.0):
    t0 = time.perf_counter()
    best, nimp, ev, done, _ = N.ig_core(p, a0, m, d, strat, T, float(budget), ls, seed, od, float(w))
    return int(best), int(nimp), float(ev), int(done), time.perf_counter() - t0


def calibrate():
    ch = d_choice()
    out = {"H": {}, "R": {}}
    inst, p, od, a0, d, strat = setup("H", 20, 4, 0, ch)
    call(p, a0, 4, d, strat, 10, 0, True, 0, od)          # aquece a compilacao
    for sc in ("H", "R"):
        for ci, (n, m) in enumerate(CLASSES):
            rs = []
            for rep in range(3):
                inst, p, od, a0, d, strat = setup(sc, n, m, rep, ch)
                cpu = {}
                for ls in (False, True):
                    reps, t0 = 0, time.process_time()
                    while time.process_time() - t0 < 2.0:
                        N.ig_core(p, a0, m, d, strat, 1000, 0.0, ls, reps, od, 1.0)
                        reps += 1
                    cpu[ls] = (time.process_time() - t0) / reps
                rs.append(cpu[True] / cpu[False])
            k = sorted(rs)[1]
            out[sc][f"{n}x{m}"] = dict(K=k, T_eq=max(1000, int(round(1000 * k))), ratios=rs)
            print(f"[calib {sc}] {n}x{m}: K_cpu = {k:.2f} -> T_eq = {out[sc][f'{n}x{m}']['T_eq']}", flush=True)
    RATIO.write_text(json.dumps(out, indent=1))


def run():
    ratio = json.loads(RATIO.read_text())
    ch = d_choice()
    done = set()
    if OUT.exists():
        done = {(r["scenario"], r["instance"], r["rep"]) for r in csv.DictReader(open(OUT, newline=""))}
    inst, p, od, a0, d, strat = setup("H", 20, 4, 0, ch)
    call(p, a0, 4, d, strat, 10, 0, True, 0, od)
    new = not OUT.exists()
    t00 = time.perf_counter()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        k = 0
        for sc in ("H", "R"):
            for ci, (n, m) in enumerate(CLASSES):
                t_eq = ratio[sc][f"{n}x{m}"]["T_eq"]
                for rep in range(100):
                    if (sc, f"rand_{n}j_{m}m", str(rep)) in done:
                        continue
                    inst, p, od, a0, d, strat = setup(sc, n, m, rep, ch)
                    rows = []

                    def rec(method, res, T_req):
                        best, nimp, ev, it, wall = res
                        rows.append(dict(scenario=sc, instance=inst.name, method=method, rep=rep, n=n, m=m, lb=inst.lower_bound, makespan=best, gap_pct=100.0 * (best - inst.lower_bound) / inst.lower_bound,
                                         n_improvements=nimp, evals=ev, iters=it, wall_s=wall, T_req=T_req))

                    r_ls = call(p, a0, m, d, strat, 1000, 0, True, seed_of(sc, ci, rep, "igls"), od)
                    rec("igls", r_ls, 1000)
                    rec("ig_T1000", call(p, a0, m, d, strat, 1000, 0, False, seed_of(sc, ci, rep, "ig_T1000"), od), 1000)
                    rec("ig_eqcpu", call(p, a0, m, d, strat, t_eq, 0, False, seed_of(sc, ci, rep, "ig_eqcpu"), od), t_eq)
                    ev_rec = r_ls[3] * d * m                  # avaliacoes da reconstrucao do igls (peso 1)
                    ev_ls = r_ls[2] - ev_rec                  # avaliacoes da busca local
                    for meth, wt in WEIGHTS.items():
                        budget = ev_ls + wt * ev_rec          # avaliacoes do igls com o peso wt
                        rec(meth, call(p, a0, m, d, strat, 10 ** 9, budget, False, seed_of(sc, ci, rep, meth), od, wt), 0)
                    w.writerows(rows)
                    k += 1
                f.flush()
                print(f"[{sc}] {n}x{m} ok ({k} instancias, {time.perf_counter() - t00:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        run()
