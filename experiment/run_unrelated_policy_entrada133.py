"""
experiment/run_unrelated_policy_entrada133.py
===============================================
Entrada 133 (Ciclo 6, frente 1): a politica (Q-Learning sobre as estrategias de destruicao) vale em REGIME NAO SATURADO, a CPU IGUAL?
Na Entrada 127 o ganho da politica em R||Cmax (1000 x 50, T = 1000) foi 0,52% (descritivo, 32 instancias), 5x maior que em n <= 500, mas o Q-Learning custou 1,40x o CPU da selecao
aleatoria. Aqui: 96 instancias NOVAS (4 distribuicoes x 24 reps; seed 500000 + 1000*dist + rep; n = 1000, m = 50), d/n = 0,02 (ajuste da Entrada 127) e limiares de estado da Entrada 118,
T = 1000. Metodos: ur_igls_random, ur_igls_ql, ur_igls_random_eq (random com T_eq = max(1000, round(1000 K)); K POR DISTRIBUICAO = razao mediana de CPU ql/random em 6 instancias de AJUSTE
separadas -- as de seed 97000+1000*dist+rep, rep 0-5 (EMENDA 1) --, medida em calibracao SEQUENCIAL com a maquina em repouso), ur_igls_fix0..3 (folga).
Saidas: results/unrelated_policy_entrada133.csv, results/unrelated_policy_cpu_ratio_entrada133.json. Retomavel.
Uso: python -m experiment.run_unrelated_policy_entrada133 --calibrate ; --run --workers 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment.run_unrelated_entrada118 import DISTS, LS_STEPS, THR  # noqa: E402
from experiment.run_unrelated_large_entrada127 import DSEL, M, N, inst_of  # noqa: E402

BASE = 500000
N_REPS = 24
RATIO = ROOT / "results" / "unrelated_policy_cpu_ratio_entrada133.json"
OUT = ROOT / "results" / "unrelated_policy_entrada133.csv"
FIELDS = ["lo", "hi", "n", "m", "rep", "method", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements", "d", "T"]
METHODS = ["ur_igls_random", "ur_igls_ql", "ur_igls_random_eq", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]


def _solve(inst, mt, rep, T):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, UrStateFn, ur_ig_run
    thr = json.loads(THR.read_text())
    sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
    d = max(2, round(json.loads(DSEL.read_text())["frac"] * N))
    kw = dict(max_iter=T, d=d, seed=rep, ls_steps=LS_STEPS)
    if mt in ("ur_igls_random", "ur_igls_random_eq"):
        return ur_ig_run(inst, RandomSelector(seed=rep, n_actions=N_UR_DESTROY), **kw), d
    if mt == "ur_igls_ql":
        return ur_ig_run(inst, QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                                              epsilon_decay=CFG.QL_EPSILON_DECAY, seed=rep, n_actions=N_UR_DESTROY), state_fn=sf, **kw), d
    return ur_ig_run(inst, FixedActionSelector(int(mt[-1])), **kw), d


def calibrate():
    # EMENDA 1 (Entrada 133, antes de qualquer dado de teste): K POR DISTRIBUICAO (mediana de 6 instancias de ajuste, rep 0-5) em vez de um K unico (mediana de 8):
    # a calibracao inicial mostrou razao ql/random de 0,98 em U[1,100] a 1,80 em U[100,200]
    out = {}
    for di in range(len(DISTS)):
        rs = []
        for rep in range(6):
            inst = inst_of(di, rep, base=97000)
            a = _solve(inst, "ur_igls_random", rep, 1000)[0]["cpu_time"]
            b = _solve(inst, "ur_igls_ql", rep, 1000)[0]["cpu_time"]
            rs.append(b / a)
        rs.sort()
        k = 0.5 * (rs[2] + rs[3])
        out[str(di)] = dict(K=k, T_eq=max(1000, int(round(1000 * k))), ratios=rs)
        print(f"[calib] dist {di} {DISTS[di]}: K = {k:.3f} -> T_eq = {out[str(di)]['T_eq']}", flush=True)
    RATIO.write_text(json.dumps(out, indent=1))


def _run(t):
    di, rep, mt, T = t["di"], t["rep"], t["method"], t["T"]
    inst = inst_of(di, rep, base=BASE)
    r, d = _solve(inst, mt, rep, T)
    lo, hi = DISTS[di]
    return dict(lo=lo, hi=hi, n=N, m=M, rep=rep, method=mt, lb=inst.lower_bound, makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"],
                n_improvements=r["n_improvements"], d=d, T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        teq = {int(k): v["T_eq"] for k, v in json.loads(RATIO.read_text()).items()}
        done = set()
        if OUT.exists():
            done = {(r["lo"], r["hi"], r["rep"], r["method"]) for r in csv.DictReader(open(OUT, newline=""))}
        tasks = [dict(di=di, rep=r, method=mt, T=teq[di] if mt == "ur_igls_random_eq" else 1000) for di in range(len(DISTS)) for r in range(N_REPS) for mt in METHODS
                 if (str(DISTS[di][0]), str(DISTS[di][1]), str(r), mt) not in done]
        tasks.sort(key=lambda t: -t["T"])
        print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)", flush=True)
        new = not OUT.exists()
        t0 = time.perf_counter()
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(_run, t): t for t in tasks}
                for i, fut in enumerate(as_completed(futs), 1):
                    try:
                        w.writerow(fut.result())
                    except Exception as e:  # noqa: BLE001
                        print("[ERRO]", futs[fut], e, flush=True)
                        continue
                    f.flush()
                    os.fsync(f.fileno())
                    if i % 50 == 0 or i == len(tasks):
                        el = time.perf_counter() - t0
                        print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
        print("concluido ->", OUT)


if __name__ == "__main__":
    main()
