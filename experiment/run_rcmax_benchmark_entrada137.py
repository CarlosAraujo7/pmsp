"""
experiment/run_rcmax_benchmark_entrada137.py
==============================================
Entrada 137 (Ciclo 8): validacao EXTERNA da decomposicao mecanismo x politica em R||Cmax no benchmark padrao de Fanjul-Peyro e Ruiz (EJOR 2010; 1.400 instancias: 7 tipos x
m in {10,20,30,40,50} x n in {100,200,500,1000} x 10 replicacoes), na copia do repositorio Fraunhofer-Chalmers-Centre/RCmax (MIT; baixada com a autorizacao de Carlos em 10/10/2026):
external_benchmarks/rcmax_fanjul_ruiz/{raw,extracted}. Referencias: melhor valor conhecido (min de CPLEX 2 h, Gurobi 300 s e LR 300 s) e se algum metodo provou otimalidade.
Divisao deterministica por celula (tipo, m, n): as 10 instancias, ordenadas por id, tem idx 0-9; idx 0-1 = AJUSTE, idx 2-6 = TESTE (5 por celula = 700), idx 7-9 = RESERVA (nao usadas).
Fases (cada uma congelada em hash antes da seguinte):
  --manifest   : results/rcmax_manifest_entrada137.csv
  --tune-d     : d/n por n em {0,02; 0,05; 0,10; 0,20; 0,30} (IG+LS aleatorio, T = 1000, ajuste; menor razao media makespan/melhor conhecido)
  --tune-arm   : por TIPO, a melhor entre 5 candidatas (4 estrategias de destruicao fixas e a mistura uniforme) no ajuste, com o d/n escolhido
  --calibrate  : razoes de CPU sequenciais (maquina em repouso) por n em instancias de ajuste de m = 30 e tipos U1a100, U100a200, JobsCorre: K_ils = igls/ils e K_ql = ql/random (T = 1000 ambos)
  --run        : teste em 700 instancias: mct, ur_igls_random, ur_igls_ql, ur_igls_random_eq (random com T = max(1000, round(1000 K_ql))), ur_ils_random (T = 1000),
                 ur_ils_random_eq (T = max(1000, round(1000 K_ils))), ur_igls_fix0..3
Saidas: results/rcmax_*_entrada137.{csv,json}. Retomavel. Uso: python -m experiment.run_rcmax_benchmark_entrada137 --manifest ; --tune-d --workers 10 ; ...
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "external_benchmarks" / "rcmax_fanjul_ruiz"
THR = ROOT / "results" / "state_calibration_unrelated_entrada118.json"
LS_STEPS = 20
FRACS = [0.02, 0.05, 0.10, 0.20, 0.30]
TYPES = ["JobsCorre", "MaqCorre", "U1000a1100", "U100a120", "U100a200", "U10a100", "U1a100"]
NS = [100, 200, 500, 1000]
MANIFEST = ROOT / "results" / "rcmax_manifest_entrada137.csv"
TUNE_D = ROOT / "results" / "rcmax_tune_d_entrada137.csv"
CHOICE_D = ROOT / "results" / "rcmax_choice_d_entrada137.json"
TUNE_ARM = ROOT / "results" / "rcmax_tune_arm_entrada137.csv"
CHOICE_ARM = ROOT / "results" / "rcmax_choice_arm_entrada137.json"
CALIB = ROOT / "results" / "rcmax_cpu_ratio_entrada137.json"
OUT = ROOT / "results" / "rcmax_test_entrada137.csv"
FIELDS = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time", "T", "d", "best_known", "lb"]
METHODS = ["mct", "ur_igls_random", "ur_igls_ql", "ur_igls_random_eq", "ur_ils_random", "ur_ils_random_eq", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]
CAND = ["ur_igls_random", "ur_igls_fix0", "ur_igls_fix1", "ur_igls_fix2", "ur_igls_fix3"]


def load_instance(typ, iid):
    from pmsp.unrelated import UnrelatedInstance
    p = np.loadtxt(BENCH / "extracted" / "instances" / typ / f"{iid}.csv", delimiter=",", dtype=np.int64, ndmin=2)
    return UnrelatedInstance(p, name=f"{typ}_{iid}")


def build_manifest():
    ref = {(r["type"], r["instance"]): r for r in csv.DictReader(open(BENCH / "raw" / "res_all_instances.csv", newline=""))}
    rows = []
    for typ in TYPES:
        cells = {}
        for f in sorted((BENCH / "extracted" / "instances" / typ).glob("*.csv"), key=lambda q: int(q.stem)):
            with open(f, newline="") as fh:
                first = fh.readline()
                n = len(first.strip().split(","))
                m = 1 + sum(1 for _ in fh)
            cells.setdefault((m, n), []).append(f.stem)
        for (m, n), ids in sorted(cells.items()):
            assert len(ids) == 10, (typ, m, n, len(ids))
            for k, iid in enumerate(ids):
                r = ref[(typ, iid)]
                vals = [int(float(r[c])) for c in ("LR value", "Gurobi value", "Cplex value")]
                opts = [int(float(r[c])) for c in ("LR opt", "Gurobi opt", "Cplex opt")]
                best = min(vals)
                proven = int(any(o == 1 and v == best for v, o in zip(vals, opts)))
                rows.append(dict(type=typ, instance=iid, m=m, n=n, idx=k, role="tune" if k < 2 else ("test" if k < 7 else "reserve"), best_known=best, proven_opt=proven,
                                 lr=vals[0], gurobi=vals[1], cplex=vals[2]))
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("manifest:", len(rows), "instancias;", {r: sum(1 for x in rows if x["role"] == r) for r in ("tune", "test", "reserve")})


def manifest(role=None):
    rows = list(csv.DictReader(open(MANIFEST, newline="")))
    return [r for r in rows if role is None or r["role"] == role]


def _solve(inst, mt, T, frac, seed):
    from experiment import config as CFG
    from pmsp.fixed_selector import FixedActionSelector
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import N_UR_DESTROY, N_UR_OPS, UrStateFn, mct_greedy, ur_ig_run, ur_ils_run
    d = max(2, round(frac * inst.n))
    if mt == "mct":
        t0 = time.perf_counter()
        s = mct_greedy(inst)
        return dict(makespan=s.makespan, cpu_time=time.perf_counter() - t0), 0
    if mt in ("ur_ils_random", "ur_ils_random_eq"):
        return ur_ils_run(inst, RandomSelector(seed=seed, n_actions=N_UR_OPS), max_iter=T, seed=seed), 0
    kw = dict(max_iter=T, d=d, seed=seed, ls_steps=LS_STEPS)
    if mt in ("ur_igls_random", "ur_igls_random_eq"):
        return ur_ig_run(inst, RandomSelector(seed=seed, n_actions=N_UR_DESTROY), **kw), d
    if mt == "ur_igls_ql":
        thr = json.loads(THR.read_text())
        sf = UrStateFn({"f1": thr["f1"], "f2": thr["f2"]})
        return ur_ig_run(inst, QLearningAgent(alpha=CFG.QL_ALPHA, gamma=CFG.QL_GAMMA, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                                              epsilon_decay=CFG.QL_EPSILON_DECAY, seed=seed, n_actions=N_UR_DESTROY), state_fn=sf, **kw), d
    return ur_ig_run(inst, FixedActionSelector(int(mt[-1])), **kw), d


def tune_d(workers):
    tasks = []
    for r in manifest("tune"):
        for fr in FRACS:
            tasks.append(dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], method=f"ur_igls_random@{fr}", T=1000, frac=fr, best_known=r["best_known"]))
    # o nome do metodo carrega a fracao (@fracao) para que a chave de retomada distinga as candidatas
    _drive_named(tasks, TUNE_D, workers)
    import pandas as pd
    d = pd.read_csv(TUNE_D)
    d["frac"] = d.method.str.split("@").str[1].astype(float)
    d["ratio"] = d.makespan / d.best_known
    choice = {}
    for n, g in d.groupby("n"):
        mean = g.groupby("frac").ratio.mean()
        choice[str(int(n))] = dict(frac=float(mean.idxmin()), means={str(k): float(v) for k, v in mean.items()})
        print(f"[tune-d] n = {int(n)} -> d/n = {mean.idxmin()} ({', '.join(f'{k}:{v:.4f}' for k, v in mean.items())})", flush=True)
    CHOICE_D.write_text(json.dumps(choice, indent=1))


def _drive_named(tasks, out, workers):
    """Como _drive, mas o nome do metodo (com @fracao) e' mantido nos resultados e na chave."""
    done = set()
    if out.exists():
        done = {(r["type"], r["instance"], r["method"]) for r in csv.DictReader(open(out, newline=""))}
    pend = [t for t in tasks if (t["type"], t["instance"], t["method"]) not in done]
    pend.sort(key=lambda t: -(int(t["n"]) * int(t["m"])))
    print(f"[tasks] {len(pend)} pendentes ({len(done)} feitas) -> {out.name}", flush=True)
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task_named, t): t for t in pend}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as e:  # noqa: BLE001
                    print("[ERRO]", futs[fut]["type"], futs[fut]["instance"], futs[fut]["method"], e, flush=True)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 100 == 0 or i == len(pend):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(pend)}] {el:.0f}s, ETA {(len(pend) - i) * el / i / 60:.1f} min", flush=True)


def _task_named(t):
    inst = load_instance(t["type"], t["instance"])
    base = t["method"].split("@")[0]
    r, d = _solve(inst, base, t["T"], t["frac"], int(t["instance"]))
    return dict(type=t["type"], instance=t["instance"], m=inst.m, n=inst.n, idx=t["idx"], method=t["method"], makespan=int(r["makespan"]), cpu_time=r["cpu_time"], T=t["T"],
                d=d, best_known=t["best_known"], lb=inst.lower_bound)


def tune_arm(workers):
    cd = json.loads(CHOICE_D.read_text())
    tasks = []
    for r in manifest("tune"):
        for c in CAND:
            tasks.append(dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=r["n"], m=r["m"], method=c, T=1000, frac=cd[r["n"]]["frac"], best_known=r["best_known"]))
    _drive_named(tasks, TUNE_ARM, workers)
    import pandas as pd
    d = pd.read_csv(TUNE_ARM)
    d["ratio"] = d.makespan / d.best_known
    choice = {}
    for typ, g in d.groupby("type"):
        mean = g.groupby("method").ratio.mean()
        choice[typ] = dict(best=mean.idxmin(), means={k: float(v) for k, v in mean.items()})
        print(f"[tune-arm] {typ} -> {mean.idxmin()} ({', '.join(f'{k[-5:]}:{v:.4f}' for k, v in mean.items())})", flush=True)
    CHOICE_ARM.write_text(json.dumps(choice, indent=1))


def calibrate():
    cd = json.loads(CHOICE_D.read_text())
    out = {}
    for n in NS:
        rk, rq = [], []
        for r in manifest("tune"):
            if int(r["n"]) != n or int(r["m"]) != 30 or r["type"] not in ("U1a100", "U100a200", "JobsCorre"):
                continue
            inst = load_instance(r["type"], r["instance"])
            frac, seed = cd[str(n)]["frac"], int(r["instance"])
            g = _solve(inst, "ur_igls_random", 1000, frac, seed)[0]["cpu_time"]
            i = _solve(inst, "ur_ils_random", 1000, frac, seed)[0]["cpu_time"]
            q = _solve(inst, "ur_igls_ql", 1000, frac, seed)[0]["cpu_time"]
            rk.append(g / i)
            rq.append(q / g)
        rk.sort()
        rq.sort()
        kil, kql = float(np.median(rk)), float(np.median(rq))
        out[str(n)] = dict(K_ils=kil, T_ils=max(1000, int(round(1000 * kil))), K_ql=kql, T_random_eq=max(1000, int(round(1000 * kql))), n_inst=len(rk), ratios_ils=rk, ratios_ql=rq)
        print(f"[calib] n = {n}: K_ils = {kil:.2f} -> T_ils = {out[str(n)]['T_ils']}; K_ql = {kql:.2f} -> T_random_eq = {out[str(n)]['T_random_eq']} ({len(rk)} instancias)", flush=True)
    CALIB.write_text(json.dumps(out, indent=1))


def run(workers):
    cd, cal = json.loads(CHOICE_D.read_text()), json.loads(CALIB.read_text())
    tasks = []
    for r in manifest("test"):
        n = r["n"]
        frac = cd[n]["frac"]
        for mt in METHODS:
            T = 1000
            if mt == "ur_ils_random_eq":
                T = cal[n]["T_ils"]
            elif mt == "ur_igls_random_eq":
                T = cal[n]["T_random_eq"]
            tasks.append(dict(type=r["type"], instance=r["instance"], idx=r["idx"], n=n, m=r["m"], method=mt, T=T, frac=frac, best_known=r["best_known"]))
    _drive_named(tasks, OUT, workers)
    print("concluido ->", OUT)


def main():
    ap = argparse.ArgumentParser()
    for k in ("manifest", "tune-d", "tune-arm", "calibrate", "run"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.manifest:
        build_manifest()
    if a.tune_d:
        tune_d(a.workers)
    if a.tune_arm:
        tune_arm(a.workers)
    if a.calibrate:
        calibrate()
    if a.run:
        run(a.workers)


if __name__ == "__main__":
    main()
