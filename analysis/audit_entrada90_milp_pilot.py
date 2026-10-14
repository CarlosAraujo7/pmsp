"""
analysis/audit_entrada90_milp_pilot.py
Piloto: MIP generico (HiGHS via scipy.milp, 10 s) vs ig_ql nas mesmas instancias; 5 instancias x 4 classes.
Diario: Entrada 90 (04/10/2026). Piloto pequeno -- NAO e uma bateria; ver diario para ressalvas.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from pmsp.instance import PMSPInstance

ROOT = str(Path(__file__).resolve().parent.parent)
res = pd.read_csv(f"{ROOT}/results/results.csv")
TL = 10.0
K = 5

def solve(inst, tl):
    p = np.asarray(inst.p, dtype=float); n, m = len(p), inst.m
    nv = n * m + 1                       # x[i,j] -> i*n+j ; last = Cmax
    c = np.zeros(nv); c[-1] = 1
    A = lil_matrix((n + m, nv)); lo = np.zeros(n + m); hi = np.zeros(n + m)
    for j in range(n):
        for i in range(m): A[j, i * n + j] = 1
        lo[j] = hi[j] = 1
    for i in range(m):
        for j in range(n): A[n + i, i * n + j] = p[j]
        A[n + i, nv - 1] = -1
        lo[n + i] = -np.inf; hi[n + i] = 0
    integrality = np.r_[np.ones(n * m), 0]
    lb = np.r_[np.zeros(n * m), inst.lower_bound]; ub = np.r_[np.ones(n * m), p.sum()]
    t0 = time.perf_counter()
    r = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integrality,
             bounds=Bounds(lb, ub), options=dict(time_limit=tl, disp=False))
    dt = time.perf_counter() - t0
    return (r.fun if r.x is not None else np.nan), dt, r.status

for cls in ["rand_20j_6m", "rand_30j_8m", "rand_50j_10m", "rand_100j_20m"]:
    out = []
    for rep in range(K):
        inst = PMSPInstance.load(f"{ROOT}/data/instances/{cls}/rep{rep:02d}.json")
        mk, dt, st = solve(inst, TL)
        sel = res[(res.instance == cls) & (res.rep == rep)].set_index("method")
        lb = inst.lower_bound
        out.append(dict(rep=rep, milp_gap=100 * (mk - lb) / lb if mk == mk else np.nan, milp_s=dt, status=st,
                        ig_ql_gap=sel.loc["ig_ql", "gap_pct"], ig_ql_s=sel.loc["ig_ql", "cpu_time"],
                        multifit_gap=sel.loc["multifit", "gap_pct"]))
    d = pd.DataFrame(out)
    print(f"== {cls}")
    print(d.round(3).to_string(index=False))
    print("mean:", d[["milp_gap", "milp_s", "ig_ql_gap", "ig_ql_s", "multifit_gap"]].mean().round(3).to_dict())
    print(flush=True)
