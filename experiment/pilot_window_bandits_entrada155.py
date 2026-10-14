import json, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Pesquisas\PMSP_Learnheuristic")
OUT = r"C:\Users\vinic\AppData\Local\Temp\claude\C--Pesquisas-PMSP-Learnheuristic\2014e0d0-2ae2-4c2c-9db8-6041232b2612\scratchpad\pilot155.csv"
METHODS = [f"fix_{k}" for k in range(5)] + ["rand_d", "ql_d", "ts_d", "ts_disc_d", "race_d"]


def one(args):
    typ, iid, n, m = args
    import experiment.run_dpolicy_entrada145 as D
    import experiment.run_rcmax_benchmark_entrada137 as B
    from pmsp.window_bandits import RacingSelector, ThompsonBernoulliSelector
    orig = D._agent

    def agent(kind, seed, q_init=None, eps0=1.0):
        if kind == "ts_d":
            return ThompsonBernoulliSelector(n_actions=5, seed=seed)
        if kind == "ts_disc_d":
            return ThompsonBernoulliSelector(n_actions=5, seed=seed, discount=0.98)
        if kind == "race_d":
            return RacingSelector(n_actions=5, seed=seed, n0=8, keep=0.5)
        return orig(kind, seed, q_init, eps0)

    D._agent = agent
    inst = B.load_instance(typ, iid)
    seed = int(iid)
    start = None
    rows = []
    for k in METHODS:
        ms, cpu, _ = D._solve(inst, k, seed)
        rows.append(dict(type=typ, instance=iid, n=n, m=m, method=k, makespan=int(ms), cpu=cpu))
    return rows


if __name__ == "__main__":
    import experiment.run_rcmax_benchmark_entrada137 as B
    man = pd.DataFrame(B.manifest("tune"))
    sel = man[man.m.isin(["10", "30", "50"]) & man.n.isin(["200", "500", "1000"])]
    jobs = [(r["type"], r["instance"], int(r["n"]), int(r["m"])) for _, r in sel.iterrows()]
    jobs.sort(key=lambda j: -j[2] * j[3])
    print(len(jobs), "instancias", flush=True)
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=11) as ex:
        for i, res in enumerate(ex.map(one, jobs), 1):
            rows += res
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    w = d.pivot_table(index=["type", "instance", "n", "m"], columns="method", values="makespan").reset_index()
    rel = lambda a, b: 100 * (a - b) / b
    fx = [f"fix_{k}" for k in range(5)]
    w["oracle"] = w[fx].min(axis=1)
    print("ganho sobre rand_d (d %):", {k: round(float(rel(w[k], w.rand_d).mean()), 3) for k in ["oracle"] + fx + ["ql_d", "ts_d", "ts_disc_d", "race_d"]})
    # braco validado por (tipo, n): argmin media no proprio ajuste (otimista)
    val = []
    for _, r in w.iterrows():
        g = w[(w["type"] == r["type"]) & (w.n == r["n"])]
        k = min(fx, key=lambda c: g[c].mean())
        val.append(r[k])
    w["val_opt"] = val
    print("val (otimista, ajuste = teste):", round(float(rel(w.val_opt, w.rand_d).mean()), 3))
    for k in ["ql_d", "ts_d", "ts_disc_d", "race_d"]:
        x = rel(w[k], w.rand_d)
        print(k, "melhor/pior", int((x < 0).sum()), int((x > 0).sum()), "por tipo:", w.assign(x=x).groupby("type").x.mean().round(3).to_dict())
