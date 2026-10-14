import sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Pesquisas\PMSP_Learnheuristic")

F = [0.01, 0.02, 0.05, 0.10, 0.20]
OUT = r"C:\Users\vinic\AppData\Local\Temp\claude\C--Pesquisas-PMSP-Learnheuristic\2014e0d0-2ae2-4c2c-9db8-6041232b2612\scratchpad\diag146.csv"


def one(args):
    typ, iid, n, m = args
    import experiment.run_rcmax_benchmark_entrada137 as B
    from pmsp.selectors import RandomSelector
    from pmsp.unrelated import _descent, ur_ig_run
    from pmsp.unrelated_lp import lp_start
    inst = B.load_instance(typ, iid)
    start, _, _ = lp_start(inst)
    _descent(start, 10 ** 6)
    seed = int(iid)

    class Rec(RandomSelector):
        def reset(self):
            super().reset()
            self.log = []

        def update(self, s, f, a, r, ns):
            self.log.append((a, r))
            super().update(s, f, a, r, ns)

    sel = Rec(seed=seed, n_actions=5)
    ur_ig_run(inst, sel, max_iter=1000, d=None, seed=seed, initial_solution=start, ls_steps=B.LS_STEPS, action_map=lambda a, nn: (0, max(2, round(F[a] * nn))))
    log = sel.log
    rows = []
    for lab, lo, hi in [("0-100", 0, 100), ("100-400", 100, 400), ("400-1000", 400, 1000)]:
        seg = log[lo:hi]
        for a in range(5):
            r = [x[1] for x in seg if x[0] == a]
            rows.append(dict(type=typ, instance=iid, n=n, m=m, window=lab, arm=a, mean_reward=float(np.mean(r)) if r else np.nan, pulls=len(r)))
    # qualidade final de cada braco fixo
    from pmsp.fixed_selector import FixedActionSelector
    for a in range(5):
        r = ur_ig_run(inst, FixedActionSelector(a), max_iter=1000, d=None, seed=seed, initial_solution=start, ls_steps=B.LS_STEPS, action_map=lambda aa, nn: (0, max(2, round(F[aa] * nn))))
        rows.append(dict(type=typ, instance=iid, n=n, m=m, window="final", arm=a, mean_reward=float(r["makespan"]), pulls=0))
    return rows


if __name__ == "__main__":
    import experiment.run_rcmax_benchmark_entrada137 as B
    man = pd.DataFrame(B.manifest("test"))
    sel = man[(man.idx == "2") & (man.m == "30") & man.n.isin(["200", "500", "1000"])]
    jobs = [(r["type"], r["instance"], int(r["n"]), int(r["m"])) for _, r in sel.iterrows()]
    print(len(jobs), "instancias", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=11) as ex:
        for res in ex.map(one, jobs):
            rows += res
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    fin = d[d.window == "final"].pivot_table(index=["type", "instance"], columns="arm", values="mean_reward")
    out = []
    for win in ["0-100", "100-400", "400-1000"]:
        rw = d[d.window == win].pivot_table(index=["type", "instance"], columns="arm", values="mean_reward")
        rho = [np.corrcoef(pd.Series(rw.loc[i].values).rank(), pd.Series(-fin.loc[i].values).rank())[0, 1] if rw.loc[i].notna().all() and fin.loc[i].nunique() > 1 else np.nan for i in rw.index]
        out.append((win, np.nanmean(rho), np.nanmean([rw.loc[i].values.argmax() == fin.loc[i].values.argmin() for i in rw.index])))
        print(win, "mean reward by arm:", rw.mean().round(3).to_dict())
    print("rank-corr(reward per pull, -final makespan) across arms, mean over instances:", [(w, round(r, 3), round(h, 3)) for w, r, h in out])
    print("final makespan relative to arm 2 by arm (mean %):", (100 * (fin.sub(fin[2], axis=0)).div(fin[2], axis=0)).mean().round(3).to_dict())
