"""
experiment/run_hyperparam_sensitivity_entrada105.py
=====================================================
WP5 do plano de revisao (diario, Entrada 105): sensibilidade do IG-QL e do controle ig_random aos hiperparametros
(nunca ajustados por classe). Uma-variavel-por-vez em torno do padrao (d = 0,3 n; alpha = 0,1; gamma = 0,9;
epsilon_decay = 0,995):
  fracao de destruicao d/n   in {0,1; 0,2; 0,4}    -> ig_ql_dXX e ig_random_dXX (controle com o MESMO d)
  alpha                      in {0,05; 0,3}        -> ig_ql_a05, ig_ql_a30
  gamma                      in {0,5; 0,99}        -> ig_ql_g50, ig_ql_g99
  decaimento de epsilon      in {0,99; 0,999}      -> ig_ql_e990, ig_ql_e999
6 classes (20,6), (50,10), (75,15), (100,20), (150,30), (300,20) x 50 repeticoes (rep 0-49), T=1000, partida LPT,
estado padrao; mesmas instancias/seeds de results.csv (instancia = PMSPInstance.random(seed=BASE_SEED+rep)).
Saida: results/hyperparam_sensitivity_entrada105.csv (retomavel). Uso:
    python -m experiment.run_hyperparam_sensitivity_entrada105 --workers 12 [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = [(20, 6), (50, 10), (75, 15), (100, 20), (150, 30), (300, 20)]
N_REPS = 50
OUT = ROOT / "results" / "hyperparam_sensitivity_entrada105.csv"
FIELDS = ["instance", "method", "rep", "n", "m", "lb", "makespan", "gap_pct", "cpu_time", "n_improvements"]
# nome -> (seletor, fracao_d, alpha, gamma, eps_decay)
CONFIGS = {
    "ig_ql_d10": ("ql", 0.1, 0.1, 0.9, 0.995), "ig_random_d10": ("rand", 0.1, None, None, None),
    "ig_ql_d20": ("ql", 0.2, 0.1, 0.9, 0.995), "ig_random_d20": ("rand", 0.2, None, None, None),
    "ig_ql_d40": ("ql", 0.4, 0.1, 0.9, 0.995), "ig_random_d40": ("rand", 0.4, None, None, None),
    "ig_ql_a05": ("ql", 0.3, 0.05, 0.9, 0.995), "ig_ql_a30": ("ql", 0.3, 0.3, 0.9, 0.995),
    "ig_ql_g50": ("ql", 0.3, 0.1, 0.5, 0.995), "ig_ql_g99": ("ql", 0.3, 0.1, 0.99, 0.995),
    "ig_ql_e990": ("ql", 0.3, 0.1, 0.9, 0.99), "ig_ql_e999": ("ql", 0.3, 0.1, 0.9, 0.999),
}


def _run(task: dict) -> dict:
    from experiment import config as CFG
    from pmsp.ig import N_DESTROY_STRATEGIES, ig_run
    from pmsp.instance import PMSPInstance
    from pmsp.qlearning import QLearningAgent
    from pmsp.selectors import RandomSelector

    n, m, rep, method = task["n"], task["m"], task["rep"], task["method"]
    kind, frac, a, g, e = CONFIGS[method]
    inst = PMSPInstance.random(n=n, m=m, seed=CFG.BASE_SEED + rep, name=f"rand_{n}j_{m}m")
    seed = CFG.BASE_SEED + rep
    if kind == "ql":
        sel = QLearningAgent(alpha=a, gamma=g, epsilon_start=CFG.QL_EPSILON_START, epsilon_min=CFG.QL_EPSILON_MIN,
                             epsilon_decay=e, seed=seed, n_actions=N_DESTROY_STRATEGIES)
    else:
        sel = RandomSelector(seed=seed, n_actions=N_DESTROY_STRATEGIES)
    r = ig_run(instance=inst, selector=sel, max_iter=1000, d=max(2, round(frac * n)), seed=seed,
               stagnation_window=CFG.STAGNATION_WINDOW)
    return dict(instance=f"rand_{n}j_{m}m", method=method, rep=rep, n=n, m=m, lb=inst.lower_bound,
                makespan=r["makespan"], gap_pct=r["gap_pct"], cpu_time=r["cpu_time"], n_improvements=r["n_improvements"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:    # config padrao reproduz ig_ql/ig_random de results.csv?
        import pandas as pd
        from experiment import config as CFG
        ref = pd.read_csv(ROOT / "results" / "results.csv")
        CONFIGS["_ql_default"] = ("ql", 0.3, 0.1, 0.9, 0.995)
        CONFIGS["_rand_default"] = ("rand", 0.3, None, None, None)
        ok = tot = 0
        for n, m in [(20, 6), (50, 10), (100, 20)]:
            for rep in range(4):
                for mine, theirs in [("_ql_default", "ig_ql"), ("_rand_default", "ig_random")]:
                    mk = _run(dict(n=n, m=m, rep=rep, method=mine))["makespan"]
                    row = ref[(ref.instance == f"rand_{n}j_{m}m") & (ref.rep == rep) & (ref.method == theirs)]
                    tot += 1
                    ok += int(int(row.makespan.iloc[0]) == mk)
        print(f"[check] {ok}/{tot} identicos a results.csv")
        return
    classes = CLASSES[:1] if args.smoke else CLASSES
    out = OUT.with_name("hyperparam_sensitivity_entrada105_SMOKE.csv") if args.smoke else OUT
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["instance"], r["method"], r["rep"]) for r in csv.DictReader(f)}
    tasks = [dict(n=n, m=m, rep=rep, method=mt) for n, m in classes for rep in range(2 if args.smoke else N_REPS)
             for mt in CONFIGS if (f"rand_{n}j_{m}m", mt, str(rep)) not in done]
    print(f"[tasks] {len(tasks)} pendentes ({len(done)} feitas)")
    new = not out.exists()
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                except Exception as exc:  # noqa: BLE001
                    print("[ERRO]", futs[fut], exc)
                    continue
                f.flush()
                os.fsync(f.fileno())
                if i % 200 == 0 or i == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"  [{i}/{len(tasks)}] {el:.0f}s, ETA {(len(tasks) - i) * el / i / 60:.1f} min", flush=True)
    print("concluido ->", out)


if __name__ == "__main__":
    main()
