"""
experiment/run_large_class_pilot.py
======================================
Roda um lote de (metodo, replica) para uma classe grande nova (150-300
tarefas), gerando baselines do zero (a classe nao existe em
results.csv) + as 3 configs de granularidade calibradas por
harvest_large_class.py. Mesmo protocolo/fabrica de selectors do
results.csv oficial (experiment.runner.make_ils_selector).

Uso:
    python3 experiment/run_large_class_pilot.py N M LABEL REP_START REP_END METHODS OUT_SUFFIX

METHODS: separado por virgula -- "lpt", baselines (ils_random/ils_ucb/
ils_rf/ils_ql) ou "granularity:K" (K em 2,3,4).

Salva em results/large_pilot_{LABEL}_{OUT_SUFFIX}.json
"""
from __future__ import annotations
import sys
import json

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent
from pmsp.bounds import StateConfig
from pmsp.ils import ils_run, lpt_run
from experiment.runner import make_ils_selector
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW,
    QL_ALPHA, QL_GAMMA, QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
    BASE_SEED,
)


def build_state_config(label, n_bins):
    with open(f"results/harvested_feature_quantile_thresholds_{label}.json") as f:
        harvested = json.load(f)
    th = harvested["thresholds_by_nbins"][str(n_bins)]
    return StateConfig(bins_f1=th["bins_f1"], bins_f2=th["bins_f2"],
                        bins_f3=th["bins_f3"], bins_f4=th["bins_f4"])


def run_one(method_spec, n, m, rep, label):
    seed = BASE_SEED + rep
    inst = PMSPInstance.random(n, m, seed=seed)
    if method_spec == "lpt":
        return lpt_run(inst)["gap_pct"]
    if method_spec.startswith("granularity:"):
        n_bins = int(method_spec.split(":")[1])
        cfg = build_state_config(label, n_bins)
        agent = QLearningAgent(
            alpha=QL_ALPHA, gamma=QL_GAMMA, epsilon_start=QL_EPSILON_START,
            epsilon_min=QL_EPSILON_MIN, epsilon_decay=QL_EPSILON_DECAY,
            seed=seed, n_states=cfg.n_states,
        )
        res = ils_run(inst, agent, max_iter=MAX_ITER,
                       perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
                       seed=seed, stagnation_window=STAGNATION_WINDOW, state_cfg=cfg)
        return res["gap_pct"]
    selector = make_ils_selector(method_spec, seed)
    res = ils_run(inst, selector, max_iter=MAX_ITER,
                   perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
                   seed=seed, stagnation_window=STAGNATION_WINDOW)
    return res["gap_pct"]


def main():
    n, m, label, rep_start, rep_end, methods_csv, out_suffix = sys.argv[1:8]
    n, m, rep_start, rep_end = int(n), int(m), int(rep_start), int(rep_end)
    methods = methods_csv.split(",")
    out = {mspec: {} for mspec in methods}
    for rep in range(rep_start, rep_end):
        for mspec in methods:
            out[mspec][rep] = run_one(mspec, n, m, rep, label)
        if (rep - rep_start + 1) % 5 == 0:
            print(f"  rep {rep+1}/{rep_end} concluido")
    out_path = f"results/large_pilot_{label}_{out_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Salvo em {out_path} ({rep_end-rep_start} reps x {len(methods)} metodos)")


if __name__ == "__main__":
    main()
