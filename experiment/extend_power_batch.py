"""
experiment/extend_power_batch.py
===================================
Worker generico para estender o piloto de granularidade (Entrada 17) a
mais replicas (seeds 30..99), para ganhar poder estatistico no teste de
Wilcoxon pareado. Roda um subconjunto (classe, faixa de reps, metodos)
por chamada -- pensado para ser chamado varias vezes em paralelo/serie
e depois agregado por merge_power_results.py.

Uso:
    python3 experiment/extend_power_batch.py N M LABEL REP_START REP_END METHODS OUT_SUFFIX

METHODS: string separada por virgula, cada item e um baseline
(ils_random/ils_ucb/ils_rf/ils_ql) ou "granularity:K" (K em {2,3,4}).

Salva em results/power_ext_{LABEL}_{OUT_SUFFIX}.json
"""
from __future__ import annotations
import sys
import json

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent
from pmsp.bounds import StateConfig
from pmsp.ils import ils_run
from experiment.runner import make_ils_selector
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW,
    QL_ALPHA, QL_GAMMA, QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
    BASE_SEED,
)

with open("results/harvested_feature_quantile_thresholds.json") as f:
    HARVESTED = json.load(f)


def build_state_config(n_bins: int) -> StateConfig:
    th = HARVESTED["thresholds_by_nbins"][str(n_bins)]
    return StateConfig(bins_f1=th["bins_f1"], bins_f2=th["bins_f2"],
                        bins_f3=th["bins_f3"], bins_f4=th["bins_f4"])


def run_one(method_spec, n, m, rep):
    seed = BASE_SEED + rep
    inst = PMSPInstance.random(n, m, seed=seed)
    if method_spec.startswith("granularity:"):
        n_bins = int(method_spec.split(":")[1])
        cfg = build_state_config(n_bins)
        agent = QLearningAgent(
            alpha=QL_ALPHA, gamma=QL_GAMMA, epsilon_start=QL_EPSILON_START,
            epsilon_min=QL_EPSILON_MIN, epsilon_decay=QL_EPSILON_DECAY,
            seed=seed, n_states=cfg.n_states,
        )
        res = ils_run(inst, agent, max_iter=MAX_ITER,
                       perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
                       seed=seed, stagnation_window=STAGNATION_WINDOW, state_cfg=cfg)
    else:
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
            gap = run_one(mspec, n, m, rep)
            out[mspec][rep] = gap
        if (rep - rep_start + 1) % 10 == 0:
            print(f"  rep {rep+1}/{rep_end} concluido")

    out_path = f"results/power_ext_{label}_{out_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Salvo em {out_path} ({rep_end - rep_start} reps x {len(methods)} metodos)")


if __name__ == "__main__":
    main()
