"""
experiment/pilot_state_granularity.py
========================================
Piloto: estudo de sensibilidade de granularidade de estado (81 vs. 16
vs. 256 estados), Entrada 16/17, diario_pesquisa.md.

Usa os limiares de QUANTIS EMPIRICOS calibrados por
harvest_feature_distribution.py (results/harvested_feature_quantile_thresholds.json)
para as 3 granularidades (n_bins=2/3/4 -> 16/81/256 estados), aplicados
com o MESMO protocolo do ils_ql ja publicado: 30 replicas independentes
por classe, cada uma com um agente Q-Learning treinado do ZERO
(cold-start, sem carry_forward), nas MESMAS 30 instancias/seeds de teste
de results.csv (BASE_SEED+i) -- comparacao direta e justa com os
baselines ja conhecidos (ils_ql, ils_random, ils_rf, ils_ucb, lpt).

Nao integrado ao pipeline principal. Nao escreve em results.csv.
Escopo: as 2 classes ja aprovadas para o piloto (rand_20j_4m, rand_50j_10m).
"""
from __future__ import annotations
import sys
import json
import numpy as np

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent
from pmsp.bounds import StateConfig
from pmsp.ils import ils_run
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW,
    QL_ALPHA, QL_GAMMA, QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
    BASE_SEED, N_INSTANCES,
)

CLASSES = [(20, 4, "rand_20j_4m"), (50, 10, "rand_50j_10m")]
N_BINS_GRID = [2, 3, 4]

with open("results/harvested_feature_quantile_thresholds.json") as f:
    HARVESTED = json.load(f)


def build_state_config(n_bins: int) -> StateConfig:
    th = HARVESTED["thresholds_by_nbins"][str(n_bins)]
    return StateConfig(
        bins_f1=th["bins_f1"], bins_f2=th["bins_f2"],
        bins_f3=th["bins_f3"], bins_f4=th["bins_f4"],
    )


def run_arm(n, m, label, n_bins, cfg):
    n_states = cfg.n_states
    gaps, coverages = [], []
    for i in range(N_INSTANCES):
        seed = BASE_SEED + i
        inst = PMSPInstance.random(n, m, seed=seed)
        agent = QLearningAgent(
            alpha=QL_ALPHA, gamma=QL_GAMMA, epsilon_start=QL_EPSILON_START,
            epsilon_min=QL_EPSILON_MIN, epsilon_decay=QL_EPSILON_DECAY,
            seed=seed, n_states=n_states,
        )
        res = ils_run(
            inst, agent, max_iter=MAX_ITER,
            perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
            seed=seed, stagnation_window=STAGNATION_WINDOW, state_cfg=cfg,
        )
        gaps.append(res["gap_pct"])
        cov = int((agent.visits.sum(axis=1) > 0).sum())
        coverages.append(cov)
    gaps = np.array(gaps)
    coverages = np.array(coverages)
    return {
        "label": label, "n": n, "m": m, "n_bins": n_bins, "n_states": n_states,
        "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
        "gap_median": float(np.median(gaps)),
        "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
        "coverage_mean": float(coverages.mean()),
        "coverage_pct_mean": round(100 * float(coverages.mean()) / n_states, 1),
        "gaps": [float(g) for g in gaps],
    }


def main():
    results = []
    for n, m, label in CLASSES:
        print(f"\n=== Classe {label} (n={n}, m={m}) ===")
        for n_bins in N_BINS_GRID:
            cfg = build_state_config(n_bins)
            r = run_arm(n, m, label, n_bins, cfg)
            results.append(r)
            print(f"  n_bins={n_bins} (|S|={r['n_states']:3d}): "
                  f"gap_mean={r['gap_mean']:.4f} gap_std={r['gap_std']:.4f} "
                  f"gap_median={r['gap_median']:.4f} | "
                  f"cobertura_media={r['coverage_mean']:.1f}/{r['n_states']} "
                  f"({r['coverage_pct_mean']}%)")

    with open("results/pilot_state_granularity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em results/pilot_state_granularity_results.json")


if __name__ == "__main__":
    main()
