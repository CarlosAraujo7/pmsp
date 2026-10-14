"""
experiment/pilot_ql_traintest_v2.py
====================================
Extensao do piloto (experiment/pilot_ql_traintest.py, ver diario_pesquisa.md
Entrada 15): testa se avaliacao com epsilon residual (em vez de greedy puro)
e/ou mais episodios de treino mudam o resultado. Nao substitui o piloto v1
-- e uma investigacao adicional, aprovada por Carlos apos ver os resultados
do v1 (conserta o artefato de interpretabilidade mas nao fecha o gap vs.
random).

Grade testada, por classe: n_train_episodes in {30, 100} x eps_eval in {0.0, 0.05}.
Grava em results/pilot_ql_traintest_v2_results.json (isolado, nao toca results.csv).
"""
from __future__ import annotations
import sys
import json
import numpy as np

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent, N_STATES
from pmsp.ils import ils_run
from pmsp.selectors import N_OPERATORS
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW,
    QL_ALPHA, QL_GAMMA, QL_EPSILON_MIN, QL_EPSILON_DECAY, BASE_SEED,
    N_INSTANCES,
)

TRAIN_SEED_OFFSET = 100_000
Q_INIT_NOISE = 1e-3
TRAIN_EPISODE_GRID = [30, 100]
EPS_EVAL_GRID = [0.0, 0.05]


def train_ql_agent(n, m, n_train_episodes, max_iter=MAX_ITER):
    agent = QLearningAgent(
        alpha=QL_ALPHA, gamma=QL_GAMMA,
        epsilon_start=1.0, epsilon_min=QL_EPSILON_MIN, epsilon_decay=QL_EPSILON_DECAY,
        q_init_noise=Q_INIT_NOISE, seed=999,
    )
    coverage_history = []
    for ep in range(n_train_episodes):
        seed = TRAIN_SEED_OFFSET + ep
        train_instance = PMSPInstance.random(n, m, seed=seed)
        agent.carry_forward()
        ils_run(
            train_instance, agent, max_iter=max_iter,
            perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
            seed=seed, stagnation_window=STAGNATION_WINDOW,
        )
        coverage_history.append(int((agent.visits.sum(axis=1) > 0).sum()))
    return agent, coverage_history


def evaluate_frozen_policy(agent_Q, n, m, eps_eval, n_test=N_INSTANCES, max_iter=MAX_ITER):
    eval_agent = QLearningAgent(
        alpha=0.0, gamma=QL_GAMMA,
        epsilon_start=eps_eval, epsilon_min=eps_eval, epsilon_decay=1.0,
        q_init=agent_Q, seed=42,
    )
    gaps = []
    for i in range(n_test):
        inst = PMSPInstance.random(n, m, seed=BASE_SEED + i)
        res = ils_run(
            inst, eval_agent, max_iter=max_iter,
            perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
            seed=BASE_SEED + i, stagnation_window=STAGNATION_WINDOW,
        )
        gaps.append(res["gap_pct"])
    return np.array(gaps)


def run_pilot_v2(n, m, label):
    print(f"\n=== Piloto v2: classe {label} (n={n}, m={m}) ===")
    class_results = {"label": label, "n": n, "m": m, "configs": []}

    for n_train in TRAIN_EPISODE_GRID:
        agent, cov_hist = train_ql_agent(n, m, n_train)
        final_cov = cov_hist[-1]
        argmax = agent.Q.argmax(axis=1)
        dom_counts = {int(a): int((argmax == a).sum()) for a in range(N_OPERATORS)}
        print(f"  [treino={n_train} eps] cobertura final: {final_cov}/{N_STATES} "
              f"({100*final_cov/N_STATES:.1f}%) | historico(a cada 10): "
              f"{cov_hist[9::10]}")
        print(f"    distribuicao argmax: {dom_counts}")

        for eps_eval in EPS_EVAL_GRID:
            gaps = evaluate_frozen_policy(agent.Q, n, m, eps_eval)
            print(f"    [avaliacao eps={eps_eval}] gap_pct: media={gaps.mean():.4f} "
                  f"desvio={gaps.std():.4f} mediana={np.median(gaps):.4f}")
            class_results["configs"].append({
                "n_train_episodes": n_train, "eps_eval": eps_eval,
                "coverage_final": final_cov, "coverage_pct": round(100*final_cov/N_STATES, 1),
                "coverage_history_every10": cov_hist[9::10],
                "dominance_counts": dom_counts,
                "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
                "gap_median": float(np.median(gaps)),
            })
    return class_results


if __name__ == "__main__":
    results = [run_pilot_v2(20, 4, "rand_20j_4m"), run_pilot_v2(50, 10, "rand_50j_10m")]
    with open("results/pilot_ql_traintest_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em results/pilot_ql_traintest_v2_results.json")

    print("\n=== Referencia: numeros ja existentes em results.csv para estas classes ===")
    print("rand_20j_4m:  ils_ql=0.053  ils_random=0.031  ils_op1_only=0.090  lpt=1.955")
    print("rand_50j_10m: ils_ql=0.641  ils_random=0.700  ils_op1_only=0.626  ils_rf=0.712  ils_ucb=1.158")
