"""
experiment/pilot_ql_traintest.py
=================================
Piloto: testa se separar treino (multiplos episodios, Q-table acumulada
via carry_forward()) de avaliacao (politica congelada, alpha=0) melhora
a cobertura de estados e muda o resultado, frente ao diagnostico de
sub-treinamento em diario_pesquisa.md, Entrada 14.

Nao faz parte do pipeline padrao (run_experiment.py). E um piloto,
rodado manualmente em 1-2 classes pequenas (a pedido de Carlos), para
decidir se vale a pena expandir. Nao escreve em results/results.csv --
grava em results/pilot_ql_traintest_results.json, isolado.
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

TRAIN_SEED_OFFSET = 100_000  # bem afastado dos seeds de avaliacao (BASE_SEED..BASE_SEED+29)
N_TRAIN_EPISODES = 30
Q_INIT_NOISE = 1e-3


def train_ql_agent(n, m, n_train_episodes=N_TRAIN_EPISODES, max_iter=MAX_ITER, verbose=True):
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
        cov = int((agent.visits.sum(axis=1) > 0).sum())
        coverage_history.append(cov)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}/{n_train_episodes}: cobertura acumulada = {cov}/{N_STATES}")
    return agent, coverage_history


def evaluate_frozen_policy(agent_Q, n, m, n_test=N_INSTANCES, max_iter=MAX_ITER, eps_eval=0.0):
    eval_agent = QLearningAgent(
        alpha=0.0, gamma=QL_GAMMA,
        epsilon_start=eps_eval, epsilon_min=eps_eval, epsilon_decay=1.0,
        q_init=agent_Q, seed=42,
    )
    gaps = []
    for i in range(n_test):
        inst = PMSPInstance.random(n, m, seed=BASE_SEED + i)  # MESMAS instancias de results.csv
        res = ils_run(
            inst, eval_agent, max_iter=max_iter,
            perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
            seed=BASE_SEED + i, stagnation_window=STAGNATION_WINDOW,
        )
        gaps.append(res["gap_pct"])
    return np.array(gaps)


def run_pilot(n, m, label):
    print(f"\n=== Piloto: classe {label} (n={n}, m={m}) ===")
    agent, cov_hist = train_ql_agent(n, m)
    final_cov = cov_hist[-1]
    print(f"  Cobertura final apos {N_TRAIN_EPISODES} episodios de treino: "
          f"{final_cov}/{N_STATES} ({100*final_cov/N_STATES:.1f}%)")

    argmax = agent.Q.argmax(axis=1)
    dom_counts = {int(a): int((argmax == a).sum()) for a in range(N_OPERATORS)}
    print(f"  Distribuicao de acao vencedora (81 estados), pos-treino: {dom_counts}")

    gaps = evaluate_frozen_policy(agent.Q, n, m)
    print(f"  ils_ql_trained -- gap_pct: media={gaps.mean():.4f} "
          f"desvio={gaps.std():.4f} mediana={np.median(gaps):.4f} "
          f"min={gaps.min():.4f} max={gaps.max():.4f}")

    return {
        "label": label, "n": n, "m": m,
        "coverage_final": final_cov, "coverage_pct": round(100*final_cov/N_STATES, 1),
        "coverage_history": cov_hist,
        "dominance_counts": dom_counts,
        "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
        "gap_median": float(np.median(gaps)),
        "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
        "gaps": [float(g) for g in gaps],
    }


if __name__ == "__main__":
    results = []
    results.append(run_pilot(20, 4, "rand_20j_4m"))
    results.append(run_pilot(50, 10, "rand_50j_10m"))

    with open("results/pilot_ql_traintest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em results/pilot_ql_traintest_results.json")
