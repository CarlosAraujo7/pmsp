"""
experiment/diag_cumulative_coverage.py
=======================================
Diagnostico complementar ao piloto (pilot_ql_traintest_v2.py).

Pergunta: agent.visits e zerado a cada reset() (chamado por ils_run()),
e carry_forward() preserva apenas Q, nao visits. Logo "coverage_final"
no piloto v1/v2 mede cobertura do ULTIMO episodio apenas, nao a UNIAO
cumulativa de estados visitados ao longo de todos os episodios de
treino. Este script mede a uniao cumulativa real, com checkpoints,
para saber se o "teto" de ~18-24/81 estados e so por episodio ou
tambem vale para a uniao ao longo do treino.

Nao escreve em results/results.csv. Diagnostico isolado, complementar
a Entrada 14/15/16 do diario.
"""
from __future__ import annotations
import sys
import json
import numpy as np

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent, N_STATES
from pmsp.ils import ils_run
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW,
    QL_ALPHA, QL_GAMMA, QL_EPSILON_MIN, QL_EPSILON_DECAY,
)

TRAIN_SEED_OFFSET = 100_000
N_EPISODES = 100
CHECKPOINTS = {10, 20, 30, 50, 75, 100}
Q_INIT_NOISE = 1e-3


def run(n, m, label):
    agent = QLearningAgent(
        alpha=QL_ALPHA, gamma=QL_GAMMA,
        epsilon_start=1.0, epsilon_min=QL_EPSILON_MIN, epsilon_decay=QL_EPSILON_DECAY,
        q_init_noise=Q_INIT_NOISE, seed=999,
    )
    union_visited = np.zeros(N_STATES, dtype=bool)
    per_episode_coverage = []
    cumulative_checkpoints = {}

    for ep in range(N_EPISODES):
        seed = TRAIN_SEED_OFFSET + ep
        inst = PMSPInstance.random(n, m, seed=seed)
        agent.carry_forward()
        ils_run(
            inst, agent, max_iter=MAX_ITER,
            perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
            seed=seed, stagnation_window=STAGNATION_WINDOW,
        )
        episode_visited = agent.visits.sum(axis=1) > 0
        per_episode_coverage.append(int(episode_visited.sum()))
        union_visited |= episode_visited
        if (ep + 1) in CHECKPOINTS:
            cumulative_checkpoints[ep + 1] = int(union_visited.sum())

    result = {
        "label": label, "n": n, "m": m,
        "per_episode_coverage": per_episode_coverage,
        "per_episode_mean": float(np.mean(per_episode_coverage)),
        "per_episode_min": int(np.min(per_episode_coverage)),
        "per_episode_max": int(np.max(per_episode_coverage)),
        "cumulative_union_checkpoints": cumulative_checkpoints,
        "cumulative_union_final": int(union_visited.sum()),
        "cumulative_union_final_pct": round(100 * int(union_visited.sum()) / N_STATES, 1),
        "never_visited_states": [int(s) for s in np.where(~union_visited)[0]],
    }
    out_path = f"results/diag_cumulative_coverage_{label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"=== {label} (n={n}, m={m}) ===")
    print(f"  Cobertura por episodio: media={result['per_episode_mean']:.1f} "
          f"min={result['per_episode_min']} max={result['per_episode_max']} (de {N_STATES})")
    print(f"  Uniao cumulativa nos checkpoints: {cumulative_checkpoints}")
    print(f"  Uniao cumulativa final (100 ep): {result['cumulative_union_final']}/{N_STATES} "
          f"({result['cumulative_union_final_pct']}%)")
    print(f"  Salvo em {out_path}")


if __name__ == "__main__":
    n, m, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    run(n, m, label)
