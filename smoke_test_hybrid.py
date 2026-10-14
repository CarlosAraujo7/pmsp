import sys

sys.path.insert(0, ".")

import numpy as np
from pmsp.instance import PMSPInstance
from pmsp.qlearning import QLearningAgent
from pmsp.hybrid import hybrid_run, pretrain_hybrid_qtable, N_HYBRID_ACTIONS

print(f"N_HYBRID_ACTIONS = {N_HYBRID_ACTIONS}")

# --- teste rapido de pretreino (bem menor que o real: 2 rounds, max_iter=50) ---
Q = pretrain_hybrid_qtable(
    source_sizes=[(20, 4)],
    n_pretrain=2,
    max_iter=50,
    base_seed=0,
    verbose=True,
)
print(f"Q pretrain shape: {Q.shape}, nonzero: {np.count_nonzero(Q)}")

# --- instancia pequena sintetica (baixa variancia) ---
inst_small = PMSPInstance.random(n=20, m=4, low=1, high=100, seed=42)

agent_cold = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05,
    epsilon_decay=0.995, seed=42, n_actions=N_HYBRID_ACTIONS,
)
r_cold = hybrid_run(instance=inst_small, selector=agent_cold, max_iter=100, seed=42, stagnation_window=50)
print(f"cold-start (n=20,m=4): gap_pct={r_cold['gap_pct']:.4f} cpu_time={r_cold['cpu_time']:.3f}s "
      f"n_improvements={r_cold['n_improvements']}")

agent_ws = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon_start=0.5, epsilon_min=0.05,
    epsilon_decay=0.995, seed=42, n_actions=N_HYBRID_ACTIONS, q_init=Q,
)
r_ws = hybrid_run(instance=inst_small, selector=agent_ws, max_iter=100, seed=42, stagnation_window=50)
print(f"warm-start (n=20,m=4): gap_pct={r_ws['gap_pct']:.4f} cpu_time={r_ws['cpu_time']:.3f}s "
      f"n_improvements={r_ws['n_improvements']}")

# --- instancia maior, alta variancia, pra garantir que nao quebra em escala maior ---
inst_big = PMSPInstance.random(n=200, m=10, low=1, high=10**8, seed=7)
agent_cold2 = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_min=0.05,
    epsilon_decay=0.995, seed=7, n_actions=N_HYBRID_ACTIONS,
)
r_big = hybrid_run(instance=inst_big, selector=agent_cold2, max_iter=200, seed=7, stagnation_window=50)
print(f"cold-start (n=200,m=10,alta variancia): gap_pct={r_big['gap_pct']:.4f} cpu_time={r_big['cpu_time']:.3f}s "
      f"n_improvements={r_big['n_improvements']}")

print("\nSMOKE TEST OK -- nenhuma excecao levantada.")
