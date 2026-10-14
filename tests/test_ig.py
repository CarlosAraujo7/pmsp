"""
tests/test_ig.py
Testes de sanidade do modulo pmsp/ig.py (Iterated Greedy + Q-Learning,
metodo ig_ql -- ver PLANO_EXPERIMENTAL_EXTENSIVO.md, Secao 1;
diario_pesquisa.md, Entrada 23).
Execute: pytest tests/test_ig.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.constructive import lpt
from pmsp.qlearning import QLearningAgent
from pmsp.ig import (
    apply_destroy_reconstruct, ig_run, N_DESTROY_STRATEGIES,
    DESTROY_STRATEGY_NAMES,
)


@pytest.fixture
def inst():
    return PMSPInstance.random(n=30, m=8, seed=42)


@pytest.fixture
def inst_small():
    return PMSPInstance.random(n=12, m=3, seed=1)


class TestDestroyReconstruct:

    def test_all_strategies_produce_valid_solution(self, inst):
        sol = lpt(inst)
        rng = np.random.default_rng(0)
        for strat in range(N_DESTROY_STRATEGIES):
            sol_new, delta = apply_destroy_reconstruct(strat, sol, d=5, rng=rng)
            assert sol_new.is_valid()

    def test_original_solution_not_mutated(self, inst):
        sol = lpt(inst)
        makespan_before = sol.makespan
        rng = np.random.default_rng(0)
        apply_destroy_reconstruct(0, sol, d=5, rng=rng)
        assert sol.makespan == makespan_before

    def test_invalid_strategy_id_raises(self, inst):
        sol = lpt(inst)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            apply_destroy_reconstruct(N_DESTROY_STRATEGIES, sol, d=3, rng=rng)
        with pytest.raises(ValueError):
            apply_destroy_reconstruct(-1, sol, d=3, rng=rng)

    def test_d_equals_n_reproduces_lpt_exactly(self, inst):
        # Regressao central (Entrada 23): destruir e reconstruir TODAS
        # as tarefas, na mesma ordem decrescente de tempo de
        # processamento que o LPT usa, deve reproduzir o LPT
        # EXATAMENTE (delta=0) -- essa e a assinatura de que a
        # reconstrucao guloso "zera" corretamente as cargas removidas
        # antes de comecar a reinserir (bug historico: nao zerava,
        # produzindo reconstrucoes sistematicamente piores que o LPT
        # mesmo destruindo a instancia inteira).
        sol = lpt(inst)
        for trial in range(5):
            rng = np.random.default_rng(trial)
            sol_new, delta = apply_destroy_reconstruct(0, sol, d=inst.n, rng=rng)
            assert delta == 0
            assert sol_new.makespan == sol.makespan

    def test_delta_can_be_positive(self, inst):
        # Com d moderado, deve ser possivel encontrar deltas positivos
        # (melhorias sobre o LPT) em pelo menos algumas tentativas --
        # se NUNCA acontecer, e sinal de que a reconstrucao esta com
        # algum problema sistematico (ver historico do bug acima).
        sol = lpt(inst)
        deltas = []
        for trial in range(30):
            rng = np.random.default_rng(trial)
            _, delta = apply_destroy_reconstruct(0, sol, d=max(2, round(0.3 * inst.n)), rng=rng)
            deltas.append(delta)
        assert max(deltas) > 0

    def test_destroy_strategy_names_length(self):
        assert len(DESTROY_STRATEGY_NAMES) == N_DESTROY_STRATEGIES

    def test_machine_concentrated_handles_small_machine(self, inst_small):
        # Maquina escolhida pode ter menos de d tarefas -- deve
        # completar de outras maquinas sem erro.
        sol = lpt(inst_small)
        rng = np.random.default_rng(0)
        sol_new, delta = apply_destroy_reconstruct(2, sol, d=8, rng=rng)
        assert sol_new.is_valid()


class TestIgRun:

    def test_returns_expected_keys(self, inst):
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res = ig_run(inst, agent, max_iter=100, seed=0)
        assert set(res.keys()) == {"makespan", "gap_pct", "cpu_time", "n_improvements", "history"}

    def test_makespan_never_worse_than_lpt(self, inst):
        # Criterio de aceitacao greedy (nao piora) garante que o melhor
        # encontrado nunca e pior que o ponto de partida (LPT).
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res = ig_run(inst, agent, max_iter=300, seed=0)
        assert res["makespan"] <= lpt(inst).makespan

    def test_makespan_ge_lower_bound(self, inst):
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res = ig_run(inst, agent, max_iter=300, seed=0)
        assert res["makespan"] >= inst.lower_bound

    def test_finds_improvements_over_many_iterations(self, inst):
        # Com o bug de reconstrucao corrigido, o IG deve encontrar
        # PELO MENOS uma melhoria sobre o LPT em 1000 iteracoes --
        # antes do conserto (Entrada 23), n_improvements ficava
        # travado em 0 para qualquer d testado.
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res = ig_run(inst, agent, max_iter=1000, seed=0)
        assert res["n_improvements"] > 0

    def test_q_table_shape_matches_n_destroy_strategies(self, inst):
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        ig_run(inst, agent, max_iter=50, seed=0)
        assert agent.Q.shape[1] == N_DESTROY_STRATEGIES

    def test_track_history_length(self, inst):
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res = ig_run(inst, agent, max_iter=80, seed=0, track_history=True)
        assert len(res["history"]) == 80

    def test_custom_d(self, inst):
        agent = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res = ig_run(inst, agent, max_iter=100, d=4, seed=0)
        assert res["makespan"] >= inst.lower_bound

    def test_reproducible_with_same_seed(self, inst):
        agent1 = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        agent2 = QLearningAgent(seed=0, n_actions=N_DESTROY_STRATEGIES)
        res1 = ig_run(inst, agent1, max_iter=200, seed=5)
        res2 = ig_run(inst, agent2, max_iter=200, seed=5)
        assert res1["makespan"] == res2["makespan"]
        assert res1["n_improvements"] == res2["n_improvements"]
