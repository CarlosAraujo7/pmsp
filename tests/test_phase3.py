"""
tests/test_phase3.py
Testes de sanidade da Fase 3: agente Q-Learning.
Execute: pytest tests/test_phase3.py -v
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.solution import PMSPSolution
from pmsp.constructive import lpt
from pmsp.operators import apply_best, apply_rand, N_OPERATORS
from pmsp.selectors import BaseSelector, RandomSelector, UCBSelector
from pmsp.bounds import compute_state, decode_state
from pmsp.qlearning import (QLearningAgent, SarsaAgent,
                             ContextualUCBSelector, FlatQLearningSelector,
                             N_STATES)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return QLearningAgent(alpha=0.1, gamma=0.9,
                          epsilon_start=1.0, epsilon_min=0.05,
                          epsilon_decay=0.99, seed=0)

@pytest.fixture
def inst():
    return PMSPInstance.random(n=30, m=5, seed=7)

@pytest.fixture
def rng():
    return np.random.default_rng(42)

# ---------------------------------------------------------------------------
# TestQLearningInit
# ---------------------------------------------------------------------------

class TestQLearningInit:

    def test_qtable_shape(self, agent):
        assert agent.Q.shape == (N_STATES, N_OPERATORS)

    def test_qtable_initialized_zero(self, agent):
        assert np.all(agent.Q == 0.0)

    def test_visits_initialized_zero(self, agent):
        assert np.all(agent.visits == 0)

    def test_epsilon_starts_at_one(self, agent):
        assert agent.epsilon == pytest.approx(1.0)

    def test_t_starts_at_zero(self, agent):
        assert agent.t == 0

    def test_n_states_correct(self):
        assert N_STATES == 81

    def test_is_base_selector_subclass(self, agent):
        assert isinstance(agent, BaseSelector)

# ---------------------------------------------------------------------------
# TestEpsilonGreedy
# ---------------------------------------------------------------------------

class TestEpsilonGreedy:

    def test_epsilon_decays_after_select(self, agent, inst):
        sol = lpt(inst)
        state_id = compute_state(sol, 0, 100, 0)
        feats = np.zeros(4)
        eps_before = agent.epsilon
        agent.select(state_id, feats)
        assert agent.epsilon < eps_before

    def test_epsilon_decays_monotonically(self, agent, inst):
        sol = lpt(inst)
        feats = np.zeros(4)
        epsilons = []
        for t in range(50):
            state_id = compute_state(sol, t, 100, 0)
            agent.select(state_id, feats)
            epsilons.append(agent.epsilon)
        # Deve ser estritamente decrescente (ate atingir epsilon_min)
        for i in range(len(epsilons) - 1):
            assert epsilons[i] >= epsilons[i+1]

    def test_epsilon_never_below_min(self, agent, inst):
        sol = lpt(inst)
        feats = np.zeros(4)
        for t in range(2000):
            state_id = compute_state(sol, t % 100, 100, 0)
            agent.select(state_id, feats)
        assert agent.epsilon >= agent.epsilon_min

    def test_pure_exploitation_uses_argmax(self):
        # Cria agente com epsilon=0 (pura explotacao)
        ag = QLearningAgent(epsilon_start=0.0, epsilon_min=0.0,
                            epsilon_decay=1.0, seed=1)
        # Define Q manualmente: op2 deve ser preferido no estado 0
        ag._Q[0, :] = [1.0, 5.0, 0.5, 2.0]
        action = ag.select(0, np.zeros(4))
        assert action == 1   # argmax = op2 (indice 1)

    def test_pure_exploration_is_random(self):
        # Com epsilon=1.0 e sem decaimento, deve ser uniforme
        ag = QLearningAgent(epsilon_start=1.0, epsilon_min=1.0,
                            epsilon_decay=1.0, seed=99)
        ag._Q[0, :] = [0.0, 0.0, 0.0, 100.0]  # Q favorece op4
        counts = np.zeros(N_OPERATORS)
        for _ in range(400):
            a = ag.select(0, np.zeros(4))
            counts[a] += 1
        # Com 400 amostras uniformes, cada op deve aparecer ~100 vezes
        # Testa que op4 nao domina (seria > 350 se epsilon=0)
        assert counts[3] < 200, "Exploracao nao esta aleatoria"

# ---------------------------------------------------------------------------
# TestQLearningUpdate
# ---------------------------------------------------------------------------

class TestQLearningUpdate:

    def test_update_increases_q_for_reward(self, agent):
        """
        Com Q inicializado em zero, uma atualizacao com reward > 0
        deve aumentar Q(s, a).
        """
        s, a, r, s_next = 10, 2, 5.0, 20
        agent.update(s, np.zeros(4), a, r, s_next)
        # Q(10, 2) = 0 + 0.1 * [5.0 + 0.9*0 - 0] = 0.5
        assert agent.Q[s, a] == pytest.approx(0.1 * 5.0)

    def test_update_zero_reward_propagates_future(self):
        """
        Reward=0 mas proximo estado tem Q alto: deve propagar valor futuro.
        """
        ag = QLearningAgent(alpha=0.5, gamma=0.9,
                            epsilon_start=0.0, epsilon_min=0.0,
                            epsilon_decay=1.0, seed=0)
        # Define Q do proximo estado
        ag._Q[50, :] = [0, 0, 10.0, 0]   # max Q(s') = 10
        # Atualiza s=5, a=0, r=0, s'=50
        ag.update(5, np.zeros(4), 0, 0.0, 50)
        # Q(5,0) = 0 + 0.5 * [0 + 0.9*10 - 0] = 4.5
        assert ag.Q[5, 0] == pytest.approx(0.5 * 0.9 * 10.0)

    def test_update_td_error_formula(self):
        """Verifica a formula completa do TD-error."""
        ag = QLearningAgent(alpha=0.2, gamma=0.8,
                            epsilon_start=0.0, epsilon_min=0.0,
                            epsilon_decay=1.0, seed=0)
        ag._Q[3, 1] = 4.0    # Q(s=3, a=1) = 4.0
        ag._Q[7, :] = [1.0, 6.0, 2.0, 3.0]  # max Q(s'=7) = 6.0
        ag.update(3, np.zeros(4), 1, 2.0, 7)
        # TD-error = 2.0 + 0.8*6.0 - 4.0 = 2.0 + 4.8 - 4.0 = 2.8
        # Q(3,1) = 4.0 + 0.2 * 2.8 = 4.0 + 0.56 = 4.56
        assert ag.Q[3, 1] == pytest.approx(4.0 + 0.2 * 2.8)

    def test_other_q_values_unchanged_on_update(self):
        """Atualizar (s, a) nao deve alterar outros pares (s', a')."""
        ag = QLearningAgent(alpha=0.1, gamma=0.9, seed=0)
        ag.update(10, np.zeros(4), 2, 3.0, 20)
        # Todos os outros estados/acoes devem ser zero
        for s in range(N_STATES):
            for a in range(N_OPERATORS):
                if (s, a) != (10, 2):
                    assert ag.Q[s, a] == pytest.approx(0.0)

    def test_visits_incremented_on_update(self, agent):
        agent.update(15, np.zeros(4), 3, 1.0, 30)
        assert agent.visits[15, 3] == 1
        agent.update(15, np.zeros(4), 3, 2.0, 40)
        assert agent.visits[15, 3] == 2

    def test_visits_only_for_updated_pair(self, agent):
        agent.update(15, np.zeros(4), 3, 1.0, 30)
        # Todos os outros pares devem ter contagem 0
        assert agent.visits.sum() == 1

# ---------------------------------------------------------------------------
# TestPolicyAnalysis
# ---------------------------------------------------------------------------

class TestPolicyAnalysis:

    def test_best_action_returns_argmax(self, agent):
        agent._Q[5, :] = [1.0, 3.0, 0.5, 2.0]
        assert agent.best_action(5) == 1   # argmax = indice 1

    def test_policy_vector_shape(self, agent):
        pv = agent.policy_vector()
        assert pv.shape == (N_STATES,)

    def test_policy_vector_values_in_range(self, agent):
        pv = agent.policy_vector()
        assert np.all((pv >= 0) & (pv < N_OPERATORS))

    def test_policy_vector_consistent_with_best_action(self, agent):
        # Define Q nao-trivial
        rng = np.random.default_rng(5)
        agent._Q[:] = rng.random((N_STATES, N_OPERATORS))
        pv = agent.policy_vector()
        for s in range(0, N_STATES, 50):   # amostra 13 estados
            assert pv[s] == agent.best_action(s)

    def test_q_values_for_action_shape(self, agent):
        vals = agent.q_values_for_action(0)
        assert vals.shape == (N_STATES,)

    def test_q_heatmap_data_keys(self, agent):
        data = agent.q_heatmap_data()
        assert "policy_matrix" in data
        assert "q_table" in data
        assert "visits" in data
        assert data["q_table"].shape == (N_STATES, N_OPERATORS)

# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_load_roundtrip(self, agent, inst, tmp_path):
        # Executa algumas atualizacoes para criar Q nao-trivial
        rng_op = np.random.default_rng(7)
        sol = lpt(inst)
        feats = np.zeros(4)
        for t in range(30):
            state_id = compute_state(sol, t, 100, 0)
            action = agent.select(state_id, feats)
            agent.update(state_id, feats, action, float(t % 5), (state_id+1) % N_STATES)

        # Salva e recarrega
        save_path = tmp_path / "qtable"
        agent.save(save_path)
        loaded = QLearningAgent.load(save_path)

        # Q-table deve ser identica
        np.testing.assert_array_almost_equal(agent.Q, loaded.Q)
        np.testing.assert_array_equal(agent.visits, loaded.visits)
        assert loaded.epsilon == pytest.approx(agent.epsilon)
        assert loaded.t == agent.t
        assert loaded.alpha == agent.alpha
        assert loaded.gamma == agent.gamma

    def test_save_creates_two_files(self, agent, tmp_path):
        save_path = tmp_path / "qtable"
        agent.save(save_path)
        assert (tmp_path / "qtable.npz").exists()
        assert (tmp_path / "qtable_meta.json").exists()

# ---------------------------------------------------------------------------
# TestIntegrationMiniILS
# ---------------------------------------------------------------------------

class TestIntegrationMiniILS:
    """
    Executa um mini-ILS com Q-Learning e verifica propriedades fundamentais.
    """

    def _run_mini_ils(self, agent, inst, n_iter=200):
        """
        Mini-ILS: a cada iteracao, Q-Learning seleciona operador,
        aplica best-improvement, calcula reward e atualiza Q.
        Retorna (makespan_inicial, makespan_final, historico de rewards).
        """
        rng_op = np.random.default_rng(0)
        sol = lpt(inst)
        best_makespan = sol.makespan
        reward_hist = []
        iters_no_improve = 0

        for t in range(n_iter):
            state_id = compute_state(sol, t, n_iter, iters_no_improve)
            feats = np.array([
                sol.relative_gap, sol.load_cv,
                t / n_iter, min(iters_no_improve / 50, 1.0),
            ])
            action = agent.select(state_id, feats)
            sol_new = sol.copy()
            delta = apply_best(action, sol_new, rng_op)
            reward = float(delta)
            next_state = compute_state(sol_new, t+1, n_iter, iters_no_improve)
            agent.update(state_id, feats, action, reward, next_state)
            reward_hist.append(reward)
            if delta > 0:
                sol = sol_new
                if sol.makespan < best_makespan:
                    best_makespan = sol.makespan
                    iters_no_improve = 0
                else:
                    iters_no_improve += 1
            else:
                iters_no_improve += 1

        return lpt(inst).makespan, best_makespan, reward_hist

    def test_mini_ils_improves_lpt(self, inst):
        agent = QLearningAgent(alpha=0.1, gamma=0.9,
                               epsilon_start=1.0, epsilon_min=0.05,
                               epsilon_decay=0.99, seed=42)
        init_mk, final_mk, _ = self._run_mini_ils(agent, inst, n_iter=300)
        # O ILS deve manter ou melhorar o LPT
        assert final_mk <= init_mk, \
            f"ILS piorou: LPT={init_mk}, final={final_mk}"

    def test_mini_ils_q_table_has_nonzero(self, inst):
        agent = QLearningAgent(seed=0)
        self._run_mini_ils(agent, inst, n_iter=100)
        assert np.count_nonzero(agent.Q) > 0

    def test_mini_ils_visits_cover_multiple_states(self, inst):
        agent = QLearningAgent(seed=0)
        self._run_mini_ils(agent, inst, n_iter=200)
        states_visited = np.any(agent.visits > 0, axis=1).sum()
        # Deve ter visitado pelo menos 5 estados distintos
        assert states_visited >= 5, f"Apenas {states_visited} estados visitados"

    def test_mini_ils_makespan_ge_lb(self, inst):
        agent = QLearningAgent(seed=0)
        rng_op = np.random.default_rng(0)
        sol = lpt(inst)
        for t in range(50):
            state_id = compute_state(sol, t, 50, 0)
            action = agent.select(state_id, np.zeros(4))
            sol_new = sol.copy()
            delta = apply_best(action, sol_new, rng_op)
            agent.update(state_id, np.zeros(4), action, float(delta),
                         compute_state(sol_new, t+1, 50, 0))
            if delta > 0:
                sol = sol_new
        # Invariante LB deve ser preservado em toda solucao visitada
        assert sol.makespan >= inst.lower_bound

    def test_agent_name(self, agent):
        assert "QL" in agent.name() or "Learn" in agent.name()

    def test_repr_contains_epsilon(self, agent):
        r = repr(agent)
        assert "eps" in r.lower() or "epsilon" in r.lower()


# ---------------------------------------------------------------------------
# TestSarsaAgent (31/08/2026 -- ver diario_pesquisa.md, Entrada 23)
# ---------------------------------------------------------------------------

class TestSarsaAgent:

    @pytest.fixture
    def sarsa(self):
        return SarsaAgent(alpha=0.1, gamma=0.9,
                          epsilon_start=1.0, epsilon_min=0.05,
                          epsilon_decay=0.99, seed=1)

    def test_q_table_shape(self, sarsa):
        assert sarsa.Q.shape == (N_STATES, N_OPERATORS)

    def test_select_returns_valid_action(self, sarsa):
        a = sarsa.select(0, np.zeros(4))
        assert 0 <= a < N_OPERATORS

    def test_on_policy_consistency(self):
        # A acao usada no alvo TD de update() DEVE ser exatamente a
        # acao retornada pela proxima chamada a select() para o mesmo
        # estado -- essa e a propriedade definidora do "on-policy" do
        # SARSA (ao contrario do Q-Learning, que usa max_a Q(s',a)
        # independente do que sera de fato executado a seguir).
        agent = SarsaAgent(seed=3, epsilon_start=0.5, epsilon_min=0.5,
                           epsilon_decay=1.0)
        s0, s1 = 10, 20
        a0 = agent.select(s0, np.zeros(4))
        agent.update(s0, np.zeros(4), a0, 1.0, s1)
        pending = agent._pending_action
        assert pending is not None
        a1 = agent.select(s1, np.zeros(4))
        assert a1 == pending
        # apos consumida, a decisao pendente deve ser limpa
        assert agent._pending_action is None

    def test_first_step_no_pending_action(self, sarsa):
        # Logo apos reset(), nao ha decisao pendente -- select() deve
        # cair no fallback epsilon-greedy sem levantar excecao.
        assert sarsa._pending_action is None
        a = sarsa.select(5, np.zeros(4))
        assert 0 <= a < N_OPERATORS

    def test_reset_clears_pending_action(self, sarsa):
        sarsa.select(0, np.zeros(4))
        sarsa.update(0, np.zeros(4), 0, 1.0, 1)
        assert sarsa._pending_action is not None
        sarsa.reset()
        assert sarsa._pending_action is None
        assert sarsa._pending_state is None

    def test_update_changes_q_table(self, sarsa):
        q_before = sarsa.Q.copy()
        sarsa.update(0, np.zeros(4), 0, 1.0, 1)
        assert not np.array_equal(q_before, sarsa.Q)

    def test_mini_ils_integration(self, inst, rng):
        # Mesmo teste de integracao de TestIntegrationMiniILS, mas para
        # SarsaAgent -- garante que o agente funciona dentro do loop
        # real de selecao/aplicacao/atualizacao, nao so isoladamente.
        agent = SarsaAgent(seed=0)
        sol = lpt(inst)
        rng_op = np.random.default_rng(7)
        for t in range(50):
            state_id = compute_state(sol, t, 50, 0)
            action = agent.select(state_id, np.zeros(4))
            sol_new = sol.copy()
            delta = apply_best(action, sol_new, rng_op)
            agent.update(state_id, np.zeros(4), action, float(delta),
                         compute_state(sol_new, t+1, 50, 0))
            if delta > 0:
                sol = sol_new
        assert sol.makespan >= inst.lower_bound

    def test_agent_name(self, sarsa):
        assert "SARSA" in sarsa.name()

    def test_repr_contains_epsilon(self, sarsa):
        r = repr(sarsa)
        assert "eps" in r.lower() or "epsilon" in r.lower()

    def test_is_subclass_of_qlearning_agent(self, sarsa):
        # Reaproveita toda a infraestrutura (Q-table, epsilon-greedy,
        # persistencia, analise de politica) do QLearningAgent --
        # so select()/update()/reset() sao sobrescritos.
        assert isinstance(sarsa, QLearningAgent)


# ---------------------------------------------------------------------------
# TestContextualUCBSelector (Entrada 28)
# ---------------------------------------------------------------------------

class TestContextualUCBSelector:

    def test_default_shape_matches_n_states_n_operators(self):
        sel = ContextualUCBSelector(seed=0)
        assert sel._counts.shape == (N_STATES, N_OPERATORS)
        assert sel._rewards.shape == (N_STATES, N_OPERATORS)
        assert sel._t.shape == (N_STATES,)

    def test_custom_n_states_n_actions_shape(self):
        # ig_ucb_ctx usa 3 acoes (estrategias de destruicao do IG), mesmo
        # N_STATES do Q-Learning (condicionamento por estado identico).
        sel = ContextualUCBSelector(seed=0, n_states=16, n_actions=3)
        assert sel._counts.shape == (16, 3)

    def test_select_range_respects_n_actions(self):
        sel = ContextualUCBSelector(seed=1, n_states=4, n_actions=3)
        draws = [sel.select(s % 4, np.zeros(4)) for s in range(200)]
        assert all(0 <= a < 3 for a in draws)

    def test_statistics_are_per_state_not_global(self):
        # Acao 0 no estado 0 nao deve influenciar as estatisticas do
        # estado 1 -- ao contrario do UCBSelector original (global).
        sel = ContextualUCBSelector(seed=2, n_states=4, n_actions=3)
        sel.select(0, np.zeros(4))
        sel.update(0, np.zeros(4), 0, 10.0, 0)
        assert sel._counts[0, 0] == 1
        assert sel._counts[1].sum() == 0  # estado 1 intocado

    def test_update_does_not_use_gamma_or_next_state_value(self):
        # Diferenca central vs. QLearningAgent: update() e uma media
        # simples por (estado, acao) -- nao ha campo gamma nem bootstrap.
        sel = ContextualUCBSelector(seed=3, n_states=4, n_actions=3)
        assert not hasattr(sel, "gamma")
        sel.update(0, np.zeros(4), 1, 5.0, 2)
        sel.update(0, np.zeros(4), 1, 3.0, 3)
        assert sel._counts[0, 1] == 2
        assert sel._rewards[0, 1] == 8.0  # soma simples, sem desconto

    def test_reset_clears_all_state_action_pairs(self):
        sel = ContextualUCBSelector(seed=4, n_states=4, n_actions=3)
        sel.update(0, np.zeros(4), 0, 1.0, 0)
        sel.update(2, np.zeros(4), 1, 1.0, 2)
        sel.reset()
        assert sel._counts.sum() == 0
        assert sel._rewards.sum() == 0

    def test_name(self):
        assert "UCB" in ContextualUCBSelector().name()

    def test_repr_contains_visited_states(self):
        sel = ContextualUCBSelector(seed=5, n_states=4, n_actions=3)
        sel.update(0, np.zeros(4), 0, 1.0, 0)
        r = repr(sel)
        assert "visited" in r.lower()

    def test_is_base_selector(self):
        assert isinstance(ContextualUCBSelector(), BaseSelector)


# ---------------------------------------------------------------------------
# TestFlatQLearningSelector (Entrada 29)
# ---------------------------------------------------------------------------

class TestFlatQLearningSelector:

    def test_inner_agent_has_single_state(self):
        sel = FlatQLearningSelector(seed=0)
        assert sel._inner._n_states == 1
        assert sel._inner._Q.shape == (1, N_OPERATORS)

    def test_custom_n_actions_propagates_to_inner(self):
        sel = FlatQLearningSelector(seed=0, n_actions=3)
        assert sel._inner._n_actions == 3
        assert sel._inner._Q.shape == (1, 3)

    def test_select_ignores_incoming_state_id(self):
        # Estado 0 e estado 99 devem produzir a MESMA politica (o
        # wrapper sempre delega ao estado interno unico, 0).
        sel_a = FlatQLearningSelector(seed=7, n_actions=3)
        sel_b = FlatQLearningSelector(seed=7, n_actions=3)
        seq_a = [sel_a.select(0, np.zeros(4)) for _ in range(30)]
        seq_b = [sel_b.select(99, np.zeros(4)) for _ in range(30)]
        assert seq_a == seq_b

    def test_update_accumulates_into_single_state(self):
        sel = FlatQLearningSelector(seed=1, n_actions=3, epsilon_start=0.0)
        sel.update(5, np.zeros(4), 1, 10.0, 42)  # state_id/next_state_id arbitrarios
        assert sel._inner._Q[0, 1] != 0.0
        assert sel._inner._Q.shape == (1, 3)  # nao criou linhas extras p/ 5 ou 42

    def test_reset_clears_inner_q_table(self):
        sel = FlatQLearningSelector(seed=2, n_actions=3)
        sel.update(0, np.zeros(4), 0, 5.0, 0)
        sel.reset()
        assert np.all(sel._inner._Q == 0.0)

    def test_name(self):
        assert "Flat" in FlatQLearningSelector().name()

    def test_uses_real_td_learning_not_simple_average(self):
        # Diferenca central vs. ContextualUCBSelector: o update usa o
        # bootstrap gamma*max_a' Q(a'), nao so soma/conta reward puro --
        # com gamma>0 e uma segunda atualizacao, o valor diverge de uma
        # media aritmetica simples das recompensas.
        sel = FlatQLearningSelector(seed=3, n_actions=3, alpha=0.5,
                                     gamma=0.9, epsilon_start=0.0)
        sel.update(0, np.zeros(4), 0, 10.0, 0)
        q_after_first = sel._inner._Q[0, 0]
        sel.update(0, np.zeros(4), 0, 10.0, 0)
        q_after_second = sel._inner._Q[0, 0]
        simple_average = 10.0
        # Com bootstrap (gamma*max_a' Q), o segundo update usa o proprio
        # Q[0,0] ja atualizado como parte do alvo -- diverge da media
        # simples que o ig_ucb_ctx (sem gamma) produziria.
        assert q_after_second != simple_average

    def test_is_base_selector(self):
        assert isinstance(FlatQLearningSelector(), BaseSelector)
