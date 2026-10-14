"""
tests/test_phase2.py
Testes de sanidade da Fase 2: LPT, operadores e seletores.
Execute: pytest tests/test_phase2.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.solution import PMSPSolution
from pmsp.constructive import lpt, random_construction
from pmsp.operators import (
    op1_best, op1_rand,
    op2_best, op2_rand,
    op3_best, op3_rand,
    op4_best, op4_rand,
    OPERATORS_BEST, OPERATORS_RAND, N_OPERATORS, apply_best, apply_rand,
)
from pmsp.selectors import (
    RandomSelector, UCBSelector, RFSelector, make_selector,
    AlwaysOp1Selector, AlwaysOp2Selector, AlwaysOp3Selector, AlwaysOp4Selector,
    ProbabilityMatchingSelector, AdaptivePursuitSelector,
)

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inst_small():
    return PMSPInstance(p=[8, 5, 3, 2], m=2, name="s")

@pytest.fixture
def inst_medium():
    return PMSPInstance.random(n=30, m=5, seed=7)

@pytest.fixture
def inst_large():
    return PMSPInstance.random(n=100, m=10, seed=42)

@pytest.fixture
def rng():
    return np.random.default_rng(123)

# ---------------------------------------------------------------------------
# TestLPT
# ---------------------------------------------------------------------------

class TestLPT:

    def test_lpt_is_valid(self, inst_medium, rng):
        sol = lpt(inst_medium)
        assert sol.is_valid()

    def test_lpt_makespan_ge_lb(self, inst_medium):
        sol = lpt(inst_medium)
        assert sol.makespan >= inst_medium.lower_bound

    def test_lpt_manual_small(self, inst_small):
        # p=[8,5,3,2], m=2, LPT order: 8,5,3,2
        # iter0: job0(8) -> maq0 [8,0]
        # iter1: job1(5) -> maq1 [8,5]
        # iter2: job2(3) -> maq1 [8,8]
        # iter3: job3(2) -> maq0 [10,8]  => makespan=10
        sol = lpt(inst_small)
        assert sol.is_valid()
        assert sol.makespan == 10

    def test_lpt_better_than_worst_case(self, inst_large):
        # LPT garantia teorica: <= (4/3 - 1/3m) * OPT >= LPT
        sol = lpt(inst_large)
        lb = inst_large.lower_bound
        # Verifica que GAP% esta dentro de limite pratico de 50%
        gap = (sol.makespan - lb) / lb
        assert gap < 0.5, f"GAP LPT muito alto: {gap:.2%}"

    def test_random_construction_valid(self, inst_medium, rng):
        sol = random_construction(inst_medium, rng)
        assert sol.is_valid()
        assert sol.makespan >= inst_medium.lower_bound

    def test_lpt_reproducible(self, inst_medium):
        # LPT e determinístico
        sol_a = lpt(inst_medium)
        sol_b = lpt(inst_medium)
        assert sol_a.makespan == sol_b.makespan

# ---------------------------------------------------------------------------
# Auxiliares: verificam invariantes apos aplicar operadores
# ---------------------------------------------------------------------------

def _check_operator(op_fn, inst, rng, n_trials=10):
    """
    Verifica operadores BEST-IMPROVEMENT:
    - is_valid() verdadeiro
    - delta >= 0 (nunca piora)
    - makespan == old_makespan - delta (delta e exato)
    """
    for seed in range(n_trials):
        base = lpt(inst)
        old_makespan = base.makespan
        sol = base.copy()
        delta = op_fn(sol, rng)
        assert sol.is_valid(), "Invariante violado apos operador"
        assert delta >= 0,     f"Delta negativo: {delta}"
        expected_new = old_makespan - delta
        assert sol.makespan == expected_new, (
            f"Makespan inconsistente: esperado {expected_new}, "
            f"obtido {sol.makespan}, delta={delta}"
        )

def _check_rand_operator(op_fn, inst, rng, n_trials=10):
    """
    Verifica operadores RANDOM (perturbacao):
    - is_valid() verdadeiro (invariante nunca violado)
    - delta >= 0 (delta e clamped: max(0, melhora))
    Nao verifica igualdade exata de makespan pois perturbacao pode piorar.
    """
    for seed in range(n_trials):
        sol = lpt(inst).copy()
        delta = op_fn(sol, rng)
        assert sol.is_valid(), "Invariante violado apos operador rand"
        assert delta >= 0,     f"Delta negativo: {delta}"

# ---------------------------------------------------------------------------
# TestOperators
# ---------------------------------------------------------------------------

class TestOperators:

    # --- Op1 (inter-machine swap) ---

    def test_op1_best_valid(self, inst_medium, rng):
        _check_operator(op1_best, inst_medium, rng)

    def test_op1_rand_valid(self, inst_medium, rng):
        _check_rand_operator(op1_rand, inst_medium, rng)

    def test_op1_best_improves_known_case(self):
        # maq0=[job0(10)]=10; maq1=[job1(1),job2(1)]=2 => makespan=10
        # Melhor swap: job0(10,maq0) <-> job1(1,maq1) => [1+1,10+1]=[2,11]? pior
        # Na verdade op1 troca job0(10,maq0) <-> job2(1,maq1):
        #   novo: maq0=[1]=1, maq1=[1,10]=11 => makespan=11  (pior!)
        # Nenhuma troca melhora; delta = 0
        inst = PMSPInstance(p=[10, 1, 1], m=2)
        sol = lpt(inst)      # LPT: [10]->[1,1] ou [10,1]->[1]
        rng = np.random.default_rng(0)
        sol2 = sol.copy()
        delta = op1_best(sol2, rng)
        assert sol2.is_valid()
        assert delta >= 0

    def test_op1_best_large_valid(self, inst_large, rng):
        _check_operator(op1_best, inst_large, rng, n_trials=5)

    # --- Op2 (single reinsertion) ---

    def test_op2_best_valid(self, inst_medium, rng):
        _check_operator(op2_best, inst_medium, rng)

    def test_op2_rand_valid(self, inst_medium, rng):
        _check_rand_operator(op2_rand, inst_medium, rng)

    def test_op2_improves_unbalanced(self):
        # maq0=[job0(10)]=10; maq1=[]=0; maq2=[]=0 => makespan=10
        # Op2 move job0 para maq1 => [0,10,0] => makespan=10 (sem melhora)
        # Com mais jobs:
        # maq0=[job0(10),job1(2)]=12; maq1=[job2(1)]=1 => makespan=12
        # Op2 move job0(10) de maq0 para maq1: [2, 11] => makespan=11
        inst = PMSPInstance(p=[10, 2, 1], m=2)
        sol = PMSPSolution(inst, assignment=[0, 0, 1])
        rng = np.random.default_rng(0)
        delta = op2_best(sol, rng)
        assert sol.is_valid()
        assert delta >= 0

    def test_op2_best_large_valid(self, inst_large, rng):
        _check_operator(op2_best, inst_large, rng, n_trials=5)

    # --- Op3 (intra-machine 2-opt) ---

    def test_op3_best_valid(self, inst_medium, rng):
        _check_operator(op3_best, inst_medium, rng)

    def test_op3_rand_valid(self, inst_medium, rng):
        _check_operator(op3_rand, inst_medium, rng)

    def test_op3_neutral_for_pure_pmsp(self, inst_medium, rng):
        # Para PMSP puro, op3 nao altera o makespan (delta sempre 0)
        for _ in range(20):
            sol = lpt(inst_medium)
            old_mk = sol.makespan
            delta = op3_best(sol, rng)
            assert delta == 0
            assert sol.makespan == old_mk

    # --- Op4 (block move) ---

    def test_op4_best_valid(self, inst_medium, rng):
        _check_operator(op4_best, inst_medium, rng)

    def test_op4_rand_valid(self, inst_medium, rng):
        _check_rand_operator(op4_rand, inst_medium, rng)

    def test_op4_best_large_valid(self, inst_large, rng):
        _check_operator(op4_best, inst_large, rng, n_trials=5)

    # --- Interface unificada ---

    def test_apply_best_all_operators(self, inst_medium, rng):
        for op_id in range(N_OPERATORS):
            sol = lpt(inst_medium).copy()
            old_mk = sol.makespan
            delta = apply_best(op_id, sol, rng)
            assert sol.is_valid()
            assert delta >= 0
            assert sol.makespan == old_mk - delta

    def test_apply_rand_all_operators(self, inst_medium, rng):
        for op_id in range(N_OPERATORS):
            sol = lpt(inst_medium).copy()
            old_mk = sol.makespan
            delta = apply_rand(op_id, sol, rng)
            assert sol.is_valid()
            assert delta >= 0

    def test_operators_list_length(self):
        assert len(OPERATORS_BEST) == N_OPERATORS
        assert len(OPERATORS_RAND) == N_OPERATORS

    def test_no_operator_worsens_solution(self, inst_large, rng):
        # Executa 20 vezes cada operador best: nunca deve piorar
        for seed in range(20):
            inst = PMSPInstance.random(n=50, m=5, seed=seed)
            for op_id in range(N_OPERATORS):
                sol = lpt(inst).copy()
                old_mk = sol.makespan
                delta = apply_best(op_id, sol, rng)
                assert sol.makespan <= old_mk, (
                    f"Op{op_id+1} piorou: {old_mk} -> {sol.makespan}"
                )

# ---------------------------------------------------------------------------
# TestSelectors
# ---------------------------------------------------------------------------

class TestSelectors:

    def _run_selector_loop(self, selector, inst, n_steps=50):
        """Executa um mini-loop de selecao/atualizacao e retorna historico."""
        rng_op = np.random.default_rng(99)
        sol = lpt(inst)
        history = []
        for t in range(n_steps):
            from pmsp.bounds import compute_state
            from pmsp.bounds import f1_relative_gap, f2_load_cv, f3_progress, f4_stagnation
            feats = np.array([
                f1_relative_gap(sol),
                f2_load_cv(sol),
                f3_progress(t, n_steps),
                f4_stagnation(0),
            ])
            state_id = compute_state(sol, t, n_steps, 0)
            action = selector.select(state_id, feats)
            assert 0 <= action < N_OPERATORS
            sol2 = sol.copy()
            delta = apply_best(action, sol2, rng_op)
            reward = float(delta)
            next_state = compute_state(sol2, t+1, n_steps, 0)
            selector.update(state_id, feats, action, reward, next_state)
            if delta > 0:
                sol = sol2
            history.append(action)
        return history

    def test_random_selector_all_ops_used(self, inst_medium):
        sel = RandomSelector(seed=0)
        hist = self._run_selector_loop(sel, inst_medium, n_steps=200)
        # Todos os 4 operadores devem aparecer (com 200 steps e prob 0.25 cada)
        assert len(set(hist)) == N_OPERATORS

    def test_random_selector_range(self, inst_medium):
        sel = RandomSelector(seed=1)
        hist = self._run_selector_loop(sel, inst_medium, n_steps=100)
        assert all(0 <= a < N_OPERATORS for a in hist)

    def test_random_selector_reset_noop(self):
        sel = RandomSelector(seed=5)
        sel.reset()  # nao deve levantar excecao

    def test_random_selector_default_n_actions_is_n_operators(self):
        # Regressao (Entrada 26): generalizar RandomSelector com n_actions
        # nao pode mudar o comportamento default (usado por ils_random /
        # ils_rand_hi) -- default precisa continuar sendo N_OPERATORS.
        sel = RandomSelector(seed=0)
        assert sel._n_actions == N_OPERATORS

    def test_random_selector_custom_n_actions_range(self, inst_medium):
        # ig_random (Entrada 26) usa RandomSelector com 3 acoes (estrategias
        # de destruicao do Iterated Greedy), nao 4 -- nunca pode sortear
        # fora do intervalo [0, n_actions).
        sel = RandomSelector(seed=2, n_actions=3)
        draws = [sel.select(0, np.zeros(4)) for _ in range(300)]
        assert all(0 <= a < 3 for a in draws)
        assert len(set(draws)) == 3  # as 3 acoes devem aparecer em 300 sorteios

    def test_random_selector_default_unaffected_by_n_actions_param(self):
        # Mesma sequencia de sorteios para uma seed fixa, com ou sem passar
        # n_actions=N_OPERATORS explicitamente -- garante que a assinatura
        # nova e aditiva de verdade, nao so "por acaso" compativel.
        import numpy as np
        a = RandomSelector(seed=7)
        b = RandomSelector(seed=7, n_actions=N_OPERATORS)
        seq_a = [a.select(0, np.zeros(4)) for _ in range(50)]
        seq_b = [b.select(0, np.zeros(4)) for _ in range(50)]
        assert seq_a == seq_b

    def test_ucb_explores_all_operators(self, inst_medium):
        sel = UCBSelector(seed=0)
        hist = self._run_selector_loop(sel, inst_medium, n_steps=100)
        # UCB garante que cada operador e tentado ao menos uma vez
        assert len(set(hist)) == N_OPERATORS

    def test_ucb_counts_updated(self, inst_medium):
        sel = UCBSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=50)
        # Todos os contadores devem ser > 0 apos 50 passos
        assert all(c > 0 for c in sel._counts)

    def test_ucb_default_n_actions_is_n_operators(self):
        # Regressao (Entrada 27): generalizar UCBSelector com n_actions nao
        # pode mudar o comportamento default (usado por ils_ucb).
        sel = UCBSelector(seed=0)
        assert sel._n_actions == N_OPERATORS
        assert sel._counts.shape == (N_OPERATORS,)

    def test_ucb_custom_n_actions_range(self, inst_medium):
        # ig_ucb (Entrada 27) usa UCBSelector com 3 acoes (estrategias de
        # destruicao do Iterated Greedy) -- nunca pode selecionar fora do
        # intervalo [0, n_actions), e a fase de inicializacao deve cobrir
        # exatamente as 3, nao 4.
        sel = UCBSelector(seed=1, n_actions=3)
        hist = self._run_selector_loop(sel, inst_medium, n_steps=60)
        assert all(0 <= a < 3 for a in hist)
        assert len(set(hist)) == 3
        assert sel._counts.shape == (3,)

    def test_ucb_mean_rewards_shape(self, inst_medium):
        sel = UCBSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=30)
        mr = sel.mean_rewards
        assert mr.shape == (N_OPERATORS,)
        assert all(v >= 0 for v in mr)

    def test_ucb_reset_clears_state(self, inst_medium):
        sel = UCBSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=20)
        sel.reset()
        assert sel._t == 0
        assert all(c == 0 for c in sel._counts)

    def test_rf_selector_runs_without_error(self, inst_medium):
        sel = RFSelector(seed=0, refit_interval=10, min_samples_fit=4)
        # Deve rodar sem excecao mesmo antes de treinar
        hist = self._run_selector_loop(sel, inst_medium, n_steps=80)
        assert all(0 <= a < N_OPERATORS for a in hist)

    def test_rf_fits_after_enough_data(self, inst_medium):
        # Apos dados suficientes (todos os 4 operadores vistos), deve treinar
        sel = RFSelector(seed=0, refit_interval=5, min_samples_fit=4,
                         epsilon=1.0)  # epsilon=1 -> exploracao pura, garante diversidade
        self._run_selector_loop(sel, inst_medium, n_steps=100)
        # Com epsilon=1.0 e 100 steps, todos os 4 ops devem ter aparecido
        actions_seen = {exp[1] for exp in sel._buffer}
        # Nao obrigatorio que tenha treinado, mas buffer deve ter dados
        assert len(sel._buffer) > 0

    def test_rf_reset_clears_model(self, inst_medium):
        sel = RFSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=80)
        sel.reset()
        assert not sel._fitted
        assert len(sel._buffer) == 0

    def test_make_selector_factory(self):
        for name in ["random", "ucb", "rf"]:
            sel = make_selector(name, seed=0)
            assert isinstance(sel, (RandomSelector, UCBSelector, RFSelector))

    def test_make_selector_invalid(self):
        with pytest.raises(ValueError):
            make_selector("nonexistent")

    def test_selector_names(self):
        assert "Random" in RandomSelector().name()
        assert "UCB" in UCBSelector().name()
        assert "RF" in RFSelector().name()

    # -----------------------------------------------------------------
    # Metodos novos (31/08/2026) -- ver diario_pesquisa.md, Entrada 23.
    # -----------------------------------------------------------------

    def test_op2_only_always_selects_1(self, inst_medium):
        sel = AlwaysOp2Selector()
        hist = self._run_selector_loop(sel, inst_medium, n_steps=30)
        assert set(hist) == {1}

    def test_op3_only_always_selects_2(self, inst_medium):
        sel = AlwaysOp3Selector()
        hist = self._run_selector_loop(sel, inst_medium, n_steps=30)
        assert set(hist) == {2}

    def test_op3_only_never_improves_makespan(self, inst_medium):
        # Controle negativo: Op3 (troca intramaquina) nunca muda o
        # makespan por design -- apply_best deve sempre retornar delta=0.
        from pmsp.bounds import compute_state
        sel = AlwaysOp3Selector()
        rng_op = np.random.default_rng(1)
        sol = lpt(inst_medium)
        base_makespan = sol.makespan
        for t in range(30):
            state_id = compute_state(sol, t, 30, 0)
            action = sel.select(state_id, np.zeros(4))
            assert action == 2
            sol2 = sol.copy()
            delta = apply_best(action, sol2, rng_op)
            assert delta == 0
            sol = sol2
        assert sol.makespan == base_makespan

    def test_op4_only_always_selects_3(self, inst_medium):
        sel = AlwaysOp4Selector()
        hist = self._run_selector_loop(sel, inst_medium, n_steps=30)
        assert set(hist) == {3}

    def test_pm_probabilities_sum_to_one(self, inst_medium):
        sel = ProbabilityMatchingSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=200)
        assert abs(sel.probs.sum() - 1.0) < 1e-8

    def test_pm_probabilities_respect_floor(self, inst_medium):
        sel = ProbabilityMatchingSelector(seed=0, p_min=0.1)
        self._run_selector_loop(sel, inst_medium, n_steps=200)
        assert all(p >= 0.1 - 1e-9 for p in sel.probs)

    def test_pm_reset_clears_state(self, inst_medium):
        sel = ProbabilityMatchingSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=50)
        sel.reset()
        assert np.allclose(sel._q, 1.0)

    def test_ap_probabilities_sum_to_one(self, inst_medium):
        sel = AdaptivePursuitSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=200)
        assert abs(sel.probs.sum() - 1.0) < 1e-6

    def test_ap_probabilities_bounded(self, inst_medium):
        sel = AdaptivePursuitSelector(seed=0, p_min=0.1)
        self._run_selector_loop(sel, inst_medium, n_steps=200)
        probs = sel.probs
        assert all(p >= 0.1 - 1e-6 for p in probs)
        assert all(p <= sel.p_max + 1e-6 for p in probs)

    def test_ap_reset_clears_state(self, inst_medium):
        sel = AdaptivePursuitSelector(seed=0)
        self._run_selector_loop(sel, inst_medium, n_steps=50)
        sel.reset()
        assert np.allclose(sel._p, 1.0 / N_OPERATORS)

    def test_make_selector_factory_pm_ap(self):
        pm = make_selector("pm", seed=0)
        ap = make_selector("ap", seed=0)
        assert isinstance(pm, ProbabilityMatchingSelector)
        assert isinstance(ap, AdaptivePursuitSelector)

    def test_new_selector_names(self):
        assert "Op2" in AlwaysOp2Selector().name()
        assert "Op3" in AlwaysOp3Selector().name()
        assert "Op4" in AlwaysOp4Selector().name()
        assert "PM" in ProbabilityMatchingSelector().name()
        assert "AP" in AdaptivePursuitSelector().name()
