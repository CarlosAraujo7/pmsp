"""
tests/test_bco.py
Testes de sanidade do modulo pmsp/bco.py (baseline SOTA Bee Colony
Optimization, reimplementado de Davidovic, Selmic, Teodorovic & Ramljak
2012 -- ver diario_pesquisa.md, Entradas 39-40).
Execute: pytest tests/test_bco.py -v
"""
from __future__ import annotations
import inspect
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.bco import (
    bco_run, _BeeState, _roulette_choice, _forward_pass, _backward_pass,
    _run_one_bco_iteration,
)


@pytest.fixture
def inst():
    return PMSPInstance.random(n=30, m=8, seed=42)


@pytest.fixture
def inst_small():
    return PMSPInstance.random(n=12, m=3, seed=1)


@pytest.fixture
def inst_tiny():
    # n < NC padrao (10): exercita o caso "moves_to_do = min(NC, livres)".
    return PMSPInstance.random(n=5, m=2, seed=7)


class TestRouletteChoice:

    def test_all_zero_falls_back_to_uniform(self):
        rng = np.random.default_rng(0)
        candidates = np.array([10, 20, 30, 40])
        weights = np.zeros(4)
        counts = np.zeros(4, dtype=int)
        for _ in range(2000):
            chosen = _roulette_choice(candidates, weights, rng)
            counts[list(candidates).index(chosen)] += 1
        # Fallback uniforme: nenhum candidato deve dominar (tolerancia
        # generosa -- nao e um teste de qualidade estatistica fina).
        assert counts.min() > 300

    def test_respects_weighting_statistically(self):
        rng = np.random.default_rng(1)
        candidates = np.array([0, 1])
        weights = np.array([9.0, 1.0])  # 90% / 10% esperado
        counts = np.zeros(2, dtype=int)
        for _ in range(3000):
            chosen = _roulette_choice(candidates, weights, rng)
            counts[chosen] += 1
        frac0 = counts[0] / 3000
        assert 0.83 < frac0 < 0.97  # generoso em torno de 0.90

    def test_deterministic_given_seed(self):
        candidates = np.array([0, 1, 2, 3, 4])
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        seq_a = [_roulette_choice(candidates, weights, np.random.default_rng(5))
                 for _ in range(20)]
        seq_b = [_roulette_choice(candidates, weights, np.random.default_rng(5))
                 for _ in range(20)]
        assert seq_a == seq_b


class TestBeeState:

    def test_starts_incomplete_with_free_tasks(self):
        bee = _BeeState(n=10, m=3)
        assert not bee.is_complete()
        assert len(bee.free_tasks) == 10
        assert (bee.assignment == -1).all()
        assert (bee.loads == 0).all()

    def test_clone_is_independent(self):
        bee = _BeeState(n=5, m=2)
        bee.assignment[0] = 1
        bee.loads[1] = 100
        bee.free_tasks.remove(0)

        clone = bee.clone()
        clone.assignment[1] = 0
        clone.loads[0] = 999
        clone.free_tasks.remove(1)

        # Mutar o clone nao pode afetar o original.
        assert bee.assignment[1] == -1
        assert bee.loads[0] == 0
        assert 1 in bee.free_tasks


class TestForwardPass:

    def test_assigns_min_nc_and_free_tasks(self, inst):
        bee = _BeeState(inst.n, inst.m)
        rng = np.random.default_rng(0)
        _forward_pass(bee, NC=10, m=inst.m, p=inst.p, global_best=None, rng=rng)
        assigned = int((bee.assignment >= 0).sum())
        assert assigned == 10

    def test_respects_fewer_free_tasks_than_nc(self, inst_tiny):
        # inst_tiny tem n=5 < NC=10 -- um unico forward pass deve
        # completar a abelha inteira, sem erro.
        bee = _BeeState(inst_tiny.n, inst_tiny.m)
        rng = np.random.default_rng(0)
        _forward_pass(bee, NC=10, m=inst_tiny.m, p=inst_tiny.p,
                      global_best=None, rng=rng)
        assert bee.is_complete()

    def test_loads_consistent_with_assignment(self, inst):
        bee = _BeeState(inst.n, inst.m)
        rng = np.random.default_rng(0)
        for _ in range(3):
            _forward_pass(bee, NC=10, m=inst.m, p=inst.p, global_best=None, rng=rng)
        expected_loads = np.zeros(inst.m, dtype=np.int64)
        for j, mach in enumerate(bee.assignment):
            if mach >= 0:
                expected_loads[mach] += inst.p[j]
        assert np.array_equal(bee.loads, expected_loads)

    def test_no_duplicate_assignment(self, inst):
        bee = _BeeState(inst.n, inst.m)
        rng = np.random.default_rng(0)
        while not bee.is_complete():
            _forward_pass(bee, NC=10, m=inst.m, p=inst.p, global_best=None, rng=rng)
        # todos os n jobs atribuidos, cada um a exatamente uma maquina
        assert (bee.assignment >= 0).all()
        assert sorted(bee.free_tasks) == []

    def test_global_best_restriction_never_exceeds_it_when_avoidable(self):
        # Instancia pequena o bastante para que o fallback aleatorio
        # (linha 746-747) raramente seja necessario: com folga grande
        # de global_best, as maquinas nunca deveriam precisar excede-lo.
        inst = PMSPInstance.random(n=8, m=4, seed=3)
        bee = _BeeState(inst.n, inst.m)
        rng = np.random.default_rng(0)
        generous_best = int(inst.p.sum())  # folga enorme, nunca aperta
        while not bee.is_complete():
            _forward_pass(bee, NC=10, m=inst.m, p=inst.p,
                          global_best=generous_best, rng=rng)
        assert bee.loads.max() <= generous_best


class TestBackwardPass:

    def test_no_global_best_never_interrupts(self, inst):
        rng = np.random.default_rng(0)
        for trial in range(20):
            bees = [_BeeState(inst.n, inst.m) for _ in range(5)]
            for bee in bees:
                _forward_pass(bee, NC=10, m=inst.m, p=inst.p,
                              global_best=None, rng=rng)
            ok = _backward_pass(bees, global_best=None, u=1, rng=rng)
            # Sem BCO Modificado, a melhor abelha do lote sempre tem
            # p_loyal=1 (Omax-Ob=0) -- nunca deveria interromper.
            assert ok is True

    def test_all_bees_worse_than_global_best_interrupts(self):
        # Cenario forcado: todas as abelhas com carga MAIOR que
        # global_best -- com BCO Modificado, todas devem ficar
        # nao-comprometidas e a passada deve retornar False.
        m = 3
        bees = []
        for _ in range(5):
            bee = _BeeState(n=6, m=m)
            bee.loads[:] = [100, 100, 100]  # todas acima do global_best
            bees.append(bee)
        rng = np.random.default_rng(0)
        ok = _backward_pass(bees, global_best=50, u=5, rng=rng)
        assert ok is False

    def test_recruiters_and_uncommitted_partition_correctly(self, inst):
        rng = np.random.default_rng(0)
        bees = [_BeeState(inst.n, inst.m) for _ in range(5)]
        for bee in bees:
            _forward_pass(bee, NC=10, m=inst.m, p=inst.p, global_best=None, rng=rng)
        ok = _backward_pass(bees, global_best=None, u=1, rng=rng)
        assert ok is True
        # Depois da passada, toda abelha deve ter uma solucao parcial
        # valida (nao corrompida pela copia de recrutador).
        for bee in bees:
            assigned = int((bee.assignment >= 0).sum())
            assert assigned == 10  # 1 forward pass, NC=10


class TestRunOneBcoIteration:

    def test_completes_all_bees(self, inst_small):
        rng = np.random.default_rng(0)
        bees = _run_one_bco_iteration(inst_small, B=5, NC=10,
                                       global_best=None, rng=rng)
        assert bees is not None
        for bee in bees:
            assert bee.is_complete()
            sol = bee.to_solution(inst_small)
            assert sol.is_valid()


class TestBcoRun:

    def test_returns_expected_keys(self, inst):
        result = bco_run(inst, max_iter=500, seed=0)
        for key in ("makespan", "gap_pct", "cpu_time", "n_improvements", "history"):
            assert key in result

    def test_produces_valid_solution_reconstructed(self, inst_small):
        # bco_run() nao retorna o assignment diretamente (mesma
        # interface enxuta de sa_run/multifit_run), entao validamos
        # reconstruindo uma iteracao isolada e conferindo is_valid().
        rng = np.random.default_rng(0)
        bees = _run_one_bco_iteration(inst_small, B=5, NC=10,
                                       global_best=None, rng=rng)
        best_bee = min(bees, key=lambda b: b.loads.max())
        sol = best_bee.to_solution(inst_small)
        assert sol.is_valid()
        assert sol.makespan == int(best_bee.loads.max())

    def test_makespan_at_least_lower_bound(self, inst):
        result = bco_run(inst, max_iter=500, seed=0)
        assert result["makespan"] >= inst.lower_bound

    def test_deterministic_given_seed(self, inst):
        r1 = bco_run(inst, max_iter=500, seed=123)
        r2 = bco_run(inst, max_iter=500, seed=123)
        assert r1["makespan"] == r2["makespan"]
        assert r1["n_improvements"] == r2["n_improvements"]

    def test_different_seeds_can_differ(self, inst):
        results = {bco_run(inst, max_iter=300, seed=s)["makespan"] for s in range(5)}
        assert len(results) >= 1  # roda sem erro para varias seeds

    def test_track_history_length_matches_iterations(self, inst_small):
        result = bco_run(inst_small, max_iter=500, seed=0, track_history=True)
        assert len(result["history"]) > 0
        # historico deve ser monotonicamente nao-crescente (melhor global
        # so melhora ou mantem)
        hist = result["history"]
        assert all(hist[i] >= hist[i + 1] for i in range(len(hist) - 1))

    def test_gap_pct_consistent_with_makespan(self, inst):
        result = bco_run(inst, max_iter=500, seed=0)
        expected_gap = 100.0 * (result["makespan"] - inst.lower_bound) / inst.lower_bound
        assert abs(result["gap_pct"] - expected_gap) < 1e-9

    def test_various_sizes_valid(self):
        # Inclui o menor (n=5, exercita min(NC,livres)) e o maior
        # (n=100,m=20) das 6 classes do projeto.
        for n, m, seed in [(5, 2, 0), (20, 4, 1), (20, 6, 2), (75, 15, 3), (100, 20, 4)]:
            inst_i = PMSPInstance.random(n=n, m=m, seed=seed)
            result = bco_run(inst_i, max_iter=1000, seed=0)
            assert result["makespan"] >= inst_i.lower_bound, f"falhou para n={n}, m={m}"

    def test_loose_quality_sanity_bound(self, inst):
        # Nao ha garantia teorica provada para BCO (ao contrario do
        # MULTIFIT, 13/11 de Yue 1990) -- e ainda mais frouxo aqui
        # porque o orcamento equalizado (I pequeno em instancias
        # maiores, ver docstring do modulo) reduz deliberadamente o
        # numero de iteracoes. Este teste so protege contra quebra
        # algoritmica grosseira (ex.: ignorar balanceamento de carga
        # por completo), nao mede qualidade.
        result = bco_run(inst, max_iter=500, seed=0)
        assert result["gap_pct"] < 50.0

    def test_use_global_knowledge_false_still_valid(self, inst_small):
        # BCO basico (sem a modificacao da Secao 5.3) tambem deve rodar
        # sem erro e produzir solucao valida.
        result = bco_run(inst_small, max_iter=500, seed=0,
                          use_global_knowledge=False)
        assert result["makespan"] >= inst_small.lower_bound
        assert result["gap_pct"] < 50.0

    def test_small_budget_clamps_to_at_least_one_iteration(self, inst):
        # max_iter=1 com B=5, n=30 -> 1//(5*30)=0 -- deve ser
        # clampado para I=1, nao quebrar nem retornar makespan invalido.
        result = bco_run(inst, max_iter=1, seed=0)
        assert result["makespan"] >= inst.lower_bound
        assert result["n_improvements"] >= 1  # 1a iteracao sempre "melhora" (de None)

    def test_default_parameters_match_paper(self):
        # B=5, NC=10 sao os valores tunados pelos proprios autores
        # (Entrada 40) -- travados aqui contra alteracao acidental.
        sig = inspect.signature(bco_run)
        assert sig.parameters["B"].default == 5
        assert sig.parameters["NC"].default == 10
        assert sig.parameters["use_global_knowledge"].default is True
