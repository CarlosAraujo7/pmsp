"""
tests/test_phase4.py
Testes de sanidade da Fase 4: motor ILS e runner experimental.
Execute: pytest tests/test_phase4.py -v
"""
from __future__ import annotations
import csv, sys, json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance
from pmsp.constructive import lpt
from pmsp.bounds import compute_features, compute_state
from pmsp.ils import ils_run, lpt_run
from pmsp.selectors import RandomSelector, UCBSelector
from pmsp.qlearning import QLearningAgent, ContextualUCBSelector, FlatQLearningSelector
from pmsp.ig import N_DESTROY_STRATEGIES
from experiment.runner import run_single, run_all, summarize, make_ils_selector
import experiment.config as CFG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inst_small():
    return PMSPInstance.random(n=20, m=4, seed=0)

@pytest.fixture
def inst_tiny():
    return PMSPInstance(p=[8, 6, 5, 4, 3, 2], m=2, name="tiny")

# ---------------------------------------------------------------------------
# TestComputeFeatures
# ---------------------------------------------------------------------------

class TestComputeFeatures:

    def test_returns_ndarray(self, inst_tiny):
        sol = lpt(inst_tiny)
        feats = compute_features(sol, 10, 100, 5)
        assert isinstance(feats, np.ndarray)

    def test_shape(self, inst_tiny):
        sol = lpt(inst_tiny)
        feats = compute_features(sol, 10, 100, 5)
        assert feats.shape == (4,)

    def test_values_in_expected_range(self, inst_tiny):
        sol = lpt(inst_tiny)
        feats = compute_features(sol, 50, 100, 25)
        f1, f2, f3, f4 = feats
        assert f1 >= 0.0          # gap relativo >= 0
        assert f2 >= 0.0          # CV >= 0
        assert 0.0 <= f3 <= 1.0   # progresso em [0,1]
        assert 0.0 <= f4 <= 1.0   # estagnacao em [0,1]

    def test_f3_reflects_progress(self, inst_tiny):
        sol = lpt(inst_tiny)
        f_early = compute_features(sol, 10, 100, 0)
        f_late  = compute_features(sol, 90, 100, 0)
        assert f_late[2] > f_early[2]   # f3 maior no final

    def test_f4_reflects_stagnation(self, inst_tiny):
        sol = lpt(inst_tiny)
        f_fresh = compute_features(sol, 0, 100, 0)
        f_stuck = compute_features(sol, 0, 100, 50)
        assert f_stuck[3] > f_fresh[3]  # f4 maior quando estagnado

# ---------------------------------------------------------------------------
# TestILSEngine
# ---------------------------------------------------------------------------

class TestILSEngine:

    def test_lpt_run_valid(self, inst_small):
        r = lpt_run(inst_small)
        assert r["makespan"] >= inst_small.lower_bound
        assert r["gap_pct"] >= 0.0
        assert r["cpu_time"] >= 0.0

    def test_ils_run_keys(self, inst_small):
        sel = RandomSelector(seed=0)
        r = ils_run(inst_small, sel, max_iter=50)
        for key in ("makespan", "gap_pct", "cpu_time", "n_improvements", "history"):
            assert key in r

    def test_ils_improves_or_matches_lpt(self, inst_small):
        lpt_mk = lpt(inst_small).makespan
        for method_cls in [RandomSelector, UCBSelector]:
            sel = method_cls(seed=42)
            r = ils_run(inst_small, sel, max_iter=200)
            assert r["makespan"] <= lpt_mk, \
                f"{method_cls.__name__} piorou o LPT"

    def test_ils_makespan_ge_lb(self, inst_small):
        sel = QLearningAgent(seed=0)
        r = ils_run(inst_small, sel, max_iter=100)
        assert r["makespan"] >= inst_small.lower_bound

    def test_ils_gap_pct_nonnegative(self, inst_small):
        sel = RandomSelector(seed=1)
        r = ils_run(inst_small, sel, max_iter=100)
        assert r["gap_pct"] >= 0.0

    def test_ils_track_history(self, inst_small):
        sel = RandomSelector(seed=2)
        max_iter = 80
        r = ils_run(inst_small, sel, max_iter=max_iter, track_history=True)
        assert len(r["history"]) == max_iter

    def test_ils_history_monotone(self, inst_small):
        sel = UCBSelector(seed=3)
        r = ils_run(inst_small, sel, max_iter=100, track_history=True)
        hist = r["history"]
        # Historico do best deve ser nao-crescente (nunca piora o best)
        for i in range(len(hist)-1):
            assert hist[i+1] <= hist[i], \
                f"Historico cresceu em iter {i}: {hist[i]} -> {hist[i+1]}"

    def test_ils_ql_agent_learns(self, inst_small):
        agent = QLearningAgent(seed=0)
        ils_run(inst_small, agent, max_iter=200)
        # Q-table deve ter valores nao-zero apos treinamento
        assert np.count_nonzero(agent.Q) > 0

    def test_ils_selector_reset_between_runs(self, inst_small):
        # Duas rodadas devem ser deterministas com a mesma seed
        agent1 = QLearningAgent(seed=7)
        r1 = ils_run(inst_small, agent1, max_iter=100)
        agent2 = QLearningAgent(seed=7)
        r2 = ils_run(inst_small, agent2, max_iter=100)
        assert r1["makespan"] == r2["makespan"]
        np.testing.assert_array_almost_equal(agent1.Q, agent2.Q)

    def test_perturbation_activates(self, inst_small):
        # Com threshold=5, perturbacao deve ativar frequentemente
        sel = RandomSelector(seed=0)
        r = ils_run(inst_small, sel, max_iter=200,
                    perturbation_threshold=5, n_perturb=2)
        # Nao deve explodir; makespan >= LB
        assert r["makespan"] >= inst_small.lower_bound

# ---------------------------------------------------------------------------
# TestRunSingle
# ---------------------------------------------------------------------------

class TestRunSingle:

    def test_run_single_lpt(self, inst_small):
        r = run_single(inst_small, "lpt", inst_id=0, max_iter=50)
        assert r["method"] == "lpt"
        assert r["makespan"] >= inst_small.lower_bound
        assert r["selector"] is None

    def test_run_single_ils_ql(self, inst_small):
        r = run_single(inst_small, "ils_ql", inst_id=0, max_iter=50)
        assert r["method"] == "ils_ql"
        assert isinstance(r["selector"], QLearningAgent)
        assert r["makespan"] >= inst_small.lower_bound

    def test_run_single_ig_random(self, inst_small):
        # Entrada 26: controle de isolamento do ig_ql -- mesmo framework
        # Iterated Greedy, selecao de estrategia de destruicao uniforme
        # aleatoria em vez de Q-Learning.
        r = run_single(inst_small, "ig_random", inst_id=0, max_iter=50)
        assert r["method"] == "ig_random"
        assert isinstance(r["selector"], RandomSelector)
        assert r["selector"]._n_actions == N_DESTROY_STRATEGIES
        assert r["makespan"] >= inst_small.lower_bound

    def test_ig_random_never_worse_than_lpt(self, inst_small):
        # Mesma propriedade ja travada para ig_ql (tests/test_ig.py) --
        # decorre do criterio de aceitacao do ig_run (nunca aceita piora),
        # nao do selector escolhido, entao deve valer igual para ig_random.
        lpt_r = run_single(inst_small, "lpt", inst_id=0)
        for inst_id in range(5):
            r = run_single(inst_small, "ig_random", inst_id=inst_id, max_iter=100)
            assert r["makespan"] <= lpt_r["makespan"]

    def test_make_ils_selector_ig_random(self):
        sel = make_ils_selector("ig_random", seed=0)
        assert isinstance(sel, RandomSelector)
        assert sel._n_actions == N_DESTROY_STRATEGIES

    def test_run_single_ig_ucb(self, inst_small):
        # Entrada 27: segunda ablacao do ig_ql -- UCB1 (bandit simples,
        # sem Q-table/TD-learning) entre as 3 estrategias de destruicao.
        r = run_single(inst_small, "ig_ucb", inst_id=0, max_iter=50)
        assert r["method"] == "ig_ucb"
        assert isinstance(r["selector"], UCBSelector)
        assert r["selector"]._n_actions == N_DESTROY_STRATEGIES
        assert r["makespan"] >= inst_small.lower_bound

    def test_ig_ucb_never_worse_than_lpt(self, inst_small):
        lpt_r = run_single(inst_small, "lpt", inst_id=0)
        for inst_id in range(5):
            r = run_single(inst_small, "ig_ucb", inst_id=inst_id, max_iter=100)
            assert r["makespan"] <= lpt_r["makespan"]

    def test_make_ils_selector_ig_ucb(self):
        sel = make_ils_selector("ig_ucb", seed=0)
        assert isinstance(sel, UCBSelector)
        assert sel._n_actions == N_DESTROY_STRATEGIES

    def test_run_single_ig_ucb_ctx(self, inst_small):
        # Entrada 28: terceira ablacao -- UCB1 condicionado por estado,
        # sem TD-learning.
        r = run_single(inst_small, "ig_ucb_ctx", inst_id=0, max_iter=50)
        assert r["method"] == "ig_ucb_ctx"
        assert isinstance(r["selector"], ContextualUCBSelector)
        assert r["selector"]._n_actions == N_DESTROY_STRATEGIES
        assert r["makespan"] >= inst_small.lower_bound

    def test_ig_ucb_ctx_never_worse_than_lpt(self, inst_small):
        lpt_r = run_single(inst_small, "lpt", inst_id=0)
        for inst_id in range(5):
            r = run_single(inst_small, "ig_ucb_ctx", inst_id=inst_id, max_iter=100)
            assert r["makespan"] <= lpt_r["makespan"]

    def test_make_ils_selector_ig_ucb_ctx(self):
        sel = make_ils_selector("ig_ucb_ctx", seed=0)
        assert isinstance(sel, ContextualUCBSelector)
        assert sel._n_actions == N_DESTROY_STRATEGIES

    def test_run_single_ig_ql_flat(self, inst_small):
        # Entrada 29: quarta celula do 2x2 -- TD-learning sem estado.
        r = run_single(inst_small, "ig_ql_flat", inst_id=0, max_iter=50)
        assert r["method"] == "ig_ql_flat"
        assert isinstance(r["selector"], FlatQLearningSelector)
        assert r["selector"]._inner._n_actions == N_DESTROY_STRATEGIES
        assert r["selector"]._inner._n_states == 1
        assert r["makespan"] >= inst_small.lower_bound

    def test_ig_ql_flat_never_worse_than_lpt(self, inst_small):
        lpt_r = run_single(inst_small, "lpt", inst_id=0)
        for inst_id in range(5):
            r = run_single(inst_small, "ig_ql_flat", inst_id=inst_id, max_iter=100)
            assert r["makespan"] <= lpt_r["makespan"]

    def test_make_ils_selector_ig_ql_flat(self):
        sel = make_ils_selector("ig_ql_flat", seed=0)
        assert isinstance(sel, FlatQLearningSelector)
        assert sel._inner._n_actions == N_DESTROY_STRATEGIES

    def test_run_single_all_methods(self, inst_small):
        for method in CFG.METHODS:
            r = run_single(inst_small, method, inst_id=0, max_iter=30)
            assert r["gap_pct"] >= 0.0
            assert r["n"] == inst_small.n

    def test_run_single_different_reps_same_lpt(self, inst_small):
        # LPT e determinístico: todos os reps devem dar o mesmo makespan
        mk0 = run_single(inst_small, "lpt", inst_id=0)["makespan"]
        mk1 = run_single(inst_small, "lpt", inst_id=5)["makespan"]
        assert mk0 == mk1

# ---------------------------------------------------------------------------
# TestRunAll (smoke test com configuracao minima)
# ---------------------------------------------------------------------------

class TestRunAll:

    def test_run_all_creates_csv(self, tmp_path, inst_small):
        csv_path = run_all(
            methods=["lpt", "ils_random"],
            instance_configs=[(20, 4)],
            n_instances=2,
            max_iter=30,
            output_dir=tmp_path,
            save_qtables=False,
            verbose=False,
        )
        assert csv_path.exists()

    def test_csv_has_correct_rows(self, tmp_path):
        methods = ["lpt", "ils_ql"]
        n_reps = 3
        csv_path = run_all(
            methods=methods,
            instance_configs=[(20, 2)],
            n_instances=n_reps,
            max_iter=20,
            output_dir=tmp_path,
            save_qtables=False,
            verbose=False,
        )
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        # 1 instancia x 2 metodos x 3 reps = 6 linhas
        assert len(rows) == 6

    def test_csv_gap_pct_nonnegative(self, tmp_path):
        csv_path = run_all(
            methods=["ils_ucb"],
            instance_configs=[(20, 2)],
            n_instances=2,
            max_iter=20,
            output_dir=tmp_path,
            save_qtables=False,
            verbose=False,
        )
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                assert float(row["gap_pct"]) >= 0.0

    def test_qtables_saved_for_ql(self, tmp_path):
        run_all(
            methods=["ils_ql"],
            instance_configs=[(20, 2)],
            n_instances=2,
            max_iter=20,
            output_dir=tmp_path,
            save_qtables=True,
            verbose=False,
        )
        qtables_dir = tmp_path / CFG.QTABLES_SUBDIR
        npz_files = list(qtables_dir.glob("*.npz"))
        # 1 instancia x 1 metodo x 2 reps = 2 arquivos
        assert len(npz_files) == 2

    def test_run_config_json_saved(self, tmp_path):
        run_all(
            methods=["lpt"],
            instance_configs=[(20, 2)],
            n_instances=1,
            max_iter=10,
            output_dir=tmp_path,
            save_qtables=False,
            verbose=False,
        )
        cfg_path = tmp_path / "run_config.json"
        assert cfg_path.exists()
        cfg = json.loads(cfg_path.read_text())
        assert cfg["n_instances"] == 1
        assert "lpt" in cfg["methods"]

    def test_summarize_runs_without_error(self, tmp_path, capsys):
        csv_path = run_all(
            methods=["lpt", "ils_ql"],
            instance_configs=[(20, 2)],
            n_instances=2,
            max_iter=20,
            output_dir=tmp_path,
            save_qtables=False,
            verbose=False,
        )
        summarize(csv_path)   # nao deve levantar excecao
        captured = capsys.readouterr()
        assert "lpt" in captured.out
        assert "ils_ql" in captured.out
