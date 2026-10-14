"""
tests/test_overflow_fix_entrada53.py
Testes de regressao para o bug de overflow int64 descoberto na Entrada
53 do diario_pesquisa.md: instancias da familia B (benchmark externo
Symmetry 2025), classe 18, tem soma de tempos de processamento
(~2.78e19) muito acima do teto do int64 (2**63-1 ~= 9.223e18) --
numpy fazia wraparound silencioso (RuntimeWarning, sem excecao) em
pmsp/constructive.py, pmsp/solution.py, pmsp/bco.py, pmsp/multifit.py.

Correcao: PMSPInstance decide o dtype do vetor `p` por soma em Python
puro (precisao arbitraria); acima de SAFE_INT64_SUM_THRESHOLD usa
dtype=object (inteiros Python nativos, imunes a overflow) em vez de
int64, e todos os acumuladores de carga derivados de `p` herdam esse
dtype. Instancias normais (soma bem abaixo do limiar) continuam em
int64 nativo -- sem qualquer mudanca de desempenho ou comportamento
(ver suite completa tests/, 259 testes preexistentes continuam
passando apos esta correcao).

Execute: pytest tests/test_overflow_fix_entrada53.py -v
"""
from __future__ import annotations

import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pmsp.instance import PMSPInstance, SAFE_INT64_SUM_THRESHOLD
from pmsp.constructive import lpt, random_construction
from pmsp.multifit import multifit, multifit_run
from pmsp.bco import bco_run
from experiment.runner import run_single
import experiment.config as CFG

EXTERNAL_B_ZIP = ROOT / "external_benchmarks" / "pcmax_symmetry2025" / "B.zip"


def _make_extreme_p(n: int, rng: np.random.Generator, scale: int = 10 ** 17) -> list[int]:
    """Gera n tempos de processamento cuja soma ultrapassa o teto do
    int64 (analogo a familia B, classe 18: D ~ U[1, 10**17])."""
    return [int(x) for x in rng.integers(1, scale, size=n, dtype=np.int64)]


def _independent_makespan(p: list[int], assignment: np.ndarray, m: int) -> int:
    """Recomputa o makespan em Python puro (precisao arbitraria), a
    partir do vetor de tempos de processamento bruto e do assignment
    devolvido pela solucao -- caminho totalmente independente de
    PMSPSolution/PMSPInstance, para nao 'testar o proprio teste'."""
    loads = [0] * m
    for j, mach in enumerate(assignment):
        loads[int(mach)] += int(p[j])
    return max(loads)


# ---------------------------------------------------------------------------
# Selecao de dtype
# ---------------------------------------------------------------------------

class TestDtypeSelection:
    def test_normal_instance_stays_int64(self):
        inst = PMSPInstance.random(n=300, m=30, seed=1)  # maior classe do projeto
        assert inst.p.dtype == np.int64

    def test_extreme_instance_uses_object_dtype(self):
        rng = np.random.default_rng(7)
        p = _make_extreme_p(50, rng)
        assert sum(p) > SAFE_INT64_SUM_THRESHOLD  # sanity do proprio gerador de teste
        inst = PMSPInstance(p=p, m=4, name="extreme_test")
        assert inst.p.dtype == object

    def test_threshold_boundary_stays_int64_just_below(self):
        # soma exatamente no limiar -> ainda int64 (comparacao e' <=)
        n = 10
        val = SAFE_INT64_SUM_THRESHOLD // n
        p = [val] * n
        inst = PMSPInstance(p=p, m=2, name="boundary_test")
        assert inst.p.dtype == np.int64


# ---------------------------------------------------------------------------
# Sem overflow (RuntimeWarning vira excecao neste teste) + correcao numerica
# ---------------------------------------------------------------------------

class TestNoOverflowExtremeInstance:
    @pytest.fixture
    def extreme_inst(self):
        rng = np.random.default_rng(123)
        p = _make_extreme_p(60, rng)
        return PMSPInstance(p=p, m=5, name="extreme_fixture"), p

    def test_lpt_no_warning_and_correct(self, extreme_inst):
        inst, p_raw = extreme_inst
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sol = lpt(inst)
        assert sol.is_valid()
        expected = _independent_makespan(p_raw, sol.assignment, inst.m)
        assert sol.makespan == expected

    def test_random_construction_no_warning_and_correct(self, extreme_inst):
        inst, p_raw = extreme_inst
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sol = random_construction(inst, rng)
        assert sol.is_valid()
        expected = _independent_makespan(p_raw, sol.assignment, inst.m)
        assert sol.makespan == expected

    def test_multifit_no_warning_and_correct(self, extreme_inst):
        inst, p_raw = extreme_inst
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sol = multifit(inst)
        assert sol.is_valid()
        expected = _independent_makespan(p_raw, sol.assignment, inst.m)
        assert sol.makespan == expected

    def test_bco_run_no_warning(self, extreme_inst):
        inst, _ = extreme_inst
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = bco_run(inst, max_iter=10, seed=42)
        assert result["makespan"] > 0
        assert result["gap_pct"] >= -1e-9  # tolerancia de ponto flutuante

    def test_lower_bound_correct_for_extreme_instance(self, extreme_inst):
        inst, p_raw = extreme_inst
        expected_lb = max(max(p_raw), -(-sum(p_raw) // inst.m))  # ceil via // negativo
        assert inst.lower_bound == expected_lb

    def test_run_single_all_methods_no_warning(self, extreme_inst):
        inst, _ = extreme_inst
        for method in CFG.METHODS:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = run_single(inst, method, inst_id=0, max_iter=15)
            assert result["makespan"] > 0, f"{method}: makespan invalido"
            assert result["gap_pct"] >= -1e-6, f"{method}: gap_pct negativo ({result['gap_pct']})"


# ---------------------------------------------------------------------------
# Instancias REAIS da familia B, classe 18 (benchmark externo Symmetry 2025)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not EXTERNAL_B_ZIP.exists(), reason="B.zip do benchmark externo nao baixado neste ambiente")
class TestRealB18Instances:
    MEMBERS = [
        "B/18/04_094_18_00.dat",
        "B/18/04_094_18_26.dat",
    ]

    def _load(self, member):
        with zipfile.ZipFile(EXTERNAL_B_ZIP) as zf:
            content = zf.read(member).decode("utf-8", errors="replace")
        tokens = content.split()
        m, n = int(tokens[0]), int(tokens[1])
        p = [int(x) for x in tokens[2:2 + n]]
        return m, n, p

    @pytest.mark.parametrize("member", MEMBERS)
    def test_real_instance_all_methods_no_warning(self, member):
        m, n, p = self._load(member)
        assert sum(p) > SAFE_INT64_SUM_THRESHOLD  # confirma que e' de fato uma instancia de risco
        inst = PMSPInstance(p=p, m=m, name=member)
        assert inst.p.dtype == object
        for method in CFG.METHODS:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = run_single(inst, method, inst_id=0, max_iter=15)
            assert result["makespan"] > 0, f"{member}/{method}: makespan invalido"
            assert result["gap_pct"] >= -1e-6, f"{member}/{method}: gap_pct negativo"

    @pytest.mark.parametrize("member", MEMBERS)
    def test_real_instance_lpt_matches_independent_recomputation(self, member):
        m, n, p = self._load(member)
        inst = PMSPInstance(p=p, m=m, name=member)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sol = lpt(inst)
        assert sol.is_valid()
        expected = _independent_makespan(p, sol.assignment, m)
        assert sol.makespan == expected
        assert sol.makespan > SAFE_INT64_SUM_THRESHOLD  # confirma magnitude realista, nao um valor "estourado" pequeno
