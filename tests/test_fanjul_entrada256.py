"""Testes do protocolo no benchmark padrao de R||Cmax de Fanjul-Peyro e Ruiz (Entrada 256): grade de acoes, leitura das instancias, sementes e execucao."""
import numpy as np
import pytest

from experiment import run_cycle32_fanjul_entrada256 as F
from pmsp import ur_numba_agents as UA

pytestmark = pytest.mark.skipif(not F.BENCH.exists(), reason="benchmark de Fanjul-Peyro e Ruiz nao extraido (external_benchmarks/rcmax_fanjul_ruiz/README.md)")


def test_action_grids_scale_with_n():
    st, d, ls = F.arm_table_b("B5", 1000)
    assert list(d) == [10, 20, 50, 100, 200] and len(st) == 5 and not ls.any()
    st, d, ls = F.arm_table_b("B15", 100)
    assert len(st) == 15 and sorted(set(st)) == [0, 1, 2] and d.min() == 2 and d.max() == 20
    st, d, ls = F.arm_table_b("B30", 200)
    assert len(st) == 30 and ls.sum() == 15 and (d[:15] == d[15:]).all()


def test_manifest_split_and_loading():
    man = F.manifest()
    assert (man.role == "tune").sum() == 280 and (man.role == "test").sum() == 700 and len(man) == 1400
    r = man[(man.role == "test") & (man.n == 100) & (man.m == 10)].iloc[0]
    P, a0, od, lb = F.load(r)
    assert P.shape == (100, 10) and P.dtype == np.int64 and lb > 0 and r.best_known >= lb * 0.99


def test_seeds_unique_across_spaces_methods_and_splits():
    seen = set()
    for sp in F.SPACES:
        for ri in (0, 700, 1399):
            for me in F.METHODS + ["arm0", "arm29"]:
                for s in range(3):
                    for tune in (False, True):
                        v = F.seed_of(sp, ri, me, s, tune)
                        assert v not in seen
                        seen.add(v)


def test_every_method_runs_and_is_deterministic():
    man = F.manifest()
    r = man[(man.role == "tune") & (man.n == 100) & (man.m == 10)].iloc[0]
    P, a0, od, lb = F.load(r)
    for sp in F.SPACES:
        qo, pub = F.frozen(sp)
        for me in F.METHODS + ["arm1"]:
            a = F.run_fixed_or_agent(sp, me, P, a0, 10, od, lb, 100, 17, 1, qo, pub)
            b = F.run_fixed_or_agent(sp, me, P, a0, 10, od, lb, 100, 17, 1, qo, pub)
            assert a == b and a[0] >= r.best_known * 0.99


def test_validated_equals_the_fixed_arm():
    man = F.manifest()
    r = man[(man.role == "tune") & (man.n == 200) & (man.m == 20)].iloc[0]
    P, a0, od, lb = F.load(r)
    st, d, ls = F.arm_table_b("B15", 200)
    ref = UA.ur_agent_core(P, a0, 20, st, d, ls, F.T, UA.AGENT_FIXED, 7, 0.1, 0.1, 5, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)[0]
    assert F.run_fixed_or_agent("B15", "val", P, a0, 20, od, lb, 200, 5, 7)[0] == int(ref)
