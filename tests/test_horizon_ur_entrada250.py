"""Testes do horizonte longo no R||Cmax (Entrada 250): sementes, epsilon escalado e determinismo."""
import json

import numpy as np

from experiment import run_cycle22_ur_entrada234 as U
from experiment import run_cycle28_horizon_ur_entrada250 as H
from experiment.run_agents_numba_entrada180 import CLASSES


def test_seeds_unique():
    seen = set()
    for sp in H.SPACES:
        for t in H.HORIZONS:
            for ci in (0, 11):
                for rep in (0, 19):
                    for me in H.METHODS:
                        for s in range(H.N_SEED):
                            v = H.seed_of(sp, t, ci, rep, me, s)
                            assert v not in seen
                            seen.add(v)
                            assert 600_000_000 <= v < 700_000_000


def test_scaled_epsilon_decay():
    assert np.allclose(H.hp_scaled(1000), U.UA.HP_DEFAULT)
    # o epsilon (1,0 -> piso 0,05) chega ao piso na mesma fracao da execucao para qualquer T
    for T in H.HORIZONS:
        hp = H.hp_scaled(T)
        steps = np.log(0.05) / np.log(hp[2])
        assert abs(steps / T - np.log(0.05) / np.log(0.995) / 1000) < 1e-9


def test_qls_equals_ql_at_default_horizon_and_methods_are_deterministic():
    n, m = CLASSES[0]
    P, a0, od, lb = U.make(n, m, 0, 0, True)
    val = json.loads(U.choice_path("S5").read_text())["val"]
    qo = tuple(json.loads(U.choice_qo_path("S5").read_text())["cfg"])
    cls = f"rand_{n}j_{m}m"
    a = H.run_method("S5", "ql", P, a0, m, od, lb, n, 1000, 4242, val[cls], qo)
    b = H.run_method("S5", "qls", P, a0, m, od, lb, n, 1000, 4242, val[cls], qo)
    assert a == b
    for me in H.METHODS:
        r1 = H.run_method("S5", me, P, a0, m, od, lb, n, 300, 17, val[cls], qo)
        r2 = H.run_method("S5", me, P, a0, m, od, lb, n, 300, 17, val[cls], qo)
        assert r1 == r2
    long_ = H.run_method("S5", "ql", P, a0, m, od, lb, n, 3000, 5, val[cls], qo)
    short = H.run_method("S5", "ql", P, a0, m, od, lb, n, 300, 5, val[cls], qo)
    assert long_[1] > short[1]               # mais iteracoes, mais avaliacoes
