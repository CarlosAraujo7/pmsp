"""Entrada 287: testa a COLA do relatorio (leitura do registro, conferencia de integridade, colunas) com um registro SINTETICO; nao abre os registros reais."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import report_entrada287_learnability as RP


def fake_cell(tmp_path, K=4, n_inst=20, n_seed=2, T=1000, break_identity=False):
    rng = np.random.default_rng(0)
    runs = n_inst * n_seed
    A = rng.integers(0, K, size=(runs, T)).astype(np.int8)
    S = rng.integers(0, 81, size=(runs, T)).astype(np.int8)
    p = np.where(A == 0, 0.02, 0.005)
    Rw = np.where(rng.random((runs, T)) < p, rng.integers(1, 4, size=(runs, T)), 0).astype(np.float32)
    C0 = np.full(runs, 500.0)
    best = C0 - Rw.sum(axis=1) + (1.0 if break_identity else 0.0)
    inst = np.repeat([f"rep{i}" for i in range(n_inst)], n_seed)
    np.savez_compressed(tmp_path / "S5_fake.npz", A=A, S=S, Rw=Rw, C0=C0, best=best, lb=np.full(runs, 480.0), n=np.full(runs, 50), m=np.full(runs, 5), inst=inst,
                        seed=np.arange(runs), arm_d=np.arange(K), arm_strat=np.zeros(K), arm_ls=np.zeros(K), coll=np.array(["sintetico"]))
    x = pd.DataFrame({"rand": 500 + rng.normal(0, 1, n_inst), "val": 499.5 + rng.normal(0, 1, n_inst), "rt": 499.9 + rng.normal(0, 1, n_inst), "ucb": 500 + rng.normal(0, 1, n_inst)})
    for a in range(K):
        x[f"arm{a}"] = 500 - 0.1 * a + rng.normal(0, 1, n_inst)
    return dict(coll="sintetico", cell="S5|fake", x=x)


def test_process_cell_glue(tmp_path, monkeypatch):
    c = fake_cell(tmp_path)
    monkeypatch.setattr(RP, "LOGS", tmp_path)
    row = RP.process_cell(0, c)
    for k in ("pos_rate", "pos_per_arm_run", "L", "kappa_ws", "tw_oracle", "tw_gain_ucb", "tw_kappa_ql", "H_phase", "H_state", "g_val", "g_rt", "g_ucb", "kap_rt", "static_fidelity"):
        assert k in row and np.isfinite(row[k]), k
    assert 0.0 < row["pos_rate"] < 0.05
    assert row["K"] == 4 and row["runs"] == 40


def test_integrity_check_catches_a_broken_log(tmp_path, monkeypatch):
    c = fake_cell(tmp_path, break_identity=True)
    monkeypatch.setattr(RP, "LOGS", tmp_path)
    with pytest.raises(AssertionError):
        RP.process_cell(0, c)
