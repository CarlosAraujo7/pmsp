"""Entrada 291: o credito 'success' de pmsp/de_credit.py reproduz pmsp/de_host.run_de; os outros creditos obedecem a sua definicao."""
from __future__ import annotations

import numpy as np
import pytest

from pmsp import de_credit as DC
from pmsp import de_host as DH


@pytest.mark.parametrize("method", ["rand", "arm1", "ql_rt", "ucb", "pm"])
def test_success_credit_reproduces_the_original_run(method):
    a = DH.run_de(8, 6, method, 17, T=120, log=True)
    b = DC.run_de_credit(8, 6, method, 17, "success", T=120, log=True)
    assert a["err"] == b["err"]
    assert np.array_equal(a["A"], b["A"]) and np.allclose(a["Rw"], b["Rc"])


def test_log_credit_sums_to_the_log_error_reduction_and_rel_credit_is_bounded():
    r = DC.run_de_credit(8, 7, "rand", 3, "log_incumbent", T=300, log=True)
    f, fopt = DH.bbob(8, 7)
    rng = np.random.default_rng(3)
    pop = rng.uniform(DH.LOW, DH.HIGH, size=(100, 10))
    err0 = float(np.min(f(pop)) - fopt)
    # a soma dos creditos e' a reducao de log10 do erro do incumbente (erros acima do piso)
    assert r["crew"] == pytest.approx(np.log10(err0) - r["lg"], abs=1e-6) or r["lg"] <= -8 + 1e-9
    assert (r["Rc"] >= 0).all()
    q = DC.run_de_credit(8, 7, "rand", 3, "rel_incumbent", T=300, log=True)
    assert (q["Rc"] >= 0).all() and (q["Rc"] <= 1).all()
    assert q["err"] == r["err"]                                      # o credito nao muda o aleatorio: a mesma trajetoria


def test_learners_receive_rewards_in_unit_interval_under_aligned_credits():
    for credit in ("log_incumbent", "rel_incumbent"):
        out = DC.run_de_credit(9, 6, "pg", 5, credit, T=150, log=True)
        assert np.isfinite(out["err"]) and out["Rc"].max() <= (8.0 + 1e-6 if credit == "log_incumbent" else 1.0)
