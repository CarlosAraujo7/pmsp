"""Entrada 291: testa a COLA do relatorio do credito alinhado com arquivos SINTETICOS (nao abre os resultados reais)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import report_entrada291_credit as RP

LEARNERS = RP.LEARNERS


def synth(tmp):
    rng = np.random.default_rng(1)
    eff = rng.normal(0, 0.5, size=(24, 5))

    def rows(insts, methods, credit=None, with_u=True):
        out = []
        for f in range(1, 25):
            for i in insts:
                base = rng.normal()
                for m in methods:
                    for s in range(3):
                        lg = base - (eff[f - 1, int(m[3:])] if m.startswith("arm") else (0.0 if m == "rand" else 0.1)) + 0.1 * rng.normal()
                        u = rng.dirichlet(np.ones(5))
                        r = dict(fun=f, inst=i, method=m, seed_i=s, lg=lg, u0=u[0], u1=u[1], u2=u[2], u3=u[3], u4=u[4])
                        if credit:
                            r["credit"] = credit
                        out.append(r)
        return pd.DataFrame(out)

    rows(range(1, 6), [f"arm{k}" for k in range(5)]).to_csv(tmp / "de_tune_entrada289.csv", index=False)
    rows(range(6, 16), ["rand"] + [f"arm{k}" for k in range(5)]).to_csv(tmp / "de_test_base_entrada289.csv", index=False)
    rows(range(6, 16), LEARNERS).to_csv(tmp / "de_test_learn_entrada289.csv", index=False)
    pd.concat([rows(range(6, 16), LEARNERS, c) for c in ("log_incumbent", "rel_incumbent")]).to_csv(tmp / "de_credit_learn_entrada291.csv", index=False)
    pd.DataFrame(dict(fun=range(1, 25), coincidem=rng.random(24) < 0.2)).to_csv(tmp / "entrada289c_alignment.csv", index=False)
    (tmp / "logs291").mkdir(exist_ok=True)
    for f in range(1, 25):
        for c in ("log_incumbent", "rel_incumbent"):
            np.savez_compressed(tmp / "logs291" / f"f{f:02d}_{c}.npz", A=rng.integers(0, 5, size=(30, 50)).astype(np.int8), Rc=rng.random((30, 50)).astype(np.float32))


def test_report_runs_end_to_end(tmp_path, monkeypatch, capsys):
    synth(tmp_path)
    monkeypatch.setattr(RP, "R", tmp_path)
    RP.main()
    out = capsys.readouterr().out
    assert "previsoes acertadas:" in out and all(f"({k})" in out for k in ("G1", "G3", "G6a", "G7"))
    S = pd.read_csv(tmp_path / "entrada291_summary.csv")
    assert len(S) == 3 * 7 and set(S.credito) == {"success", "log_incumbent", "rel_incumbent"}
