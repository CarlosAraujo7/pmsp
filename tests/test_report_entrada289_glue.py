"""Entrada 289: testa a COLA do relatorio do hospedeiro de DE com CSVs SINTETICOS (nao abre os resultados reais)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import report_entrada289_de as RP

LEARNERS = RP.LEARNERS


def synth(tmp):
    rng = np.random.default_rng(0)
    arm_eff = rng.normal(0, 0.4, size=(24, 5))
    def rows(insts, methods):
        out = []
        for f in range(1, 25):
            for i in insts:
                base = rng.normal(0, 1)
                for m in methods:
                    for s in range(3):
                        if m == "rand":
                            lg = base + 0.1 * rng.normal()
                        elif m.startswith("arm"):
                            lg = base - arm_eff[f - 1, int(m[3:])] + 0.1 * rng.normal()
                        else:
                            lg = base - 0.1 + 0.1 * rng.normal()
                        u = rng.dirichlet(np.ones(5))
                        out.append(dict(fun=f, inst=i, method=m, seed_i=s, err=10 ** lg, lg=lg, rew=float(rng.uniform(50, 150)), u0=u[0], u1=u[1], u2=u[2], u3=u[3], u4=u[4]))
        return pd.DataFrame(out)
    rows(range(1, 6), [f"arm{k}" for k in range(5)]).to_csv(tmp / "de_tune_entrada289.csv", index=False)
    rows(range(6, 16), ["rand"] + [f"arm{k}" for k in range(5)]).to_csv(tmp / "de_test_base_entrada289.csv", index=False)
    rows(range(6, 16), LEARNERS).to_csv(tmp / "de_test_learn_entrada289.csv", index=False)
    tw = pd.DataFrame(dict(fun=range(1, 25), succ_per_arm_run=rng.uniform(100, 3000, 24), G_uniform=1.0, G_max=2.0, **{f"kappa_{nm}": rng.uniform(0, 1, 24) for nm in ("ucb", "pm", "ap", "etc", "ql1")}))
    tw.to_csv(tmp / "entrada289_twin_predictions.csv", index=False)


def test_report_runs_end_to_end(tmp_path, monkeypatch, capsys):
    synth(tmp_path)
    monkeypatch.setattr(RP, "R", tmp_path)
    RP.main()
    out = capsys.readouterr().out
    assert "previsoes acertadas:" in out
    for k in ("D1", "D5", "D8", "D12"):
        assert f"({k})" in out
    assert (tmp_path / "entrada289_de.txt").exists() and (tmp_path / "entrada289_cells.csv").exists()
    cells = pd.read_csv(tmp_path / "entrada289_cells.csv")
    assert len(cells) == 24 and {"gain_val", "g_ql_rt", "val_beats_rand"} <= set(cells.columns)
