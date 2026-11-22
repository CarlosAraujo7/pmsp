"""Testes da logica de analysis/report_entrada286_estimands.py com dados SINTETICOS (nao tocam os resultados reais)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.report_entrada286_estimands import boot_cluster, coupling, fieller, rules_table


def test_fieller_covers_true_ratio():
    rng = np.random.default_rng(1)
    hits, n = 0, 600
    for _ in range(n):
        k = 25
        b = rng.gamma(3.0, 0.5, size=k)                  # denominadores positivos
        a = 0.2 * b + rng.normal(0, 0.05, size=k)        # razao verdadeira 0,2
        lo, hi = fieller(a, b)
        hits += (lo <= 0.2 <= hi)
    cov = hits / n
    assert 0.84 <= cov <= 0.96, cov                      # IC90 nominal


def test_fieller_unbounded_when_denominator_is_not_significant():
    a = np.array([0.1, -0.1, 0.05, -0.05, 0.02, -0.02])
    b = np.array([0.1, -0.1, 0.05, -0.05, 0.03, -0.03])
    lo, hi = fieller(a, b)
    assert np.isnan(lo) and np.isnan(hi)


def test_boot_cluster_interval_contains_point_estimate():
    cl = np.repeat(np.arange(10), 3)
    v = np.random.default_rng(0).normal(1.0, 0.3, size=30)
    lo, hi = boot_cluster(cl, lambda ix: v[ix].mean(), nb=500)
    assert lo < v.mean() < hi


def test_rules_table_counts():
    T = pd.DataFrame(dict(
        a_conf=[True, True, False, False], a_lo=[-0.50, -0.10, 0.00, -0.02], a_hi=[-0.20, -0.004, 0.06, 0.02],
        b_conf=[True, False, False, False], b_lo=[-1.0, -0.1, 0.0, 0.0], b_hi=[-0.5, 0.1, 0.1, 0.1]))
    r = rules_table(T, "t")
    r05 = r[r.margem == 0.05].iloc[0]
    # celula 0: gain CI (0,20; 0,50): alem da margem nas duas regras; celula 1: d CI (-0,10; -0,004): gain (0,004; 0,10): alem na regra antiga, NAO na estrita (limite inferior 0,004 < 0,05);
    # celula 2: d CI (0; 0,06) e nao confirmada (o rt nao e' melhor); celula 3: nao confirmada; paridade (IC dentro de +-0,05): celula 3 (-0,02; 0,02) e celula 1 nao (-0,10)
    assert int(r05.rt_alem_antiga) == 2 and int(r05.rt_alem_estrita) == 1
    assert int(r05.rt_paridade_TOST) == 1
    assert int(r05.val_bate_rt_antiga) == 1 and int(r05.val_bate_rt_estrita) == 1


def test_decoupling_removes_spurious_correlation_from_the_shared_random_arm():
    # 30 celulas SEM correlacao verdadeira entre o ganho da validada e o do rt, mas o braco aleatorio e' compartilhado e ruidoso:
    # o Spearman acoplado deve ficar positivo e o desacoplado (metades disjuntas de instancias) perto de zero.
    rng = np.random.default_rng(5)
    cells = []
    for c in range(30):
        n = 80
        base = 100.0 + rng.normal(0, 3, size=n)
        rand = base * (1 + rng.normal(0, 0.02, size=n))             # ruido do braco aleatorio
        val = base * (1 - 0.0005 * rng.uniform(0, 1))               # ganho verdadeiro minusculo, independente entre celulas
        rt = base * (1 - 0.0005 * rng.uniform(0, 1))
        cells.append(dict(coll="sintetico", cell=f"c{c}", x=pd.DataFrame(dict(rand=rand, val=val, rt=rt))))
    CP = coupling(cells, "teste", nsplit=60)
    row = CP[CP.escopo == "todas"].iloc[0]
    assert row.S_acoplado_meio > 0.3, row
    assert abs(row.S_desacoplado_meio) < 0.25, row
    assert row.S_acoplado_meio - row.S_desacoplado_meio > 0.25
