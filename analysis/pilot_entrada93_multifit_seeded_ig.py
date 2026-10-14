"""
analysis/pilot_entrada93_multifit_seeded_ig.py
================================================
PILOTO EXPLORATORIO (Entrada 93, 04/10/2026) -- NAO gera alegacoes.
Pergunta: iniciar o ig_ql a partir da melhor solucao entre LPT e MULTIFIT
(em vez de so' LPT) elimina a perda do ig_ql para o MULTIFIT em n>=75, sem
piorar as classes pequenas? Como o IG nunca piora a solucao corrente, o
resultado e' <= MULTIFIT por construcao; o que se mede e' (a) se ainda
melhora o MULTIFIT e (b) o efeito nas classes pequenas.

Metodo: monkeypatch de pmsp.ig.lpt (o ig_run chama `lpt(instance)` como
ponto de partida) -- nenhum arquivo de pmsp/ e' alterado. Mesmas
instancias/seeds de results/results.csv (PMSPInstance.random(seed=rep)).
N=30 por classe, 1 execucao por instancia.

Uso: python -m analysis.pilot_entrada93_multifit_seeded_ig
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pmsp.ig as ig_mod  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402
from pmsp.multifit import multifit  # noqa: E402
from experiment.runner import run_single  # noqa: E402

_lpt = ig_mod.lpt


def best_start(inst):
    a, b = _lpt(inst), multifit(inst)
    return b if b.makespan < a.makespan else a


CLASSES = [(20, 6), (30, 8), (50, 10), (60, 20), (75, 15), (100, 20), (150, 30), (250, 25), (300, 20)]
K = 30
res = pd.read_csv(ROOT / "results" / "results.csv")
rows = []
for n, m in CLASSES:
    name = f"rand_{n}j_{m}m"
    out = []
    for rep in range(K):
        inst = PMSPInstance.random(n=n, m=m, seed=rep, name=name)
        ig_mod.lpt = best_start
        r = run_single(inst, "ig_ql", rep, max_iter=1000)
        ig_mod.lpt = _lpt
        s = res[(res.instance == name) & (res.rep == rep)].set_index("method")
        out.append(dict(seeded=r["gap_pct"], ig_ql=s.loc["ig_ql", "gap_pct"], multifit=s.loc["multifit", "gap_pct"],
                        seeded_mk=r["makespan"], mf_mk=s.loc["multifit", "makespan"], q_mk=s.loc["ig_ql", "makespan"]))
    d = pd.DataFrame(out)
    rows.append(dict(cls=name, seeded=d.seeded.mean(), ig_ql=d.ig_ql.mean(), multifit=d.multifit.mean(),
                     seeded_beats_mf=int((d.seeded_mk < d.mf_mk).sum()), seeded_worse_mf=int((d.seeded_mk > d.mf_mk).sum()),
                     seeded_beats_igql=int((d.seeded_mk < d.q_mk).sum()), seeded_worse_igql=int((d.seeded_mk > d.q_mk).sum())))
    print(rows[-1], flush=True)
print(pd.DataFrame(rows).round(4).to_string(index=False))
