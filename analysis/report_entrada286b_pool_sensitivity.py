"""
analysis/report_entrada286b_pool_sensitivity.py (Entrada 286b, Ciclo 53) -- POS HOC
=====================================================================================
O pool sem rotulo da Entrada 286 deu um resultado que as previsoes P10 e P11 nao esperavam: a acao estatica UNICA escolhida pelo criterio pre-registrado (menor lacuna media ao limite inferior, todas as instancias de ajuste com o
mesmo peso) fica PIOR que o aleatorio no teste (S5: -0,072 pp; S15: -0,081 pp) e o rt a supera alem da margem. Esse criterio da mais peso as classes de lacuna grande; pode ser um criterio ruim e nao uma propriedade dos pools.
Esta analise e' POS HOC (nao ha' previsoes) e so' pergunta se a conclusao resiste a outros criterios de escolha da acao unica:
  c0  pre-registrado: menor lacuna media ao limite inferior (todas as instancias);
  c1  balanceado por classe: menor media, sobre as classes, da lacuna RELATIVA a media das acoes da classe (aproxima o ganho sobre o aleatorio);
  c2  por postos: menor posto medio entre as classes;
  c3  retrospectivo (usa os dados de TESTE; trapaca, e' um LIMITE SUPERIOR para qualquer acao unica): maior ganho medio sobre o aleatorio, mesmo peso por classe.
Para cada criterio: ganho do pool (media das 12 classes, pontos), ganho do rt contra a acao unica e IC90 estratificado (instancias dentro de cada classe; 2.000 reamostras).
Uso: python -m analysis.report_entrada286b_pool_sensitivity
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.cells_instance_tables import load_cells  # noqa: E402
from analysis.report_entrada180_agents import rel  # noqa: E402

R = ROOT / "results"
NB = 2000


def main():
    rng = np.random.default_rng(2862)
    cells = load_cells(include_tsp=False)
    rows, L = [], ["=" * 110, "SENSIBILIDADE DO POOL SEM ROTULO AO CRITERIO DE ESCOLHA DA ACAO UNICA (Entrada 286b, POS HOC)", "=" * 110]
    for sp in ("S5", "S15"):
        cs = [c for c in cells if c["coll"] == "sintetico" and c["space"] == sp]
        tune = pd.read_csv(R / f"agents_tune_{sp}_entrada180.csv")
        tune = tune[tune.method.str.startswith("arm")].copy()
        tune["a"] = tune.method.str[3:].astype(int)
        tune["gap"] = 100.0 * (tune.makespan - tune.lb) / tune.lb
        G = tune.groupby(["cls", "a"]).gap.mean().unstack("a")                      # classes x acoes
        rel_gap = G.sub(G.mean(axis=1), axis=0).div(G.mean(axis=1), axis=0)
        ranks = G.rank(axis=1)
        A = G.shape[1]
        test_gain = {}                                                               # ganho de teste de cada acao por classe
        for c in cs:
            cl = c["cell"].split("|", 1)[1]
            x = c["x"]
            test_gain[cl] = np.array([float(-rel(x[f"arm{a}"], x.rand).mean()) for a in range(A)])
        TG = pd.DataFrame(test_gain).T.loc[G.index]
        choose = {"c0 menor lacuna media (pre-registrado)": int(tune.groupby("a").gap.mean().idxmin()),
                  "c1 balanceado por classe (lacuna relativa)": int(rel_gap.mean(axis=0).idxmin()),
                  "c2 posto medio": int(ranks.mean(axis=0).idxmin()),
                  "c3 retrospectivo no teste (limite superior)": int(TG.mean(axis=0).idxmax())}
        cls_ = sorted(G.index)
        series_all = {}
        for c in cs:
            cl = c["cell"].split("|", 1)[1]
            x = c["x"]
            series_all[cl] = x
        idx = {cl: rng.integers(0, len(series_all[cl]), size=(NB, len(series_all[cl]))) for cl in cls_}
        vals = {cl: np.asarray(-rel(series_all[cl].val, series_all[cl].rand)) for cl in cls_}
        rts = {cl: np.asarray(-rel(series_all[cl].rt, series_all[cl].rand)) for cl in cls_}
        for name, a in choose.items():
            pool = {cl: np.asarray(-rel(series_all[cl][f"arm{a}"], series_all[cl].rand)) for cl in cls_}
            rt_vs = {cl: np.asarray(-rel(series_all[cl].rt, series_all[cl][f"arm{a}"])) for cl in cls_}
            for tag, ser in (("pool", pool), ("val", vals), ("rt", rts), ("rt_vs_pool", rt_vs)):
                pt = float(np.mean([ser[cl].mean() for cl in cls_]))
                bs = np.mean([ser[cl][idx[cl]].mean(axis=1) for cl in cls_], axis=0)
                lo, hi = np.percentile(bs, [5, 95])
                rows.append(dict(espaco=sp, criterio=name, acao=a, metrica=tag, ganho_pp=pt, ci90_lo=float(lo), ci90_hi=float(hi)))
        L += [f"--- {sp}: acoes unicas por criterio: " + "; ".join(f"{k.split()[0]} -> {v}" for k, v in choose.items()) + f"; acoes validadas distintas: {len(set(int(G.loc[cl].idxmin()) for cl in cls_))} ---"]
    D = pd.DataFrame(rows)
    piv = D.pivot_table(index=["espaco", "criterio", "acao"], columns="metrica", values=["ganho_pp", "ci90_lo", "ci90_hi"]).round(4)
    L += ["", D[D.metrica.isin(["pool", "rt_vs_pool"])].round(4).to_string(index=False), "",
          "leitura: 'pool' e' o ganho da acao unica sobre o aleatorio; 'rt_vs_pool' e' o ganho do rt sobre essa acao unica (positivo = o rt e' melhor); 'val' e 'rt' (ganho sobre o aleatorio) nao dependem do criterio:",
          D[(D.metrica.isin(["val", "rt"])) & D.criterio.str.startswith("c0")].round(4).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada286b_pool_sensitivity.txt").write_text(text + "\n", encoding="utf-8")
    D.to_csv(R / "entrada286b_pool_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
