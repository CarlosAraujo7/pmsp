"""
analysis/report_entrada115_external_confirm2.py
==================================================
SEGUNDO teste confirmatorio externo (estado informativo + replicacao do original). Diario: Entrada 115.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada115_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: instancias NUNCA executadas (manifest_confirm2_entrada115.csv: B 980, C 1.050 -- 75 classes de C nao amostradas antes --, E 660);
limiares do estado relativo calibrados em instancias disjuntas (manifest_calib_entrada115.csv), pooled entre familias.
Estatistica por instancia: diferenca de GAP (pontos percentuais) entre metodos pareados; Wilcoxon bicaudal por familia, diferencas nulas
descartadas; 'confirmada' = p_holm<0,05 e media da diferenca < 0 (Holm sobre os 6 testes abaixo).
  R_B, R_C, R_E : ig_ql_rel vs ig_random   (o Q-Learning com ESTADO INFORMATIVO supera a selecao aleatoria?)
  O_B, O_C, O_E : ig_ql     vs ig_random   (replicacao independente do teste da Entrada 90 com o estado original, instancias novas)
Leitura pre-especificada:
  * 'efeito pequeno do Q-Learning replica' = O_B, O_C, O_E confirmadas (3/3), em 'sentido ... 2/3' = parcial;
  * 'estado informativo ajuda' = R_f confirmada em >=2 familias E, pooled, ig_ql_rel vs ig_ql com d relativo de GAP <= -5% (p<0,05, sem Holm);
    caso contrario 'estado informativo nao agrega ao original'.
Descritivas: GAP medio por familia e metodo; diferencas em % do GAP; contagens melhor/pior/empate; rank-biserial.
Uso: python -m analysis.report_entrada115_external_confirm2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

EXT = ROOT / "external_benchmarks" / "pcmax_symmetry2025"


def test(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    rel = 100 * (x.mean() - y.mean()) / y.mean() if y.mean() > 0 else float("nan")
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_diff_pp=float(d.mean()),
                rel_gap_pct=float(rel), rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    d = pd.read_csv(EXT / "results_external_confirm2_entrada115.csv").drop_duplicates(["member", "method"])
    man = pd.read_csv(EXT / "manifest_confirm2_entrada115.csv")
    w = d.pivot(index="member", columns="method", values="gap_pct").reset_index().merge(man[["member", "family"]], on="member")
    assert len(w) == len(man) and w[["ig_ql_rel", "ig_ql", "ig_random"]].isna().sum().sum() == 0, (len(w), len(man))
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "SEGUNDO TESTE CONFIRMATORIO EXTERNO -- estado informativo e replicacao, Entrada 115", "=" * 110,
         f"instancias: {len(w)} {w.groupby('family').size().to_dict()}", "limiares pooled: " + open(ROOT / "results" / "state_calibration_external_entrada115.json").read().replace("\n", " ")[:260]]
    rows = []
    for f, g in w.groupby("family"):
        for pid, a in [("R", "ig_ql_rel"), ("O", "ig_ql")]:
            t = test(g[a], g.ig_random)
            t.update(id=f"{pid}_{f}", family=f, comparison=f"{a} vs ig_random")
            rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_diff_pp < 0)
    L += ["\n--- PRIMARIA (Holm sobre 6) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_diff_pp", "rel_gap_pct", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    nO = int(P[P.id.str.startswith("O")].confirmada.sum())
    nR = int(P[P.id.str.startswith("R")].confirmada.sum())
    L.append(f"\nREPLICACAO do efeito pequeno (O_*): {nO}/3 confirmadas -> {'REPLICADO' if nO == 3 else 'PARCIAL' if nO >= 2 else 'NAO replicado'}")
    pooled = test(w.ig_ql_rel, w.ig_ql)
    L.append(f"ESTADO INFORMATIVO: R_* confirmadas = {nR}/3; pooled ig_ql_rel vs ig_ql (original): d GAP = {pooled['rel_gap_pct']:+.2f}% (p={pooled['p_raw']:.3g}) -> "
             + ("'estado informativo ajuda'" if (nR >= 2 and pooled["rel_gap_pct"] <= -5 and pooled["p_raw"] < 0.05) else "'estado informativo NAO agrega ao original'"))
    sec = []
    for f, g in list(w.groupby("family")) + [("ALL", w)]:
        t = test(g.ig_ql_rel, g.ig_ql)
        t.update(family=f, comparison="ig_ql_rel vs ig_ql")
        sec.append(t)
        for a in ["ig_ql_rel", "ig_ql", "ig_random"]:
            pass
    S = pd.DataFrame(sec)
    L += ["\n--- ig_ql_rel vs ig_ql (descritiva) ---", S[["family", "n", "better", "worse", "tie", "mean_diff_pp", "rel_gap_pct", "rank_biserial", "p_raw"]].to_string(index=False)]
    G = w.groupby("family")[["ig_random", "ig_ql", "ig_ql_rel"]].mean()
    G.loc["ALL"] = w[["ig_random", "ig_ql", "ig_ql_rel"]].mean()
    L += ["\n--- GAP medio (%) ---", G.to_string()]
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada115_external_confirm2.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada115_external_confirm2_primary.csv", index=False)


if __name__ == "__main__":
    main()
