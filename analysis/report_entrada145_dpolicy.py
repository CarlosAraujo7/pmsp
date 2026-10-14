"""
analysis/report_entrada145_dpolicy.py
=======================================
Teste PROSPECTIVO da lei da folga no espaco de acoes 'tamanho de destruicao' em R||Cmax (benchmark padrao; LP-IG). Diario: Entrada 145.

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada145_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/dpolicy_test_entrada145.csv (700 instancias idx 2-6 x 9 metodos: fix_0..fix_4, rand_d, ql_d, ucb_d, ql_d_pre). A tabela Q de ql_d_pre vem das instancias de AJUSTE do mesmo (tipo, n).
Braco validado: val_n = a fracao escolhida por n no ajuste do LP-IG (results/lpig_choice_d_entrada141.json); val_tn = a fracao de menor razao media makespan/melhor conhecido por (tipo, n) nas 10 instancias de
ajuste do grupo (results/lpig_tune_d_entrada141.csv). d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; bootstrap IC90.
Familia primaria (Holm sobre 5, alpha = 0,05; 'confirmada' = p_holm < 0,05 e o sinal pre-fixado):
  D1  folga:               val_n vs rand_d              (d < 0: escolher d offline captura folga sobre o sorteio)
  D2  captura online:      ql_d vs rand_d               (d < 0)
  D3  online vs validacao: ql_d vs val_tn               (d < 0: o Q-Learning online VENCE a validacao por (tipo, n))
  D4  offline vs validacao: ql_d_pre vs val_tn          (d < 0: o Q-Learning pre-treinado VENCE a validacao por (tipo, n))
  D5  lei da folga:        Spearman entre a folga da celula (tipo x n; 28 celulas; folga = d medio do oraculo por instancia dos 5 bracos fixos vs rand_d) e o ganho do Q-Learning online da celula
                           (d medio de ql_d vs rand_d); p unilateral (rho > 0); confirmada = p_holm < 0,05 e rho >= 0,5
Leituras pre-fixadas: 'o aprendizado online captura a folga' = D2 confirmada, com razao de captura = d(ql_d vs rand_d) / d(val_n vs rand_d) reportada ('majoritaria' se >= 0,5);
'o aprendizado vence a validacao' = D3 ou D4 confirmada; 'a lei da folga se confirma prospectivamente' = D5 confirmada.
Descritivas: razao de captura de ql_d, ucb_d, ql_d_pre e val_tn; folga restante do oraculo por instancia sobre val_tn; GAP ao melhor conhecido por metodo; CPU medio; por tipo.
Uso: python -m analysis.report_entrada145_dpolicy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

R = ROOT / "results"
F = [0.01, 0.02, 0.05, 0.10, 0.20]
KEY = ["type", "instance"]


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    d = pd.read_csv(R / "dpolicy_test_entrada145.csv")
    d["instance"] = d.instance.astype(str)
    man = pd.read_csv(R / "rcmax_manifest_entrada137.csv", dtype={"instance": str})
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index().merge(man[KEY + ["best_known"]], on=KEY)
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 700 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    # bracos validados (so' com dados de ajuste)
    ch_n = json.loads((R / "lpig_choice_d_entrada141.json").read_text())
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv")
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    val_tn = {(t, int(n)): F.index(float(g.groupby("frac").ratio.mean().idxmin())) for (t, n), g in tune.groupby(["type", "n"])}
    w["val_n"] = [r[f"fix_{F.index(ch_n[str(r['n'])]['frac'])}"] for _, r in w.iterrows()]
    w["val_tn"] = [r[f"fix_{val_tn[(r['type'], int(r['n']))]}"] for _, r in w.iterrows()]
    fx = [f"fix_{k}" for k in range(5)]
    w["oracle"] = w[fx].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "TESTE PROSPECTIVO DA LEI DA FOLGA: POLITICA SOBRE O TAMANHO DE DESTRUICAO EM R||Cmax -- Entrada 145", "=" * 110, f"instancias de teste: {len(w)}"]
    rows = []
    for pid, lab, a, b in [("D1", "val_n vs rand_d (folga)", "val_n", "rand_d"), ("D2", "ql_d vs rand_d (captura online)", "ql_d", "rand_d"), ("D3", "ql_d vs val_tn (online vs validacao)", "ql_d", "val_tn"),
                           ("D4", "ql_d_pre vs val_tn (pre-treino vs validacao)", "ql_d_pre", "val_tn")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    w["cell"] = w["type"] + "|" + w["n"].astype(str)
    cell = pd.DataFrame({"hd": rel(w.oracle, w.rand_d), "gain": rel(w.ql_d, w.rand_d), "cell": w.cell}).groupby("cell").mean()
    rho, p_two = sstats.spearmanr(cell.hd, cell.gain)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    rows.append(dict(id="D5", comparison=f"Spearman(folga, ganho do ql_d) entre {len(cell)} celulas (tipo x n)", n=len(cell), better=np.nan, worse=np.nan, tie=np.nan, mean_d_pct=float(rho), ci90_lo=np.nan, ci90_hi=np.nan,
                     rank_biserial=np.nan, p_raw=float(p_one)))
    P = pd.DataFrame(rows).set_index("id").loc[["D1", "D2", "D3", "D4", "D5"]].reset_index()
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = [bool(r.reject and ((r.mean_d_pct >= 0.5) if r.id == "D5" else (r.mean_d_pct < 0))) for r in P.itertuples()]
    L += ["\n--- PRIMARIA (Holm sobre 5; D5: mean_d_pct = rho de Spearman, p unilateral) ---",
          P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    c = P.set_index("id")
    gain_val = rel(w.val_n, w.rand_d).mean()
    cap = lambda k: rel(w[k], w.rand_d).mean() / gain_val
    L.append(f"\nLeituras pre-fixadas: 'o aprendizado online captura a folga' (D2): {'SIM' if c.loc['D2', 'confirmada'] else 'NAO'} (razao de captura {cap('ql_d'):.2f}; "
             f"{'majoritaria' if cap('ql_d') >= 0.5 else 'minoritaria'}) | 'o aprendizado vence a validacao' (D3 ou D4): {'SIM' if (c.loc['D3', 'confirmada'] or c.loc['D4', 'confirmada']) else 'NAO'} | "
             f"'a lei da folga se confirma prospectivamente' (D5): {'SIM' if c.loc['D5', 'confirmada'] else 'NAO'} (rho = {rho:.3f})")
    L.append("\nGanho sobre rand_d (d %): " + ", ".join(f"{k} {rel(w[k], w.rand_d).mean():+.3f}" for k in ["val_n", "val_tn", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    L.append("Razao de captura (ganho / ganho de val_n): " + ", ".join(f"{k} {cap(k):.2f}" for k in ["val_tn", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    L.append(f"Folga restante do oraculo por instancia sobre val_tn: d = {rel(w.oracle, w.val_tn).mean():+.3f}%")
    L.append("GAP medio ao melhor conhecido (%): " + ", ".join(f"{k} {100 * ((w[k] - w.best_known) / w.best_known).mean():.3f}" for k in ["rand_d", "val_n", "val_tn", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    L.append("CPU medio por execucao (s): " + ", ".join(f"{k} {cpu[k].mean():.2f}" for k in ["rand_d", "fix_0", "ql_d", "ucb_d", "ql_d_pre"]))
    sec = []
    for typ, g in w.groupby("type"):
        sec.append(dict(type=typ, folga=rel(g.oracle, g.rand_d).mean(), val_tn=rel(g.val_tn, g.rand_d).mean(), ql_d=rel(g.ql_d, g.rand_d).mean(), ql_pre=rel(g.ql_d_pre, g.rand_d).mean(), ql_vs_val=rel(g.ql_d, g.val_tn).mean(),
                        pre_vs_val=rel(g.ql_d_pre, g.val_tn).mean()))
    L += ["\n--- por tipo (d % sobre rand_d, exceto as duas ultimas colunas: sobre val_tn) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada145_dpolicy.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada145_dpolicy_primary.csv", index=False)
    cell.to_csv(R / "entrada145_dpolicy_cells.csv")


if __name__ == "__main__":
    main()
