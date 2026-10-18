"""
analysis/report_entrada159_ctx.py
===================================
Aprendizes de OUTRA CLASSE no espaco 'tamanho de destruicao' do LP-IG em R||Cmax (Ciclo 12, frente C). Diario: Entrada 159 (pre-registro) e 160 (resultados).

PRE-REGISTRO (congelado ANTES de executar o teste; hashes em results/entrada159_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------
Dados: results/dpolicy_test_entrada145.csv (700 instancias de teste; bracos fixos fix_0..4, rand_d, ql_d, ucb_d, ql_d_pre) + results/ctx_test_entrada159.csv (linucb_d; 700 execucoes novas, alpha escolhido no ajuste) +
results/ctx_choice_entrada159.json (bracos previstos no teste pelo seletor instancia-condicional treinado OFFLINE so' nas 280 instancias de ajuste; modelo escolhido por validacao cruzada no ajuste).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90. val_tn = braco validado por (tipo, n) no ajuste (como na Entrada 145);
val_ctx = braco previsto pelo seletor offline instancia-condicional (features da matriz de tempos, sem o tipo do gerador).
Familia primaria (Holm sobre 4, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  C1  seletor offline instancia-condicional vs validacao por (tipo, n):        val_ctx vs val_tn
  C2  o seletor offline instancia-condicional captura a folga sobre o sorteio: val_ctx vs rand_d
  C3  aprendiz online de aproximacao de funcao captura a folga:               linucb_d vs rand_d
  C4  a validacao por (tipo, n) vence o aprendiz online contextual:           val_tn vs linucb_d
Leituras pre-fixadas: 'um aprendiz de outra classe muda a conclusao da atribuicao' = C1 confirmada (offline instancia-condicional supera a validacao por celula) OU C3 confirmada com razao de captura >= 0,5;
caso contrario, 'o resultado nao e' artefato do aprendiz tabular' = C3 nao confirmada E C4 confirmada. Equivalencia: IC90 dentro de +-0,05% => 'equivalente'.
Previsoes registradas antes de rodar (a validacao cruzada do AJUSTE ja' tinha sido vista ao registrar: razao media do braco escolhido por kNN 1,01044, ridge 1,0113, floresta 1,01078, melhor braco por celula 1,00995,
melhor braco global 1,01171, oraculo 1,00743): C1 NAO confirmada, com d >= 0 (as features nao agregam ao tipo e n); C2 confirmada (d ~ -0,15%); C3 NAO confirmada (d >= 0); C4 confirmada (d ~ -0,2%).
Descritivas: razoes de captura (ganho sobre rand_d de cada metodo / ganho de val_tn), ql_d, ucb_d, ql_d_pre, oraculo, Spearman (folga da celula x ganho) para val_ctx e linucb_d (28 celulas), CPU de linucb_d vs rand_d.
Uso: python -m analysis.report_entrada159_ctx
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
    d = pd.read_csv(R / "dpolicy_test_entrada145.csv", dtype={"instance": str})
    e = pd.read_csv(R / "ctx_test_entrada159.csv", dtype={"instance": str})
    d = pd.concat([d[["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time"]], e[["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time"]]])
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 700 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    val_tn = {(t, int(n)): F.index(float(g.groupby("frac").ratio.mean().idxmin())) for (t, n), g in tune.groupby(["type", "n"])}
    ch = json.loads((R / "ctx_choice_entrada159.json").read_text())
    w["val_tn"] = [r[f"fix_{val_tn[(r['type'], int(r['n']))]}"] for _, r in w.iterrows()]
    w["val_ctx"] = [r[f"fix_{ch['pred'][r['type'] + '|' + r['instance']]}"] for _, r in w.iterrows()]
    fx = [f"fix_{k}" for k in range(5)]
    w["oracle"] = w[fx].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "APRENDIZES DE OUTRA CLASSE: tamanho de destruicao do LP-IG em R||Cmax -- Entrada 159/160", "=" * 110,
         f"instancias de teste: {len(w)} | modelo offline escolhido por CV no ajuste: {ch['model']} | CV (razao media do braco escolhido): " + ", ".join(f"{k} {v:.5f}" for k, v in ch["cv_mean_ratio"].items())]
    rows = []
    for pid, lab, a, b in [("C1", "val_ctx vs val_tn (offline instancia-condicional vs validacao por celula)", "val_ctx", "val_tn"), ("C2", "val_ctx vs rand_d (captura da folga)", "val_ctx", "rand_d"),
                           ("C3", "linucb_d vs rand_d (aprendiz online contextual)", "linucb_d", "rand_d"), ("C4", "val_tn vs linucb_d (validacao vs online contextual)", "val_tn", "linucb_d")]:
        t = test(w[a], w[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["equivalente_0.05"] = (P.ci90_lo > -0.05) & (P.ci90_hi < 0.05)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05"]].to_string(index=False)]
    c = P.set_index("id")
    gain = lambda k: rel(w[k], w.rand_d).mean()
    cap = lambda k: gain(k) / gain("val_tn")
    changed = bool(c.loc["C1", "confirmada"] or (c.loc["C3", "confirmada"] and cap("linucb_d") >= 0.5))
    L.append(f"\nLeitura pre-fixada: 'um aprendiz de outra classe muda a conclusao' (C1 ou C3 com captura >= 0,5): {'SIM' if changed else 'NAO'} | "
             f"'nao e' artefato do aprendiz tabular' (C3 nao e C4 sim): {'SIM' if ((not c.loc['C3', 'confirmada']) and c.loc['C4', 'confirmada']) else 'NAO'}")
    L.append("\nGanho sobre rand_d (d %): " + ", ".join(f"{k} {gain(k):+.3f}" for k in ["val_tn", "val_ctx", "linucb_d", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    L.append("Razao de captura (ganho / ganho de val_tn): " + ", ".join(f"{k} {cap(k):.2f}" for k in ["val_ctx", "linucb_d", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    w["cell"] = w["type"] + "|" + w["n"].astype(str)
    hd = rel(w.oracle, w.rand_d)
    for k in ["val_ctx", "linucb_d"]:
        cell = pd.DataFrame({"hd": hd, "g": rel(w[k], w.rand_d), "cell": w.cell}).groupby("cell").mean()
        rho, p = sstats.spearmanr(cell.hd, cell.g)
        L.append(f"Spearman(folga da celula, ganho de {k}) em {len(cell)} celulas: rho = {rho:.3f} (p bicaudal {p:.3g}); sinal: ganho negativo = melhor que o sorteio")
    L.append(f"CPU medio (s): rand_d {cpu.rand_d.mean():.2f}, linucb_d {cpu.linucb_d.mean():.2f}, ql_d {cpu.ql_d.mean():.2f} | razao linucb_d/rand_d {cpu.linucb_d.mean() / cpu.rand_d.mean():.2f}")
    sel = np.bincount(list(ch["pred"].values()), minlength=5)
    L.append("bracos previstos pelo seletor offline no teste (d/n 0,01 ... 0,20): " + ", ".join(f"{F[i]:g}: {sel[i]}" for i in range(5)))
    text = "\n".join(L)
    print(text)
    (R / "entrada159_ctx.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada159_ctx_primary.csv", index=False)


if __name__ == "__main__":
    main()
