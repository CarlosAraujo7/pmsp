"""
analysis/report_entrada162_neural.py
======================================
Aprendizes neurais no espaco 'tamanho de destruicao' do LP-IG em R||Cmax (Ciclo 13, frente E). Diario: Entrada 163 (pre-registro) e 165 (resultados).

PRE-REGISTRO (congelado ANTES de executar a fase --run; hashes em results/entrada163_prereg_hashes.txt)
--------------------------------------------------------------------------------------------------------
Dados: results/dpolicy_test_entrada145.csv (700 instancias de teste; bracos fixos fix_0..4, rand_d, ql_d, ucb_d, ql_d_pre) + results/nq_test_entrada162.csv (nq_d: Q-learning com rede neural online; lr escolhido no ajuste) +
results/neural_choice_entrada162.json (val_mlp: braco previsto por um MLP treinado OFFLINE so' nas 280 instancias de ajuste, features da matriz de tempos sem o tipo; alvos padronizados por braco).
Nota de desenho (somente dados de ajuste, antes de qualquer teste): a 1a versao do MLP, com alvos brutos (razoes ~1,01), previa quase uma constante (razao media dentro da amostra 1,01424); a padronizacao dos alvos por braco corrigiu
(dentro da amostra 1,00972; validacao cruzada de 5 dobras 1,01070, a ser comparada com kNN 1,01044 e melhor braco por celula 1,00995 da Entrada 159).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90. val_tn = braco validado por (tipo, n) no ajuste (como na Entrada 145).
Familia primaria (Holm sobre 4, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  N1  nq_d vs rand_d        (a politica neural online captura a folga)
  N2  val_tn vs nq_d        (a validacao por celula vence a politica neural online)
  N3  val_mlp vs val_tn     (o seletor MLP offline instancia-condicional supera a validacao por celula)
  N4  val_mlp vs rand_d     (o seletor MLP offline captura a folga sobre o sorteio)
Leituras pre-fixadas: 'um aprendiz neural muda a conclusao' = N3 confirmada OU (N1 confirmada com captura >= 0,5 do ganho validado); 'a falha nao e' de classe de aprendiz (nem com rede neural)' = N1 nao confirmada E N2 confirmada.
Previsoes registradas antes de rodar: N1 NAO confirmada (d >= -0,05%; provavelmente perto de +0,05 a +0,15 como o Q-learning tabular); N2 confirmada (d em [-0,35; -0,15]%); N3 NAO confirmada com d >= 0 (em [0; +0,15]%);
N4 confirmada com d em [-0,12; -0,04]%.
Descritivas: razoes de captura, Spearman (folga x ganho) por celula para val_mlp e nq_d, CPU de nq_d vs rand_d, bracos previstos.
Uso: python -m analysis.report_entrada162_neural
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
    cols = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time"]
    d = pd.concat([pd.read_csv(R / "dpolicy_test_entrada145.csv", dtype={"instance": str})[cols], pd.read_csv(R / "nq_test_entrada162.csv", dtype={"instance": str})[cols]])
    w = d.pivot(index=KEY + ["m", "n"], columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=KEY, columns="method", values="cpu_time")
    assert len(w) == 700 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    val_tn = {(t, int(n)): F.index(float(g.groupby("frac").ratio.mean().idxmin())) for (t, n), g in tune.groupby(["type", "n"])}
    ch = json.loads((R / "neural_choice_entrada162.json").read_text())
    chn = json.loads((R / "nq_choice_entrada162.json").read_text())
    w["val_tn"] = [r[f"fix_{val_tn[(r['type'], int(r['n']))]}"] for _, r in w.iterrows()]
    w["val_mlp"] = [r[f"fix_{ch['pred'][r['type'] + '|' + r['instance']]}"] for _, r in w.iterrows()]
    fx = [f"fix_{k}" for k in range(5)]
    w["oracle"] = w[fx].min(axis=1)
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "APRENDIZES NEURAIS: tamanho de destruicao do LP-IG em R||Cmax -- Entrada 163/165", "=" * 110,
         f"instancias de teste: {len(w)} | lr da rede online escolhido no ajuste: {chn['lr']} (razao media {chn['mean_ratio']}) | MLP offline: razao no ajuste dentro da amostra {ch['in_sample_mean_ratio']:.5f}, "
         f"validacao cruzada {ch['cv_mean_ratio']:.5f}"]
    rows = []
    for pid, lab, a, b in [("N1", "nq_d vs rand_d (rede neural online)", "nq_d", "rand_d"), ("N2", "val_tn vs nq_d (validacao vs rede neural online)", "val_tn", "nq_d"),
                           ("N3", "val_mlp vs val_tn (MLP offline vs validacao por celula)", "val_mlp", "val_tn"), ("N4", "val_mlp vs rand_d (MLP offline, captura)", "val_mlp", "rand_d")]:
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
    changed = bool(c.loc["N3", "confirmada"] or (c.loc["N1", "confirmada"] and cap("nq_d") >= 0.5))
    L.append(f"\nLeituras pre-fixadas: 'um aprendiz neural muda a conclusao' (N3, ou N1 com captura >= 0,5): {'SIM' if changed else 'NAO'} | "
             f"'a falha nao e' de classe de aprendiz' (N1 nao e N2 sim): {'SIM' if ((not c.loc['N1', 'confirmada']) and c.loc['N2', 'confirmada']) else 'NAO'}")
    L.append("\nGanho sobre rand_d (d %): " + ", ".join(f"{k} {gain(k):+.3f}" for k in ["val_tn", "val_mlp", "nq_d", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    L.append("Razao de captura (ganho / ganho de val_tn): " + ", ".join(f"{k} {cap(k):.2f}" for k in ["val_mlp", "nq_d", "ql_d", "ucb_d", "ql_d_pre", "oracle"]))
    w["cell"] = w["type"] + "|" + w["n"].astype(str)
    hd = rel(w.oracle, w.rand_d)
    for k in ["val_mlp", "nq_d"]:
        cell = pd.DataFrame({"hd": hd, "g": rel(w[k], w.rand_d), "cell": w.cell}).groupby("cell").mean()
        rho, p = sstats.spearmanr(cell.hd, cell.g)
        L.append(f"Spearman(folga da celula, ganho de {k}) em {len(cell)} celulas: rho = {rho:.3f} (p bicaudal {p:.3g}); ganho negativo = melhor que o sorteio")
    L.append(f"CPU medio (s): rand_d {cpu.rand_d.mean():.2f}, nq_d {cpu.nq_d.mean():.2f} | razao nq_d/rand_d {cpu.nq_d.mean() / cpu.rand_d.mean():.2f}")
    sel = np.bincount(list(ch["pred"].values()), minlength=5)
    L.append("bracos previstos pelo MLP offline no teste (d/n 0,01 ... 0,20): " + ", ".join(f"{F[i]:g}: {sel[i]}" for i in range(5)))
    text = "\n".join(L)
    print(text)
    (R / "entrada162_neural.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada162_neural_primary.csv", index=False)


if __name__ == "__main__":
    main()
