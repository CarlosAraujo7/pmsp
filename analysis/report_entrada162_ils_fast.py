"""
analysis/report_entrada162_ils_fast.py
========================================
O IG vence o ILS (mecanismo) quando o ILS usa operadores eficientes? Ciclo 13, frente D3. Diario: Entrada 163 (pre-registro) e 164 (resultados).

PRE-REGISTRO (congelado ANTES de executar a fase --run; hashes em results/entrada163_prereg_hashes.txt)
--------------------------------------------------------------------------------------------------------
Dados: ig_random (T = 1000; results/results.csv; 12 classes x 100 reps) vs ils_fast_eqT (ILS com operadores vetorizados, T = 1000) e ils_fast_eqCPU (T_ils = round(1000/K_ils), K_ils = CPU/iter do ILS rapido / CPU/iter do IG;
calibracao sequencial) em results/ils_eq_fast_entrada162.csv. d = 100*(A-B)/B (negativo: A = IG melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
Familia primaria (Holm sobre 2, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0, o IG vence):
  E1  ig_random vs ils_fast_eqCPU  (CPU igual, ILS eficiente)
  E2  ig_random vs ils_fast_eqT    (iteracoes iguais, ILS eficiente)
Leitura pre-fixada: 'o achado do mecanismo (IG > ILS) e' robusto a implementacao do ILS' = E1 confirmada. Descritivas: K_ils por classe, T_ils, razao de CPU realizada, por classe, comparacao com o ILS padrao (lacos) quando existir.
Previsoes registradas antes de rodar: K_ils (ILS rapido / IG, CPU por iteracao) entre 0,4 e 4; E1 confirmada com d em [-0,6; -0,1]%; E2 confirmada com d perto do valor ja' conhecido (cerca de -0,5%), pois a iteracoes iguais a velocidade e' irrelevante.
Uso: python -m analysis.report_entrada162_ils_fast
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


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    res = pd.read_csv(R / "results.csv")
    e = pd.read_csv(R / "ils_eq_fast_entrada162.csv")
    ver = pd.read_csv(R / "ils_fast_verify_entrada162.csv")
    ratio = json.loads((R / "ils_fast_cpu_ratio_entrada162.json").read_text())
    keep = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[res.method == "ig_random"][keep], e[keep]]).drop_duplicates(["instance", "rep", "method"])
    W = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().dropna(subset=["ig_random", "ils_fast_eqT", "ils_fast_eqCPU"])
    assert len(W) == 1200, len(W)
    cpu = pd.concat([res[res.method == "ig_random"][["instance", "rep", "method", "cpu_time"]], e[["instance", "rep", "method", "cpu_time"]]]).drop_duplicates(["instance", "rep", "method"])
    cpu = cpu.pivot(index=["instance", "rep"], columns="method", values="cpu_time")
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "O IG VENCE O ILS QUANDO O ILS USA OPERADORES EFICIENTES? -- Entrada 163/164", "=" * 110,
         f"equivalencia (--verify): {int(ver.equal.sum())}/{len(ver)} makespans identicos (T = 150); razao de CPU padrao/rapido {ver.cpu_standard.sum() / ver.cpu_fast.sum():.1f}"]
    rows = []
    for pid, lab, b in [("E1", "ig_random vs ils_fast_eqCPU (CPU igual)", "ils_fast_eqCPU"), ("E2", "ig_random vs ils_fast_eqT (iteracoes iguais)", "ils_fast_eqT")]:
        t = test(W.ig_random, W[b])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    L += ["\n--- PRIMARIA (Holm sobre 2; confirmada = o IG vence) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False)]
    L.append(f"\nLeitura pre-fixada: 'o achado do mecanismo (IG > ILS) e' robusto a implementacao do ILS' (E1 confirmada): {'SIM' if P.set_index('id').loc['E1', 'confirmada'] else 'NAO'}")
    kk = pd.DataFrame([dict(classe=c, K_ils=v["K_ils"], T_ils=v["T_ils"]) for c, v in ratio.items()])
    L += ["\n--- calibracao ---", kk.to_string(index=False), f"razao de CPU realizada ils_fast_eqCPU / ig_random: {cpu.ils_fast_eqCPU.mean() / cpu.ig_random.mean():.2f}; ils_fast_eqT / ig_random: {cpu.ils_fast_eqT.mean() / cpu.ig_random.mean():.2f}"]
    sec = []
    for c, g in W.groupby("instance"):
        sec.append(dict(classe=c, d_CPU_igual=rel(g.ig_random, g.ils_fast_eqCPU).mean(), d_iter_igual=rel(g.ig_random, g.ils_fast_eqT).mean()))
    L += ["\n--- por classe (d %; negativo: IG melhor) ---", pd.DataFrame(sec).to_string(index=False)]
    text = "\n".join(L)
    print(text)
    (R / "entrada162_ils_fast.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada162_ils_fast_primary.csv", index=False)


if __name__ == "__main__":
    main()
