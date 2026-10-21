"""
analysis/report_entrada178_eqcpu_fast.py
==========================================
IG versus ILS a CPU igual com o ILS vetorizado, distribuicoes fina U[1,10^4] e grossa U[1,100] (Ciclo 14, frente J2). Diario: Entrada 178 (pre-registro) e 179 (resultados).

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada178_prereg_hashes.txt)
------------------------------------------------------------------------------------------
Contexto: a subsecao 'IG versus ILS at equal CPU time' do suplemento (Entrada 110) deu ao IG T = 1000 K(n) iteracoes, K = CPU(ILS)/CPU(IG) medido com operadores do ILS em laços (K ate' 24,8), e concluiu que a vantagem do ILS em tempos finos persiste a CPU igual
(d = +0,086% e +0,138%) e que a do IG em tempos grossos cresce (-0,703% e -0,676%). Com o ILS vetorizado (resultados identicos, 6,2 vezes mais rapido) a razao e' <= 1,1 (Entrada 174) e o ILS recebe MAIS iteracoes a CPU igual. Aqui: IG a T = 1000 (results/new_distributions_entrada108.csv,
metodos ig_ql e ig_random) contra o ILS vetorizado com T_ILS = round(1000 / K_fast(n)) (experiment/run_equal_cpu_fast_entrada178.py; K_fast congelado la'; results/equal_cpu_fast_entrada178.csv; mesmas instancias, 5 tamanhos x 30 reps = 150 por celula).
d = 100*(IG - ILS)/ILS em makespan (negativo: IG melhor); Wilcoxon pareado bicaudal; nulos descartados; Holm sobre 4; 'confirmada' = p_holm < 0,05 (a direcao e' avaliada separadamente).
Testes (Holm 4): E1 u1_10000 ig_ql vs ils_ql_eqfast; E2 u1_10000 ig_random vs ils_random_eqfast; E3 u1_100 ig_ql vs ils_ql_eqfast; E4 u1_100 ig_random vs ils_random_eqfast.
Leituras pre-fixadas: 'a vantagem do ILS em tempos finos persiste e nao diminui' = E1 e E2 com d > 0 e p_holm < 0,05 e d(E1), d(E2) >= os da Entrada 110 (+0,086 e +0,138); 'a vantagem do IG em tempos grossos persiste, mas diminui' = E3 e E4 com d < 0, p_holm < 0,05 e |d| menor que os da Entrada 110 (0,703 e 0,676).
Previsoes registradas antes de rodar: E1 d em [+0,08; +0,25]; E2 d em [+0,14; +0,45]; E3 d em [-0,55; -0,15]; E4 d em [-0,55; -0,15]; todos os quatro Holm-significativos; as duas leituras SIM.
Uso: python -m analysis.report_entrada178_eqcpu_fast
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

R = ROOT / "results"


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def main() -> None:
    nd = pd.read_csv(R / "new_distributions_entrada108.csv")
    fast = pd.read_csv(R / "equal_cpu_fast_entrada178.csv")
    old = pd.read_csv(R / "entrada110_equal_cpu_primary.csv")
    key = ["dist", "n", "m", "rep"]
    rows = []
    for pid, dist, ig, ils in [("E1", "u1_10000", "ig_ql", "ils_ql_eqfast"), ("E2", "u1_10000", "ig_random", "ils_random_eqfast"), ("E3", "u1_100", "ig_ql", "ils_ql_eqfast"), ("E4", "u1_100", "ig_random", "ils_random_eqfast")]:
        a = nd[(nd.method == ig) & (nd.dist == dist)][key + ["makespan"]].rename(columns={"makespan": "ig"})
        b = fast[(fast.method == ils) & (fast.dist == dist)][key + ["makespan", "T"]].rename(columns={"makespan": "ils"})
        m = a.merge(b, on=key)
        assert len(m) == 150, (pid, len(m))
        d = rel(m.ig, m.ils)
        nz = d[d != 0]
        p = float(sstats.wilcoxon(nz).pvalue)
        rows.append(dict(id=pid, dist=dist, comparison=f"{ig} vs {ils}", n=len(m), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()),
                         rank_biserial=rank_biserial_paired(m.ig, m.ils), p_raw=p, T_ils_min=int(m["T"].min()), T_ils_max=int(m["T"].max())))
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    o = {("u1_10000", "ql"): old.iloc[0], ("u1_10000", "rand"): old.iloc[1], ("u1_100", "ql"): old.iloc[2], ("u1_100", "rand"): old.iloc[3]}
    P["d_entrada110_lacos"] = [float(old[(old.dist == r.dist) & (old.comparison.str.contains("ql" if "ql" in r.comparison.split()[0] else "random"))].mean_d_pct.iloc[0]) for r in P.itertuples()]
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "IG VERSUS ILS A CPU IGUAL COM O ILS VETORIZADO (tempos finos e grossos) -- Entrada 178/179", "=" * 110,
         P[["id", "dist", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "d_entrada110_lacos", "rank_biserial", "p_raw", "p_holm", "reject", "T_ils_min", "T_ils_max"]].to_string(index=False)]
    e = P.set_index("id")
    fine = bool((e.loc[["E1", "E2"], "mean_d_pct"] > 0).all() and e.loc[["E1", "E2"], "reject"].all() and e.loc["E1", "mean_d_pct"] >= e.loc["E1", "d_entrada110_lacos"] and e.loc["E2", "mean_d_pct"] >= e.loc["E2", "d_entrada110_lacos"])
    coarse = bool((e.loc[["E3", "E4"], "mean_d_pct"] < 0).all() and e.loc[["E3", "E4"], "reject"].all() and (e.loc[["E3", "E4"], "mean_d_pct"].abs() < e.loc[["E3", "E4"], "d_entrada110_lacos"].abs()).all())
    L += ["", f"Leitura pre-fixada 'a vantagem do ILS em tempos finos persiste e nao diminui': {'SIM' if fine else 'NAO'}",
          f"Leitura pre-fixada 'a vantagem do IG em tempos grossos persiste, mas diminui': {'SIM' if coarse else 'NAO'}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada178_eqcpu_fast.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada178_eqcpu_fast_primary.csv", index=False)


if __name__ == "__main__":
    main()
