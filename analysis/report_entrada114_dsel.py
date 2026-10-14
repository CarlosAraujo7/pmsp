"""
analysis/report_entrada114_dsel.py
====================================
A politica aprendida importa quando o espaco de acoes tem folga? (acao = tamanho de destruicao d). Diario: Entrada 114.

PRE-REGISTRO (congelado ANTES de executar; hashes em results/entrada114_prereg_hashes.txt)
-----------------------------------------------------------------------------------------
Dados: TESTE = 12 classes x 100 repeticoes = 1.200 instancias (mesmas de results.csv); AJUSTE = 12 classes x 30 instancias
SEPARADAS (seeds 90000+), so' os 5 metodos de d fixo. Em cada classe, `cvfixed` = o d fixo de menor makespan medio no AJUSTE,
avaliado no TESTE (validacao fora da amostra). d relativo = 100*(Cmax_A - Cmax_B)/Cmax_B (negativo: A melhor); Wilcoxon pareado
bicaudal pooled sobre as 1.200 instancias, diferencas nulas descartadas.

Familia confirmatoria (Holm sobre 4, alpha=0,05; "confirmada" = p_holm<0,05 e d<0 [e "relevante" se d<=-0,05%]):
  A1  cvfixed vs igd_random   (existe folga fora da amostra no tamanho de destruicao? previsao: SIM, relevante)
  A2  igd_ql_cal vs igd_random   (o Q-Learning captura parte dessa folga? SEM previsao direcional pre-fixada: ambos os resultados
                                  sao informativos -- ver regras de leitura)
  A3  igd_ucb vs igd_random   (idem para um bandit sem estado)
  A4  igd_ql_cal vs cvfixed   (bicaudal: o aprendizado supera/alcanca o ajuste fixo por validacao?)
Regras de leitura pre-especificadas:
  * "A folga explica o nulo" e' SUPORTADA se A1 for confirmada e relevante (a folga existe quando a acao e' d) E, ao mesmo tempo, o
    nulo das 3 estrategias (Entrada 104, folga 0,04%) se mantem -- i.e. a falta de folga e' propriedade do espaco de acoes.
  * "A politica importa quando ha folga" e' SUPORTADA se A1 e A2 forem confirmadas e relevantes; REFUTADA (a politica nao
    captura folga mesmo quando ela existe) se A1 confirmada/relevante e A2 nao.
Secundarias (descritivas): igd_ql_def vs igd_random; fracao da folga capturada = (media_random - media_X)/(media_random - media_cvfixed)
por classe e pooled; melhor fixo no proprio TESTE (otimista) e oraculo por instancia (min dos 5); cvfixed e igd_ql_cal vs os padroes do
artigo (ig_random e ig_ql, d=0,3 n com 3 estrategias; results.csv); d escolhido por classe.

Uso: python -m analysis.report_entrada114_dsel
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

MARGIN = 0.05
FIX = ["igd_fix05", "igd_fix10", "igd_fix20", "igd_fix30", "igd_fix40"]
FR = dict(zip(FIX, [0.05, 0.10, 0.20, 0.30, 0.40]))


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()),
                ci90_lo=float(lo), ci90_hi=float(hi), rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    te = pd.read_csv(ROOT / "results" / "dsel_test_entrada114.csv")
    tu = pd.read_csv(ROOT / "results" / "dsel_tuning_entrada114.csv")
    res = pd.read_csv(ROOT / "results" / "results.csv")
    w = te.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    assert len(w) == 1200 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    choice = {cls: min(FIX, key=lambda k: g[g.method == k].makespan.mean()) for cls, g in tu.groupby("instance")}
    w["cvfixed"] = [r[choice[r["instance"]]] for _, r in w.iterrows()]
    w["best_in_test"] = [min(r[k] for k in FIX) for _, r in w.iterrows()]   # oraculo por instancia
    bt = {cls: min(FIX, key=lambda k: g[k].mean()) for cls, g in w.groupby("instance")}
    w["bestfixed_test"] = [r[bt[r["instance"]]] for _, r in w.iterrows()]
    base = res[res.method.isin(["ig_random", "ig_ql"])][["instance", "rep", "method", "makespan"]].pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    w = w.merge(base, on=["instance", "rep"], how="left")
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "A POLITICA IMPORTA QUANDO HA FOLGA? (acao = tamanho de destruicao) -- teste pre-registrado, Entrada 114", "=" * 110,
         f"instancias de teste: {len(w)}", "d escolhido por classe (ajuste): " + ", ".join(f"{c.split('_')[1][:-1]}x{c.split('_')[2][:-1]}:{FR[k]:.2f}" for c, k in sorted(choice.items(), key=lambda kv: int(kv[0].split('_')[1][:-1])))]
    prim = []
    for pid, lab, a, b in [("A1", "cvfixed vs igd_random", w.cvfixed, w.igd_random), ("A2", "igd_ql_cal vs igd_random", w.igd_ql_cal, w.igd_random),
                           ("A3", "igd_ucb vs igd_random", w.igd_ucb, w.igd_random), ("A4", "igd_ql_cal vs cvfixed", w.igd_ql_cal, w.cvfixed)]:
        t = test(a, b)
        t.update(id=pid, comparison=lab)
        prim.append(t)
    P = pd.DataFrame(prim)
    h = holm_bonferroni(P.p_raw.fillna(1.0).tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["relevante"] = P.confirmada & (P.mean_d_pct <= -MARGIN)
    L += ["\n--- PRIMARIA (Holm sobre 4) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "relevante"]].to_string(index=False)]
    a1 = bool(P.loc[P.id == "A1", "relevante"].iloc[0])
    a2 = bool(P.loc[P.id == "A2", "relevante"].iloc[0])
    L.append(f"\nREGRA 'a folga existe no espaco de acoes d' (A1 confirmada e relevante): {'SUPORTADA' if a1 else 'NAO suportada'}")
    L.append(f"REGRA 'a politica importa quando ha folga' (A1 e A2 confirmadas e relevantes): " + ("SUPORTADA" if (a1 and a2) else ("REFUTADA (folga existe mas a politica nao a captura)" if a1 else "NAO aplicavel (A1 nao confirmada)")))
    # secundarias
    sec = []
    for lab, a, b in [("igd_ql_def vs igd_random", w.igd_ql_def, w.igd_random), ("best fixed in TEST (optimistic) vs igd_random", w.bestfixed_test, w.igd_random),
                      ("per-instance oracle of the 5 d vs igd_random", w.best_in_test, w.igd_random),
                      ("cvfixed vs ig_random (default d=0.3n, 3 strategies)", w.cvfixed, w.ig_random), ("igd_ql_cal vs ig_random", w.igd_ql_cal, w.ig_random),
                      ("cvfixed vs ig_ql (default)", w.cvfixed, w.ig_ql), ("igd_ql_cal vs ig_ql (default)", w.igd_ql_cal, w.ig_ql),
                      ("igd_fix30 vs igd_random", w.igd_fix30, w.igd_random)]:
        ok = ~(pd.isna(a) | pd.isna(b))
        t = test(np.asarray(a)[ok], np.asarray(b)[ok])
        t["comparison"] = lab
        sec.append(t)
    S = pd.DataFrame(sec)
    L += ["\n--- SECUNDARIAS (descritivas, sem Holm) ---", S[["comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw"]].to_string(index=False)]
    rows = []
    for cls, g in w.groupby("instance"):
        mr, mc = g.igd_random.mean(), g.cvfixed.mean()
        row = dict(instance=cls, d_cv=FR[choice[cls]], mean_random=mr, mean_cvfixed=mc, mean_ql=g.igd_ql_cal.mean(), mean_ucb=g.igd_ucb.mean())
        row["headroom_pct"] = 100 * (mr - mc) / mr
        den = mr - mc
        row["ql_captures"] = (mr - g.igd_ql_cal.mean()) / den if abs(den) > 1e-9 else np.nan
        row["ucb_captures"] = (mr - g.igd_ucb.mean()) / den if abs(den) > 1e-9 else np.nan
        for k in FIX:
            row[k] = 100 * (g[k].mean() - mr) / mr
        rows.append(row)
    C = pd.DataFrame(rows).sort_values("instance")
    L += ["\n--- por classe: folga (random -> cvfixed, %), fracao capturada por QL/UCB, d relativo de cada d fixo vs random (%) ---", C.to_string(index=False)]
    pm, pc = w.igd_random.mean(), w.cvfixed.mean()
    L.append(f"\nfracao pooled da folga capturada: QL = {(pm - w.igd_ql_cal.mean()) / (pm - pc):.3f}; UCB = {(pm - w.igd_ucb.mean()) / (pm - pc):.3f} (em makespan medio; classes de tamanhos distintos pesam diferente)")
    text = "\n".join(L)
    print(text)
    (ROOT / "results" / "entrada114_dsel.txt").write_text(text, encoding="utf-8")
    P.to_csv(ROOT / "results" / "entrada114_dsel_primary.csv", index=False)
    C.to_csv(ROOT / "results" / "entrada114_dsel_by_class.csv", index=False)


if __name__ == "__main__":
    main()
