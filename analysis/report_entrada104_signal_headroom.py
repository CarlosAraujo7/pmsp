"""
analysis/report_entrada104_signal_headroom.py
===============================================
WP4: folga (headroom) e sinal de aprendizado no IG-QL. Diario: Entrada 104.

PRE-REGISTRO (congelado ANTES de executar as variantes; hashes em results/entrada104_prereg_hashes.txt)
---------------------------------------------------------------------------------------------------------
Dados: 12 classes sinteticas x 100 repeticoes = 1.200 instancias, T=1000, partida LPT (como o ig_ql original),
pareadas com ig_random (results.csv). d = 100*(Cmax_A - Cmax_B)/Cmax_B (negativo: A melhor). Wilcoxon pareado
bicaudal pooled, diferencas nulas descartadas.

(A) FOLGA. ig_fix{0,1,2} = IG com cada estrategia de destruicao fixa. Para cada classe, a melhor estrategia fixa
    (menor makespan medio, ESCOLHIDA NA PROPRIA AMOSTRA = cota otimista) vs ig_random; "folga por classe" = d medio.
    Folga agregada = media (ponderada por instancias) das folgas por classe. Tambem reportado o oraculo POR INSTANCIA
    (min das 3 fixas em cada instancia; cota superior inatingivel por qualquer seletor sem previsao).
    Regra: explicacao A (estrategias quase intercambiaveis) SUPORTADA se a folga agregada da melhor fixa for
    > -0,05% (a melhor fixa nao supera a selecao aleatoria em 0,05% ou mais).
(B) SINAL. Familia confirmatoria (Holm sobre 3, alpha=0,05), cada uma vs ig_random:
      B1 ig_ql_dense_cal       (recompensa densa, online)
      B2 ig_ql_ws_cal          (recompensa original, Q pre-treinada por classe)
      B3 ig_ql_dense_ws_cal    (densa + pre-treinada)
    "confirmada" = p_holm < 0,05 E d_medio < 0. Explicacao B (esparsidade/falta de treino) SUPORTADA se alguma
    variante e' confirmada com d_medio <= -0,05%; REFUTADA se nenhuma e'.
Equivalencia (TOST via bootstrap): IC percentil de 90% (2000 reamostragens, seed 0) do d medio dentro de
(-0,05; +0,05) => equivalente a ig_random na margem de 0,05%. Calculada para B1-B3 e, como referencia, para
ig_ql (estado padrao) e ig_ql_cal armazenados.
Secundarias (descritivas): por classe; variantes vs ig_ql_cal; razao de recompensas nao nulas nao e' recalculada aqui.

Uso: python -m analysis.report_entrada104_signal_headroom [--smoke]
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


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y, seed=0):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()),
                mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi), rank_biserial=rank_biserial_paired(x, y),
                p_raw=p, equivalent=bool(lo > -MARGIN and hi < MARGIN))


def main() -> None:
    smoke = "--smoke" in sys.argv
    res = pd.read_csv(ROOT / "results" / "results.csv")
    rec = pd.read_csv(ROOT / "results" / "state_recalibration_entrada98.csv")
    new = pd.read_csv(ROOT / "results" / ("signal_headroom_entrada104_SMOKE.csv" if smoke else "signal_headroom_entrada104.csv"))
    cols = ["instance", "rep", "method", "makespan"]
    d = pd.concat([res[res.method.isin(["ig_random", "ig_ql"])][cols], rec[rec.method == "ig_ql_cal"][cols], new[cols]])
    d = d.drop_duplicates(["instance", "rep", "method"])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan")
    need = ["ig_random", "ig_fix0", "ig_fix1", "ig_fix2", "ig_ql_dense_cal", "ig_ql_ws_cal", "ig_ql_dense_ws_cal", "ig_ql_cal"]
    w = w.dropna(subset=need).reset_index()
    if not smoke:
        assert len(w) == 1200, len(w)
    L = ["=" * 110, "FOLGA E SINAL DE APRENDIZADO NO IG-QL -- teste pre-registrado, Entrada 104", "=" * 110, f"instancias: {len(w)}"]
    pd.set_option("display.width", 230); pd.set_option("display.float_format", lambda v: f"{v:.4g}")

    # (A) folga
    fix = ["ig_fix0", "ig_fix1", "ig_fix2"]
    rows, best_k = [], {}
    for cls, g in w.groupby("instance"):
        means = {k: g[k].mean() for k in fix}
        kb = min(means, key=means.get)
        t = test(g[kb], g.ig_random)
        rows.append(dict(instance=cls, best_fixed=kb, **{f"d_{k[-4:]}": rel(g[k], g.ig_random).mean() for k in fix},
                         d_best=t["mean_d_pct"], n=len(g)))
        best_k[cls] = kb
    A = pd.DataFrame(rows)
    w["best_fixed_insample"] = [r[best_k[r["instance"]]] for _, r in w.iterrows()]
    agg = float(np.average(A.d_best, weights=A.n))
    oracle = test(w[fix].min(axis=1), w.ig_random)
    sel = test(w.best_fixed_insample, w.ig_random)
    L += ["\n--- (A) FOLGA: IG com estrategia fixa vs ig_random (d medio %, por classe) ---", A.to_string(index=False),
          f"\nfolga agregada da melhor fixa por classe (otimista, escolhida na amostra): {agg:.4f}%  [IC90 {sel['ci90_lo']:.4f}, {sel['ci90_hi']:.4f}]",
          f"oraculo POR INSTANCIA (min das 3 fixas; inatingivel): d = {oracle['mean_d_pct']:.4f}%  melhor/pior/empate = {oracle['better']}/{oracle['worse']}/{oracle['tie']}",
          f"REGRA (A): folga agregada > -0,05% ? {'SIM -> explicacao A SUPORTADA' if agg > -MARGIN else 'NAO -> explicacao A NAO suportada'}"]

    # (B) sinal
    B = []
    for pid, k in [("B1", "ig_ql_dense_cal"), ("B2", "ig_ql_ws_cal"), ("B3", "ig_ql_dense_ws_cal")]:
        t = test(w[k], w.ig_random); t.update(id=pid, method=k); B.append(t)
    B = pd.DataFrame(B)
    h = holm_bonferroni(B.p_raw.fillna(1.0).tolist(), alpha=0.05)
    B["p_holm"], B["reject"] = h["p_adjusted"], h["reject"]
    B["confirmada"] = B.reject & (B.mean_d_pct < 0)
    ref = []
    for k in ["ig_ql", "ig_ql_cal"]:      # EMENDA 1: ig_ql em results.csv tem so' 30 reps em (20,4) -> usar pares disponiveis
        wk = w.dropna(subset=[k])
        t = test(wk[k], wk.ig_random); t.update(id="ref", method=k, p_holm=float("nan")); ref.append(t)
    cols_b = ["id", "method", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalent"]
    L += ["\n--- (B) SINAL: variantes de Q-Learning vs ig_random (Holm sobre 3; d<0 => variante melhor) ---",
          B.reindex(columns=cols_b).to_string(index=False), "\nreferencias armazenadas (sem Holm):",
          pd.DataFrame(ref).reindex(columns=cols_b).to_string(index=False)]
    rel_ok = B[(B.confirmada) & (B.mean_d_pct <= -MARGIN)]
    L.append(f"\nREGRA (B): variantes confirmadas e com d <= -0,05%: {list(rel_ok.id) or 'nenhuma'} -> explicacao B "
             f"{'SUPORTADA' if len(rel_ok) else 'REFUTADA'}")
    sec = []
    for cls, g in w.groupby("instance"):
        for pid, k in [("B1", "ig_ql_dense_cal"), ("B2", "ig_ql_ws_cal"), ("B3", "ig_ql_dense_ws_cal")]:
            t = test(g[k], g.ig_random); t.update(instance=cls, id=pid); sec.append(t)
    sec = pd.DataFrame(sec)
    hh = holm_bonferroni(sec.p_raw.fillna(1.0).tolist(), alpha=0.05)
    sec["p_holm"] = hh["p_adjusted"]
    L += ["\n--- secundaria por classe (Holm sobre 36) ---",
          sec[["instance", "id", "n", "better", "worse", "tie", "mean_d_pct", "p_raw", "p_holm"]].to_string(index=False)]
    text = "\n".join(L)
    print(text)
    if not smoke:
        (ROOT / "results" / "entrada104_signal_headroom.txt").write_text(text, encoding="utf-8")
        A.to_csv(ROOT / "results" / "entrada104_headroom_by_class.csv", index=False)
        B.to_csv(ROOT / "results" / "entrada104_signal_primary.csv", index=False)
        sec.to_csv(ROOT / "results" / "entrada104_signal_by_class.csv", index=False)


if __name__ == "__main__":
    main()
