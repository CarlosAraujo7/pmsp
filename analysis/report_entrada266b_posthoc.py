"""
analysis/report_entrada266b_posthoc.py
=======================================
Entrada 266, leitura POS HOC (NAO registrada): a transferencia da acao estatica a um TAMANHO n nao visto (LOSO) contra os aprendizes online, no benchmark de Fanjul-Peyro e Ruiz, espaco B30.
Motivo: no relatorio registrado (analysis/report_entrada266_granularity.py) a previsao K7 (LOSO-G0 ganha do aleatorio nos tres tamanhos de quatro) falhou e o ganho medio de LOSO-G0 (0,34) e de LOSO-Gt (0,38) ficou da ordem do melhor aprendiz
online (0,39). Aqui as quatro celulas de tamanho (175 instancias cada, os sete tipos agrupados) comparam, sem correcao alem do Holm sobre os quatro tamanhos, o melhor aprendiz online de cada tamanho com LOSO-G0, com LOSO-Gt e com a politica
neural. Tudo e' calculado dos mesmos arquivos do relatorio registrado; nada e' reexecutado.
Uso: python -m analysis.report_entrada266b_posthoc
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada266_granularity import LEARN, NSIZES, TYPES, pick, read, tuning  # noqa: E402

R = ROOT / "results"
pd.set_option("display.width", 230)


def beyond(F_):
    return F_.confirmada & ~F_["equivalente_0.05"]


def main():
    key = ["type", "instance"]
    arms = read("arms_test_B30_s*of*_entrada266.csv")
    WA = arms.groupby(key + ["arm"]).makespan.mean().unstack("arm")
    WA.columns = [f"arm{c}" for c in WA.columns]
    meta = arms.drop_duplicates(key)[key + ["n"]]
    T_ = tuning("B30")
    fr = read("fr_test_B30_s*of4_entrada256.csv")
    W = fr.groupby(key + ["method"]).makespan.mean().unstack("method").reset_index().merge(meta, on=key)
    W = W.merge(read("qperm_fanjul_B30_s*of*_entrada260.csv").groupby(key).makespan.mean().rename("ql_perm").reset_index(), on=key)
    W = W.merge(read("nn_test_B30_s*of4_entrada262.csv").groupby(key + ["method"]).makespan.mean().unstack("method").reset_index(), on=key)
    W = W.merge(WA.reset_index(), on=key)
    loso0 = {n: pick(T_[T_.n != n]) for n in NSIZES}
    losoT = {n: pick(T_[T_.n != n], "type") for n in NSIZES}
    W["LOSO_G0"] = [float(r[f"arm{loso0[r.n]}"]) for _, r in W.iterrows()]
    W["LOSO_Gt"] = [float(r[f"arm{losoT[r.n][r['type']]}"]) for _, r in W.iterrows()]
    g0 = pick(T_)
    W["G0"] = float("nan")
    W["G0"] = [float(r[f"arm{g0}"]) for _, r in W.iterrows()]
    online = LEARN + ["ql_perm"]
    rows, TL, T0, TT, TN = [], [], [], [], []
    for n in NSIZES:
        x = W[W.n == n]
        g = {k: float(-rel(x[k], x.rand).mean()) for k in ["G0", "LOSO_G0", "LOSO_Gt", "nn"] + online}
        kb = max(LEARN, key=lambda k: g[k])
        rows.append(dict(n=n, **g, best_online=kb, g_best_online=g[kb]))
        a = test(x[kb], x.LOSO_G0); a.update(id=f"best-vs-LOSO_G0-n{n}"); T0.append(a)
        b = test(x[kb], x.LOSO_Gt); b.update(id=f"best-vs-LOSO_Gt-n{n}"); TT.append(b)
        c = test(x["nn"], x.LOSO_G0); c.update(id=f"nn-vs-LOSO_G0-n{n}"); TN.append(c)
        d = test(x.LOSO_G0, x.rand); d.update(id=f"LOSO_G0-vs-rand-n{n}"); TL.append(d)
    C = pd.DataFrame(rows)
    F0, FT, FN, FL = family(T0), family(TT), family(TN), family(TL)
    L = ["=" * 110, "LEITURA POS HOC: TRANSFERENCIA A UM TAMANHO NAO VISTO (Entrada 266b)", "=" * 110,
         "ganhos sobre o aleatorio (pontos) por tamanho; os sete tipos agrupados (175 instancias por tamanho); 'best_online' = o melhor dos sete aprendizes online no tamanho", C.round(3).to_string(index=False), "",
         "--- o melhor aprendiz online contra LOSO-G0 (Holm sobre 4 tamanhos; negativo = o aprendiz e' melhor) ---", F0[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
         "--- o melhor aprendiz online contra LOSO-Gt ---", FT[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
         "--- a politica neural contra LOSO-G0 ---", FN[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
         "--- LOSO-G0 contra o aleatorio ---", FL[["id", "n", "mean_d_pct", "ci90_lo", "ci90_hi", "p_holm", "confirmada", "equivalente_0.05"]].round(4).to_string(index=False), "",
         f"tamanhos em que o melhor aprendiz online bate LOSO-G0 alem da margem: {int(beyond(F0).sum())} de 4; bate LOSO-Gt: {int(beyond(FT).sum())} de 4; a politica neural bate LOSO-G0: {int(beyond(FN).sum())} de 4",
         f"tamanhos em que LOSO-G0 PERDE para o aleatorio (d > 0 com o IC90 acima de zero): {int((FL.ci90_lo > 0).sum())} de 4"]
    text = "\n".join(L)
    print(text)
    (R / "entrada266b_posthoc.txt").write_text(text + "\n", encoding="utf-8")
    C.to_csv(R / "entrada266b_posthoc_cells.csv", index=False)
    pd.concat([F0, FT, FN, FL]).to_csv(R / "entrada266b_posthoc_tests.csv", index=False)


if __name__ == "__main__":
    main()
