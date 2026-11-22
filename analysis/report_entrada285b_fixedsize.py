"""
analysis/report_entrada285b_fixedsize.py
=========================================
Entrada 285b (Ciclo 52): O ILS CLASSICO COM O TAMANHO DA PERTURBACAO AJUSTADO. Na Entrada 285 o R2 (reinsercao aleatoria + descida ate' o otimo local) sorteava o tamanho da perturbacao entre cinco (d/n de 0,05 a 0,40), mistura em que a maioria dos sorteios e'
pesada demais para a descida reparar; um ILS bem ajustado usaria um tamanho fixo. Aqui o R2 roda com cada tamanho fixo (F0 a F4) e, por classe, fica o MELHOR (menor lacuna media ao limite inferior nas PROPRIAS instancias de teste: uma escolha oraculo que FAVORECE o R2 e
e' conservadora para a conclusao). Comparacoes com o IG simples (G0, escolha aleatoria entre os cinco tamanhos): a iteracoes iguais com o G0 de 1.000 iteracoes da Entrada 284 e a CPU igual com H_k* (G0 com round(1000 K_k*) iteracoes, K_k = CPU(F_k)/CPU(G0)).
Testes como nas Entradas 284 e 285 (ganho = -d, media das 5 sementes por instancia, Wilcoxon pareado, IC90 bootstrap, Holm sobre as 12 classes por contraste; 'alem da margem' = confirmado e IC90 nao contido em +-0,05):
  S1  G0 contra R2* (R2 com o melhor tamanho fixo da classe), iteracoes iguais     S2  H* contra R2*, CPU igual     S3  R2* contra R2 (tamanho sorteado, Entrada 285), iteracoes iguais

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada285b_prereg_hashes.txt):
(C1) o melhor tamanho fixo do R2 e' o menor (d/n = 0,05) em pelo menos 8 das 12 classes. (65%)
(C2) S1: o IG simples vence o R2* alem da margem em pelo menos 4 das 12 classes a iteracoes iguais. (60%)
(C3) S2: com a CPU do R2*, o IG simples vence-o alem da margem em pelo menos 5 classes e perde em no maximo 2. (60%)
(C4) a lacuna media ao limite inferior do R2* (media sobre as classes) e' pelo menos 0,40%, isto e', mais de 0,12 acima a do G0 (0,271%). (65%)
(C5) S3: ajustar o tamanho melhora o ILS: ganho medio do R2* sobre o R2 de tamanho sorteado >= 0,15. (70%)
Uso: python -m analysis.report_entrada285b_fixedsize
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, test  # noqa: E402

R = ROOT / "results"
NK = 5
pd.set_option("display.width", 230)


def main():
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "fixedsize_test_s*of*_entrada285b.csv")))], ignore_index=True)
    D4 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "ablation_test_s*of*_entrada284.csv")))], ignore_index=True)
    D5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "fulldescent_test_s*of*_entrada285.csv")))], ignore_index=True)
    assert not D.duplicated(["cls", "rep", "arm", "seed_i"]).any()
    G0 = D4[D4.arm == "G0"].assign(arm="G0")
    R2r = D5[D5.arm == "R2"].assign(arm="R2rand")
    A = pd.concat([D, G0, R2r], ignore_index=True)
    W = A.groupby(["cls", "rep", "arm"]).makespan.mean().unstack()
    gapi = A.assign(g=(A.makespan - A.lb) / A.lb).groupby(["cls", "rep", "arm"]).g.mean().unstack()
    classes = list(dict.fromkeys(D.cls))
    K = json.loads((R / "fixedsize_cpu_ratio_entrada285b.json").read_text())
    best = {}
    tab = []
    for c in classes:
        g = {k: 100 * float(gapi.loc[c][f"F{k}"].mean()) for k in range(NK)}
        kb = min(range(NK), key=lambda k: (round(g[k], 12), k))
        best[c] = kb
        tab.append(dict(cls=c, **{f"gap_F{k}": g[k] for k in range(NK)}, gap_G0=100 * float(gapi.loc[c]["G0"].mean()), gap_R2rand=100 * float(gapi.loc[c]["R2rand"].mean()), melhor_k=kb,
                        K_melhor=K[c.replace("rand_", "").replace("j_", "x").replace("m", "")][f"F{kb}"]))
    T = pd.DataFrame(tab)
    L = ["=" * 110, "O ILS CLASSICO COM O TAMANHO DA PERTURBACAO AJUSTADO (Entrada 285b)", "=" * 110, f"execucoes novas: {len(D)}; instancias: {W.dropna(subset=['G0']).shape[0]}",
         "--- lacuna media ao limite inferior (%) por classe: R2 com cada tamanho fixo (F0 = 0,05 n ... F4 = 0,40 n), G0 e R2 de tamanho sorteado; melhor tamanho (oraculo nos dados de teste) e o K dele ---", T.to_string(index=False, float_format=lambda v: f"{v:.3f}")]
    # series por instancia
    def series(c, which):
        kb = best[c]
        if which == "R2star":
            return W.loc[c][f"F{kb}"]
        if which == "Hstar":
            return W.loc[c][f"H{kb}"]
        return W.loc[c][which]
    CONTRASTS = [("S1", "G0", "R2star"), ("S2", "Hstar", "R2star"), ("S3", "R2star", "R2rand")]
    res = {}
    for cid, a, b in CONTRASTS:
        rows = []
        for c in classes:
            t = test(series(c, a), series(c, b))
            t["id"] = f"{cid}-{c}"
            rows.append(t)
        F = family(rows)
        eq = (F.ci90_lo > -0.05) & (F.ci90_hi < 0.05)
        res[cid] = dict(contraste=f"{a} contra {b}", ganho_medio=float(-F.mean_d_pct.mean()), ganho_min=float(-F.mean_d_pct.max()), ganho_max=float(-F.mean_d_pct.min()),
                        vence_alem=int((F.confirmada & ~eq).sum()), perde_alem=int((F.reject & (F.mean_d_pct > 0) & ~eq).sum()), paridade=int((F.confirmada & eq).sum()), sem_confirmar=int((~F.reject).sum()))
        F.to_csv(R / f"entrada285b_tests_{cid}.csv", index=False)
    Rt = pd.DataFrame(res).T
    L += ["", "--- contrastes: ganho medio sobre as 12 classes (pontos), classes em que A vence B alem da margem, perde alem da margem, paridade, sem confirmar ---", Rt.to_string(float_format=lambda v: f"{v:.3f}")]
    Rt.to_csv(R / "entrada285b_contrasts.csv")
    T.to_csv(R / "entrada285b_by_class.csv", index=False)
    gap_star = float(np.mean([T.loc[T.cls == c, f"gap_F{best[c]}"].iloc[0] for c in classes]))
    gap_g0 = float(T.gap_G0.mean())
    n_small = sum(1 for c in classes if best[c] == 0)
    yn = lambda b: "SIM" if b else "NAO"
    p = []
    c1 = n_small >= 8; p.append(c1)
    c2 = res["S1"]["vence_alem"] >= 4; p.append(c2)
    c3 = res["S2"]["vence_alem"] >= 5 and res["S2"]["perde_alem"] <= 2; p.append(c3)
    c4 = gap_star >= 0.40; p.append(c4)
    c5 = res["S3"]["ganho_medio"] >= 0.15; p.append(c5)
    L += ["", "--- previsoes ---",
          f"(C1) classes em que o melhor tamanho fixo e' o menor: {n_small} de 12 (previsto >= 8): {yn(c1)}",
          f"(C2) S1: o IG simples vence o R2* alem da margem em {res['S1']['vence_alem']} de 12 e perde em {res['S1']['perde_alem']} (previsto >= 4): {yn(c2)}",
          f"(C3) S2: a CPU igual o IG simples vence em {res['S2']['vence_alem']} e perde em {res['S2']['perde_alem']} (previsto >= 5 e <= 2): {yn(c3)}",
          f"(C4) lacuna media do R2*: {gap_star:.3f}% contra {gap_g0:.3f}% do G0 (previsto R2* >= 0,40%): {yn(c4)}",
          f"(C5) S3: ganho medio do R2* sobre o R2 de tamanho sorteado: {res['S3']['ganho_medio']:.3f} (previsto >= 0,15): {yn(c5)}",
          f"previsoes acertadas: {sum(p)} de {len(p)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada285b_fixedsize.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
