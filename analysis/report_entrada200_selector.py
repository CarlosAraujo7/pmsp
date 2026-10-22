"""
analysis/report_entrada200_selector.py
========================================
Ciclo 18, frente B: o SELETOR OFFLINE condicionado a instancia (ponto 2 de Carlos: um aprendiz que EXPLORE a folga). Diario: Entrada 200 (pre-registro) e 202 (resultados).
Delineamento: experiment/run_cycle18_selector_entrada200.py. Politica aprendida OFFLINE (ExtraTrees multi-saida sobre os atributos de pmsp/pcmax_features.py; alvo: desempenho relativo centrado por instancia de cada acao fixa; 5.720 instancias de treino B, C, E de manifest.csv, 3 sementes por acao, disjuntas de todo ajuste e teste),
aplicada a cada instancia de TESTE (as 2.550 da Entrada 189 'discovery' e as 3.950 da Entrada 190 'heldout'): uma acao fixa por instancia, 5 sementes, T = 1000, espacos S5 e S15. Comparada com a acao validada ESTATICA por (familia, tercil de n) (a 'val' dos arquivos existentes), com o Q-Learning e a roleta ALNS online (a iteracoes iguais) e com o aleatorio.
Validacao cruzada SO' no treino (5 dobras; results/selector_cv_entrada200.json; vista ANTES de fixar as previsoes; nenhum dado de teste foi lido): desempenho relativo medio (%, menor e' melhor) do seletor, da melhor acao fixa da familia (no proprio treino, favoravel a ela) e do oraculo (viesado): S5 -0,294, -0,217, -0,309; S15 -1,320, -1,245, -1,341;
por familia o seletor supera a melhor fixa da familia por 0,07 a 0,09 pontos em S5 e S15.

PRE-REGISTRO (congelado ANTES de rodar o seletor no teste; hashes em results/entrada200_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Estatistica (a das Entradas 180-199): unidade = media das sementes por instancia; d = 100 (A - B) / B (negativo: A melhor); Wilcoxon pareado bilateral, IC90 bootstrap, equivalencia a 0 se o IC90 cabe em +-0,05; 'confirmada' = p_holm < 0,05 e d < 0. Familias (Holm sobre 4 = 2 espacos x 2 conjuntos): Cval seletor contra a validada estatica; Cql seletor contra o Q-Learning;
Calns seletor contra a roleta ALNS; Crand seletor contra o aleatorio.
Hipoteses e previsoes:
(H1) 'um seletor OFFLINE condicionado a instancia supera a acao validada estatica': Cval confirmada nos quatro pares (S5 e S15; discovery e heldout) e com d medio <= -0,02 nos quatro.
(P2) d medio de Cval em [-0,12; -0,02] nos quatro pares. (P3) Cql confirmada nos quatro pares (seletor contra o aprendiz online mais usado). (P4) Calns confirmada nos quatro pares (contra o aprendiz online mais forte do estagio 2).
(P5) captura da folga: na familia C do held-out com S15, a fracao do ganho do oraculo sobre o aleatorio que o seletor captura (media dos ganhos / ganho medio do oraculo) e' >= 0,85, e a da validada estatica e' menor que a do seletor.
(P6) o seletor nunca e' pior que o aleatorio: em cada uma das 12 celulas (familia x espaco x conjunto) d medio contra o aleatorio < 0.
(P7) custo neutro: razao das avaliacoes medias do seletor sobre as da validada em [0,8; 1,25] em cada espaco e conjunto (se fora, a comparacao a iteracoes iguais nao e' a CPU igual e o resultado e' reportado como tal).
Leitura S ('o seletor offline explora a folga que o aprendizado online nao explora') = H1 e P3 e P4. Leitura S- ('ganho presente mas dentro da margem'): H1 confirmada nos quatro pares sem d medio <= -0,02 em algum. Se Cval nao for confirmada em algum par, a politica offline NAO supera a validada nesse par e o artigo reporta paridade.
Uso: python -m analysis.report_entrada200_selector
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
SETS = {"discovery": ("pub_{sp}_s*of*_entrada189.csv", 2550), "heldout": ("pubh_{sp}_s*of*_entrada190.csv", 3950)}
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load(sp, conj):
    pat, expect = SETS[conj]
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat.format(sp=sp))))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method")
    ev = d[d.method.isin(["val"])].groupby("cls").evals.mean().rename("evals_val")
    orc = d[d.method.str.startswith("arm")].groupby("cls").makespan.min().rename("oracle")
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"sel_{sp}_{conj}_s*of*_entrada200.csv")))], ignore_index=True)
    sm = s.groupby("cls").makespan.mean().rename("sel")
    se = s.groupby("cls").evals.mean().rename("evals_sel")
    w = ms.join(ev).join(orc).join(sm, how="inner").join(se).reset_index()
    assert len(w) == expect, (sp, conj, len(w), expect)
    w["fam"] = w.cls.str.split("|").str[0]
    return w


def main():
    L = ["=" * 110, "SELETOR OFFLINE CONDICIONADO A INSTANCIA (Entrada 200)", "=" * 110]
    W = {(sp, cj): load(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    fams = {k: [] for k in ("Cval", "Cql", "Calns", "Crand")}
    ref = {"Cval": "val", "Cql": "ql", "Calns": "alns", "Crand": "rand"}
    for (sp, cj), w in W.items():
        for fid, col in ref.items():
            t = test(w["sel"], w[col]); t.update(id=f"{fid}-{sp}-{cj}"); fams[fid].append(t)
    F = {k: family(v) for k, v in fams.items()}
    for k, X in F.items():
        L += [f"\n--- {k}: seletor contra {ref[k]} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    rows = []
    for (sp, cj), w in W.items():
        for fam, x in list(w.groupby("fam")) + [("all", w)]:
            g = lambda col: float(-rel(x[col], x.rand).mean())
            rows.append(dict(space=sp, set=cj, fam=fam, n=len(x), sel=g("sel"), val=g("val"), ql=g("ql"), alns=g("alns"), oracle=g("oracle"), sel_minus_val=float(rel(x.sel, x.val).mean()), cap_sel=g("sel") / g("oracle") if g("oracle") > 0 else np.nan, cap_val=g("val") / g("oracle") if g("oracle") > 0 else np.nan,
                             evals_ratio=float(x.evals_sel.mean() / x.evals_val.mean())))
    D = pd.DataFrame(rows)
    L += ["\n--- ganho sobre o aleatorio (%, positivo: melhor), por familia ---", D.to_string(index=False)]
    cv, cq, ca = F["Cval"], F["Cql"], F["Calns"]
    h1 = bool(cv.confirmada.all() and (cv.mean_d_pct <= -0.02).all())
    p2 = bool(((cv.mean_d_pct >= -0.12) & (cv.mean_d_pct <= -0.02)).all())
    p3, p4 = bool(cq.confirmada.all()), bool(ca.confirmada.all())
    c = D[(D.space == "S15") & (D.set == "heldout") & (D.fam == "C")].iloc[0]
    p5 = bool(c.cap_sel >= 0.85 and c.cap_sel > c.cap_val)
    cells = D[D.fam != "all"]
    p6 = bool((cells.sel > 0).all())
    al = D[D.fam == "all"]
    p7 = bool(((al.evals_ratio >= 0.8) & (al.evals_ratio <= 1.25)).all())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(H1) Cval confirmada nos quatro pares e d medio <= -0,02 nos quatro: {yn(h1)} (d medio: " + ", ".join(f"{v:+.3f}" for v in cv.mean_d_pct) + ")", f"(P2) d medio de Cval em [-0,12; -0,02] nos quatro: {yn(p2)}", f"(P3) Cql confirmada nos quatro pares: {yn(p3)}", f"(P4) Calns confirmada nos quatro pares: {yn(p4)}",
          f"(P5) familia C, held-out, S15: captura do seletor {c.cap_sel:.3f} (>= 0,85) e da validada {c.cap_val:.3f}: {yn(p5)}", f"(P6) seletor melhor que o aleatorio nas 12 celulas (familia x espaco x conjunto): {yn(p6)} (minimo ganho {cells.sel.min():+.3f})", f"(P7) razao de avaliacoes seletor/validada em [0,8; 1,25] nos quatro: {yn(p7)} (" + ", ".join(f"{v:.2f}" for v in al.evals_ratio) + ")",
          f"S (o seletor offline supera a validada e os aprendizes online): {yn(h1 and p3 and p4)}   |   S- (H1 confirmada sem d <= -0,02 em algum): {yn(bool(cv.confirmada.all()) and not h1)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada200_selector.txt").write_text(text, encoding="utf-8")
    D.to_csv(R / "entrada200_selector_by_family.csv", index=False)
    for k, X in F.items():
        X.to_csv(R / f"entrada200_selector_{k}.csv", index=False)


if __name__ == "__main__":
    main()
