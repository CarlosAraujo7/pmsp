"""
analysis/report_entrada215_robustness.py
==========================================
Ciclo 19, frente D: ROBUSTEZ ESTATISTICA das conclusoes (critica 6 da revisao cega #2: empates descartados no Wilcoxon, sem correcao entre baterias, medias agrupadas escondem heterogeneidade por classe). Diario: Entrada 215 (pre-registro) e 216 (resultados). So' analise de dados ja' existentes.
(i) Pares de CABECALHO (os que o artigo afirma): por conjunto e espaco, seletor vs validada, seletor vs Q-Learning, Q-Learning vs aleatorio, validada vs Q-Learning (publicadas descoberta e held-out; S5 e S15), Q-Learning vs aleatorio e validada vs Q-Learning nas sinteticas (S5 e S15; 1.200 instancias) e no flowshop (sinteticas e Taillard; F5 e F15).
Cada par e' testado de tres modos sobre a media das sementes por instancia: Wilcoxon com empates descartados (o das Entradas 180-211), Wilcoxon com zeros de PRATT (os empates entram nos postos) e teste de PERMUTACAO pareado das medias (inversao aleatoria dos sinais; 20.000 permutacoes; os empates contribuem zero); Holm sobre todos os pares de cada modo.
(ii) Controle GLOBAL: o p bruto de TODOS os testes das familias registradas (Entradas 180-211) entra em Holm e em Benjamini-Yekutieli (dependencia arbitraria) a 0,05; conta-se quantos dos testes 'confirmada' sobrevivem.
(iii) Heterogeneidade por classe: nas publicadas, em cada celula (familia x tercil de n) e espaco, a fracao de celulas em que o seletor supera a validada e o sinal do ganho do Q-Learning.

PRE-REGISTRO (congelado ANTES de rodar este relatorio pela primeira vez; hashes em results/entrada215_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
(D1) A significancia (Holm) dos pares de cabecalho concorda entre os tres modos em >= 95% dos pares, e nenhum par de cabecalho 'confirmado' pelo Wilcoxon com empates descartados perde a significancia no teste de permutacao.
(D2) Controle global: >= 95% dos testes 'confirmada' continuam significativos em Holm global e em Benjamini-Yekutieli global.
(D3) Heterogeneidade: o seletor supera a validada em >= 70% das celulas (familia x tercil) em cada espaco e conjunto (descoberta e held-out).
(D4) O ganho do Q-Learning sobre o aleatorio no held-out e' positivo em pelo menos 5 das 6 celulas da familia C (3 tercis x 2 espacos) e negativo em pelo menos 5 das 6 da familia E.
Uso: python -m analysis.report_entrada215_robustness
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import rel  # noqa: E402
from analysis.report_entrada197_pfsp import load_test  # noqa: E402
from analysis.report_entrada200_selector import SETS, load  # noqa: E402

R = ROOT / "results"
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
rng = np.random.default_rng(0)


def three_tests(a, b):
    d = rel(a, b)
    nz = d[d != 0]
    p_w = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan
    p_pr = float(sstats.wilcoxon(d, zero_method="pratt").pvalue) if len(nz) >= 6 else np.nan
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(20000, len(nz)))
    perm = np.abs((signs * np.abs(nz)).sum(axis=1) / len(d))
    p_pm = float((1 + (perm >= obs - 1e-12).sum()) / (1 + len(perm)))
    return dict(n=len(d), mean_d=float(d.mean()), tie=int((d == 0).sum()), p_wilcoxon=p_w, p_pratt=p_pr, p_perm=p_pm)


def holm_adj(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def by_adj(p):
    p = np.asarray(p, float)
    m = len(p)
    c = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    prev = 1.0
    for k, i in enumerate(order):
        rank = m - k
        prev = min(prev, p[i] * m * c / rank)
        adj[i] = min(1.0, prev)
    return adj


def main():
    L = ["=" * 110, "ROBUSTEZ ESTATISTICA (Entrada 215)", "=" * 110]
    rows = []

    def add(label, a, b):
        t = three_tests(a, b)
        t["claim"] = label
        rows.append(t)
    for sp in ("S5", "S15"):
        for cj in SETS:
            w = load(sp, cj)
            for lab, (a, b) in {"sel vs val": ("sel", "val"), "sel vs ql": ("sel", "ql"), "ql vs rand": ("ql", "rand"), "val vs ql": ("val", "ql")}.items():
                add(f"pub {cj} {sp}: {lab}", w[a], w[b])
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
        add(f"syn {sp}: ql vs rand", ms.ql, ms.rand)
        add(f"syn {sp}: val vs ql", ms.val, ms.ql)
    for sp in ("F5", "F15"):
        w = load_test(sp)
        for s in ("syn", "ta"):
            x = w[w.set == s]
            add(f"flowshop {s} {sp}: ql vs rand", x.ql, x.rand)
            add(f"flowshop {s} {sp}: val vs ql", x.val, x.ql)
    T = pd.DataFrame(rows)
    for col in ("p_wilcoxon", "p_pratt", "p_perm"):
        T["h_" + col[2:]] = holm_adj(T[col].fillna(1.0))
    T["sig_w"], T["sig_pr"], T["sig_pm"] = T.h_wilcoxon < 0.05, T.h_pratt < 0.05, T.h_perm < 0.05
    L += ["\n--- pares de cabecalho: p brutos e significancia apos Holm sobre os pares (por modo) ---", T[["claim", "n", "tie", "mean_d", "p_wilcoxon", "p_pratt", "p_perm", "sig_w", "sig_pr", "sig_pm"]].to_string(index=False)]
    agree = float(((T.sig_w == T.sig_pr) & (T.sig_w == T.sig_pm)).mean())
    lost = int((T.sig_w & ~T.sig_pm).sum())
    L.append(f"\nconcordancia dos tres modos: {agree:.3f}; pares significativos no Wilcoxon e nao na permutacao: {lost}")
    # ---- controle global
    files = [f for f in sorted(glob.glob(str(R / "entrada*_*.csv"))) if not any(k in f for k in ("cells", "k.csv", "headroom", "family", "summary", "power", "plant", "by_family", "desc", "pooled", "corr", "by_set"))]
    allp = []
    for f in files:
        try:
            x = pd.read_csv(f)
        except Exception:
            continue
        if {"p_raw", "mean_d_pct", "id"} <= set(x.columns):
            x = x[["id", "p_raw", "mean_d_pct"] + (["confirmada"] if "confirmada" in x.columns else [])].copy()
            x["file"] = Path(f).name
            allp.append(x)
    G = pd.concat(allp, ignore_index=True)
    G = G[G.p_raw.notna()].reset_index(drop=True)
    if "confirmada" not in G.columns:
        G["confirmada"] = False
    G["confirmada"] = G.confirmada.fillna(False).astype(bool)
    G["p_holm_global"] = holm_adj(G.p_raw)
    G["p_by_global"] = by_adj(G.p_raw)
    conf = G[G.confirmada]
    sh, sb = float((conf.p_holm_global < 0.05).mean()), float((conf.p_by_global < 0.05).mean())
    L.append(f"\ncontrole global: {len(G)} testes registrados em {G.file.nunique()} arquivos; {len(conf)} 'confirmada'; sobrevivem a Holm global {int((conf.p_holm_global < 0.05).sum())} ({sh:.3f}) e a Benjamini-Yekutieli global {int((conf.p_by_global < 0.05).sum())} ({sb:.3f})")
    drop = conf[(conf.p_by_global >= 0.05)][["file", "id", "p_raw", "p_by_global"]]
    if len(drop):
        L += ["testes 'confirmada' que NAO sobrevivem ao BY global:", drop.to_string(index=False)]
    # ---- heterogeneidade por classe
    het = []
    for sp in ("S5", "S15"):
        for cj in SETS:
            w = load(sp, cj)
            w["grp"] = w.cls.str.split("|").str[1].astype(int)
            for (fam, g), x in w.groupby(["fam", "grp"]):
                het.append(dict(space=sp, set=cj, fam=fam, grp=g, n=len(x), d_sel_val=float(rel(x.sel, x.val).mean()), g_ql=float(-rel(x.ql, x.rand).mean())))
    H = pd.DataFrame(het)
    L += ["\n--- celulas (familia x tercil de n) ---", H.to_string(index=False)]
    frac = {(sp, cj): float((x.d_sel_val < 0).mean()) for (sp, cj), x in H.groupby(["space", "set"])}
    held = H[H.set == "heldout"]
    cpos = int((held[held.fam == "C"].g_ql > 0).sum())
    eneg = int((held[held.fam == "E"].g_ql < 0).sum())
    yn = lambda b: "SIM" if b else "NAO"
    d1 = agree >= 0.95 and lost == 0
    d2 = sh >= 0.95 and sb >= 0.95
    d3 = all(v >= 0.70 for v in frac.values())
    d4 = cpos >= 5 and eneg >= 5
    L += ["\n--- PREVISOES PRE-FIXADAS ---", f"(D1) concordancia >= 0,95 e nenhum par perde a significancia na permutacao: {yn(d1)} (concordancia {agree:.3f}; perdidos {lost})", f"(D2) >= 95% dos 'confirmada' sobrevivem a Holm e a BY globais: {yn(d2)} ({sh:.3f}; {sb:.3f})",
          f"(D3) seletor melhor que a validada em >= 70% das celulas em cada espaco e conjunto: {yn(d3)} (" + ", ".join(f"{k[0]}/{k[1]} {v:.2f}" for k, v in frac.items()) + ")", f"(D4) held-out: Q-Learning positivo em >= 5 das 6 celulas da familia C e negativo em >= 5 das 6 da E: {yn(d4)} (C positivas {cpos}/6; E negativas {eneg}/6)"]
    text = "\n".join(L)
    print(text)
    (R / "entrada215_robustness.txt").write_text(text, encoding="utf-8")
    T.to_csv(R / "entrada215_robustness_pairs.csv", index=False)
    H.to_csv(R / "entrada215_robustness_cells.csv", index=False)
    G.to_csv(R / "entrada215_robustness_global.csv", index=False)


if __name__ == "__main__":
    main()
