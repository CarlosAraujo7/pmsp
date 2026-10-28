"""
analysis/report_entrada209_cforacle.py
========================================
Ciclo 19, frente A: o ORACULO COM VALIDACAO CRUZADA DE SEMENTES e a regra da folga reavaliada. Diario: Entrada 209 (pre-registro) e 210 (resultados). Runner: experiment/run_cycle19_cforacle_entrada209.py.
Definicoes. Folga ENVIESADA (a das Entradas 180-197) de uma instancia: -rel(min_a mk[a, semente 0], media das sementes do aleatorio) -- o melhor de uma semente por acao. Folga com VALIDACAO CRUZADA (cf): a acao a* = argmin_a media(mk[a, sementes {0, 1}]) e o oraculo cf e' media(mk[a*, sementes {2, 3}]);
folga cf = -rel(oraculo cf, aleatorio) (sementes independentes de a*: sem vies de selecao). Celulas: P||Cmax sintetico = classe x espaco (S5, S15); publicadas = (familia, tercil de n) x espaco (descoberta e held-out, como na Entrada 192); flowshop = classe x espaco (F5, F15; sinteticas e Taillard).
Ganho do Q-Learning da celula (g_Q) = -rel(ql, aleatorio) medio nas instancias da celula, a iteracoes iguais (arquivos das Entradas 180, 189, 190 e 197; so' ha' Q-Learning no padrao).

PRE-REGISTRO (congelado ANTES de rodar as sementes novas; hashes em results/entrada209_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
(A1) a folga cf e' menor que a enviesada em TODOS os dez pares (conjunto x espaco): syn S5, S15; descoberta S5, S15; held-out S5, S15; flowshop sinteticas F5, F15; Taillard F5, F15.
(A2) o vies relativo (enviesada - cf)/enviesada e' MAIOR no flowshop (sinteticas, classes grandes n >= 50 e m >= 10) que nas celulas da familia C, tercis 1-2, do held-out publicado: >= 0,5 em F5 e F15 contra <= 0,35 em S5 e S15.
(A3) regra da folga com a folga cf, held-out publicado, nove celulas por espaco: Spearman(folga cf, g_Q) >= +0,5 em S5 e S15.
(A4) flowshop sinteticas, nove celulas por espaco: Spearman(folga cf, g_Q) >= -0,3 em F5 e F15 (a correlacao negativa de -0,93 e -0,78 era em boa parte ruido).
(A5) flowshop sinteticas, classes grandes: folga cf < 0,10 pontos em F5 e F15 (quase toda a folga era ruido de sementes).
(A6) regra unificada: um limiar h* da folga cf e' escolhido nas celulas da DESCOBERTA publicada (S5 e S15; maior acuracia balanceada de 'g_Q > 0'; ponto medio) e aplicado SEM mudanca a todas as demais celulas (held-out publicado, P||Cmax sintetico, flowshop sinteticas): acuracia >= 0,75 na uniao dessas celulas.
Leitura U ('com a folga sem vies, a regra vale nos dois problemas') = A3 e A4 e A6.
Uso: python -m analysis.report_entrada209_cforacle
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

R = ROOT / "results"
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def arms_table(old, new):
    """old: linhas das acoes com a semente 0 (colunas cls, method='arm<k>', makespan); new: cfo (cls, arm, seed_i, makespan). Retorna (selecao, avaliacao, enviesada) por instancia (cls): menor makespan de uma semente; media das sementes {0,1}; media das sementes {2,3}."""
    o = old[old.method.str.startswith("arm")].copy()
    o["arm"] = o.method.str[3:].astype(int)
    o = o.groupby(["cls", "arm"]).makespan.mean().rename("s0").reset_index()
    n = new.pivot_table(index=["cls", "arm"], columns="seed_i", values="makespan").reset_index()
    n.columns = ["cls", "arm"] + [f"s{c}" for c in n.columns[2:]]
    a = o.merge(n, on=["cls", "arm"])
    a["sel"] = (a.s0 + a.s1) / 2
    a["ev"] = (a.s2 + a.s3) / 2
    g = a.groupby("cls")
    best_idx = g.sel.idxmin()
    return pd.DataFrame(dict(biased=g.s0.min(), cf=a.loc[best_idx].set_index("cls").ev))


def load_syn(sp):
    d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
    d["cls"] = d.cls + "#" + d.rep.astype(str)
    new = pd.read_csv(R / f"cfo_syn_{sp}_entrada209.csv")
    new["cls"] = new.cls + "#" + new.rep.astype(str)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method")
    A = arms_table(d, new)
    w = ms.join(A)
    w["cell"] = [c.split("#")[0] for c in w.index]
    return w.reset_index()


def load_pub(conj, sp):
    pat = {"discovery": f"pub_{sp}_s*of*_entrada189.csv", "heldout": f"pubh_{sp}_s*of*_entrada190.csv"}[conj]
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat)))], ignore_index=True)
    new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"cfo_{conj}_{sp}_s*of*_entrada209.csv")))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method")
    w = ms.join(arms_table(d, new))
    parts = [c.split("|") for c in w.index]
    w["cell"] = [f"{p[0]}|{p[1]}" for p in parts]
    return w.reset_index()


def load_pfsp(sp):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"pfsp_test_{sp}_s*of*_entrada197.csv")))], ignore_index=True)
    new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"cfo_pfsp_{sp}_s*of*_entrada209.csv")))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method")
    w = ms.join(arms_table(d, new))
    w["set"] = [c.split("|")[0] for c in w.index]
    w["cell"] = [c.split("|")[1] for c in w.index]
    return w.reset_index()


def cells(w, problem, setname, sp, key="cell", extra=None):
    rows = []
    for c, x in w.groupby(key):
        rows.append(dict(problem=problem, set=setname, space=sp, cell=c, n=len(x), h_biased=float(-rel(x.biased, x.rand).mean()), h_cf=float(-rel(x.cf, x.rand).mean()), gq=float(-rel(x.ql, x.rand).mean())))
    return rows


def main():
    L = ["=" * 110, "ORACULO COM VALIDACAO CRUZADA DE SEMENTES (Entrada 209)", "=" * 110]
    rows, pooled = [], []
    W = {}
    for sp in ("S5", "S15"):
        W[("syn", sp)] = load_syn(sp)
        rows += cells(W[("syn", sp)], "pcmax", "syn", sp)
        for cj in ("discovery", "heldout"):
            W[(cj, sp)] = load_pub(cj, sp)
            rows += cells(W[(cj, sp)], "pcmax", cj, sp)
    for sp in ("F5", "F15"):
        w = load_pfsp(sp)
        for s in ("syn", "ta"):
            W[("pfsp-" + s, sp)] = w[w.set == s]
            rows += cells(w[w.set == s], "pfsp", s, sp)
    C = pd.DataFrame(rows)
    for (setname, sp), w in W.items():
        subs = [("all", w)]
        if setname == "pfsp-syn":
            subs.append(("large", w[w.cell.isin(["50x10", "50x20", "100x10", "100x20"])]))
        if setname == "heldout":
            subs.append(("C12", w[w.cell.isin(["C|1", "C|2"])]))
        for sub, x in subs:
            hb, hc = float(-rel(x.biased, x.rand).mean()), float(-rel(x.cf, x.rand).mean())
            pooled.append(dict(set=setname, space=sp, sub=sub, n=len(x), h_biased=hb, h_cf=hc, bias_rel=(hb - hc) / hb if hb > 0 else np.nan))
    Pd = pd.DataFrame(pooled)
    L += ["\n--- folga enviesada x com validacao cruzada (agrupadas por instancia; em pontos percentuais) ---", Pd.to_string(index=False)]
    # ---- correlacoes
    cor = []
    for (pr, st, sp), x in C.groupby(["problem", "set", "space"]):
        if len(x) >= 5:
            r1, r2 = sstats.spearmanr(x.h_biased, x.gq), sstats.spearmanr(x.h_cf, x.gq)
            cor.append(dict(problem=pr, set=st, space=sp, n_cells=len(x), rho_biased=r1.statistic, rho_cf=r2.statistic, p_cf=r2.pvalue))
    Co = pd.DataFrame(cor)
    L += ["\n--- Spearman(folga, ganho do Q-Learning) por conjunto e espaco ---", Co.to_string(index=False)]
    # ---- limiar unificado
    disc = C[(C.set == "discovery")]
    v = disc.sort_values("h_cf")
    y = (v.gq > 0).to_numpy()
    hv = v.h_cf.to_numpy()
    best, hs = -1, None
    for i in range(len(hv) - 1):
        t = 0.5 * (hv[i] + hv[i + 1])
        pred = hv >= t
        b = 0.5 * ((pred & y).sum() / max(y.sum(), 1) + (~pred & ~y).sum() / max((~y).sum(), 1))
        if b > best:
            best, hs = b, t
    rest = C[(C.set == "heldout") | ((C.problem == "pcmax") & (C.set == "syn")) | ((C.problem == "pfsp") & (C.set == "syn"))]
    acc = float(((rest.h_cf >= hs) == (rest.gq > 0)).mean())
    parts = {k: float(((x.h_cf >= hs) == (x.gq > 0)).mean()) for k, x in rest.groupby(["problem", "set"])}
    L += [f"\nlimiar h* escolhido na descoberta (acuracia balanceada {best:.3f}): {hs:.4f} pontos; acuracia nas demais celulas ({len(rest)}): {acc:.3f}; por conjunto: " + ", ".join(f"{k[0]}/{k[1]} {v:.2f}" for k, v in parts.items())]
    # ---- previsoes
    yn = lambda b: "SIM" if b else "NAO"
    a1 = bool((Pd[Pd["sub"] == "all"].h_cf < Pd[Pd["sub"] == "all"].h_biased).all()) and len(Pd[Pd["sub"] == "all"]) == 10
    fl = Pd[(Pd.set == "pfsp-syn") & (Pd["sub"] == "large")]
    pc = Pd[(Pd.set == "heldout") & (Pd["sub"] == "C12")]
    a2 = bool((fl.bias_rel >= 0.5).all() and (pc.bias_rel <= 0.35).all())
    ch = Co[(Co.set == "heldout")]
    a3 = bool((ch.rho_cf >= 0.5).all())
    cf = Co[(Co.problem == "pfsp") & (Co.set == "syn")]
    a4 = bool((cf.rho_cf >= -0.3).all())
    a5 = bool((fl.h_cf < 0.10).all())
    a6 = acc >= 0.75
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(A1) folga cf < enviesada nos dez pares: {yn(a1)}",
          f"(A2) vies relativo flowshop (classes grandes) >= 0,5 e familia C held-out <= 0,35: {yn(a2)} (flowshop " + ", ".join(f"{v:.2f}" for v in fl.bias_rel) + "; C12 held-out " + ", ".join(f"{v:.2f}" for v in pc.bias_rel) + ")",
          f"(A3) Spearman(folga cf, g_Q) >= +0,5 no held-out publicado em S5 e S15: {yn(a3)} (" + ", ".join(f"{v:+.2f}" for v in ch.rho_cf) + ")", f"(A4) flowshop sinteticas: Spearman(folga cf, g_Q) >= -0,3 em F5 e F15: {yn(a4)} (" + ", ".join(f"{v:+.2f}" for v in cf.rho_cf) + ")",
          f"(A5) flowshop sinteticas, classes grandes: folga cf < 0,10 em F5 e F15: {yn(a5)} (" + ", ".join(f"{v:.3f}" for v in fl.h_cf) + ")", f"(A6) limiar unificado: acuracia >= 0,75 nas demais celulas: {yn(a6)} ({acc:.3f})", f"U (a regra vale nos dois problemas com a folga sem vies) = A3 e A4 e A6: {yn(a3 and a4 and a6)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada209_cforacle.txt").write_text(text, encoding="utf-8")
    C.to_csv(R / "entrada209_cforacle_cells.csv", index=False)
    Pd.to_csv(R / "entrada209_cforacle_pooled.csv", index=False)
    Co.to_csv(R / "entrada209_cforacle_corr.csv", index=False)


if __name__ == "__main__":
    main()
