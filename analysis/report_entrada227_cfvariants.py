"""
analysis/report_entrada227_cfvariants.py
==========================================
Entrada 227 (Ciclo 21, frente B): analise POS HOC (NAO pre-registrada; criticas 1 e 2 da revisao cega #4) do estimador da folga e da regra do limiar. Le apenas dados ja' existentes (as quatro sementes por acao fixa das Entradas 180/189/190/197 e 209); nada e' rodado.
Perguntas: (1) a folga cf com 2+2 sementes e' uma cota inferior FROUXA: o que acontece com a folga e com a regra se a acao do oraculo e' escolhida em MAIS sementes (3 e avalia em 1, quatro dobras) ou em MENOS (1 e avalia em 3), ou se a escolha 2+2 e' simetrizada?  (2) quanta incerteza tem a acuracia da regra quando as celulas de um mesmo conglomerado (a mesma classe nos dois espacos S5 e S15) sao reamostradas juntas (bootstrap por conglomerado, 30 conglomerados nas 60 celulas fora da amostra) e ela supera a taxa de base?  (3) metricas DENTRO de cada hospedeiro.
Variantes (por instancia, depois media por celula): one-seed = menor makespan da semente 0 (enviesada); cf13 = escolhe em uma semente, avalia na media das outras tres (4 dobras); cf22 = escolhe em {0,1}, avalia em {2,3} (a das Entradas 209/210); cf22s = media de cf22 e da troca de papeis; cf31 = escolhe na media de tres, avalia na quarta (4 dobras).
Regra: limiar = ponto medio que maximiza a acuracia balanceada nas 18 celulas da DESCOBERTA publicada (como na Entrada 210), aplicado sem mudanca as 60 celulas restantes (held-out publicado, sinteticas P||Cmax, flowshop sinteticas); 'Q ganha' = ganho do Q-Learning sobre o aleatorio a iteracoes iguais > 0.
Uso: python -m analysis.report_entrada227_cfvariants   -> results/entrada227_cfvariants.txt, results/entrada227_cfvariants_cells.csv
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
from analysis import report_entrada209_cforacle as M  # noqa: E402
from analysis.report_entrada180_agents import rel  # noqa: E402

R = ROOT / "results"
VARS = ["biased", "cf13", "cf22", "cf22s", "cf31"]
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def arms_variants(old, new):
    """igual a arms_table da Entrada 209, com as variantes de escolha/avaliacao das quatro sementes por acao"""
    o = old[old.method.str.startswith("arm")].copy()
    o["arm"] = o.method.str[3:].astype(int)
    o = o.groupby(["cls", "arm"]).makespan.mean().rename("s0").reset_index()
    n = new.pivot_table(index=["cls", "arm"], columns="seed_i", values="makespan").reset_index()
    n.columns = ["cls", "arm"] + [f"s{c}" for c in n.columns[2:]]
    a = o.merge(n, on=["cls", "arm"]).sort_values(["cls", "arm"])
    cls = a.cls.to_numpy()
    uniq, start = np.unique(cls, return_index=True)
    cnt = np.diff(np.append(start, len(cls)))
    assert (cnt == cnt[0]).all(), "numero de acoes diferente entre instancias"
    X = a[["s0", "s1", "s2", "s3"]].to_numpy(float).reshape(len(uniq), cnt[0], 4)  # (instancia, acao, semente)
    ix = np.arange(len(uniq))

    def pick(sel, ev):
        return ev[ix, sel.argmin(axis=1)]

    out = {"biased": X[:, :, 0].min(axis=1)}
    f = X[:, :, [0, 1]].mean(axis=2), X[:, :, [2, 3]].mean(axis=2)
    out["cf22"] = pick(f[0], f[1])
    out["cf22s"] = 0.5 * (pick(f[0], f[1]) + pick(f[1], f[0]))
    out["cf31"] = np.mean([pick(np.delete(X, j, axis=2).mean(axis=2), X[:, :, j]) for j in range(4)], axis=0)
    out["cf13"] = np.mean([pick(X[:, :, j], np.delete(X, j, axis=2).mean(axis=2)) for j in range(4)], axis=0)
    r = pd.DataFrame(out, index=uniq)
    r.index.name = "cls"
    r["cf"] = r.cf22
    return r


def raw(kind, sp, cj=None):
    """execucoes brutas por semente (para separar as sementes do controle aleatorio)"""
    if kind == "syn":
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        d["cls"] = d.cls + "#" + d.rep.astype(str)
        return d
    pat = {"discovery": f"pub_{sp}_s*of*_entrada189.csv", "heldout": f"pubh_{sp}_s*of*_entrada190.csv", "pfsp": f"pfsp_test_{sp}_s*of*_entrada197.csv"}[cj]
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat)))], ignore_index=True)


def attach(w, d):
    """controle aleatorio INDEPENDENTE: a primeira metade das sementes serve a folga, a segunda ao ganho do Q-Learning"""
    r = d[d.method == "rand"].pivot_table(index="cls", columns="seed_i", values="makespan")
    ks = sorted(r.columns)
    k = (len(ks) + 1) // 2
    add = pd.DataFrame({"rand_h": r[ks[:k]].mean(axis=1), "rand_g": r[ks[k:]].mean(axis=1)}).reset_index()
    return w.merge(add, on="cls", how="left")


def cells_of(w, problem, setname, sp):
    return [dict(problem=problem, set=setname, space=sp, cell=c, n=len(x), gq=float(-rel(x.ql, x.rand).mean()), h_ind=float(-rel(x.cf22, x.rand_h).mean()), gq_ind=float(-rel(x.ql, x.rand_g).mean()), **{v: float(-rel(x[v], x.rand).mean()) for v in VARS}) for c, x in w.groupby("cell")]


def choose(h, y):
    o = np.argsort(h)
    h, y = np.asarray(h)[o], np.asarray(y)[o]
    best, hs = -1.0, None
    for i in range(len(h) - 1):
        t = 0.5 * (h[i] + h[i + 1])
        p = h >= t
        b = 0.5 * ((p & y).sum() / max(y.sum(), 1) + (~p & ~y).sum() / max((~y).sum(), 1))
        if b > best:
            best, hs = b, t
    return hs, best


def metrics(h, y, t):
    h, y = np.asarray(h), np.asarray(y)
    p = h >= t
    tpr = (p & y).sum() / y.sum() if y.sum() else np.nan
    tnr = (~p & ~y).sum() / (~y).sum() if (~y).sum() else np.nan
    return float((p == y).mean()), float(np.nanmean([tpr, tnr])) if y.sum() and (~y).sum() else np.nan, float(max(y.mean(), 1 - y.mean()))


def main():
    M.arms_table = arms_variants
    rows, pooled = [], []
    for sp in ("S5", "S15"):
        w = attach(M.load_syn(sp), raw("syn", sp))
        rows += cells_of(w, "pcmax", "syn", sp)
        pooled.append(dict(set="syn", space=sp, n=len(w), **{v: float(-rel(w[v], w.rand).mean()) for v in VARS}))
        for cj in ("discovery", "heldout"):
            w = attach(M.load_pub(cj, sp), raw("pub", sp, cj))
            rows += cells_of(w, "pcmax", cj, sp)
            pooled.append(dict(set=cj, space=sp, n=len(w), **{v: float(-rel(w[v], w.rand).mean()) for v in VARS}))
    for sp in ("F5", "F15"):
        w = attach(M.load_pfsp(sp), raw("pfsp", sp, "pfsp"))
        for s in ("syn", "ta"):
            x = w[w.set == s]
            rows += cells_of(x, "pfsp", s, sp)
            pooled.append(dict(set="pfsp-" + s, space=sp, n=len(x), **{v: float(-rel(x[v], x.rand).mean()) for v in VARS}))
    C, Pd = pd.DataFrame(rows), pd.DataFrame(pooled)
    C["win"] = C.gq > 0
    C["cluster"] = C.problem + "|" + C.set + "|" + C.cell
    L = ["=" * 110, "VARIANTES DO ESTIMADOR DA FOLGA E ROBUSTEZ DA REGRA (Entrada 227; POS HOC)", "=" * 110,
         "\n--- folga agregada por conjunto e espaco (pontos percentuais; one-seed = cota superior, as outras sao cotas inferiores em esperanca) ---", Pd.to_string(index=False)]
    disc = C[C.set == "discovery"]
    test = C[(C.set == "heldout") | ((C.problem == "pcmax") & (C.set == "syn")) | ((C.problem == "pfsp") & (C.set == "syn"))].copy()
    hosts = {"pcmax(held-out+sinteticas)": test[test.problem == "pcmax"], "flowshop sintetico": test[test.problem == "pfsp"]}
    rng = np.random.default_rng(20261023)
    res = []
    L.append(f"\n--- a regra com cada variante: limiar escolhido nas {len(disc)} celulas da descoberta, aplicado as {len(test)} restantes ({test.cluster.nunique()} conglomerados) ---")
    for v in VARS:
        t, b0 = choose(disc[v], disc.win.to_numpy())
        acc, bal, base = metrics(test[v], test.win.to_numpy(), t)
        clusters = test.cluster.unique()
        grp = {c: test[test.cluster == c] for c in clusters}
        bs = []
        for _ in range(4000):
            pick = rng.choice(len(clusters), len(clusters))
            x = pd.concat([grp[clusters[i]] for i in pick])
            a_, b_, base_ = metrics(x[v], x.win.to_numpy(), t)
            bs.append((a_, b_, a_ - base_))
        bs = np.array(bs, float)
        lo = lambda k: float(np.nanpercentile(bs[:, k], 2.5))
        hi = lambda k: float(np.nanpercentile(bs[:, k], 97.5))
        per = {}
        for hn, x in hosts.items():
            a_, b_, bs_ = metrics(x[v], x.win.to_numpy(), t)
            r_ = sstats.spearmanr(x[v], x.gq).statistic if len(x) > 3 else np.nan
            per[hn] = (a_, b_, bs_, r_)
        res.append(dict(variante=v, limiar=t, bal_descoberta=b0, acuracia=acc, balanceada=bal, base=base, ac_menos_base=acc - base,
                        ic95_acuracia=f"[{lo(0):.2f}, {hi(0):.2f}]", ic95_balanceada=f"[{lo(1):.2f}, {hi(1):.2f}]", ic95_exc_base=f"[{lo(2):+.2f}, {hi(2):+.2f}]", p_exc_le0=float((bs[:, 2] <= 0).mean()),
                        pcmax_bal=per["pcmax(held-out+sinteticas)"][1], pcmax_rho=per["pcmax(held-out+sinteticas)"][3], flow_acc=per["flowshop sintetico"][0], flow_bal=per["flowshop sintetico"][1], flow_rho=per["flowshop sintetico"][3]))
    Rr = pd.DataFrame(res)
    L.append(Rr.to_string(index=False))
    # controle aleatorio independente: a folga usa a primeira metade das sementes do aleatorio, o ganho do Q-Learning a segunda (resposta a revisao cega #6)
    C["win_ind"] = C.gq_ind > 0
    disc2 = C[C.set == "discovery"]
    test2 = C.loc[test.index]
    t2, b2 = choose(disc2.h_ind, disc2.win_ind.to_numpy())
    a2, bal2, base2 = metrics(test2.h_ind, test2.win_ind.to_numpy(), t2)
    clusters = test2.cluster.unique()
    grp = {c: test2[test2.cluster == c] for c in clusters}
    bs2 = []
    for _ in range(4000):
        pick = rng.choice(len(clusters), len(clusters))
        x = pd.concat([grp[clusters[i]] for i in pick])
        bs2.append(metrics(x.h_ind, x.win_ind.to_numpy(), t2)[1])
    bs2 = np.array(bs2, float)
    pc2 = test2[test2.problem == "pcmax"]
    rho2 = sstats.spearmanr(pc2.h_ind, pc2.gq_ind).statistic
    L.append(f"\n--- controle aleatorio independente (a folga com as sementes {{1a metade}} do aleatorio, o ganho do Q com as da 2a metade) ---\nlimiar {t2:.4f}; acuracia {a2:.3f}, balanceada {bal2:.3f} (IC95 por conglomerado [{np.nanpercentile(bs2, 2.5):.2f}, {np.nanpercentile(bs2, 97.5):.2f}]), base {base2:.3f}; Spearman no P||Cmax (held-out + sinteticas) {rho2:+.3f}")
    L.append("\nLeitura: 'exc_base' = acuracia menos a taxa da classe majoritaria; p_exc_le0 = fracao das reamostras por conglomerado em que a regra nao supera a taxa de base; pcmax_* e flow_* = metricas DENTRO do hospedeiro (limiar fixo da descoberta).")
    text = "\n".join(L)
    print(text)
    (R / "entrada227_cfvariants.txt").write_text(text, encoding="utf-8")
    C.drop(columns=["cluster"]).to_csv(R / "entrada227_cfvariants_cells.csv", index=False)


if __name__ == "__main__":
    main()
