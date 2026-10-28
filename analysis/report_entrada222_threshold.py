"""
analysis/report_entrada222_threshold.py
=========================================
Entrada 222 (Ciclo 20): verificacao POS HOC (NAO pre-registrada; pedida pela revisao cega #3) da regra do limiar h* da folga cross-fitted. Le apenas tabelas ja' existentes (results/entrada209_cforacle_cells.csv e results/entrada219_ga_cells.csv); nada e' rodado.
Perguntas: (1) qual a taxa de base ('o Q-Learning NAO ganha') nas celulas fora da amostra e qual a acuracia balanceada do limiar escolhido na descoberta?  (2) a regra discrimina DENTRO de cada conjunto (inclusive o flowshop)?  (3) o limiar e' estavel se um hospedeiro inteiro e' deixado de fora (LOHO: P||Cmax, flowshop sintetico, algoritmo genetico)?
Celula 'ganha' = ganho do Q-Learning sobre o aleatorio a iteracoes iguais > 0; predicao da regra = folga cf >= limiar. Limiar = ponto medio que maximiza a acuracia balanceada (mesmo procedimento da Entrada 210).
Uso: python -m analysis.report_entrada222_threshold   -> results/entrada222_threshold.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


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


def score(h, y, t):
    h, y = np.asarray(h), np.asarray(y)
    p = h >= t
    acc = float((p == y).mean())
    tpr = (p & y).sum() / y.sum() if y.sum() else float("nan")
    tnr = (~p & ~y).sum() / (~y).sum() if (~y).sum() else float("nan")
    bal = float(np.nanmean([tpr, tnr])) if y.sum() and (~y).sum() else float("nan")
    base = float(max(y.mean(), 1 - y.mean()))
    return acc, bal, base, int(len(y))


def main():
    C = pd.read_csv(R / "entrada209_cforacle_cells.csv")
    G = pd.read_csv(R / "entrada219_ga_cells.csv")
    C["win"] = C.gq > 0
    G["win"] = G.ql > 0
    L = ["=" * 110, "LIMIAR h*: verificacao POS HOC (Entrada 222; nao pre-registrada)", "=" * 110]
    disc = C[C.set == "discovery"]
    hs, bal0 = choose(disc.h_cf, disc.win.to_numpy())
    rest = C[(C.set == "heldout") | ((C.problem == "pcmax") & (C.set == "syn")) | ((C.problem == "pfsp") & (C.set == "syn"))]
    L.append(f"h* (descoberta, 18 celulas; acuracia balanceada {bal0:.3f}) = {hs:.4f}")
    acc, bal, base, n = score(rest.h_cf, rest.win.to_numpy(), hs)
    L.append(f"fora da amostra ({n} celulas): acuracia {acc:.3f}, acuracia balanceada {bal:.3f}, taxa de base da classe majoritaria {base:.3f}, fracao em que o Q ganha {rest.win.mean():.3f}")
    L.append("\npor conjunto (limiar fixo h*): acuracia | balanceada | base | celulas | fracao que ganha")
    for (pr, st), x in list(rest.groupby(["problem", "set"])) + [(("pfsp", "ta"), C[(C.problem == "pfsp") & (C.set == "ta")])]:
        a, b, bs, k = score(x.h_cf, x.win.to_numpy(), hs)
        L.append(f"  {pr}/{st}: {a:.3f} | {b:.3f} | {bs:.3f} | {k} | {x.win.mean():.3f}" + ("  (Taillard: nao entra nas 60)" if st == "ta" else ""))
    a, b, bs, k = score(G.h_cf, G.win.to_numpy(), hs)
    L.append(f"  algoritmo genetico: {a:.3f} | {b:.3f} | {bs:.3f} | {k} | {G.win.mean():.3f}  (nao entra nas 60)")
    allc = pd.concat([rest[["h_cf", "win"]], G[["h_cf", "win"]]])
    a, b, bs, k = score(allc.h_cf, allc.win.to_numpy(), hs)
    L.append(f"incluindo o AG ({k} celulas): acuracia {a:.3f}, balanceada {b:.3f}, base {bs:.3f}")
    L.append("\nLOHO (treina o limiar em dois hospedeiros, testa no terceiro; hospedeiros: P||Cmax = descoberta + held-out + sinteticas; flowshop sintetico; AG):")
    H = {
        "pcmax": C[C.problem == "pcmax"][["h_cf", "win"]],
        "flowshop": C[(C.problem == "pfsp") & (C.set == "syn")][["h_cf", "win"]],
        "ga": G[["h_cf", "win"]],
    }
    for out in H:
        tr = pd.concat([v for k, v in H.items() if k != out])
        t, bt = choose(tr.h_cf, tr.win.to_numpy())
        a, b, bs, k = score(H[out].h_cf, H[out].win.to_numpy(), t)
        L.append(f"  testa {out} ({k} celulas): limiar de treino {t:.4f} (balanceada no treino {bt:.3f}); teste: acuracia {a:.3f}, balanceada {b:.3f}, base {bs:.3f}")
    # ---- Ciclo 21 (Entrada 229, tambem pos hoc): os tempos finos (bateria E, Entrada 228) e a leitura de CONDICAO NECESSARIA em todas as celulas
    F = pd.read_csv(R / "entrada228_cf_fine_cells.csv")
    F["win"] = F.gq > 0
    a, b, bs, k = score(F.h_cf, F.win.to_numpy(), hs)
    L.append(f"\ntempos finos (Entrada 228, {k} celulas classe x espaco): acuracia {a:.3f}, balanceada {b:.3f}, base {bs:.3f}, fracao em que o Q ganha {F.win.mean():.3f}")
    A = pd.concat([C[["problem", "set", "space", "cell", "h_cf", "gq"]], G.assign(problem="ga", set="ga")[["problem", "set", "space", "cell", "h_cf"]].assign(gq=G.ql), F.assign(problem="pcmax", set="fine")[["problem", "set", "space", "cell", "h_cf", "gq"]]], ignore_index=True)
    L.append(f"\ncondicao necessaria, TODAS as {len(A)} celulas (P||Cmax sinteticas, descoberta, held-out, flowshop sintetico e Taillard, AG, tempos finos); margem de equivalencia 0,05:")
    for nm, x in [("todas", A)] + list(A.groupby("set")):
        lo, hi = x[x.h_cf < hs], x[x.h_cf >= hs]
        L.append(f"  {nm}: {len(x)} celulas | abaixo de h*: {len(lo)}, ganho do Q > 0: {int((lo.gq > 0).sum())}, ganho > 0,05: {int((lo.gq > 0.05).sum())}, maior ganho {lo.gq.max() if len(lo) else float('nan'):+.3f} | acima de h*: {len(hi)}, ganho > 0: {int((hi.gq > 0).sum())}, ganho > 0,05: {int((hi.gq > 0.05).sum())}")
    hi = A[A.h_cf >= hs]
    cap = (hi.gq / hi.h_cf)
    L.append(f"  acima de h* ({len(hi)} celulas): razao ganho do Q / folga cf: mediana {cap.median():.2f}, quartis [{cap.quantile(0.25):.2f}, {cap.quantile(0.75):.2f}]; ganho do Q negativo em {int((hi.gq < 0).sum())}")
    # Ciclo 21 (resposta a revisao #5): o mesmo, FORA das 18 celulas da descoberta com que h* foi escolhido
    B = A[A.set != "discovery"]
    lo, hi = B[B.h_cf < hs], B[B.h_cf >= hs]
    L.append(f"\nfora da descoberta, {len(B)} celulas | abaixo de h*: {len(lo)}, ganho do Q > 0: {int((lo.gq > 0).sum())}, ganho > 0,05: {int((lo.gq > 0.05).sum())}, maior ganho {lo.gq.max():+.3f} | acima de h*: {len(hi)}, ganho > 0: {int((hi.gq > 0).sum())}, ganho > 0,05: {int((hi.gq > 0.05).sum())}, ganho negativo: {int((hi.gq < 0).sum())}, razao ganho/folga cf mediana {(hi.gq / hi.h_cf).median():.2f}")
    # Ciclo 22 (revisoes cegas #6): sensibilidade da leitura unilateral a margem
    for mg in (0.02, 0.10):
        L.append(f"margem {mg:.2f}: abaixo de h* {int((lo.gq > mg).sum())} de {len(lo)} | acima de h* {int((hi.gq > mg).sum())} de {len(hi)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada222_threshold.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
