"""
analysis/report_entrada283b_cec69.py (Entrada 283b, POS HOC declarada)
=======================================================================
A Entrada 283 usou as 85 celulas da Entrada 280. O artigo do CEC descreve so' os hospedeiros P||Cmax (sintetico e publicado) e R||Cmax (benchmark padrao e familias publicadas): 69 celulas, sem as 16 do TSP (que fica no manuscrito de periodico).
Este relatorio repete, nessas 69 celulas, a razao de captura e o Spearman com bootstrap por classe (31 classes), a sensibilidade da margem e as contagens validada x desempate aleatorio. RESTRICAO POS HOC (escolhida depois de ver os resultados das 85 celulas, para casar com o escopo do artigo):
sem previsoes registradas; os numeros das 85 celulas continuam valendo e estao em results/entrada283_reanalysis.txt. Uso: python -m analysis.report_entrada283b_cec69
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
B = 4000


def boot(x, v, cl, fn, seed=2832):
    rng = np.random.default_rng(seed)
    keys = pd.unique(cl)
    groups = [np.flatnonzero(cl == k) for k in keys]
    pt = fn(x, v)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        s = fn(x[idx], v[idx])
        if np.isfinite(s):
            out.append(s)
    lo, hi = np.percentile(out, [5, 95]) if len(out) > 50 else (np.nan, np.nan)
    return float(pt), float(lo), float(hi), len(groups)


def cap(x, v):
    return float(x.sum() / v.sum()) if v.sum() != 0 else np.nan


def sp(x, v):
    if np.ptp(x) == 0 or np.ptp(v) == 0:
        return np.nan
    return float(ss.spearmanr(x, v).statistic)


def main():
    C = pd.read_csv(R / "entrada283_cells.csv")
    T = pd.read_csv(R / "entrada280_tiesall_tests.csv")
    C = C[C.coll != "tsp"].reset_index(drop=True)
    L = ["=" * 110, "AS 69 CELULAS DOS HOSPEDEIROS DO ARTIGO DO CEC (sem o TSP): captura, Spearman e margem (Entrada 283b, POS HOC)", "=" * 110]
    rows = []
    for name, d in [("todas as 69", C)] + [(k, C[C.coll == k]) for k in ("padrao", "familias", "sintetico", "publicado")]:
        v = d.g_val.to_numpy()
        cl = d.cluster.to_numpy()
        for lab, col in (("padrao", "g_ql"), ("permutada", "g_qp"), ("desempate aleatorio", "g_rt")):
            x = d[col].to_numpy()
            c = boot(x, v, cl, cap)
            s = boot(x, v, cl, sp)
            rows.append(dict(grupo=name, aprendiz=lab, celulas=len(d), classes=c[3], captura=c[0], cap_lo=c[1], cap_hi=c[2], spearman=s[0], sp_lo=s[1], sp_hi=s[2]))
    Rr = pd.DataFrame(rows)
    L += [Rr.to_string(index=False, float_format=lambda a: f"{a:.3f}")]
    Rr.to_csv(R / "entrada283b_capture69.csv", index=False)
    L += ["", "--- margem (desempate aleatorio ganha do aleatorio alem da margem m) ---"]
    for m in (0.02, 0.05, 0.10):
        L.append(f"margem {m:.2f}: {int(C[f'rt_beyond_{m:.2f}'].sum())} de {len(C)}")
    low = C[C.g_val <= 0.05]
    L.append(f"celulas com a validada <= 0,05: {len(low)}; nelas o desempate aleatorio ganha alem da margem em {int(low['rt_beyond_0.05'].sum())}")
    B_ = T[T.id.str.startswith("B-")].copy()
    B_["cell"] = B_.id.str[2:]
    B_ = B_.merge(C[["coll", "cell"]], on=["coll", "cell"])
    val_better = B_[B_.confirmada]
    rt_better = B_[B_.reject & (B_.mean_d_pct > 0)]
    L.append(f"validada bate o desempate aleatorio (Holm) em {len(val_better)} de {len(B_)}; o desempate aleatorio bate a validada em {len(rt_better)} (diferencas {sorted((rt_better.mean_d_pct).round(3).tolist())}); cells: {rt_better.cell.tolist()}")
    ql_beyond = int(((C.g_ql > 0.05)).sum())
    L.append(f"celulas em que o Q-Learning padrao ganha mais de 0,05 pontos (estimativa pontual): {ql_beyond} de {len(C)}")
    L.append(f"ganho medio sobre as 69 celulas: validada {C.g_val.mean():.3f}; padrao {C.g_ql.mean():.3f}; permutada {C.g_qp.mean():.3f}; desempate aleatorio {C.g_rt.mean():.3f}")
    L.append(f"maior razao rt/validada entre as celulas com validada > 0,05: {float((C[C.g_val > 0.05].g_rt / C[C.g_val > 0.05].g_val).max()):.3f}")
    text = "\n".join(L)
    print(text)
    (R / "entrada283b_cec69.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
