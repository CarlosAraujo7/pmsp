"""
analysis/published_cells.py (Entrada 192, Ciclo 17)
=====================================================
Dados derivados das instancias publicadas (estagio 2 = 'descoberta', Entrada 189; estagio 3 = 'replicacao' held-out, Entrada 190) para a figura e a tabela do artigo do CEC: por celula (conjunto x espaco x familia x tercil de n) a folga do oraculo e o d medio de cada metodo contra o aleatorio a iteracoes iguais,
e por subconjunto (todas, familia C grupos 1-2, familia E) o d a iteracoes iguais e a CPU igual do Q-Learning e da roleta ALNS. Nao refaz nenhum teste: so' agrega as saidas ja' congeladas. Uso: python -m analysis.published_cells
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import rel  # noqa: E402
from analysis.report_entrada189_published import per_instance  # noqa: E402

R = ROOT / "results"
SETS = {"discovery": "pub_{sp}_s*of*_entrada189.csv", "replication": "pubh_{sp}_s*of*_entrada190.csv"}
SUBSETS = {"all": lambda w: w.fam != "", "C12": lambda w: (w.fam == "C") & w.grp.isin([1, 2]), "E": lambda w: w.fam == "E"}


def main():
    cells, summ = [], []
    for name, pat in SETS.items():
        for sp in ("S5", "S15"):
            d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat.format(sp=sp))))], ignore_index=True)
            w = per_instance(d)
            w["fam"] = w.cls.str.split("|").str[0]
            w["grp"] = w.cls.str.split("|").str[1].astype(int)
            for (f, g), x in w.groupby(["fam", "grp"]):
                cells.append(dict(set=name, space=sp, fam=f, grp=g, n=len(x), folga=float(rel(x.oracle, x.rand).mean()), val=float(rel(x.val, x.rand).mean()), ql=float(rel(x.ql, x.rand).mean()), alns=float(rel(x.alns, x.rand).mean()), ts=float(rel(x.ts, x.rand).mean())))
            for sub, m in SUBSETS.items():
                x = w[m(w)]
                rec = dict(set=name, space=sp, subset=sub, n=len(x), folga=float(rel(x.oracle, x.rand).mean()))
                for k in ("ql", "alns", "ts", "val"):
                    rec[f"{k}_iter"] = float(rel(x[k], x.rand).mean())
                for k in ("ql", "alns"):
                    rec[f"{k}_cpu"] = float(rel(x[k], x[f"rand_eq_{k}"]).mean())
                rec["val_vs_ql"] = float(rel(x.val, x.ql).mean())
                summ.append(rec)
    pd.DataFrame(cells).to_csv(R / "entrada192_published_cells.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(R / "entrada192_published_summary.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
