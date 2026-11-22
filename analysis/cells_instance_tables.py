"""
analysis/cells_instance_tables.py (Entrada 286, Ciclo 53)
==========================================================
Tabelas por instancia das 85 celulas das Entradas 280/283 (69 do artigo do CEC, sem o TSP).

A logica de carga e' uma COPIA da de analysis/report_entrada280_tiesall.py (arquivo congelado, nao editado): cada celula e' um DataFrame com uma linha por instancia (medias sobre as sementes) e uma coluna
por metodo (rand, val, rt, ql, qp, ucb, pg, arm0..., conforme a colecao). Aqui as colunas extras (ucb, pg, arm*) NAO sao descartadas, porque as analises dos Ciclos 53 e 54 as usam.

Uso: from analysis.cells_instance_tables import load_cells; cells = load_cells(include_tsp=True)
Retorna uma lista de dicts (coll, cell, space, x) na mesma ordem de report_entrada280_tiesall (padrao, familias, sintetico, publicado, tsp).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada189_published import per_instance  # noqa: E402

R = ROOT / "results"
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]


def read(pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pattern)))], ignore_index=True)


def load_cells(include_tsp: bool = True) -> list[dict]:
    out: list[dict] = []
    # ---------------------------------------------------------------- benchmark padrao (21 celulas)
    for sp in ("B5", "B15", "B30"):
        d = read(f"fr_test_{sp}_s*of4_entrada256.csv")
        w = d.groupby(["type", "instance", "method"]).makespan.mean().unstack("method").reset_index()
        q = read(f"qperm_fanjul_{sp}_s*of*_entrada260.csv").groupby(["type", "instance"]).makespan.mean().rename("qp").reset_index()
        rt = read("ties280_b5_B5_s*of12_entrada280.csv" if sp == "B5" else f"ties_fanjul_{sp}_s*of12_entrada279.csv")
        rt = rt[rt.method == "ql_rt"].groupby(["type", "instance"]).makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on=["type", "instance"]).merge(rt, on=["type", "instance"])
        assert len(w) == 700, (sp, len(w))
        for ty in TYPES:
            out.append(dict(coll="padrao", cell=f"{sp}|{ty}", space=sp, x=w[w.type == ty].reset_index(drop=True)))
    # ---------------------------------------------------------------- familias publicadas de R||Cmax (6)
    for sp in ("S5", "S15"):
        d = read(f"ur_ext_test_{sp}_s*of*_entrada254.csv")
        w = d.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
        q = read(f"qperm_families_{sp}_s*of*_entrada260.csv").groupby(["cls", "rep"]).makespan.mean().rename("qp").reset_index()
        rt = read(f"ties280_families_{sp}_s*of12_entrada280.csv").groupby(["cls", "rep"]).makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on=["cls", "rep"]).merge(rt, on=["cls", "rep"])
        w["fam"] = w.cls.str.split("|").str[0]
        for fam in ("unc", "cj", "cm"):
            out.append(dict(coll="familias", cell=f"{sp}|{fam}", space=sp, x=w[w.fam == fam].reset_index(drop=True)))
    # ---------------------------------------------------------------- P||Cmax sintetico (24)
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        w = d.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
        q = read(f"qperm_pcmax_{sp}_s*of*_entrada261.csv").groupby(["cls", "rep"]).makespan.mean().rename("qp").reset_index()
        rt = read(f"ties280_pcmax_{sp}_s*of12_entrada280.csv").groupby(["cls", "rep"]).makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on=["cls", "rep"]).merge(rt, on=["cls", "rep"])
        for cl in sorted(w.cls.unique()):
            out.append(dict(coll="sintetico", cell=f"{sp}|{cl}", space=sp, x=w[w.cls == cl].reset_index(drop=True)))
    # ---------------------------------------------------------------- P||Cmax publicado held-out (18)
    info = json.loads((R / "pubh_groups_entrada190.json").read_text())
    for sp in ("S5", "S15"):
        d = read(f"pubh_{sp}_s*of*_entrada190.csv")
        w = per_instance(d)
        q = read(f"qpermpub_{sp}_s*of*_entrada264.csv").groupby("cls").makespan.mean().rename("qp").reset_index()
        rt = read(f"ties280_pub_{sp}_s*of12_entrada280.csv").groupby("cls").makespan.mean().rename("rt").reset_index()
        w = w.merge(q, on="cls").merge(rt, on="cls")
        assert len(w) == len(info["test"]), (sp, len(w), len(info["test"]))
        w["fam"] = w.cls.str.split("|").str[0]
        w["grp"] = w.cls.str.split("|").str[1].astype(int)
        for (f, g), x in w.groupby(["fam", "grp"]):
            out.append(dict(coll="publicado", cell=f"{sp}|{f}|{g}", space=sp, x=x.reset_index(drop=True)))
    # ---------------------------------------------------------------- TSP (16)
    if include_tsp:
        d = read("tsp_test_s*of*_entrada267.csv")
        ch = json.loads((R / "tsp_choice_entrada267.json").read_text())
        rtd = read("ties280_tsp_s*of12_entrada280.csv")
        key = ["inst", "kind", "n"]
        arms = d[d.method.str.startswith("arm")].copy()
        arms["a"] = arms.method.str[3:].astype(int)
        WA = arms.groupby(key + ["a"]).length.mean().unstack("a")
        WA.columns = [f"arm{c}" for c in WA.columns]
        WA = WA.reset_index()
        for sp in ("T5", "T15"):
            x = d[(d.space == sp) & ~d.method.str.startswith("arm")]
            w = x.groupby(key + ["method"]).length.mean().unstack("method").reset_index().merge(WA, on=key)
            w["cls"] = w.kind + "|" + w.n.astype(str)
            w["val"] = [float(r[f"arm{ch[sp][r.cls]}"]) for _, r in w.iterrows()]
            w = w.merge(rtd[rtd.space == sp].groupby(key).length.mean().rename("rt").reset_index(), on=key)
            w["qp"] = w["ql_perm"]
            for (k, n), g in w.groupby(["kind", "n"]):
                out.append(dict(coll="tsp", cell=f"{sp}|{k}|{n}", space=sp, x=g.reset_index(drop=True)))
    return out


if __name__ == "__main__":
    cs = load_cells()
    print(len(cs), "celulas")
    for c in cs[:3] + cs[-2:]:
        print(c["coll"], c["cell"], c["x"].shape, [k for k in c["x"].columns][:12])
