"""
pmsp/instance_tsplib.py (Entrada 270)
Leitor de instancias da TSPLIB (Reinelt, 1991) do tipo EUC_2D para o ILS do TSP (pmsp/tsp_numba.py): distancias arredondadas ao inteiro mais proximo, como na TSPLIB, de modo que os comprimentos de tour sao comparaveis aos valores
otimos publicados (arquivo `solutions`). Os arquivos sao lidos como TEXTO (nada e' executado).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .instance_tsp import dist_matrix, neighbor_lists

TSPLIB = Path(__file__).resolve().parent.parent / "external_benchmarks" / "tsplib"


def optima() -> dict:
    out = {}
    for ln in (TSPLIB / "solutions_mastqe.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\S+)\s*:\s*(\d+)", ln)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def load_tsplib(name: str, K: int = 10):
    """(D arredondada, listas de vizinhos) da instancia EUC_2D `name`."""
    txt = (TSPLIB / f"{name}.tsp").read_text(encoding="latin-1")
    assert re.search(r"EDGE_WEIGHT_TYPE\s*:\s*EUC_2D", txt), name
    body = txt.split("NODE_COORD_SECTION")[1].split("EOF")[0].split()
    pts = np.array(body, dtype=float).reshape(-1, 3)[:, 1:3]
    D = np.ascontiguousarray(np.floor(dist_matrix(pts) + 0.5))
    return D, neighbor_lists(D, K)
