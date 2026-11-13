"""Entrada 270: leitor da TSPLIB e valores otimos; o ILS do TSP em uma instancia publicada nunca fica abaixo do otimo."""
import json

import numpy as np

from pmsp import tsp_numba as TS
from pmsp.instance_tsplib import TSPLIB, load_tsplib, optima


def test_manifest_and_optima_cover_the_37_instances():
    man = json.loads((TSPLIB / "manifest_tsplib.json").read_text())["instances"]
    opt = optima()
    assert len(man) == 37
    assert all(m["name"] in opt for m in man)
    assert all(m["edge_weight_type"] == "EUC_2D" for m in man)


def test_loader_gives_rounded_symmetric_distances():
    D, NL = load_tsplib("kroA200")
    assert D.shape == (200, 200) and NL.shape == (200, 10)
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0)
    assert np.all(D == np.floor(D))                      # distancias inteiras como na TSPLIB
    assert all(D[i, NL[i, 0]] <= D[i, NL[i, 1]] for i in range(200))


def test_ils_on_kroA200_is_a_valid_tour_not_below_the_optimum_and_close_to_it():
    D, NL = load_tsplib("kroA200")
    tour0, _ = TS.construct(D, NL)
    kinds, sizes = TS.arm_table("T15")
    r = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 500, TS.AGENT_FIXED, 4, 0.1, 0.0, 3, np.array([0.1, 0.9, 0.995, 0.05, 0.0]), 0.0)
    opt = optima()["kroA200"]
    assert r[0] >= opt - 1e-6                            # nenhum tour e' menor que o otimo (confere o arquivo e o leitor)
    assert r[0] <= 1.06 * opt
