"""Entradas 274 e 275: o nucleo aquecido do ILS do TSP (pmsp/tsp_warm.py) repete o agente 11 do nucleo original com um vetor constante, congela a tabela com alfa = 0 e usa a priori por acao."""
import numpy as np

from pmsp import tsp_numba as TS
from pmsp import tsp_warm as W
from pmsp.instance_tsp import build

HP = np.array([0.1, 0.9, 1.0, 0.05, 0.0])


def _inst(seed=3, n=150):
    D, NL = build("unif", n, seed)
    tour0, _ = TS.construct(D, NL)
    return D, NL, tour0


def test_constant_prior_equals_the_original_optimistic_agent():
    D, NL, tour0 = _inst()
    kinds, sizes = TS.arm_table("T15")
    for seed in (11, 12):
        a = TS.tsp_agent_core(D, NL, tour0, kinds, sizes, 120, TS.AGENT_QO, 0, 0.1, 0.0, seed, HP, 0.02)
        b = W.tsp_qo_prior_core(D, NL, tour0, kinds, sizes, 120, seed, HP, np.full(15, 0.02))
        assert float(a[0]) == float(b[0]) and float(a[2]) == float(b[2]) and a[3].tolist() == b[3].tolist()


def test_frozen_table_follows_the_prior_with_probability_one_minus_epsilon():
    D, NL, tour0 = _inst(5, 120)
    kinds, sizes = TS.arm_table("T15")
    sc = np.zeros(15)
    sc[7] = 1.0
    hp = HP.copy()
    hp[0] = 0.0
    used = W.tsp_qo_prior_core(D, NL, tour0, kinds, sizes, 400, 21, hp, 0.02 * sc)[3]
    # a acao de maior Q0 e' escolhida em ~ 1 - eps + eps/15 das iteracoes (eps = 0,05)
    assert used.sum() == 400 and used[7] >= 370 and used[7] <= 400


def test_learning_moves_away_from_a_wrong_prior():
    D, NL, tour0 = _inst(7, 150)
    kinds, sizes = TS.arm_table("T15")
    sc = np.zeros(15)
    sc[0] = 1.0
    frozen = HP.copy()
    frozen[0] = 0.0
    u_frozen = W.tsp_qo_prior_core(D, NL, tour0, kinds, sizes, 600, 31, frozen, 0.02 * sc)[3]
    u_learn = W.tsp_qo_prior_core(D, NL, tour0, kinds, sizes, 600, 31, HP, 0.02 * sc)[3]
    assert u_frozen[0] > 540 and u_learn[0] < u_frozen[0]


def test_prior_scores_range_and_ties():
    s = W.prior_scores([1.0, 0.9, 0.95])
    assert abs(s.max() - 1.0) < 1e-12 and abs(s.min()) < 1e-12 and int(np.argmax(s)) == 1
    assert W.prior_scores([0.5, 0.5, 0.5]).tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------- o corredor (Entradas 274 e 275): prioris e sementes
import glob
import json
from pathlib import Path

import pandas as pd

from experiment import run_cycle40_tsp_entrada267 as E
from experiment import run_cycle45_warmtsp_entrada274 as S
from experiment import run_cycle46_warmtsplib_entrada275 as P

R = Path(__file__).resolve().parent.parent / "results"


def _tune():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "tsp_tune_s*of*_entrada267.csv")))], ignore_index=True)
    d["kind"] = d.cls.str.split("|").str[0]
    d["n"] = d.cls.str.split("|").str[1].astype(int)
    return d


def test_priors_of_entry_274_have_the_right_shape_and_range():
    pri = json.loads((R / "tspwarm_prior_entrada274.json").read_text())
    for sp, na in (("T15", 15), ("T5", 5)):
        for kind in ("full", "loso"):
            assert len(pri[sp][kind]) == 8
            for v in pri[sp][kind].values():
                assert len(v) == na and abs(max(v) - 1.0) < 1e-12 and abs(min(v)) < 1e-12


def test_loso_prior_excludes_the_size_and_rev_is_the_reverse():
    pri = json.loads((R / "tspwarm_prior_entrada274.json").read_text())
    d = _tune()
    x = d[(d.kind == "unif") & (d.n != 500)]
    assert np.allclose(pri["T15"]["loso"]["unif|500"], S.score_of(x))
    x5 = d[(d.arm < 5) & (d.kind == "clust") & (d.n != 2000)]
    assert np.allclose(pri["T5"]["loso"]["clust|2000"], S.score_of(x5))
    full = np.asarray(pri["T15"]["full"]["clust|500"])
    rev = S.vector(pri, "T15", "clust|500", "rev")
    assert int(np.argmax(rev)) == int(np.argmin(full)) and np.allclose(rev + full, 1.0)


def test_tsplib_priors_use_only_synthetic_tuning_data():
    pri = json.loads((R / "tsplibwarm_prior_entrada275.json").read_text())
    d = _tune()
    assert sorted(pri["T15"]["synn"]) == ["1000", "200", "2000", "500"]
    assert np.allclose(pri["T15"]["syn0"], S.score_of(d))
    assert np.allclose(pri["T5"]["synn"]["1000"], S.score_of(d[(d.arm < 5) & (d.n == 1000)]))


def test_seeds_of_the_warm_variants_are_distinct_and_in_range():
    seeds = {(sp, inst, k, s): (E.seed_of(sp, inst, "qo", s) + 74_000_000 + k * 1_000_000) % (2 ** 31 - 1) for sp in ("T5", "T15") for inst in (0, 1, 399) for k in range(6) for s in range(3)}
    assert len(set(seeds.values())) == len(seeds) and max(seeds.values()) < 2 ** 31
    old = {E.seed_of(sp, inst, m, s) for sp in ("T5", "T15") for inst in (0, 1, 399) for m in E.METHODS for s in range(3)}
    assert not (old & set(seeds.values()))
    assert len(S.VARIANTS) == 6 and len(P.VARIANTS) == 4
