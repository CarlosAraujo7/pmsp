"""
analysis/audit_entrada98_feature_dynamics.py
==============================================
Entrada 98 (desenho do item 1 da prioridade 0). Mede como f1 (gap relativo)
e f2 (CV das cargas) se comportam DENTRO das execucoes de ig_random e
ils_random: quantis, variabilidade intra-corrida vs inter-corrida, numero de
valores distintos por corrida. Motivo: um agente que aprende do zero em cada
corrida so' pode usar features que variam ao longo da corrida.
Instancias de CALIBRACAO (seeds 10000+), distintas das de avaliacao (0..99).
Monkeypatch de compute_features em pmsp.ig / pmsp.ils; nada em pmsp/ e'
alterado. Uso: python -m analysis.audit_entrada98_feature_dynamics
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pmsp.ig as ig_mod  # noqa: E402
import pmsp.ils as ils_mod  # noqa: E402
from experiment import config as CFG  # noqa: E402
from experiment.runner import make_ils_selector  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402

LOG: list = []


def make_logger(orig):
    def f(solution, t, max_iter, iters_no_improve, stagnation_window=50):
        v = orig(solution, t, max_iter, iters_no_improve, stagnation_window)
        LOG.append((float(v[0]), float(v[1])))
        return v
    return f


ig_orig, ils_orig = ig_mod.compute_features, ils_mod.compute_features
rows = []
for n, m in [(20, 6), (50, 10), (100, 20), (300, 20)]:
    for fw, run in [("IG", "ig"), ("ILS", "ils")]:
        if fw == "ILS" and n == 300:
            continue                      # ILS em 300x20 e' caro; nao necessario para o desenho
        per_run = []
        for k in range(12):
            inst = PMSPInstance.random(n=n, m=m, seed=10000 + k, name=f"cal_{n}_{m}")
            seed = CFG.BASE_SEED + 10000 + k
            LOG.clear()
            if fw == "IG":
                ig_mod.compute_features = make_logger(ig_orig)
                ig_mod.ig_run(instance=inst, selector=make_ils_selector("ig_random", seed), max_iter=1000, seed=seed,
                              stagnation_window=CFG.STAGNATION_WINDOW)
                ig_mod.compute_features = ig_orig
            else:
                ils_mod.compute_features = make_logger(ils_orig)
                ils_mod.ils_run(instance=inst, selector=make_ils_selector("ils_random", seed), max_iter=1000,
                                perturbation_threshold=CFG.PERTURBATION_THRESHOLD, n_perturb=CFG.N_PERTURB, seed=seed,
                                stagnation_window=CFG.STAGNATION_WINDOW)
                ils_mod.compute_features = ils_orig
            a = np.array(LOG)
            per_run.append(a)
        f1 = [r[:, 0] for r in per_run]; f2 = [r[:, 1] for r in per_run]
        allf1, allf2 = np.concatenate(f1), np.concatenate(f2)
        rows.append(dict(
            cls=f"{n}x{m}", fw=fw,
            f1_q=np.round(np.quantile(allf1, [.05, .5, .95]), 4).tolist(),
            f2_q=np.round(np.quantile(allf2, [.05, .5, .95]), 4).tolist(),
            f1_within_sd=round(float(np.mean([x.std() for x in f1])), 5), f1_between_sd=round(float(np.std([x.mean() for x in f1])), 5),
            f2_within_sd=round(float(np.mean([x.std() for x in f2])), 5), f2_between_sd=round(float(np.std([x.mean() for x in f2])), 5),
            f1_distinct_per_run=round(float(np.mean([len(set(np.round(x, 9))) for x in f1])), 1),
            f2_distinct_per_run=round(float(np.mean([len(set(np.round(x, 9))) for x in f2])), 1)))
        print(rows[-1], flush=True)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
