"""
experiment/harvest_large_class.py
====================================
Igual a harvest_feature_distribution.py (Entrada 17), mas parametrizado
para UMA classe qualquer via CLI -- usado para testar o redesenho de
granularidade nas classes maiores do plano aprovado (150-300 tarefas),
a pedido de Carlos apos o resultado nulo em rand_20j_4m/rand_50j_10m
com N=100 (Entrada 18).

Harvest especifico da classe (nao agrupado com as classes pequenas) --
escolha deliberada para isolar se a dinamica de busca em escala maior
e suficientemente diferente para mudar a conclusao. Documentado na
Entrada 19.

Uso: python3 experiment/harvest_large_class.py N M LABEL
Salva em results/harvested_feature_quantile_thresholds_{LABEL}.json
"""
from __future__ import annotations
import sys
import json
import numpy as np

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.selectors import BaseSelector, RandomSelector
from pmsp.bounds import _BINS_F1, _BINS_F2, _BINS_F3, _BINS_F4
from pmsp.ils import ils_run
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW, N_INSTANCES,
)

HARVEST_SEED_OFFSET = 400_000  # faixa propria para classes grandes
N_HARVEST_EPISODES = N_INSTANCES
N_BINS_GRID = [2, 3, 4]


class FeatureLoggerWrapper(BaseSelector):
    def __init__(self, inner):
        self._inner = inner
        self.log = []

    def select(self, state_id, features):
        self.log.append(np.asarray(features, dtype=np.float64).copy())
        return self._inner.select(state_id, features)

    def update(self, state_id, features, action, reward, next_state_id):
        self._inner.update(state_id, features, action, reward, next_state_id)

    def reset(self):
        self._inner.reset()

    def name(self):
        return f"FeatureLogger({self._inner.name()})"


def quantile_thresholds(values, n_bins):
    qs = [k / n_bins for k in range(1, n_bins)]
    return [float(np.quantile(values, q)) for q in qs]


def main():
    n, m, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    print(f"Coletando (f1,f2,f3,f4) para {label} (n={n}, m={m}), "
          f"{N_HARVEST_EPISODES} episodios via ILS-Random...")
    all_feats = []
    for ep in range(N_HARVEST_EPISODES):
        seed = HARVEST_SEED_OFFSET + ep
        inst = PMSPInstance.random(n, m, seed=seed)
        logger = FeatureLoggerWrapper(RandomSelector(seed=seed))
        ils_run(inst, logger, max_iter=MAX_ITER,
                perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
                seed=seed, stagnation_window=STAGNATION_WINDOW)
        all_feats.extend(logger.log)
    feats = np.array(all_feats)
    n_samples = feats.shape[0]
    print(f"Total de amostras: {n_samples}")

    names = ["f1_gap", "f2_load_cv", "f3_progress", "f4_stagnation"]
    stats = {}
    for i, name in enumerate(names):
        col = feats[:, i]
        stats[name] = {"min": float(col.min()), "max": float(col.max()),
                        "mean": float(col.mean()), "median": float(np.median(col))}
        print(f"  {name}: min={stats[name]['min']:.4f} mediana={stats[name]['median']:.4f} "
              f"media={stats[name]['mean']:.4f} max={stats[name]['max']:.4f}")

    thresholds_by_nbins = {}
    for n_bins in N_BINS_GRID:
        th = {
            "bins_f1": quantile_thresholds(feats[:, 0], n_bins),
            "bins_f2": quantile_thresholds(feats[:, 1], n_bins),
            "bins_f3": quantile_thresholds(feats[:, 2], n_bins),
            "bins_f4": quantile_thresholds(feats[:, 3], n_bins),
        }
        thresholds_by_nbins[str(n_bins)] = th
        print(f"n_bins={n_bins}: {th}")

    print(f"\nComparacao com limiares manuais (n_bins=3): "
          f"_BINS_F1={_BINS_F1} vs {[round(x,4) for x in thresholds_by_nbins['3']['bins_f1']]} | "
          f"_BINS_F2={_BINS_F2} vs {[round(x,4) for x in thresholds_by_nbins['3']['bins_f2']]}")

    out = {"n_samples": n_samples, "n": n, "m": m, "label": label,
           "harvest_seed_offset": HARVEST_SEED_OFFSET,
           "feature_stats": stats, "thresholds_by_nbins": thresholds_by_nbins}
    out_path = f"results/harvested_feature_quantile_thresholds_{label}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
