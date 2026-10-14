"""
experiment/harvest_feature_distribution.py
============================================
Coleta a distribuicao empirica das 4 features (f1..f4) ao longo de
trajetorias reais do ILS, para calibrar limiares de discretizacao por
QUANTIS EMPIRICOS em diferentes granularidades (N=2/16 estados,
N=3/81, N=4/256) -- Entrada 16, diario_pesquisa.md.

Por que quantis, e nao os limiares manuais _BINS_F1..F4 originais:
usar o MESMO criterio (quantis empiricos) nas 3 granularidades isola
"granularidade" como a unica variavel manipulada -- os limiares
manuais de 3 bins nao sao equidistantes nem foram derivados por uma
regra generalizavel para 2/4 bins sem escolhas ad hoc adicionais.

Por que RandomSelector para coletar: f3 (progresso) e f4 (estagnacao)
sao determinadas pela dinamica do LOOP do ILS (iter/max_iter,
contador de estagnacao), praticamente independentes do seletor. f1/f2
dependem da trajetoria de busca -- RandomSelector da uma amostra
exploratoria, nao enviesada por uma politica ja aprendida (evita
circularidade: nao calibramos os limiares com base em COMO o proprio
Q-Learning eventualmente vai se comportar).

Escopo: apenas as 2 classes ja aprovadas para o piloto (rand_20j_4m,
rand_50j_10m) -- nao as 6 classes completas do plano. Nao escreve em
results.csv. Salva em results/harvested_feature_quantile_thresholds.json.
"""
from __future__ import annotations
import sys
import json
import numpy as np

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.selectors import BaseSelector, RandomSelector
from pmsp.bounds import compute_features, _BINS_F1, _BINS_F2, _BINS_F3, _BINS_F4
from pmsp.ils import ils_run
from experiment.config import (
    MAX_ITER, PERTURBATION_THRESHOLD, N_PERTURB, STAGNATION_WINDOW, N_INSTANCES,
)

HARVEST_SEED_OFFSET = 300_000  # faixa propria, sem sobreposicao com treino (100000+) ou teste (0..29)
N_HARVEST_EPISODES = N_INSTANCES  # 30, mesmo N das demais rodadas
CLASSES = [(20, 4, "rand_20j_4m"), (50, 10, "rand_50j_10m")]
N_BINS_GRID = [2, 3, 4]


class FeatureLoggerWrapper(BaseSelector):
    """Encapsula um selector interno e registra as features vistas em
    cada select(), sem alterar o comportamento do selector encapsulado."""

    def __init__(self, inner: BaseSelector):
        self._inner = inner
        self.log: list[np.ndarray] = []

    def select(self, state_id, features):
        self.log.append(np.asarray(features, dtype=np.float64).copy())
        return self._inner.select(state_id, features)

    def update(self, state_id, features, action, reward, next_state_id):
        self._inner.update(state_id, features, action, reward, next_state_id)

    def reset(self):
        self._inner.reset()

    def name(self):
        return f"FeatureLogger({self._inner.name()})"


def harvest() -> np.ndarray:
    all_feats = []
    for n, m, label in CLASSES:
        for ep in range(N_HARVEST_EPISODES):
            seed = HARVEST_SEED_OFFSET + ep
            inst = PMSPInstance.random(n, m, seed=seed)
            logger = FeatureLoggerWrapper(RandomSelector(seed=seed))
            ils_run(
                inst, logger, max_iter=MAX_ITER,
                perturbation_threshold=PERTURBATION_THRESHOLD, n_perturb=N_PERTURB,
                seed=seed, stagnation_window=STAGNATION_WINDOW,
            )
            all_feats.extend(logger.log)
        print(f"  {label}: {N_HARVEST_EPISODES} episodios coletados.")
    return np.array(all_feats)  # shape (N_amostras, 4)


def quantile_thresholds(values: np.ndarray, n_bins: int) -> list[float]:
    qs = [k / n_bins for k in range(1, n_bins)]
    return [float(np.quantile(values, q)) for q in qs]


def main():
    print(f"Coletando amostras de (f1,f2,f3,f4) via RandomSelector "
          f"({len(CLASSES)} classes x {N_HARVEST_EPISODES} episodios x ate {MAX_ITER} passos)...")
    feats = harvest()
    n_samples = feats.shape[0]
    print(f"\nTotal de amostras coletadas: {n_samples}")

    names = ["f1_gap", "f2_load_cv", "f3_progress", "f4_stagnation"]
    stats = {}
    for i, name in enumerate(names):
        col = feats[:, i]
        stats[name] = {
            "min": float(col.min()), "max": float(col.max()),
            "mean": float(col.mean()), "median": float(np.median(col)),
        }
        print(f"  {name}: min={stats[name]['min']:.4f} "
              f"mediana={stats[name]['median']:.4f} "
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
        print(f"\nLimiares de quantis para n_bins={n_bins} (n_states={n_bins**4}):")
        for k, v in th.items():
            print(f"  {k}: {[round(x, 4) for x in v]}")

    print(f"\nComparacao com limiares manuais originais (n_bins=3, |S|=81):")
    print(f"  _BINS_F1 manual={_BINS_F1}  vs. quantil={[round(x,4) for x in thresholds_by_nbins['3']['bins_f1']]}")
    print(f"  _BINS_F2 manual={_BINS_F2}  vs. quantil={[round(x,4) for x in thresholds_by_nbins['3']['bins_f2']]}")
    print(f"  _BINS_F3 manual={_BINS_F3}  vs. quantil={[round(x,4) for x in thresholds_by_nbins['3']['bins_f3']]}")
    print(f"  _BINS_F4 manual={_BINS_F4}  vs. quantil={[round(x,4) for x in thresholds_by_nbins['3']['bins_f4']]}")

    out = {
        "n_samples": n_samples,
        "harvest_seed_offset": HARVEST_SEED_OFFSET,
        "n_harvest_episodes_per_class": N_HARVEST_EPISODES,
        "classes": [c[2] for c in CLASSES],
        "feature_stats": stats,
        "thresholds_by_nbins": thresholds_by_nbins,
    }
    with open("results/harvested_feature_quantile_thresholds.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSalvo em results/harvested_feature_quantile_thresholds.json")


if __name__ == "__main__":
    main()
