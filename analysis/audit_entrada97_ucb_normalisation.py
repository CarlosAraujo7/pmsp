"""
analysis/audit_entrada97_ucb_normalisation.py
================================================
PILOTO EXPLORATORIO (Entrada 97) -- sem alegacoes. O UCBSelector do projeto
usa a recompensa BRUTA (reducao de makespan em unidades de p_j), mas o UCB1
(Auer et al. 2002) assume recompensas em [0,1]. Pergunta: a pior performance
do ils_ucb/ig_ucb (e a vantagem "ils_ql > ils_ucb") e' em parte artefato da
escala da recompensa?

Variante UCBNorm: mesmo UCBSelector, recompensa dividida pela maior recompensa
observada ate' o passo (maxima corrente; 0 -> 0), o que a coloca em [0,1] sem
olhar a instancia. Chama ils_run/ig_run diretamente com a mesma configuracao
de experiment.runner.run_single; nenhum arquivo de pmsp/ ou experiment/ e'
alterado. Mesmas instancias/seeds de results/results.csv. N=30 por classe.

Uso: python -m analysis.audit_entrada97_ucb_normalisation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment import config as CFG  # noqa: E402
from pmsp.ig import ig_run  # noqa: E402
from pmsp.ils import ils_run  # noqa: E402
from pmsp.instance import PMSPInstance  # noqa: E402
from pmsp.selectors import UCBSelector  # noqa: E402


class UCBNorm(UCBSelector):
    def reset(self):
        super().reset()
        self._rmax = 0.0

    def update(self, state_id, features, action, reward, next_state_id):
        self._rmax = max(self._rmax, float(reward))
        r = float(reward) / self._rmax if self._rmax > 0 else 0.0
        super().update(state_id, features, action, r, next_state_id)


res = pd.read_csv(ROOT / "results" / "results.csv")
K = 30
rows = []
for n, m in [(20, 6), (50, 10), (100, 20)]:
    name = f"rand_{n}j_{m}m"
    g_ils, g_ig = [], []
    for rep in range(K):
        inst = PMSPInstance.random(n=n, m=m, seed=rep, name=name)
        seed = CFG.BASE_SEED + rep
        r1 = ils_run(instance=inst, selector=UCBNorm(C=CFG.UCB_C, seed=seed), max_iter=CFG.MAX_ITER,
                     perturbation_threshold=CFG.PERTURBATION_THRESHOLD, n_perturb=CFG.N_PERTURB, seed=seed,
                     stagnation_window=CFG.STAGNATION_WINDOW)
        r2 = ig_run(instance=inst, selector=UCBNorm(C=CFG.UCB_C, seed=seed, n_actions=3), max_iter=CFG.MAX_ITER,
                    seed=seed, stagnation_window=CFG.STAGNATION_WINDOW)
        g_ils.append(r1["gap_pct"]); g_ig.append(r2["gap_pct"])
    s = res[(res.instance == name) & (res.rep < K)]
    mean = lambda meth: s[s.method == meth].set_index("rep").gap_pct.loc[range(K)]
    d = dict(cls=name)
    for lab, vals, ref_u, ref_r, ref_q in [("ILS", g_ils, "ils_ucb", "ils_random", "ils_ql"), ("IG", g_ig, "ig_ucb", "ig_random", "ig_ql")]:
        v = pd.Series(vals, index=range(K))
        d[f"{lab}_ucb_raw"] = mean(ref_u).mean()
        d[f"{lab}_ucb_norm"] = v.mean()
        d[f"{lab}_random"] = mean(ref_r).mean()
        d[f"{lab}_ql"] = mean(ref_q).mean()
        diff = (v - mean(ref_u)).values
        d[f"{lab}_p_norm_vs_raw"] = wilcoxon(diff[diff != 0]).pvalue if (diff != 0).sum() > 5 else float("nan")
        diff2 = (v - mean(ref_q)).values
        d[f"{lab}_p_norm_vs_ql"] = wilcoxon(diff2[diff2 != 0]).pvalue if (diff2 != 0).sum() > 5 else float("nan")
    rows.append(d)
    print(d, flush=True)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(4).T.to_string())
