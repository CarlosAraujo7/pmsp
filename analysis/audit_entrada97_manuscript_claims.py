"""
analysis/audit_entrada97_manuscript_claims.py
================================================
Auditoria numerica (05/10/2026, Entrada 97) de alegacoes do manuscrito
contra results/results.csv e o codigo. Imprime:
  A. cobertura metodo x classe (verifica "23 metodos em 12 classes", N por celula)
  B. familia F1 (ig_ql vs LPT + 7 ILS, 12 classes, Holm sobre 96): n significativas,
     favoraveis, celulas nao significativas por classe; variante "6 competidores"
  C. familia F3 (ablacao, 48) e F4 (ig_random vs ils_random, 12)
  D. ILS (N=30): ils_ql vs ils_random / ils_ucb nas 6 classes originais
  E. MAX_ITER_BY_SIZE (orcamento dos -HI)
  F. colapso do espaco de estados: estados visitados e bins de f1/f2/f3/f4 em
     execucoes reais do ig_ql (instrumentado, nenhum arquivo de pmsp/ alterado)
Uso: python -m analysis.audit_entrada97_manuscript_claims
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import pairwise_wilcoxon_family  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
df = pd.read_csv(ROOT / "results" / "results.csv")
order = df.groupby("instance").n.first().sort_values().index.tolist()

print("=" * 100, "\nA. COBERTURA metodo x classe (N de replicas)\n" + "=" * 100)
cov = df.pivot_table(index="method", columns="instance", values="rep", aggfunc="count").reindex(columns=order)
print(cov.fillna(0).astype(int).to_string())
print("metodos:", df.method.nunique(), "| classes:", df.instance.nunique(),
      "| celulas vazias:", int(cov.isna().sum().sum()))

NR = 199
ils7 = ["ils_random", "ils_ucb", "ils_rf", "ils_ql", "ils_rand_hi", "ils_ql_ws", "ils_ql_hi"]
print("\n" + "=" * 100, "\nB. F1: ig_ql vs LPT + 7 ILS (Holm sobre 96)\n" + "=" * 100)
F1 = pairwise_wilcoxon_family(df, "ig_ql", ["lpt"] + ils7, family_name="F1", n_resamples=NR)
valid = F1.p_raw.notna()
print("testes validos:", int(valid.sum()), "| p_raw<0,05:", int((F1.p_raw < .05).sum()),
      "| Holm-sig:", int(F1.reject_holm.sum()),
      "| Holm-sig favoraveis a ig_ql:", int((F1.reject_holm & (F1.direction == "ref_better")).sum()),
      "| Holm-sig contra ig_ql:", int((F1.reject_holm & (F1.direction == "ref_worse")).sum()))
ns = F1[~F1.reject_holm]
print("celulas NAO significativas por classe:", ns.groupby(["n", "m"]).size().to_dict())
print("celulas nao significativas por competidor:", ns.groupby("other").size().to_dict())
six = ["ils_ql", "ils_ql_hi", "ils_ql_ws", "ils_rand_hi", "ils_rf", "ils_ucb"]
s6 = F1[F1.other.isin(six)]
print(f"6 competidores 'fortes' x 12 classes = {len(s6)} comparacoes; Holm-sig (da familia de 96): {int(s6.reject_holm.sum())}, "
      f"favoraveis: {int((s6.reject_holm & (s6.direction == 'ref_better')).sum())}")

print("\n" + "=" * 100, "\nC. F3 (ablacao, 48) e F4 (ig_random vs ils_random, 12)\n" + "=" * 100)
F3 = pairwise_wilcoxon_family(df, "ig_ql", ["ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat"], family_name="F3", n_resamples=NR)
print("F3: testes", int(F3.p_raw.notna().sum()), "| p_raw<0,05:", int((F3.p_raw < .05).sum()), "| Holm-sig:", int(F3.reject_holm.sum()),
      "| menor p_holm:", float(F3.p_holm.min()), F3.loc[F3.p_holm.idxmin(), ["other", "n", "m"]].tolist())
F4 = pairwise_wilcoxon_family(df, "ig_random", ["ils_random"], family_name="F4", n_resamples=NR)
print("F4: Holm-sig:", int(F4.reject_holm.sum()), "de", int(F4.p_raw.notna().sum()),
      "| favoraveis a ig_random:", int((F4.reject_holm & (F4.direction == "ref_better")).sum()),
      "| nao sig. em:", F4[~F4.reject_holm][["n", "m"]].values.tolist())

print("\n" + "=" * 100, "\nD. ILS (classes originais, N=30): ils_ql vs ils_random / ils_ucb / ils_rf\n" + "=" * 100)
orig = [(20, 4), (20, 6), (30, 8), (50, 10), (75, 15), (100, 20)]
D = pairwise_wilcoxon_family(df, "ils_ql", ["ils_random", "ils_ucb", "ils_rf"], sizes=orig, family_name="D", n_resamples=NR)
print(D[["other", "n", "m", "n_pairs", "mean_ref", "mean_other", "p_raw", "direction"]].round(4).to_string(index=False))
print("ils_ql vs ils_random: p_raw minimo", float(D[D.other == "ils_random"].p_raw.min()),
      "| vs ils_ucb: p_raw maximo", float(D[D.other == "ils_ucb"].p_raw.max()))

print("\n" + "=" * 100, "\nE. MAX_ITER_BY_SIZE\n" + "=" * 100)
from experiment import config as CFG  # noqa: E402
print(CFG.MAX_ITER_BY_SIZE)

print("\n" + "=" * 100, "\nF. COLAPSO DO ESPACO DE ESTADOS (ig_ql instrumentado)\n" + "=" * 100)
from pmsp.instance import PMSPInstance  # noqa: E402
from pmsp.qlearning import QLearningAgent  # noqa: E402
from experiment.runner import run_single  # noqa: E402

visited = []
orig_select = QLearningAgent.select


def logged_select(self, state_id, features=None):
    visited.append(int(state_id))
    return orig_select(self, state_id, features)


QLearningAgent.select = logged_select
rows = []
for n, m in [(20, 6), (50, 10), (100, 20), (300, 20)]:
    per_run_distinct, bins = [], Counter()
    for rep in range(8):
        visited.clear()
        run_single(PMSPInstance.random(n=n, m=m, seed=rep, name=f"rand_{n}j_{m}m"), "ig_ql", rep, max_iter=1000)
        per_run_distinct.append(len(set(visited)))
        for s in visited:
            bins[(s // 27, (s // 9) % 3, (s // 3) % 3, s % 3)] += 1
    tot = sum(bins.values())
    share = lambda k: [round(sum(v for key, v in bins.items() if key[k] == b) / tot, 3) for b in range(3)]
    rows.append(dict(cls=f"{n}x{m}", distinct_states_mean=round(float(np.mean(per_run_distinct)), 1),
                     f1_bins=share(0), f2_bins=share(1), f3_bins=share(2), f4_bins=share(3),
                     states_with_any_visit=len({(k) for k in bins})))
print(pd.DataFrame(rows).to_string(index=False))
print("(f1 = gap vs LB, f2 = CV das cargas, f3 = progresso, f4 = estagnacao; bins 0/1/2; participacao de cada bin nos passos)")
