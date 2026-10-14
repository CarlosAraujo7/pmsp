"""Quick analysis script — run from PMSP_Learnheuristic/ directory."""
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv('results/results.csv')

print("=== GAP mean ± std ===")
g = df.groupby(['method','n','m'])['gap_pct'].agg(['mean','std']).round(4)
print(g.unstack(['n','m']).to_string())

print("\n=== CPU mean (s) ===")
c = df.groupby(['method','n','m'])['cpu_time'].mean().round(3)
print(c.unstack(['n','m']).to_string())

print("\n=== %Opt (gap_pct == 0) ===")
p = df.groupby(['method','n','m'])['gap_pct'].apply(lambda s: 100*(s==0).mean()).round(1)
print(p.unstack(['n','m']).to_string())

print("\n=== Wilcoxon: ILS-QL vs others (p-value, two-sided) ===")
configs = df[['n','m']].drop_duplicates().values.tolist()
competitors = ['ils_random','ils_rand_hi','ils_ql_ws','ils_ql_hi','ils_rf','ils_ucb']
rows = []
for (n, m) in configs:
    sub = df[(df.n == n) & (df.m == m)]
    ql = sub[sub.method == 'ils_ql']['gap_pct'].values
    for comp in competitors:
        other = sub[sub.method == comp]['gap_pct'].values
        if len(ql) > 0 and len(other) > 0:
            try:
                _, pv = wilcoxon(ql, other)
                rows.append({'config': f"n={n},m={m}", 'vs': comp, 'p': round(pv, 4)})
            except Exception:
                pass
wdf = pd.DataFrame(rows)
if not wdf.empty:
    print(wdf.pivot(index='vs', columns='config', values='p').to_string())

print("\n=== Overall mean GAP per method ===")
print(df.groupby('method')['gap_pct'].agg(['mean','std','min','max']).round(4).to_string())

print("\n=== n_improvements mean (proxy for convergence) ===")
ni = df.groupby(['method','n','m'])['n_improvements'].mean().round(1)
print(ni.unstack(['n','m']).to_string())
