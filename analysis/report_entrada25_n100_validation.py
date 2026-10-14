"""
Validacao em N=100 de ig_ql vs. os dois concorrentes mais fortes
(ils_rand_hi, ils_ql_hi), nas 3 classes escolhidas para a extensao
(rand_30j_8m, rand_50j_10m, rand_100j_20m) -- Entrada 25 do diario.
Roda a partir da raiz do repositorio: python3 analysis/report_entrada25_n100_validation.py
"""
import pandas as pd
from scipy import stats

df = pd.read_csv('results/results.csv')
classes = ['rand_30j_8m', 'rand_50j_10m', 'rand_100j_20m']
refs = ['ils_rand_hi', 'ils_ql_hi']

out = []
out.append("=== Comparacao N=30 (reps 0-29) vs N=100 (reps 0-99): ig_ql vs concorrentes fortes ===")
out.append(f"{'classe':>16} {'vs':>12} {'N':>5} {'gap_igql':>10} {'gap_ref':>10} {'wins':>5} {'loss':>5} {'ties':>5} {'p-value':>12}")

results_summary = []
for c in classes:
    for ref in refs:
        for N, label in [(30, 'N=30'), (100, 'N=100')]:
            igql = df[(df.instance == c) & (df.method == 'ig_ql')].copy()
            igql['rep'] = igql['rep'].astype(int)
            igql = igql[igql['rep'] < N].set_index('rep')['gap_pct'].sort_index()

            refser = df[(df.instance == c) & (df.method == ref)].copy()
            refser['rep'] = refser['rep'].astype(int)
            refser = refser[refser['rep'] < N].set_index('rep')['gap_pct'].sort_index()

            common = igql.index.intersection(refser.index)
            assert len(common) == N, f"esperado {N} pares, achei {len(common)} em {c}/{ref}/{label}"
            a = igql.loc[common].values
            b = refser.loc[common].values
            wins = int((a < b).sum())
            loss = int((a > b).sum())
            ties = int((a == b).sum())
            stat, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
            out.append(f"{c:>16} {ref:>12} {label:>5} {a.mean():>10.4f} {b.mean():>10.4f} {wins:>5} {loss:>5} {ties:>5} {p:>12.6f}")
            results_summary.append((c, ref, label, a.mean(), b.mean(), wins, loss, ties, p))
    out.append("")

out.append("=== Resumo: p-valor N=30 (subconjunto) vs N=100 (completo), por classe x concorrente ===")
for c in classes:
    for ref in refs:
        p30 = [r[8] for r in results_summary if r[0] == c and r[1] == ref and r[2] == 'N=30'][0]
        p100 = [r[8] for r in results_summary if r[0] == c and r[1] == ref and r[2] == 'N=100'][0]
        sig30 = "sig" if p30 < 0.05 else "ns"
        sig100 = "sig" if p100 < 0.05 else "ns"
        out.append(f"  {c:>16} vs {ref:>12}: N=30 p={p30:.6f} ({sig30})  ->  N=100 p={p100:.6f} ({sig100})")

text = "\n".join(out)
with open('results/entrada25_n100_validation_report.txt', 'w') as f:
    f.write(text)
print(text)
