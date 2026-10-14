"""
Relatorio de analise da bateria completa dos 7 metodos novos (Entrada 24 do diario).
Roda a partir da raiz do repositorio: python3 analysis/report_entrada24_ig_ql_battery.py
Gera results/entrada24_battery_report.txt com: tabela de gap medio por metodo x classe,
contagem de replicas, sanity check (ig_ql nunca pior que LPT), Wilcoxon pareado
(ig_ql vs cada baseline, por classe, mesma convencao de analysis/tables.py), comparacao
de tempo de CPU, e n_improvements medio.
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('results/results.csv')
classes = ['rand_20j_4m', 'rand_20j_6m', 'rand_30j_8m', 'rand_50j_10m', 'rand_75j_15m', 'rand_100j_20m']

baselines = ['lpt', 'ils_random', 'ils_ucb', 'ils_rf', 'ils_ql', 'ils_ql_hi', 'ils_rand_hi', 'ils_ql_ws']
new_methods = ['ils_sarsa', 'ils_pm', 'ils_ap', 'ils_op2_only', 'ils_op3_only', 'ils_op4_only', 'ig_ql']
all_methods = baselines + new_methods

out = []
out.append("=== TABELA gap_pct MEDIO (%) por metodo x classe ===")
pivot = df[df.method.isin(all_methods)].pivot_table(index='method', columns='instance', values='gap_pct', aggfunc='mean')
pivot = pivot[classes].reindex(all_methods)
out.append(pivot.round(4).to_string())

out.append("")
out.append("=== N de replicas por metodo x classe (confirmar 30 em tudo) ===")
cnt = df[df.method.isin(all_methods)].pivot_table(index='method', columns='instance', values='rep', aggfunc='count')
cnt = cnt[classes].reindex(all_methods)
out.append(cnt.to_string())

out.append("")
out.append("=== SANITY CHECK: ig_ql nunca pior que LPT (makespan_ig_ql <= makespan_lpt) ===")
viol_total = 0
for c in classes:
    lpt_c = df[(df.instance == c) & (df.method == 'lpt')].set_index('rep')['makespan']
    igql_c = df[(df.instance == c) & (df.method == 'ig_ql')].set_index('rep')['makespan']
    common = lpt_c.index.intersection(igql_c.index)
    viol = (igql_c.loc[common] > lpt_c.loc[common]).sum()
    viol_total += viol
    out.append(f"  {c}: violacoes={viol} (n pares={len(common)})")
out.append(f"TOTAL violacoes: {viol_total} (esperado: 0)")

out.append("")
out.append("=== WILCOXON pareado (rep): ig_ql vs cada baseline, por classe ===")
out.append(f"{'classe':>16} {'vs':>14} {'gap_igql':>10} {'gap_ref':>10} {'wins':>5} {'loss':>5} {'ties':>5} {'p-value':>12}")
compare_refs = ['ils_random', 'ils_ucb', 'ils_rf', 'ils_ql', 'ils_ql_hi', 'ils_rand_hi', 'ils_ql_ws', 'ils_sarsa', 'ils_pm', 'ils_ap']
for c in classes:
    igql = df[(df.instance == c) & (df.method == 'ig_ql')].set_index('rep')['gap_pct'].sort_index()
    for ref in compare_refs:
        refser = df[(df.instance == c) & (df.method == ref)].set_index('rep')['gap_pct'].sort_index()
        common = igql.index.intersection(refser.index)
        a = igql.loc[common].values
        b = refser.loc[common].values
        wins = int((a < b).sum())
        loss = int((a > b).sum())
        ties = int((a == b).sum())
        if wins == 0 and loss == 0:
            p = float('nan')
        else:
            try:
                stat, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
            except Exception:
                p = float('nan')
        out.append(f"{c:>16} {ref:>14} {a.mean():>10.4f} {b.mean():>10.4f} {wins:>5} {loss:>5} {ties:>5} {p:>12.6f}")

out.append("")
out.append("=== CPU TIME: media (s) por metodo x classe ===")
cpu_methods = ['ils_random', 'ils_ql', 'ils_ql_hi', 'ils_rand_hi', 'ig_ql']
cpu_pivot = df[df.method.isin(cpu_methods)].pivot_table(index='method', columns='instance', values='cpu_time', aggfunc='mean')
cpu_pivot = cpu_pivot[classes].reindex(cpu_methods)
out.append(cpu_pivot.round(4).to_string())

out.append("")
out.append("=== RAZAO cpu_time: ig_ql / ils_ql por classe ===")
for c in classes:
    igql_t = df[(df.instance == c) & (df.method == 'ig_ql')]['cpu_time'].mean()
    ilsql_t = df[(df.instance == c) & (df.method == 'ils_ql')]['cpu_time'].mean()
    out.append(f"  {c}: ig_ql={igql_t:.4f}s  ils_ql={ilsql_t:.4f}s  razao={igql_t / ilsql_t:.4f}")

out.append("")
out.append("=== n_improvements medio: ig_ql vs ils_ql vs ils_random ===")
for c in classes:
    for m in ['ils_random', 'ils_ql', 'ig_ql']:
        v = df[(df.instance == c) & (df.method == m)]['n_improvements'].mean()
        out.append(f"  {c} {m}: n_improvements_mean={v:.2f}")

out.append("")
out.append("=== 7 metodos novos: gap_pct medio, todas as 6 classes agregadas ===")
agg = df[df.method.isin(new_methods + ['ils_random', 'ils_ql'])].groupby('method')['gap_pct'].agg(['mean', 'std', 'count'])
out.append(agg.round(4).to_string())

text = "\n".join(out)
with open('results/entrada24_battery_report.txt', 'w') as f:
    f.write(text)
print(text)
