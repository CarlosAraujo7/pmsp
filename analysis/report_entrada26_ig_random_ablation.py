"""
Entrada 26: isolar se o ganho do ig_ql vem do mecanismo de busca (Iterated
Greedy: destruir/reconstruir) em si, ou da selecao via Q-Learning entre as
3 estrategias de destruicao -- comparando ig_ql vs. ig_random (mesmo
framework, selecao uniforme aleatoria em vez de Q-Learning) e contra os
baselines ILS originais, nas 6 classes validadas.
Roda a partir da raiz do repositorio: python3 analysis/report_entrada26_ig_random_ablation.py
"""
import pandas as pd
from scipy import stats

df = pd.read_csv('results/results.csv')
classes = ['rand_20j_4m', 'rand_20j_6m', 'rand_30j_8m', 'rand_50j_10m', 'rand_75j_15m', 'rand_100j_20m']

out = []

out.append("=== gap_pct medio: ig_ql vs ig_random vs ils_random, N=30 (reps 0-29), 6 classes ===")
out.append(f"{'classe':>16} {'ig_ql':>10} {'ig_random':>10} {'ils_random':>10}")
for c in classes:
    def g(m, N=30):
        s = df[(df.instance == c) & (df.method == m)].copy()
        s['rep'] = s['rep'].astype(int)
        return s[s['rep'] < N]['gap_pct']
    out.append(f"{c:>16} {g('ig_ql').mean():>10.4f} {g('ig_random').mean():>10.4f} {g('ils_random').mean():>10.4f}")

out.append("")
out.append("=== WILCOXON pareado (rep), N=30: ig_ql vs ig_random ===")
out.append(f"{'classe':>16} {'gap_igql':>10} {'gap_igrand':>10} {'wins_ql':>8} {'loss_ql':>8} {'ties':>5} {'p-value':>12}")
for c in classes:
    a_s = df[(df.instance == c) & (df.method == 'ig_ql')].copy()
    a_s['rep'] = a_s['rep'].astype(int)
    a_s = a_s[a_s['rep'] < 30].set_index('rep')['gap_pct'].sort_index()
    b_s = df[(df.instance == c) & (df.method == 'ig_random')].copy()
    b_s['rep'] = b_s['rep'].astype(int)
    b_s = b_s[b_s['rep'] < 30].set_index('rep')['gap_pct'].sort_index()
    common = a_s.index.intersection(b_s.index)
    a = a_s.loc[common].values
    b = b_s.loc[common].values
    wins = int((a < b).sum())
    loss = int((a > b).sum())
    ties = int((a == b).sum())
    if wins == 0 and loss == 0:
        p = float('nan')
    else:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    out.append(f"{c:>16} {a.mean():>10.4f} {b.mean():>10.4f} {wins:>8} {loss:>8} {ties:>5} {p:>12.6f}")

out.append("")
out.append("=== WILCOXON pareado (rep), N=100 (onde disponivel): ig_ql vs ig_random ===")
out.append(f"{'classe':>16} {'gap_igql':>10} {'gap_igrand':>10} {'wins_ql':>8} {'loss_ql':>8} {'ties':>5} {'p-value':>12}")
classes_n100 = ['rand_30j_8m', 'rand_50j_10m', 'rand_100j_20m']
for c in classes_n100:
    a_s = df[(df.instance == c) & (df.method == 'ig_ql')].copy()
    a_s['rep'] = a_s['rep'].astype(int)
    a_s = a_s[a_s['rep'] < 100].set_index('rep')['gap_pct'].sort_index()
    b_s = df[(df.instance == c) & (df.method == 'ig_random')].copy()
    b_s['rep'] = b_s['rep'].astype(int)
    b_s = b_s[b_s['rep'] < 100].set_index('rep')['gap_pct'].sort_index()
    common = a_s.index.intersection(b_s.index)
    assert len(common) == 100, f"esperava 100 pares em {c}, achei {len(common)}"
    a = a_s.loc[common].values
    b = b_s.loc[common].values
    wins = int((a < b).sum())
    loss = int((a > b).sum())
    ties = int((a == b).sum())
    stat, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    out.append(f"{c:>16} {a.mean():>10.4f} {b.mean():>10.4f} {wins:>8} {loss:>8} {ties:>5} {p:>12.6f}")

out.append("")
out.append("=== ig_random vs ils_random (N=30, 6 classes): o IG ainda vence o ILS sem nenhum aprendizado? ===")
out.append(f"{'classe':>16} {'gap_igrand':>10} {'gap_ilsrand':>10} {'wins':>5} {'loss':>5} {'ties':>5} {'p-value':>12}")
for c in classes:
    a_s = df[(df.instance == c) & (df.method == 'ig_random')].copy()
    a_s['rep'] = a_s['rep'].astype(int)
    a_s = a_s[a_s['rep'] < 30].set_index('rep')['gap_pct'].sort_index()
    b_s = df[(df.instance == c) & (df.method == 'ils_random')].copy()
    b_s['rep'] = b_s['rep'].astype(int)
    b_s = b_s[b_s['rep'] < 30].set_index('rep')['gap_pct'].sort_index()
    common = a_s.index.intersection(b_s.index)
    a = a_s.loc[common].values
    b = b_s.loc[common].values
    wins = int((a < b).sum())
    loss = int((a > b).sum())
    ties = int((a == b).sum())
    if wins == 0 and loss == 0:
        p = float('nan')
    else:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    out.append(f"{c:>16} {a.mean():>10.4f} {b.mean():>10.4f} {wins:>5} {loss:>5} {ties:>5} {p:>12.6f}")

out.append("")
out.append("=== sanity check: ig_random nunca pior que LPT (N=30, 6 classes) ===")
viol_total = 0
for c in classes:
    lpt_c = df[(df.instance == c) & (df.method == 'lpt')].set_index('rep')['makespan']
    igr_c = df[(df.instance == c) & (df.method == 'ig_random')].copy()
    igr_c['rep'] = igr_c['rep'].astype(int)
    igr_c = igr_c[igr_c['rep'] < 30].set_index('rep')['makespan']
    common = lpt_c.index.intersection(igr_c.index)
    viol = (igr_c.loc[common] > lpt_c.loc[common]).sum()
    viol_total += viol
    out.append(f"  {c}: violacoes={viol} (n pares={len(common)})")
out.append(f"TOTAL violacoes: {viol_total} (esperado: 0)")

text = "\n".join(out)
with open('results/entrada26_ig_random_report.txt', 'w') as f:
    f.write(text)
print(text)
