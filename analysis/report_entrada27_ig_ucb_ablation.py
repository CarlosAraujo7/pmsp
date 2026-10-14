"""
Entrada 27: segunda ablacao do ig_ql -- ig_ucb (UCB1, bandit simples sem
Q-table/TD-learning) entre as 3 estrategias de destruicao. Testa se a
pequena vantagem do ig_ql sobre o ig_random (Entrada 26) precisa de
aprendizado sequencial de verdade, ou se qualquer selecao adaptativa
simples (mesmo sem TD-learning) ja captura a maior parte dela.
Roda a partir da raiz do repositorio: python3 analysis/report_entrada27_ig_ucb_ablation.py
"""
import pandas as pd
from scipy import stats

df = pd.read_csv('results/results.csv')
classes = ['rand_20j_4m', 'rand_20j_6m', 'rand_30j_8m', 'rand_50j_10m', 'rand_75j_15m', 'rand_100j_20m']
classes_ql_n100 = ['rand_30j_8m', 'rand_50j_10m', 'rand_100j_20m']

def series(method, c, N):
    s = df[(df.instance == c) & (df.method == method)].copy()
    s['rep'] = s['rep'].astype(int)
    return s[s['rep'] < N].set_index('rep')['gap_pct'].sort_index()

def wilcoxon_pair(a_s, b_s):
    common = a_s.index.intersection(b_s.index)
    a = a_s.loc[common].values
    b = b_s.loc[common].values
    wins = int((a < b).sum())
    loss = int((a > b).sum())
    ties = int((a == b).sum())
    if wins == 0 and loss == 0:
        p = float('nan')
    else:
        _, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    return a.mean(), b.mean(), wins, loss, ties, p, len(common)

out = []
out.append("=== gap_pct medio (N=100 em todas): ig_ql*, ig_random, ig_ucb ===")
out.append("(* ig_ql em N=100 so nas 3 classes marcadas; N=30 nas outras 3)")
out.append(f"{'classe':>16} {'ig_ql':>10} {'N_ql':>5} {'ig_random':>10} {'ig_ucb':>10}")
for c in classes:
    n_ql = 100 if c in classes_ql_n100 else 30
    gql = series('ig_ql', c, n_ql).mean()
    gr = series('ig_random', c, 100).mean()
    gu = series('ig_ucb', c, 100).mean()
    out.append(f"{c:>16} {gql:>10.4f} {n_ql:>5} {gr:>10.4f} {gu:>10.4f}")

out.append("")
out.append("=== WILCOXON: ig_ucb vs ig_random (N=100, todas as 6 classes) ===")
out.append(f"{'classe':>16} {'gap_ucb':>10} {'gap_rand':>10} {'wins':>5} {'loss':>5} {'ties':>5} {'p-value':>12}")
for c in classes:
    a_s = series('ig_ucb', c, 100)
    b_s = series('ig_random', c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {bm:>10.4f} {w:>5} {l:>5} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== WILCOXON: ig_ucb vs ig_ql (N=100 nas 3 classes estendidas) ===")
out.append(f"{'classe':>16} {'gap_ucb':>10} {'gap_ql':>10} {'wins_ucb':>8} {'loss_ucb':>8} {'ties':>5} {'p-value':>12}")
for c in classes_ql_n100:
    a_s = series('ig_ucb', c, 100)
    b_s = series('ig_ql', c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {bm:>10.4f} {w:>8} {l:>8} {t:>5} {p:>12.6f} (N={n})")

out.append("")
out.append("=== WILCOXON: ig_ucb vs ig_ql (N=30, as outras 3 classes) ===")
out.append(f"{'classe':>16} {'gap_ucb':>10} {'gap_ql':>10} {'wins_ucb':>8} {'loss_ucb':>8} {'ties':>5} {'p-value':>12}")
for c in classes:
    if c in classes_ql_n100:
        continue
    a_s = series('ig_ucb', c, 30)
    b_s = series('ig_ql', c, 30)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {bm:>10.4f} {w:>8} {l:>8} {t:>5} {p:>12.6f} (N={n})")

out.append("")
out.append("=== sanity check: ig_ucb nunca pior que LPT (N=30, 6 classes) ===")
viol_total = 0
for c in classes:
    lpt_c = df[(df.instance == c) & (df.method == 'lpt')].set_index('rep')['makespan']
    igu_c = df[(df.instance == c) & (df.method == 'ig_ucb')].copy()
    igu_c['rep'] = igu_c['rep'].astype(int)
    igu_c = igu_c[igu_c['rep'] < 30].set_index('rep')['makespan']
    common = lpt_c.index.intersection(igu_c.index)
    viol = (igu_c.loc[common] > lpt_c.loc[common]).sum()
    viol_total += viol
    out.append(f"  {c}: violacoes={viol} (n pares={len(common)})")
out.append(f"TOTAL violacoes: {viol_total} (esperado: 0)")

text = "\n".join(out)
with open('results/entrada27_ig_ucb_report.txt', 'w') as f:
    f.write(text)
print(text)
