"""
Entrada 29: fecha o desenho fatorial 2x2 (estado x TD-learning) das
Entradas 27-28 com o ig_ql_flat (TD-learning sem estado, a celula que
faltava). As 4 celulas:
    SEM estado | ig_ucb      (sem TD) | ig_ql_flat (COM TD)
    COM estado | ig_ucb_ctx  (sem TD) | ig_ql      (COM TD)
mais ig_random como piso de referencia (sem TD, sem estado, sem NENHUMA
forma de adaptacao, nem UCB).
Roda a partir da raiz do repositorio: python3 analysis/report_entrada29_factorial_2x2.py
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
out.append("=== TABELA 2x2 completa, gap_pct medio (N=100 exceto ig_ql*) ===")
out.append("(* ig_ql em N=100 so nas 3 classes marcadas; N=30 nas outras 3)")
out.append(f"{'classe':>16} {'ig_random':>10} {'ig_ucb':>9} {'ig_ucb_ctx':>11} {'ig_ql_flat':>11} {'ig_ql':>9} {'N_ql':>5}")
for c in classes:
    n_ql = 100 if c in classes_ql_n100 else 30
    gr = series('ig_random', c, 100).mean()
    gu = series('ig_ucb', c, 100).mean()
    guc = series('ig_ucb_ctx', c, 100).mean()
    gf = series('ig_ql_flat', c, 100).mean()
    gql = series('ig_ql', c, n_ql).mean()
    out.append(f"{c:>16} {gr:>10.4f} {gu:>9.4f} {guc:>11.4f} {gf:>11.4f} {gql:>9.4f} {n_ql:>5}")

out.append("")
out.append("=== EFEITO DO TD (sem estado): ig_ql_flat vs ig_ucb (N=100, 6 classes) ===")
out.append(f"{'classe':>16} {'gap_flat':>10} {'gap_ucb':>10} {'wins_flat':>9} {'loss_flat':>9} {'ties':>5} {'p-value':>12}")
for c in classes:
    a_s = series('ig_ql_flat', c, 100)
    b_s = series('ig_ucb', c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {bm:>10.4f} {w:>9} {l:>9} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== EFEITO DO ESTADO (com TD): ig_ql vs ig_ql_flat ===")
out.append(f"{'classe':>16} {'gap_ql':>10} {'N':>4} {'gap_flat':>10} {'wins_ql':>8} {'loss_ql':>8} {'ties':>5} {'p-value':>12}")
for c in classes:
    n_ql = 100 if c in classes_ql_n100 else 30
    a_s = series('ig_ql', c, n_ql)
    b_s = series('ig_ql_flat', c, n_ql)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {n_ql:>4} {bm:>10.4f} {w:>8} {l:>8} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== ig_ql_flat vs ig_random (N=100, 6 classes) -- TD sozinho bate a aleatoriedade pura? ===")
out.append(f"{'classe':>16} {'gap_flat':>10} {'gap_rand':>10} {'wins':>5} {'loss':>5} {'ties':>5} {'p-value':>12}")
for c in classes:
    a_s = series('ig_ql_flat', c, 100)
    b_s = series('ig_random', c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {bm:>10.4f} {w:>5} {l:>5} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== ig_ql_flat vs ig_ucb_ctx (TD-sem-estado vs estado-sem-TD -- qual fator isolado pesa mais?) ===")
out.append(f"{'classe':>16} {'gap_flat':>10} {'gap_ctx':>10} {'wins_flat':>9} {'loss_flat':>9} {'ties':>5} {'p-value':>12}")
for c in classes:
    a_s = series('ig_ql_flat', c, 100)
    b_s = series('ig_ucb_ctx', c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {am:>10.4f} {bm:>10.4f} {w:>9} {l:>9} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== 'ANOVA' informal -- media dos 4 cantos, agregada nas 6 classes (so descritivo, nao teste formal) ===")
means = {}
for m in ['ig_ucb', 'ig_ucb_ctx', 'ig_ql_flat']:
    vals = [series(m, c, 100).mean() for c in classes]
    means[m] = sum(vals) / len(vals)
ql_vals = [series('ig_ql', c, 100 if c in classes_ql_n100 else 30).mean() for c in classes]
means['ig_ql'] = sum(ql_vals) / len(ql_vals)
rand_vals = [series('ig_random', c, 100).mean() for c in classes]
means['ig_random'] = sum(rand_vals) / len(rand_vals)
for k, v in means.items():
    out.append(f"  media(6 classes) {k:>12}: {v:.4f}")
out.append(f"  efeito medio do TD (sem estado)   = ig_ucb - ig_ql_flat = {means['ig_ucb']-means['ig_ql_flat']:+.4f}")
out.append(f"  efeito medio do estado (com TD)   = ig_ql_flat - ig_ql = {means['ig_ql_flat']-means['ig_ql']:+.4f}")
out.append(f"  efeito medio do estado (sem TD)   = ig_ucb - ig_ucb_ctx = {means['ig_ucb']-means['ig_ucb_ctx']:+.4f}")

out.append("")
out.append("=== sanity check: ig_ql_flat nunca pior que LPT (N=30, 6 classes) ===")
viol_total = 0
for c in classes:
    lpt_c = df[(df.instance == c) & (df.method == 'lpt')].set_index('rep')['makespan']
    x_c = df[(df.instance == c) & (df.method == 'ig_ql_flat')].copy()
    x_c['rep'] = x_c['rep'].astype(int)
    x_c = x_c[x_c['rep'] < 30].set_index('rep')['makespan']
    common = lpt_c.index.intersection(x_c.index)
    viol = (x_c.loc[common] > lpt_c.loc[common]).sum()
    viol_total += viol
    out.append(f"  {c}: violacoes={viol} (n pares={len(common)})")
out.append(f"TOTAL violacoes: {viol_total} (esperado: 0)")

text = "\n".join(out)
with open('results/entrada29_factorial_report.txt', 'w') as f:
    f.write(text)
print(text)
