"""
Entrada 30: fecha a lacuna apontada na Entrada 29 -- estende ig_ql,
ils_rand_hi e ils_ql_hi de N=30 para N=100 nas 2 classes que faltavam
(rand_20j_6m, rand_75j_15m; usa o mesmo driver crash-seguro da Entrada 25,
que ja agrupa exatamente esses 3 metodos). Depois de rodar, so
rand_20j_4m (classe trivial, gap ~0 para todos) continua em N=30 para
esses 3 metodos.

Duas analises:
(A) Igual ao script da Entrada 25: ig_ql vs ils_rand_hi/ils_ql_hi,
    comparando N=30 (subconjunto) vs N=100 (completo), agora nas 2
    classes novas -- fecha de vez a ressalva da Atualizacao 12.
(B) Recalcula a tabela fatorial 2x2 completa da Entrada 29, agora com
    ig_ql em N=100 em 5 das 6 classes (todas exceto rand_20j_4m) --
    resolve a ambiguidade "achado que exige honestidade extra".

Roda a partir da raiz do repositorio: python3 analysis/report_entrada30_ig_ql_n100_full.py
"""
import pandas as pd
from scipy import stats

df = pd.read_csv("results/results.csv")

# ---------------------------------------------------------------
# (A) N=30 vs N=100, ig_ql vs concorrentes fortes, nas 2 classes novas
# ---------------------------------------------------------------
classes_new = ["rand_20j_6m", "rand_75j_15m"]
refs = ["ils_rand_hi", "ils_ql_hi"]

out = []
out.append("=== PARTE A: N=30 (subconjunto) vs N=100 (completo) -- ig_ql vs concorrentes fortes ===")
out.append("(mesmo protocolo da Entrada 25, agora nas 2 classes que faltavam)")
out.append(f"{'classe':>16} {'vs':>12} {'N':>5} {'gap_igql':>10} {'gap_ref':>10} {'wins':>5} {'loss':>5} {'ties':>5} {'p-value':>12}")

results_summary = []
for c in classes_new:
    for ref in refs:
        for N, label in [(30, "N=30"), (100, "N=100")]:
            igql = df[(df.instance == c) & (df.method == "ig_ql")].copy()
            igql["rep"] = igql["rep"].astype(int)
            igql = igql[igql["rep"] < N].set_index("rep")["gap_pct"].sort_index()

            refser = df[(df.instance == c) & (df.method == ref)].copy()
            refser["rep"] = refser["rep"].astype(int)
            refser = refser[refser["rep"] < N].set_index("rep")["gap_pct"].sort_index()

            common = igql.index.intersection(refser.index)
            assert len(common) == N, f"esperado {N} pares, achei {len(common)} em {c}/{ref}/{label}"
            a = igql.loc[common].values
            b = refser.loc[common].values
            wins = int((a < b).sum())
            loss = int((a > b).sum())
            ties = int((a == b).sum())
            stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            out.append(f"{c:>16} {ref:>12} {label:>5} {a.mean():>10.4f} {b.mean():>10.4f} {wins:>5} {loss:>5} {ties:>5} {p:>12.6f}")
            results_summary.append((c, ref, label, a.mean(), b.mean(), wins, loss, ties, p))
    out.append("")

out.append("=== Resumo: p-valor N=30 (subconjunto) vs N=100 (completo), por classe x concorrente ===")
for c in classes_new:
    for ref in refs:
        p30 = [r[8] for r in results_summary if r[0] == c and r[1] == ref and r[2] == "N=30"][0]
        p100 = [r[8] for r in results_summary if r[0] == c and r[1] == ref and r[2] == "N=100"][0]
        sig30 = "sig" if p30 < 0.05 else "ns"
        sig100 = "sig" if p100 < 0.05 else "ns"
        out.append(f"  {c:>16} vs {ref:>12}: N=30 p={p30:.6f} ({sig30})  ->  N=100 p={p100:.6f} ({sig100})")

# ---------------------------------------------------------------
# (B) Tabela fatorial 2x2 completa, recalculada com ig_ql em N=100
#     em 5 das 6 classes (repete o desenho da Entrada 29)
# ---------------------------------------------------------------
classes_all = ["rand_20j_4m", "rand_20j_6m", "rand_30j_8m", "rand_50j_10m", "rand_75j_15m", "rand_100j_20m"]
classes_ql_n100 = ["rand_20j_6m", "rand_30j_8m", "rand_50j_10m", "rand_75j_15m", "rand_100j_20m"]

def series(method, c, N):
    s = df[(df.instance == c) & (df.method == method)].copy()
    s["rep"] = s["rep"].astype(int)
    return s[s["rep"] < N].set_index("rep")["gap_pct"].sort_index()

def wilcoxon_pair(a_s, b_s):
    common = a_s.index.intersection(b_s.index)
    a = a_s.loc[common].values
    b = b_s.loc[common].values
    wins = int((a < b).sum())
    loss = int((a > b).sum())
    ties = int((a == b).sum())
    if wins == 0 and loss == 0:
        p = float("nan")
    else:
        _, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return a.mean(), b.mean(), wins, loss, ties, p, len(common)

out.append("")
out.append("=== PARTE B: TABELA 2x2 completa, gap_pct medio (N=100 exceto ig_ql* em rand_20j_4m) ===")
out.append("(* rand_20j_4m e classe trivial -- gap ~0 para quase todos os metodos, nunca foi motivo de preocupacao)")
out.append(f"{'classe':>16} {'ig_random':>10} {'ig_ucb':>9} {'ig_ucb_ctx':>11} {'ig_ql_flat':>11} {'ig_ql':>9} {'N_ql':>5}")
for c in classes_all:
    n_ql = 100 if c in classes_ql_n100 else 30
    gr = series("ig_random", c, 100).mean()
    gu = series("ig_ucb", c, 100).mean()
    guc = series("ig_ucb_ctx", c, 100).mean()
    gf = series("ig_ql_flat", c, 100).mean()
    gql = series("ig_ql", c, n_ql).mean()
    out.append(f"{c:>16} {gr:>10.4f} {gu:>9.4f} {guc:>11.4f} {gf:>11.4f} {gql:>9.4f} {n_ql:>5}")

out.append("")
out.append("=== ig_ql vs ig_ucb_ctx, agora em N=100 nas 2 classes que geravam a ambiguidade ===")
out.append(f"{'classe':>16} {'N':>4} {'gap_ql':>10} {'gap_ctx':>10} {'wins_ql':>8} {'loss_ql':>8} {'ties':>5} {'p-value':>12}")
for c in ["rand_20j_6m", "rand_75j_15m"]:
    n_ql = 100
    a_s = series("ig_ql", c, n_ql)
    b_s = series("ig_ucb_ctx", c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {n_ql:>4} {am:>10.4f} {bm:>10.4f} {w:>8} {l:>8} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== ig_ql vs ig_ql_flat (efeito do estado, com TD) -- TODAS as 6 classes, poder atualizado ===")
out.append(f"{'classe':>16} {'N':>4} {'gap_ql':>10} {'gap_flat':>10} {'wins_ql':>8} {'loss_ql':>8} {'ties':>5} {'p-value':>12}")
for c in classes_all:
    n_ql = 100 if c in classes_ql_n100 else 30
    a_s = series("ig_ql", c, n_ql)
    b_s = series("ig_ql_flat", c, n_ql)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {n_ql:>4} {am:>10.4f} {bm:>10.4f} {w:>8} {l:>8} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== ig_ql vs ig_random -- TODAS as 6 classes, poder atualizado ===")
out.append(f"{'classe':>16} {'N':>4} {'gap_ql':>10} {'gap_rand':>10} {'wins_ql':>8} {'loss_ql':>8} {'ties':>5} {'p-value':>12}")
for c in classes_all:
    n_ql = 100 if c in classes_ql_n100 else 30
    a_s = series("ig_ql", c, n_ql)
    b_s = series("ig_random", c, 100)
    am, bm, w, l, t, p, n = wilcoxon_pair(a_s, b_s)
    out.append(f"{c:>16} {n_ql:>4} {am:>10.4f} {bm:>10.4f} {w:>8} {l:>8} {t:>5} {p:>12.6f}")

out.append("")
out.append("=== 'ANOVA' informal -- media dos 5 metodos, agregada nas 6 classes (so descritivo) ===")
means = {}
for m in ["ig_ucb", "ig_ucb_ctx", "ig_ql_flat", "ig_random"]:
    vals = [series(m, c, 100).mean() for c in classes_all]
    means[m] = sum(vals) / len(vals)
ql_vals = [series("ig_ql", c, 100 if c in classes_ql_n100 else 30).mean() for c in classes_all]
means["ig_ql"] = sum(ql_vals) / len(ql_vals)
for k, v in sorted(means.items(), key=lambda kv: kv[1]):
    out.append(f"  media(6 classes) {k:>12}: {v:.4f}")

# ---------------------------------------------------------------
# Sanity check + confundimento de CPU nas 2 classes novas
# ---------------------------------------------------------------
out.append("")
out.append("=== sanity check: ig_ql/ils_rand_hi/ils_ql_hi nunca piores que LPT (reps 30-99, 2 classes novas) ===")
viol_total = 0
for c in classes_new:
    lpt_c = df[(df.instance == c) & (df.method == "lpt")].set_index("rep")["makespan"]
    for m in ["ig_ql", "ils_rand_hi", "ils_ql_hi"]:
        x_c = df[(df.instance == c) & (df.method == m)].copy()
        x_c["rep"] = x_c["rep"].astype(int)
        x_c = x_c[(x_c["rep"] >= 30) & (x_c["rep"] < 100)].set_index("rep")["makespan"]
        common = lpt_c.index.intersection(x_c.index)
        viol = (x_c.loc[common] > lpt_c.loc[common]).sum()
        viol_total += viol
        out.append(f"  {c} / {m}: violacoes={viol} (n pares={len(common)})")
out.append(f"TOTAL violacoes: {viol_total} (esperado: 0)")

out.append("")
out.append("=== confundimento por tempo de CPU (reps 30-99, 2 classes novas): ig_ql vs concorrentes ===")
for c in classes_new:
    for m in ["ils_rand_hi", "ils_ql_hi"]:
        cpu_ql = df[(df.instance == c) & (df.method == "ig_ql") & (df.rep >= 30) & (df.rep < 100)]["cpu_time"]
        cpu_m = df[(df.instance == c) & (df.method == m) & (df.rep >= 30) & (df.rep < 100)]["cpu_time"]
        out.append(f"  {c} / ig_ql cpu_mean={cpu_ql.mean():.3f}s  vs {m} cpu_mean={cpu_m.mean():.3f}s  (razao ig_ql/{m}={cpu_ql.mean()/cpu_m.mean():.2f}x)")

text = "\n".join(out)
with open("results/entrada30_ig_ql_n100_full_report.txt", "w") as f:
    f.write(text)
print(text)
