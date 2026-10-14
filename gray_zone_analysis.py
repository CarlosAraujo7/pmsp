import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Caracterizacao da "zona cinzenta" pro mecanismo com gatilho de confianca
# (Task #79, continuacao da Entrada 64, a pedido de Carlos "pode prosseguir
# com ela"). Reusa _scratch_dup_ratio_validation_full.csv (1.612 instancias,
# ja calculado) -- de graca, sem simulacao nova -- pra decidir a LARGURA da
# faixa ao redor do threshold=0,1 onde vale a pena gastar um probe empirico
# (baixa confianca), antes de implementar o mecanismo de verdade.
#
# Duas perguntas respondidas aqui:
# 1. A acuracia do dup_ratio realmente piora perto do limiar? (justifica a
#    ideia toda -- se nao piorar, gatilho de confianca nao faz sentido)
# 2. Pra cada largura de faixa candidata: quantas instancias cairiam nela
#    (= quantas pagariam o custo do probe) e qual o TETO TEORICO de melhora
#    (se essas instancias fossem perfeitamente resolvidas por um oraculo,
#    em vez de pela regra de dup_ratio) -- serve pra escolher uma faixa
#    estreita o suficiente pra nao inflar o custo medio, mas que capture a
#    maior parte do ganho possivel.
# ---------------------------------------------------------------------------

THRESHOLD = 0.1
df = pd.read_csv("_scratch_dup_ratio_validation_full.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
df["pred_thr"] = np.where(df.dup_ratio > THRESHOLD, "ig", "ils")
df["gap_thr"] = np.where(df.pred_thr == "ig", df.ig_ql, df.ils_ql)
df["dist_to_thr"] = (df.dup_ratio - THRESHOLD).abs()

print(f"gap_pct medio da regra atual (threshold={THRESHOLD}), TODAS as 1612: {df.gap_thr.mean():.4f}")
print(f"oracle_best medio (teto teorico absoluto): {df.oracle_best.mean():.4f}")

# ---------------------------------------------------------------------
# 1. Acuracia da regra de threshold, em funcao da distancia ao limiar
#    (bins) -- confirma (ou refuta) a premissa de que "perto do limiar
#    = mais erro"
# ---------------------------------------------------------------------
print("\n=== acuracia da regra de threshold=0,1, por faixa de distancia ao limiar ===")
bins = [0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 1.0]
df["dist_bin"] = pd.cut(df.dist_to_thr, bins=bins, include_lowest=True)
acc_by_dist = df.groupby("dist_bin", observed=True).apply(
    lambda g: pd.Series({
        "n": len(g),
        "acc": (g.pred_thr == g.actual_winner).mean(),
        "gap_thr_medio": g.gap_thr.mean(),
        "oracle_medio": g.oracle_best.mean(),
        "regret_medio": (g.gap_thr - g.oracle_best).mean(),
    })
)
print(acc_by_dist.round(4))

# ---------------------------------------------------------------------
# 2. Sweep de largura de faixa (gray_zone) -- teto teorico de melhora
#    se as instancias DENTRO da faixa fossem substituidas por
#    oracle_best (resolvidas perfeitamente), e o resto continuasse
#    dispachado normalmente pela regra de threshold
# ---------------------------------------------------------------------
print("\n=== sweep de largura de faixa (gray_zone) -- teto teorico ===")
gap_atual = df.gap_thr.mean()
rows = []
for w in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    in_zone = df.dist_to_thr <= w
    gap_ceiling = np.where(in_zone, df.oracle_best, df.gap_thr).mean()
    acc_in_zone = (df[in_zone].pred_thr == df[in_zone].actual_winner).mean() if in_zone.sum() > 0 else np.nan
    rows.append((w, in_zone.sum(), in_zone.mean() * 100, acc_in_zone, gap_ceiling, gap_atual - gap_ceiling))
sweep = pd.DataFrame(rows, columns=["gray_zone_w", "n_in_zone", "pct_in_zone", "acc_in_zone", "gap_ceiling_teorico", "melhora_teorica_max"])
print(sweep.round(4).to_string(index=False))

# ---------------------------------------------------------------------
# 3. Mesma coisa, por familia -- confirma se a zona cinzenta se
#    concentra nalguma familia especifica (ex.: E, ja conhecida como
#    fraca)
# ---------------------------------------------------------------------
print("\n=== zona cinzenta (w=0,05) por familia ===")
w = 0.05
in_zone = df.dist_to_thr <= w
for fam, g in df.groupby("family"):
    gz = g[g.dist_to_thr <= w]
    print(f"  familia {fam}: {len(gz)}/{len(g)} ({len(gz)/len(g)*100:.1f}%) na zona cinzenta, "
          f"acc dentro={( gz.pred_thr==gz.actual_winner).mean() if len(gz)>0 else float('nan'):.4f}, "
          f"acc fora={(g[~(g.dist_to_thr<=w)].pred_thr==g[~(g.dist_to_thr<=w)].actual_winner).mean():.4f}")

print("\nDONE")
