import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Busca de regra melhorada para apriori_select (Task #78, a pedido de Carlos:
# "pode explorar mais a fundo, com o objetivo de melhorar o que temos" --
# apos a Entrada 63 apontar apriori_select como o mais solido dos 4
# candidatos, mas ainda com a fraqueza conhecida na familia E, Entradas
# 61-62).
#
# Ideia central: dup_ratio/actual_winner/ils_ql/ig_ql JA SAO CONHECIDOS pra
# TODAS as 1.612 instancias da bateria externa (_scratch_dup_ratio_
# validation_full.csv, ja calculado, nenhuma simulacao nova necessaria
# aqui) -- entao QUALQUER regra de despacho (limiar diferente, limiares
# por familia, ou uma regra mais rica com mais features) pode ser avaliada
# de GRACA: a regra so decide "usar ils_ql ou ig_ql", e os dois valores ja
# existem. Isso permite testar centenas de variantes sem rodar nenhuma
# busca nova -- SO' a validacao final do candidato escolhido (proxima
# etapa, script separado) precisa de uma rodada real de apriori_run().
#
# Metrica de otimizacao: gap_pct MEDIO resultante (nao acuracia de
# classificacao) -- uma regra pode "errar" o vencedor exato e ainda
# assim ser boa, se o erro for barato (as duas opcoes quase empatadas);
# e pode "acertar" muito e ainda ser ruim, se os erros raros forem caros.
# Acuracia de classificacao (usada nas Entradas 61-62) e' um proxy, nao o
# alvo real -- aqui otimizamos o alvo real diretamente.
#
# Protocolo anti-overfitting: split treino/teste ESTRATIFICADO por
# familia (70/30, seed fixa) -- toda escolha de limiar/regra e' feita
# olhando SO' o treino; o numero reportado como "resultado" e' sempre o
# do teste (dados nunca vistos durante a escolha da regra).
# ---------------------------------------------------------------------------

SEED = 2026
df = pd.read_csv("_scratch_dup_ratio_validation_full.csv")
df["oracle_best"] = df[["ils_ql", "ig_ql"]].min(axis=1)
df["nm_ratio"] = df.n / df.m
df["log_n"] = np.log(df.n)

train, test = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df.family)
print(f"treino: {len(train)}  teste: {len(test)}")
print("treino por familia:\n", train.family.value_counts())
print("teste por familia:\n", test.family.value_counts())


def gap_of_rule(sub, choose_ig_mask):
    """gap_pct medio resultante se a regra escolhe ig_ql onde choose_ig_mask
    e' True, ils_ql caso contrario."""
    chosen = np.where(choose_ig_mask, sub.ig_ql, sub.ils_ql)
    return chosen.mean()


def acc_of_rule(sub, choose_ig_mask):
    pred = np.where(choose_ig_mask, "ig", "ils")
    return (pred == sub.actual_winner.values).mean()


# ---------------------------------------------------------------------
# 1. Baseline atual: limiar global fixo = 0.1 (default do codigo hoje)
# ---------------------------------------------------------------------
print("\n=== BASELINE (threshold=0.1 global, o default atual do codigo) ===")
mask_test_01 = test.dup_ratio.values > 0.1
print(f"  TESTE: gap_pct medio={gap_of_rule(test, mask_test_01):.4f}  acc={acc_of_rule(test, mask_test_01):.4f}")
mask_all_01 = df.dup_ratio.values > 0.1
print(f"  TODAS 1612: gap_pct medio={gap_of_rule(df, mask_all_01):.4f}  acc={acc_of_rule(df, mask_all_01):.4f}")

# ---------------------------------------------------------------------
# 2. Sweep de limiar GLOBAL no TREINO, otimizando gap_pct (nao acuracia)
# ---------------------------------------------------------------------
print("\n=== sweep de limiar global no TREINO (otimizando gap_pct medio) ===")
thresholds = np.arange(0.0, 0.61, 0.01)
best_thr, best_gap = None, np.inf
results = []
for thr in thresholds:
    mask = train.dup_ratio.values > thr
    g = gap_of_rule(train, mask)
    a = acc_of_rule(train, mask)
    results.append((thr, g, a))
    if g < best_gap:
        best_gap, best_thr = g, thr
res_df = pd.DataFrame(results, columns=["threshold", "gap_pct_train", "acc_train"])
print(res_df.iloc[::5].to_string(index=False))  # amostra a cada 5 pra nao poluir
print(f"\nMELHOR limiar global por gap_pct no TREINO: {best_thr:.2f} (gap_train={best_gap:.4f})")
best_thr_acc = res_df.loc[res_df.acc_train.idxmax(), "threshold"]
print(f"MELHOR limiar global por ACURACIA no TREINO: {best_thr_acc:.2f} (p/ comparacao)")

print(f"\n=== limiar {best_thr:.2f} (tunado por gap_pct) avaliado no TESTE (nunca visto) ===")
mask_test_tuned = test.dup_ratio.values > best_thr
print(f"  TESTE: gap_pct medio={gap_of_rule(test, mask_test_tuned):.4f}  acc={acc_of_rule(test, mask_test_tuned):.4f}")

# ---------------------------------------------------------------------
# 3. Limiares POR FAMILIA (escolhidos so' com dados de treino daquela
#    familia), avaliados no teste
# ---------------------------------------------------------------------
print("\n=== limiares POR FAMILIA (escolhidos no treino, por familia) ===")
family_thresholds = {}
for fam, g_train in train.groupby("family"):
    best_f_thr, best_f_gap = None, np.inf
    for thr in thresholds:
        mask = g_train.dup_ratio.values > thr
        g = gap_of_rule(g_train, mask)
        if g < best_f_gap:
            best_f_gap, best_f_thr = g, thr
    family_thresholds[fam] = best_f_thr
    print(f"  familia {fam}: melhor limiar (treino)={best_f_thr:.2f}  gap_train={best_f_gap:.4f}")

print("\n=== limiares por familia avaliados no TESTE, por familia e GLOBAL ===")
test = test.copy()
test["thr_by_family"] = test.family.map(family_thresholds)
mask_test_perfam = test.dup_ratio.values > test.thr_by_family.values
per_fam_rows = []
for fam, g_test in test.groupby("family"):
    m = g_test.dup_ratio.values > family_thresholds[fam]
    per_fam_rows.append((fam, gap_of_rule(g_test, m), acc_of_rule(g_test, m), len(g_test)))
print(pd.DataFrame(per_fam_rows, columns=["family", "gap_pct_test", "acc_test", "n"]).round(4).to_string(index=False))
print(f"GLOBAL (por familia) no TESTE: gap_pct={gap_of_rule(test, mask_test_perfam):.4f}  acc={acc_of_rule(test, mask_test_perfam):.4f}")

# ---------------------------------------------------------------------
# 4. Regra mais rica: arvore de decisao rasa (profundidade 3) com
#    [dup_ratio, n, m, n/m, log(n)] -- treino/teste, comparar honestamente
# ---------------------------------------------------------------------
print("\n=== arvore de decisao rasa (profundidade 3), features=[dup_ratio,n,m,nm_ratio,log_n] ===")
FEATS = ["dup_ratio", "n", "m", "nm_ratio", "log_n"]
Xtr, ytr = train[FEATS], train.actual_winner
Xte, yte = test[FEATS], test.actual_winner
tree = DecisionTreeClassifier(max_depth=3, random_state=SEED, min_samples_leaf=20)
tree.fit(Xtr, ytr)
pred_tree_test = tree.predict(Xte)
mask_tree_test = pred_tree_test == "ig"
print(f"  TESTE: gap_pct medio={gap_of_rule(test, mask_tree_test):.4f}  acc={(pred_tree_test==yte.values).mean():.4f}")
print("\n  arvore aprendida (pra inspecao de interpretabilidade):")
print(export_text(tree, feature_names=FEATS))

# ---------------------------------------------------------------------
# 5. Regressao logistica com as mesmas features, normalizadas
# ---------------------------------------------------------------------
print("=== regressao logistica, mesmas features (normalizadas) ===")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(Xtr)
logit = LogisticRegression(random_state=SEED, max_iter=1000)
logit.fit(scaler.transform(Xtr), ytr)
pred_logit_test = logit.predict(scaler.transform(Xte))
mask_logit_test = pred_logit_test == "ig"
print(f"  TESTE: gap_pct medio={gap_of_rule(test, mask_logit_test):.4f}  acc={(pred_logit_test==yte.values).mean():.4f}")
print(f"  coeficientes: {dict(zip(FEATS, logit.coef_[0].round(3)))}")

# ---------------------------------------------------------------------
# 6. Resumo final -- todas as opcoes lado a lado, no TESTE, contra
#    oracle_best e contra sempre-ils/sempre-ig
# ---------------------------------------------------------------------
print("\n=== RESUMO FINAL: todas as opcoes no TESTE (nunca visto durante a escolha) ===")
oracle_test = test.oracle_best.mean()
always_ils_test = test.ils_ql.mean()
always_ig_test = test.ig_ql.mean()
summary = [
    ("oracle_best (teto teorico)", oracle_test, np.nan),
    ("sempre ils_ql", always_ils_test, np.nan),
    ("sempre ig_ql", always_ig_test, np.nan),
    ("threshold=0.1 (atual)", gap_of_rule(test, mask_test_01), acc_of_rule(test, mask_test_01)),
    (f"threshold={best_thr:.2f} (tunado, global)", gap_of_rule(test, mask_test_tuned), acc_of_rule(test, mask_test_tuned)),
    ("limiares por familia (tunados)", gap_of_rule(test, mask_test_perfam), acc_of_rule(test, mask_test_perfam)),
    ("arvore rasa (depth=3)", gap_of_rule(test, mask_tree_test), (pred_tree_test==yte.values).mean()),
    ("regressao logistica", gap_of_rule(test, mask_logit_test), (pred_logit_test==yte.values).mean()),
]
print(pd.DataFrame(summary, columns=["regra", "gap_pct_TESTE", "acc_TESTE"]).round(4).to_string(index=False))

print("\nfamily_thresholds encontrados:", family_thresholds)
print("DONE")
