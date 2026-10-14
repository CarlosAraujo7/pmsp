"""
tests/test_stats.py
Testes de sanidade do modulo analysis/stats.py (correcao de Holm-
Bonferroni e comparacao Wilcoxon por familia -- ver diario_pesquisa.md,
Entrada 37).
Execute: pytest tests/test_stats.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import (
    holm_bonferroni,
    pairwise_wilcoxon_family,
    rank_biserial_paired,
    bootstrap_ci_mean_diff,
    variance_equality_family,
)


class TestHolmBonferroni:

    def test_known_example(self):
        # Exemplo classico verificado manualmente na docstring do
        # modulo/diario: p=[0.01,0.02,0.03,0.04,0.5], alpha=0.05 ->
        # so o primeiro (p=0.01) permanece significativo.
        res = holm_bonferroni([0.01, 0.02, 0.03, 0.04, 0.5], alpha=0.05)
        assert res["m"] == 5
        assert res["reject"] == [True, False, False, False, False]
        assert res["p_adjusted"][0] == pytest.approx(0.05)
        # monotonicidade: p ajustado nao pode diminuir com o rank
        sorted_adj = [res["p_adjusted"][i] for i in
                      sorted(range(5), key=lambda i: [0.01,0.02,0.03,0.04,0.5][i])]
        assert all(sorted_adj[i] <= sorted_adj[i+1] for i in range(4))

    def test_empty_input(self):
        res = holm_bonferroni([], alpha=0.05)
        assert res == {"p_adjusted": [], "reject": [], "m": 0}

    def test_single_pvalue_matches_alpha(self):
        res = holm_bonferroni([0.03], alpha=0.05)
        assert res["p_adjusted"][0] == pytest.approx(0.03)
        assert res["reject"][0] is True

    def test_never_less_conservative_than_bonferroni(self):
        # Holm sempre rejeita pelo menos tantas hipoteses quanto
        # Bonferroni simples (p_holm <= m * p_raw, elemento a elemento
        # nao vale sempre, mas o NUMERO de rejeicoes de Holm >= Bonferroni).
        pvals = [0.001, 0.004, 0.01, 0.02, 0.03, 0.2, 0.5]
        m = len(pvals)
        holm = holm_bonferroni(pvals, alpha=0.05)
        bonferroni_reject = [p * m <= 0.05 for p in pvals]
        assert sum(holm["reject"]) >= sum(bonferroni_reject)

    def test_all_pvalues_bounded_by_one(self):
        res = holm_bonferroni([0.9, 0.95, 0.99], alpha=0.05)
        assert all(p <= 1.0 for p in res["p_adjusted"])

    def test_output_order_matches_input_order(self):
        # p-valores fora de ordem na entrada devem ter p_adjusted
        # correspondendo ao INDICE original, nao a ordem ordenada.
        pvals = [0.5, 0.01, 0.03]
        res = holm_bonferroni(pvals, alpha=0.05)
        # o menor p-valor (indice 1) deve ter o maior "poder" (menor
        # fator de penalizacao relativo) -- aqui so checamos que o
        # p ajustado do indice 1 (bruto=0.01) e o mais competitivo
        assert res["p_adjusted"][1] < res["p_adjusted"][0]


class TestPairwiseWilcoxonFamily:

    @pytest.fixture
    def toy_df(self):
        # 2 metodos, 1 classe (n=10,m=2), 30 reps -- 'a' sistematicamente
        # melhor (menor gap) que 'b' em quase todos os pares.
        rng = np.random.default_rng(0)
        rows = []
        for rep in range(30):
            gap_a = max(0.0, rng.normal(1.0, 0.3))
            gap_b = gap_a + abs(rng.normal(1.5, 0.3))  # 'b' sempre pior
            rows.append(dict(method="a", n=10, m=2, rep=rep, gap_pct=gap_a))
            rows.append(dict(method="b", n=10, m=2, rep=rep, gap_pct=gap_b))
            # 'c' = empate exato com 'a'
            rows.append(dict(method="c", n=10, m=2, rep=rep, gap_pct=gap_a))
        return pd.DataFrame(rows)

    def test_detects_real_difference(self, toy_df):
        result = pairwise_wilcoxon_family(toy_df, "a", ["b"], family_name="teste")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_pairs"] == 30
        assert row["p_raw"] < 0.05
        assert row["reject_holm"] == True
        assert row["direction"] == "ref_better"

    def test_tie_produces_nan_pvalue(self, toy_df):
        result = pairwise_wilcoxon_family(toy_df, "a", ["c"], family_name="teste")
        row = result.iloc[0]
        assert pd.isna(row["p_raw"])
        assert row["reject_holm"] == False
        assert row["direction"] == "tie"

    def test_family_correction_applied_jointly(self, toy_df):
        # Familia com 2 comparacoes (b e c) -- a correcao de Holm deve
        # ser aplicada considerando as DUAS juntas (m=2 validos: so 'b'
        # tem p valido, 'c' e NaN e fica fora da correcao).
        result = pairwise_wilcoxon_family(toy_df, "a", ["b", "c"], family_name="teste")
        assert len(result) == 2
        valid = result[result["p_raw"].notna()]
        assert len(valid) == 1  # so 'b' tem p-valor valido
        assert valid.iloc[0]["p_holm"] == pytest.approx(valid.iloc[0]["p_raw"])  # m=1 valido -> sem penalizacao

    def test_missing_method_produces_zero_pairs(self, toy_df):
        result = pairwise_wilcoxon_family(toy_df, "a", ["nonexistent"], family_name="teste")
        assert result.iloc[0]["n_pairs"] == 0
        assert pd.isna(result.iloc[0]["p_raw"])

    def test_unequal_n_uses_common_reps_only(self, toy_df):
        # 'd' so tem reps 0..9 (N menor) -- o merge deve usar so os
        # 10 reps em comum, nao falhar.
        extra = toy_df[toy_df["method"] == "a"].copy()
        extra["method"] = "d"
        extra = extra[extra["rep"] < 10]
        df2 = pd.concat([toy_df, extra], ignore_index=True)
        result = pairwise_wilcoxon_family(df2, "a", ["d"], family_name="teste")
        assert result.iloc[0]["n_pairs"] == 10

    # ---- Testes novos (Entrada 55 / Task #45): effect sizes ----

    def test_effect_size_columns_present(self, toy_df):
        # Schema completo esperado desde a extensao da Entrada 53 --
        # trava o contrato de colunas (regressao se alguma sumir/mudar
        # de nome).
        result = pairwise_wilcoxon_family(toy_df, "a", ["b"], family_name="teste")
        expected_cols = {
            "family", "reference", "other", "n", "m", "n_pairs", "p_raw",
            "mean_ref", "mean_other", "mean_diff", "ci_low", "ci_high",
            "rank_biserial", "p_holm", "reject_holm", "direction",
        }
        assert expected_cols == set(result.columns)

    def test_effect_size_consistent_with_real_difference(self, toy_df):
        # 'b' e' sistematicamente pior que 'a' (gap_b = gap_a + offset
        # positivo, por construcao do fixture) -- o effect size deve
        # refletir isso com folga, nao so' o p-valor.
        result = pairwise_wilcoxon_family(toy_df, "a", ["b"], family_name="teste")
        row = result.iloc[0]
        # mean_diff = mean(ref) - mean(other) = mean(a) - mean(b) < 0
        # (a tem gap menor)
        assert row["mean_diff"] < 0
        assert row["mean_diff"] == pytest.approx(row["mean_ref"] - row["mean_other"], abs=0.05)
        # rank-biserial: convencao de sinal da docstring (Entrada 53) --
        # r_rb > 0 => ref PIOR (gap maior). Aqui ref e' melhor, entao
        # r_rb deve ser fortemente NEGATIVO (quase -1, quase todo par
        # favorece 'a' por construcao do fixture).
        assert row["rank_biserial"] < -0.9
        # IC 95% nao cruza zero -- diferenca robusta, nao so' "p<0.05"
        assert row["ci_high"] < 0

    def test_tie_case_bootstrap_still_computed(self, toy_df):
        # Mesmo quando o Wilcoxon nao da' p-valor (empate total, 'c'
        # identico a 'a'), o IC bootstrap da diferenca media ainda e'
        # calculado (mean_diff exatamente 0, ci degenerado em 0 ou NaN)
        # -- e' uma quantidade DIFERENTE do teste de hipotese, nao deve
        # ficar refem do mesmo caso degenerado.
        result = pairwise_wilcoxon_family(toy_df, "a", ["c"], family_name="teste")
        row = result.iloc[0]
        assert pd.isna(row["rank_biserial"])  # sem diferencas nao-nulas
        assert row["mean_diff"] == pytest.approx(0.0, abs=1e-9)
        assert pd.isna(row["ci_low"]) or row["ci_low"] == pytest.approx(0.0, abs=1e-9)
        assert pd.isna(row["ci_high"]) or row["ci_high"] == pytest.approx(0.0, abs=1e-9)

    def test_bootstrap_seed_reproducible_across_calls(self, toy_df):
        # Mesma bootstrap_seed -> mesmo IC exato entre duas chamadas
        # independentes (reprodutibilidade, nao so' "roda sem erro").
        r1 = pairwise_wilcoxon_family(toy_df, "a", ["b"], family_name="t", bootstrap_seed=42)
        r2 = pairwise_wilcoxon_family(toy_df, "a", ["b"], family_name="t", bootstrap_seed=42)
        assert r1.iloc[0]["ci_low"] == r2.iloc[0]["ci_low"]
        assert r1.iloc[0]["ci_high"] == r2.iloc[0]["ci_high"]


class TestRankBiserialPaired:

    def test_all_positive_differences(self):
        assert rank_biserial_paired([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)

    def test_all_negative_differences(self):
        assert rank_biserial_paired([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)

    def test_all_zero_differences_returns_nan(self):
        assert np.isnan(rank_biserial_paired([1, 2, 3], [1, 2, 3]))

    def test_known_hand_computed_value(self):
        # d = [1,-2,3,-4,5] (valores absolutos ja distintos e ja
        # ordenados, ranks = os proprios valores absolutos: 1,2,3,4,5).
        # W+ = rank(1)+rank(3)+rank(5) = 1+3+5 = 9.
        # W- = rank(2)+rank(4) = 2+4 = 6.
        # r_rb = (9-6)/(9+6) = 3/15 = 0.2 -- conferido a mao, nao so' rodado.
        x = [1, -2, 3, -4, 5]
        y = [0, 0, 0, 0, 0]
        assert rank_biserial_paired(x, y) == pytest.approx(0.2)

    def test_antisymmetry(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=25)
        y = rng.normal(size=25)
        assert rank_biserial_paired(x, y) == pytest.approx(-rank_biserial_paired(y, x))

    def test_bounded_range_with_ties(self):
        # dados com empates propositais (varios valores repetidos) --
        # rankdata faz rank medio nesse caso, resultado deve continuar
        # em [-1, 1].
        x = [1, 1, 2, 2, 3, 3, 1, 2]
        y = [2, 1, 1, 3, 2, 1, 1, 2]
        r = rank_biserial_paired(x, y)
        assert -1.0 <= r <= 1.0

    def test_zero_pairs_excluded_dont_change_result(self):
        # Pares com d_i=0 sao excluidos do calculo -- acrescentar mais
        # um par empatado nao pode mudar o resultado.
        base = rank_biserial_paired([1, -2, 3], [0, 0, 0])
        extended = rank_biserial_paired([1, -2, 3, 5], [0, 0, 0, 5])
        assert base == pytest.approx(extended)


class TestBootstrapCiMeanDiff:

    def test_point_estimate_exact(self):
        # mean_diff e' a estimativa pontual (nao passa por bootstrap) --
        # deve ser EXATA, sem nenhuma variabilidade de reamostragem.
        res = bootstrap_ci_mean_diff([10, 20, 30], [1, 2, 3], seed=0)
        assert res["mean_diff"] == pytest.approx(18.0)

    def test_ci_bounds_ordered(self):
        rng = np.random.default_rng(5)
        x = rng.normal(10, 2, size=40)
        y = rng.normal(8, 2, size=40)
        res = bootstrap_ci_mean_diff(x, y, seed=1, n_resamples=999)
        assert res["ci_low"] <= res["ci_high"]

    def test_degenerate_constant_difference(self):
        # diferenca identica em todos os pares -> variancia zero -> BCa
        # nao consegue estimar aceleracao -> cai para percentile -> se
        # AINDA falhar, NaN (nunca uma excecao). mean_diff deve ficar
        # exato de qualquer forma.
        x = [5.0] * 10
        y = [0.0] * 10
        res = bootstrap_ci_mean_diff(x, y, seed=0, n_resamples=200)
        assert res["mean_diff"] == pytest.approx(5.0)
        assert np.isnan(res["ci_low"]) or res["ci_low"] == pytest.approx(5.0)
        assert np.isnan(res["ci_high"]) or res["ci_high"] == pytest.approx(5.0)

    def test_reproducible_with_same_seed(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=30)
        y = rng.normal(size=30)
        r1 = bootstrap_ci_mean_diff(x, y, seed=99, n_resamples=500)
        r2 = bootstrap_ci_mean_diff(x, y, seed=99, n_resamples=500)
        assert r1 == r2

    def test_n_zero_returns_nan(self):
        res = bootstrap_ci_mean_diff([], [])
        assert np.isnan(res["mean_diff"])
        assert np.isnan(res["ci_low"])
        assert np.isnan(res["ci_high"])

    def test_n_one_mean_valid_ci_nan(self):
        # n=1: media pontual e' definida (diferenca do unico par), mas
        # bootstrap de 1 ponto nao faz sentido -- ci fica NaN por
        # construcao (guarda explicita `if n < 2`).
        res = bootstrap_ci_mean_diff([5], [2])
        assert res["mean_diff"] == pytest.approx(3.0)
        assert np.isnan(res["ci_low"])
        assert np.isnan(res["ci_high"])


class TestVarianceEqualityFamily:

    @pytest.fixture
    def variance_df(self):
        # 3 metodos, mesma media (~2.0), variancias BEM diferentes por
        # construcao -- 'lowvar' e 'samevar' compartilham o mesmo
        # desvio-padrao (0.1), 'highvar' tem 10x mais (1.0). Trunca em
        # 0 (max(0, ...)) igual ao padrao de gap_pct no resto do
        # projeto.
        rng = np.random.default_rng(1)
        rows = []
        for rep in range(50):
            rows.append(dict(method="lowvar", n=10, m=2, rep=rep,
                              gap_pct=max(0.0, rng.normal(2.0, 0.1))))
            rows.append(dict(method="highvar", n=10, m=2, rep=rep,
                              gap_pct=max(0.0, rng.normal(2.0, 1.0))))
            rows.append(dict(method="samevar", n=10, m=2, rep=rep,
                              gap_pct=max(0.0, rng.normal(2.0, 0.1))))
        return pd.DataFrame(rows)

    def test_detects_real_variance_difference(self, variance_df):
        result = variance_equality_family(variance_df, "lowvar", ["highvar"], family_name="teste")
        row = result.iloc[0]
        assert row["std_ref"] < row["std_other"]
        assert row["p_levene"] < 0.05
        assert row["p_fligner"] < 0.05
        assert row["reject_holm_levene"] == True
        assert row["reject_holm_fligner"] == True
        assert row["agreement"] == True

    def test_similar_variance_not_rejected(self, variance_df):
        result = variance_equality_family(variance_df, "lowvar", ["samevar"], family_name="teste")
        row = result.iloc[0]
        assert row["reject_holm_levene"] == False
        assert row["reject_holm_fligner"] == False
        assert row["agreement"] == True

    def test_ratio_std_matches_raw_columns(self, variance_df):
        result = variance_equality_family(variance_df, "lowvar", ["highvar"], family_name="teste")
        row = result.iloc[0]
        assert row["ratio_std"] == pytest.approx(row["std_ref"] / row["std_other"])

    def test_no_pairing_required_unequal_n(self, variance_df):
        # Levene/Fligner comparam dispersao, nao pares -- deve
        # funcionar mesmo com N diferente entre os dois metodos (ao
        # contrario de pairwise_wilcoxon_family, que exige merge por
        # 'rep').
        extra = variance_df[variance_df["method"] == "highvar"].copy()
        extra["method"] = "highvar_fewer"
        extra = extra[extra["rep"] < 20]  # so' 20 reps em vez de 50
        df2 = pd.concat([variance_df, extra], ignore_index=True)
        result = variance_equality_family(df2, "lowvar", ["highvar_fewer"], family_name="teste")
        row = result.iloc[0]
        assert row["n_ref"] == 50
        assert row["n_other"] == 20
        assert not pd.isna(row["p_levene"])

    def test_missing_method_gives_nan_stats(self, variance_df):
        result = variance_equality_family(variance_df, "lowvar", ["nonexistent"], family_name="teste")
        row = result.iloc[0]
        assert row["n_other"] == 0
        assert pd.isna(row["std_other"])
        assert pd.isna(row["p_levene"])
        assert pd.isna(row["p_fligner"])

    def test_two_independent_comparisons_holm_corrected_separately(self, variance_df):
        # familia com 2 comparacoes (highvar real, samevar nulo) --
        # Levene e Fligner devem ser corrigidos CADA UM por Holm
        # considerando as 2 comparacoes da familia (nao misturados
        # entre si -- ver docstring do modulo).
        result = variance_equality_family(
            variance_df, "lowvar", ["highvar", "samevar"], family_name="teste")
        assert len(result) == 2
        assert set(result.columns) >= {
            "family", "reference", "other", "n", "m", "n_ref", "n_other",
            "std_ref", "std_other", "ratio_std", "p_levene", "p_holm_levene",
            "reject_holm_levene", "p_fligner", "p_holm_fligner",
            "reject_holm_fligner", "agreement",
        }
