"""
analysis/report_entrada55_variance_ig_ql_vs_random.py
========================================================
Teste FORMAL de igualdade de variancia entre ig_ql e ig_random, motivado
pela hipotese levantada de forma independente por 2 consultas ao Gemini
(diario_pesquisa.md, Entrada 47): "Q-Learning reduz a VARIANCIA do
gap%, nao necessariamente a MEDIA". A Entrada 47 so' fez uma checagem
descritiva rapida (comparacao informal de desvio-padrao, sem teste de
hipotese) em todas as 6 classes originais do projeto -- este script
aplica o teste formal (Levene centrado na mediana + Fligner-Killeen,
cada um Holm-corrigido separadamente -- ver analysis/stats.py,
variance_equality_family, Entrada 53/Task #45) a TODAS as classes
disponiveis em results.csv.

Relatorio vivo (regenerado/sobrescrito a cada nova bateria -- mesmo
padrao de report_entrada37_sota_and_holm.py, nao versionado por
entrada ao contrario do diario/plano).

Execute: python3 -m analysis.report_entrada55_variance_ig_ql_vs_random
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import variance_equality_family

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")


def fmt(df):
    cols = ["family", "reference", "other", "n", "m", "n_ref", "n_other",
            "std_ref", "std_other", "ratio_std", "p_levene", "p_holm_levene",
            "reject_holm_levene", "p_fligner", "p_holm_fligner",
            "reject_holm_fligner", "agreement"]
    return df[cols].to_string(index=False)


def main():
    df = pd.read_csv("results/results.csv")

    result = variance_equality_family(
        df, reference_method="ig_ql", other_methods=["ig_random"],
        family_name="V1_ig_ql_vs_ig_random_variance",
    )

    lines = []
    lines.append("=" * 100)
    lines.append("TESTE FORMAL DE IGUALDADE DE VARIANCIA: ig_ql vs. ig_random (hipotese da Entrada 47)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Hipotese (diario_pesquisa.md, Entrada 47 -- 2 consultas INDEPENDENTES ao Gemini,")
    lines.append("uma sem saber da outra): Q-Learning reduz a VARIANCIA do gap%, nao necessariamente")
    lines.append("a media. H0: variancias iguais. Levene (centrado na mediana, robusto a")
    lines.append("nao-normalidade) e Fligner-Killeen (nao-parametrico) lado a lado -- concordancia")
    lines.append("entre os dois e' evidencia mais forte que qualquer um isolado. Cada teste")
    lines.append("Holm-corrigido SEPARADAMENTE (ver docstring de variance_equality_family).")
    lines.append("")
    lines.append(fmt(result))
    lines.append("")

    n_classes = len(result)
    n_ratio_below_1 = int((result["ratio_std"] < 1).sum())
    n_reject_both = int((result["reject_holm_levene"] & result["reject_holm_fligner"]).sum())
    n_reject_either = int((result["reject_holm_levene"] | result["reject_holm_fligner"]).sum())
    n_agree = int(result["agreement"].sum())

    lines.append("=" * 100)
    lines.append("RESUMO")
    lines.append("=" * 100)
    lines.append(f"Classes testadas: {n_classes}")
    lines.append(f"Classes com ratio_std < 1 (ig_ql com MENOS variancia que ig_random): {n_ratio_below_1}/{n_classes}")
    lines.append(f"Classes onde os DOIS testes (Levene E Fligner) rejeitam H0 apos Holm: {n_reject_both}/{n_classes}")
    lines.append(f"Classes onde PELO MENOS UM dos dois rejeita H0 apos Holm: {n_reject_either}/{n_classes}")
    lines.append(f"Classes onde os dois testes CONCORDAM na decisao (rejeitar ou nao): {n_agree}/{n_classes}")

    text = "\n".join(lines)
    print(text)
    out_path = Path("results/entrada55_variance_ig_ql_vs_random_report.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"\n\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
