"""
analysis/report_entrada37_sota_and_holm.py
============================================
Relatorio vivo (regenerado/sobrescrito a cada nova bateria SOTA -- ver
diario_pesquisa.md, Entrada 38: decisao deliberada de nao versionar
este arquivo por entrada, ao contrario do diario/plano, que sao
append-only): (A) comparacao empirica real do ig_ql contra os baselines
SOTA reimplementados (MULTIFIT -- Entrada 37; SA de Li & Perkins 2024
-- Entrada 37; BCO de Davidovic et al. 2012, variante Modificada --
Entradas 39-40); (B) correcao de Holm-Bonferroni aplicada
retroativamente as 4 familias de hipoteses do projeto (decisao de
Carlos). Ver analysis/stats.py para a implementacao.

Execute: python3 -m analysis.report_entrada37_sota_and_holm
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import pairwise_wilcoxon_family
from analysis.tables import latex_wilcoxon_family_table

pd.set_option("display.width", 220)  # mais largo -- 4 colunas novas de effect size (Entrada 55)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")


def fmt(df):
    # mean_diff/ci_low/ci_high/rank_biserial: effect sizes acrescentados
    # na Entrada 53 (Task #45) -- ver diario_pesquisa.md, Entrada 55,
    # para a leitura destes numeros aplicada as 4 familias abaixo
    # (motivacao original: distinguir "nao significativo" de
    # "comprovadamente nulo" na familia F3, ver LEI_GERAL_PESQUISA.md).
    cols = ["family", "reference", "other", "n", "m", "n_pairs",
            "mean_ref", "mean_other", "mean_diff", "ci_low", "ci_high",
            "rank_biserial", "p_raw", "p_holm", "reject_holm", "direction"]
    return df[cols].to_string(index=False)


def main():
    df = pd.read_csv("results/results.csv")

    families = []

    fam1 = pairwise_wilcoxon_family(
        df, reference_method="ig_ql",
        other_methods=["lpt", "ils_random", "ils_ucb", "ils_rf", "ils_ql",
                        "ils_rand_hi", "ils_ql_ws", "ils_ql_hi"],
        family_name="F1_headline_ig_ql_vs_ils",
    )
    families.append(("F1 -- ig_ql vs. baselines ILS (alegacao central do artigo)", fam1))

    fam2 = pairwise_wilcoxon_family(
        df, reference_method="ig_ql",
        other_methods=["multifit", "sa_li_perkins", "bco"],
        family_name="F2_sota_ig_ql_vs_published",
    )
    families.append(("F2 -- ig_ql vs. baselines SOTA publicados (MULTIFIT+SA Entrada 37, BCO Entradas 40-44, N=100 desde a Entrada 44)", fam2))

    fam3 = pairwise_wilcoxon_family(
        df, reference_method="ig_ql",
        other_methods=["ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat"],
        family_name="F3_ablation_mechanism_vs_learning",
    )
    families.append(("F3 -- ablacao ig_ql vs. controles (decomposicao mecanismo/aprendizado)", fam3))

    fam4 = pairwise_wilcoxon_family(
        df, reference_method="ig_random",
        other_methods=["ils_random"],
        family_name="F4_mechanism_isolation",
    )
    families.append(("F4 -- isolamento de mecanismo (ig_random vs. ils_random, Atualizacao 13)", fam4))

    lines = []
    lines.append("=" * 100)
    lines.append("RELATORIO SOTA (MULTIFIT + SA Li&Perkins 2024 + BCO Davidovic et al. 2012) + CORRECAO DE HOLM-BONFERRONI")
    lines.append("=" * 100)
    for title, fam_df in families:
        lines.append("")
        lines.append(f"--- {title} ---")
        lines.append(f"    tamanho da familia (testes com p valido): {fam_df['p_raw'].notna().sum()}")
        lines.append(fmt(fam_df))

    lines.append("")
    lines.append("=" * 100)
    lines.append("RESUMO: quantas comparacoes MUDAM de significativas (p_raw<0.05) para")
    lines.append("NAO significativas (p_holm>=0.05) apos a correcao, por familia:")
    lines.append("=" * 100)
    all_fams = pd.concat([f for _, f in families], ignore_index=True)
    for fam_name, group in all_fams.groupby("family"):
        raw_sig = (group["p_raw"] < 0.05).sum()
        holm_sig = group["reject_holm"].sum()
        flipped = ((group["p_raw"] < 0.05) & (~group["reject_holm"])).sum()
        lines.append(f"  {fam_name:38s} | sig. bruto: {raw_sig:2d} | sig. Holm: {holm_sig:2d} | "
                      f"mudaram p/ nao-sig.: {flipped}")

    text = "\n".join(lines)
    print(text)
    out_path = Path("results/entrada37_sota_and_holm_report.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"\n\nSalvo em {out_path}")

    # Tabelas LaTeX para o artigo (Entrada 75) -- F1 (headline) e F2
    # (SOTA) sao as duas familias que o texto do artigo vai precisar
    # primeiro (Task #44); F3/F4 ficam disponiveis nas variaveis
    # fam3/fam4 acima se a reescrita tambem precisar delas.
    tables_dir = Path("results/tables")
    p1 = latex_wilcoxon_family_table(
        fam1, tables_dir / "f1_headline_ig_ql_vs_ils.tex",
        reference_label=r"\textsc{IG-QL}", label="tab:f1_headline",
    )
    p2 = latex_wilcoxon_family_table(
        fam2, tables_dir / "f2_sota_ig_ql_vs_published.tex",
        reference_label=r"\textsc{IG-QL}", label="tab:f2_sota",
    )
    # Tabelas F3/F4 (Entrada 82 -- Task #44/Fase D): a decisao de Carlos
    # (Entrada 80, "tudo mesmo") passou a exigir F3/F4 tambem como
    # tabelas formatadas no corpo do artigo, nao so disponiveis como
    # variaveis em memoria (nota da Entrada 75 dizia "ficam disponiveis
    # se a reescrita tambem precisar" -- agora precisa).
    p3 = latex_wilcoxon_family_table(
        fam3, tables_dir / "f3_ablation_ig_ql_vs_controls.tex",
        reference_label=r"\textsc{IG-QL}", label="tab:f3_ablation",
    )
    p4 = latex_wilcoxon_family_table(
        fam4, tables_dir / "f4_mechanism_isolation.tex",
        reference_label=r"\textsc{IG-Random}", label="tab:f4_isolation",
    )
    print(f"Tabelas LaTeX (Entrada 75): {p1}, {p2}")
    print(f"Tabelas LaTeX (Entrada 82): {p3}, {p4}")


if __name__ == "__main__":
    main()
