"""
analysis/report_entrada58_external_validation.py
===================================================
Relatorio vivo (mesma convencao do report_entrada37_sota_and_holm.py --
regenerado/sobrescrito, nao versionado por entrada -- ver diario,
Entrada 38): comparacao estatistica formal do ig_ql contra os 3
baselines SOTA, as 4 ablacoes IG e os 4 metodos ILS, usando os
resultados da bateria externa (Symmetry 2025, familias B/C/E/F,
1.500 instancias -- ver diario_pesquisa.md, Entradas 52-57).

Diferenca estrutural em relacao ao report_entrada37 (que roda nas
instancias SINTETICAS do projeto): la', UMA familia de Holm cobre
todas as classes (n,m) do projeto inteiro. Aqui, cada FAMILIA EXTERNA
(B, C, E ou F) e' tratada como uma familia de Holm SEPARADA por tipo
de comparacao -- B, C, E, F sao distribuicoes de instancia distintas
(origens diferentes dentro do benchmark Symmetry 2025), entao "ig_ql
generaliza pra familia B" e "ig_ql generaliza pra familia C" sao
alegacoes empiricas distintas, nao uma unica alegacao "generaliza pra
instancias externas em geral".

Nota tecnica importante: pairwise_wilcoxon_family() agrupa por (n,m),
nao por subfolder -- e a familia F tem 3 pares (n,m) usados por DOIS
subfolders diferentes cada (FFU e FFNU coincidem em (50,25), (100,5) e
(1000,5) -- confirmado por inspecao direta do manifest.csv). Agrupar
so' por (n,m) juntaria instancias de FFU com FFNU incorretamente no
pareamento do Wilcoxon. Por isso, este script chama
pairwise_wilcoxon_family() UMA VEZ POR SUBFOLDER (que e' sempre um
identificador de classe valido, nunca colide) e refaz a correcao de
Holm manualmente sobre os p-valores brutos concatenados de toda a
familia externa -- reaproveita o pareamento/effect-size ja testado de
pairwise_wilcoxon_family, so' corrige o agrupamento de Holm.

Execute: python3 -m analysis.report_entrada58_external_validation
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import pairwise_wilcoxon_family, holm_bonferroni

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")

RESULTS_CSV = ROOT / "external_benchmarks" / "pcmax_symmetry2025" / "results_external_symmetry2025.csv"
MANIFEST_CSV = ROOT / "external_benchmarks" / "pcmax_symmetry2025" / "manifest.csv"


def load_clean_external_results() -> pd.DataFrame:
    """
    Carrega results_external_symmetry2025.csv, restrito as 1.500
    instancias do manifest ATUAL, com exatamente 1 linha por
    (member, method) -- resolve as 52 duplicatas conhecidas (Entrada
    57: mesma instancia recebeu inst_id/rep diferente em geracoes
    anteriores do manifest) mantendo APENAS a linha cujo rep bate com
    o inst_id oficial do manifest atual para aquele member.
    """
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest_inst_id = manifest.set_index("member")["inst_id"].astype(str).to_dict()
    manifest_members = set(manifest_inst_id)

    df = pd.read_csv(RESULTS_CSV)
    df = df[df["member"].isin(manifest_members)].copy()
    df["rep"] = df["rep"].astype(str)
    df["_official_rep"] = df["member"].map(manifest_inst_id)
    df["_is_official"] = df["rep"] == df["_official_rep"]

    before = len(df)
    # dentro de cada (member,method): se existir linha com rep oficial,
    # fica com ela; senao (so' existe linha de uma geracao antiga do
    # manifest, nunca reprocessada -- resume por (member,method) nao
    # exige rep==oficial), fica com QUALQUER linha disponivel (sao
    # validas, so' tem uma tag de rep antiga). Nunca descarta um
    # (member,method) que so' tem 1 linha, so' resolve empate quando ha
    # mais de uma.
    df = (df.sort_values("_is_official", ascending=False)
            .drop_duplicates(subset=["member", "method"], keep="first"))
    after = len(df)
    dup_resolved = before - after
    n_nonofficial_kept = int((~df["_is_official"]).sum())
    if dup_resolved:
        print(f"[dedup] {dup_resolved} linhas duplicadas descartadas "
              f"(mantida a de rep oficial quando havia mais de uma)")
    if n_nonofficial_kept:
        print(f"[dedup] {n_nonofficial_kept} (member,method) so' tinham linha de rep "
              f"nao-oficial (geracao antiga do manifest, nunca reprocessada) -- mantidas, sao dados validos")
    df = df.drop(columns=["_official_rep", "_is_official"])

    n_expected = len(manifest) * 13
    assert len(df) == n_expected, (
        f"esperado {n_expected} linhas (1500 membros x 13 metodos), "
        f"encontrado {len(df)} -- bateria externa incompleta ou dedup incorreto"
    )
    got = df.groupby("member")["method"].nunique()
    assert (got == 13).all(), "algum membro nao tem os 13 metodos"
    return df


def family_by_subfolder(df: pd.DataFrame, reference: str, others: list[str],
                          family_name: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Mesma logica do pairwise_wilcoxon_family, mas agrupando por
    subfolder (nunca colide) em vez de (n,m) (colide na familia F --
    ver docstring do modulo). Reusa pairwise_wilcoxon_family() por
    subfolder para o pareamento/effect-size, descarta o p_holm de
    cada chamada individual (grupo pequeno demais) e recalcula Holm
    UMA VEZ sobre os p_raw concatenados de toda a familia externa.
    """
    rows = []
    for subfolder, sub_df in df.groupby("subfolder"):
        partial = pairwise_wilcoxon_family(
            sub_df, reference_method=reference, other_methods=others,
            family_name=family_name, alpha=alpha,
        )
        partial["subfolder"] = subfolder
        rows.append(partial)
    result = pd.concat(rows, ignore_index=True)

    valid = result["p_raw"].notna()
    holm = holm_bonferroni(result.loc[valid, "p_raw"].tolist(), alpha=alpha)
    p_holm_full = np.full(len(result), np.nan)
    reject_full = np.zeros(len(result), dtype=bool)
    p_holm_full[valid.to_numpy()] = holm["p_adjusted"]
    reject_full[valid.to_numpy()] = holm["reject"]
    result["p_holm"] = p_holm_full
    result["reject_holm"] = reject_full
    return result


def fmt(df: pd.DataFrame) -> str:
    cols = ["subfolder", "reference", "other", "n", "m", "n_pairs",
            "mean_ref", "mean_other", "mean_diff", "ci_low", "ci_high",
            "rank_biserial", "p_raw", "p_holm", "reject_holm", "direction"]
    return df[cols].to_string(index=False)


def main():
    df = load_clean_external_results()
    print(f"[dados] {df['member'].nunique()} instancias x {df['method'].nunique()} "
          f"metodos = {len(df)} linhas (100% da bateria externa, deduplicada)\n")

    SOTA = ["multifit", "sa_li_perkins", "bco"]
    ABLATIONS = ["ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql_flat"]
    ILS = ["ils_random", "ils_ucb", "ils_ql", "ils_rf"]

    comparisons = [
        ("SOTA", SOTA, "ig_ql vs. baselines SOTA publicados (MULTIFIT, SA Li&Perkins 2024, BCO Davidovic et al. 2012)"),
        ("ABLATION", ABLATIONS, "ig_ql vs. ablacoes IG (isola a contribuicao do Q-Learning dentro do mecanismo IG)"),
        ("ILS", ILS, "ig_ql (mecanismo IG) vs. familia ILS (isola mecanismo de busca: IG vs. ILS)"),
    ]

    all_results = {}
    lines = []
    lines.append("=" * 100)
    lines.append("VALIDACAO EXTERNA (Symmetry 2025, familias B/C/E/F, 1.500 instancias) -- ig_ql vs. SOTA/ablacoes/ILS")
    lines.append("Holm-Bonferroni aplicado SEPARADAMENTE por familia externa (B, C, E, F) e por tipo de comparacao")
    lines.append("=" * 100)

    summary_rows = []
    for fam in ["B", "C", "E", "F"]:
        fam_df = df[df["family"] == fam]
        n_subfolders = fam_df["subfolder"].nunique()
        n_inst = fam_df["member"].nunique()
        lines.append(f"\n{'#'*100}")
        lines.append(f"# FAMILIA EXTERNA {fam}  ({n_subfolders} subpastas/classes, {n_inst} instancias, "
                      f"n={fam_df['n'].min()}-{fam_df['n'].max()})")
        lines.append(f"{'#'*100}")

        for tag, others, desc in comparisons:
            key = f"{fam}_{tag}"
            res = family_by_subfolder(fam_df, "ig_ql", others, family_name=key)
            all_results[key] = res
            lines.append(f"\n--- {fam} / {tag}: {desc} ---")
            lines.append(f"    testes com p valido: {res['p_raw'].notna().sum()} / {len(res)}")
            lines.append(fmt(res))

            raw_sig = (res["p_raw"] < 0.05).sum()
            holm_sig = int(res["reject_holm"].sum())
            summary_rows.append(dict(familia=fam, comparacao=tag, n_testes=len(res),
                                       sig_bruto=int(raw_sig), sig_holm=holm_sig))

    lines.append(f"\n{'='*100}")
    lines.append("RESUMO GERAL: significancia bruta vs. pos-Holm, por familia externa x tipo de comparacao")
    lines.append("="*100)
    summary_df = pd.DataFrame(summary_rows)
    lines.append(summary_df.to_string(index=False))

    # direcao das comparacoes Holm-significativas: ig_ql favoravel ou desfavoravel?
    lines.append(f"\n{'='*100}")
    lines.append("DIRECAO das comparacoes Holm-significativas (ig_ql melhor = mean_ref < mean_other, i.e. 'ref_better'):")
    lines.append("="*100)
    for key, res in all_results.items():
        sig = res[res["reject_holm"]]
        if len(sig):
            fav = (sig["direction"] == "ref_better").sum()
            unfav = (sig["direction"] == "ref_worse").sum()
            lines.append(f"  {key:12s}: {len(sig):2d} sig. Holm -- ig_ql favoravel: {fav:2d} | ig_ql desfavoravel: {unfav:2d}")
        else:
            lines.append(f"  {key:12s}: 0 sig. Holm")

    text = "\n".join(lines)
    print(text)
    out_path = ROOT / "results" / "entrada58_external_validation_report.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"\n\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
