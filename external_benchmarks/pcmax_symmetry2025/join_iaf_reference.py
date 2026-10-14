"""
join_iaf_reference.py
======================
Liga as solucoes de referencia do iAF (Results_iAF.xlsx, metodo exato de
fluxo em arcos iterativo -- Ostojic et al. 2025) as instancias .dat das
familias C e F do benchmark Symmetry 2025, e VERIFICA o join por evidencia
independente. Entrada 90, item 4. (A Entrada 52 havia descartado o join
por falta de confianca na correspondencia arquivo-linha.)

Regras de ligacao:
  * F: a planilha (abas FFU/FFNU) traz o NOME DO ARQUIVO (coluna File) --
    join exato.
  * C: abas C_<classe> com colunas Class, Run, n, m, LB, Z; o arquivo e'
    C/C<classe>/C_<classe>_<NNN>.dat. A hipotese Run == NNN e' aceita para
    uma aba SO' se TODAS as condicoes abaixo valerem (verificacao
    independente, nao presuncao):
      (a) n,m do arquivo coincidem com os da planilha em 100% das linhas;
      (b) LB_sheet >= LB_ours em 100% das linhas (nosso LB e' o limite
          classico fraco; a planilha pode usar um mais forte, nunca
          mais fraco);
      (c) ZERO violacoes de "makespan do melhor dos nossos metodos < Z"
          nas instancias da aba que ja foram executadas (Z e' uma
          solucao viavel; se o join estivesse errado, ~40% das linhas
          violariam -- medido empiricamente com deslocamento +1 como
          controle: 72/160, 105/279, 37/120 ... violacoes).
    Valores de LB identicos em 100% das linhas (abas C_5, C_8..C_14) sao
    evidencia adicional, nao requisito.

Verificacoes adicionais impressas: (i) fracao de linhas em que Z < nosso
LB (impossivel se Z e' viavel); (ii) fracao em que a planilha marca
Gap > 0 (iAF nao provou otimalidade -- Z e' so' um limite superior).

Saida: iaf_reference.csv (family, member, class_sheet, Z, LB_sheet,
LB_ours, gap_sheet, n, m).

Uso: python join_iaf_reference.py
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
XL = pd.ExcelFile(HERE / "Results_iAF.xlsx")
_res = pd.read_csv(HERE / "results_external_symmetry2025.csv").drop_duplicates(["member", "method"])
BEST_MK = _res.groupby("member").makespan.min()   # melhor makespan entre os metodos ja executados


def parse_dat(text: str):
    t = text.split()
    m, n = int(t[0]), int(t[1])
    p = [int(x) for x in t[2:2 + n]]
    return m, n, p


def our_lb(m: int, p: list[int]) -> int:
    s = sum(p)
    return max(max(p), -(-s // m))


rows = []
# ---------------- F ----------------
zf = zipfile.ZipFile(HERE / "F.zip")
fmembers = {Path(n).name: n for n in zf.namelist() if n.endswith(".dat")}
for sh in ["FFU", "FFNU"]:
    d = XL.parse(sh)
    for r in d.itertuples(index=False):
        name = f"{r.File}.dat"
        if name not in fmembers:
            continue
        m, n, p = parse_dat(zf.read(fmembers[name]).decode())
        rows.append(dict(family="F", member=fmembers[name], class_sheet=sh, Z=r.Z, LB_sheet=r.LB,
                         LB_ours=our_lb(m, p), gap_sheet=r.Gap, n=n, m=m))
print(f"F: {sum(1 for r in rows if r['family']=='F')} linhas ligadas por nome de arquivo")

# ---------------- C ----------------
zc = zipfile.ZipFile(HERE / "C.zip")
cnames = {n for n in zc.namelist() if n.endswith(".dat")}
cache: dict[str, tuple] = {}


def get(member):
    if member not in cache:
        cache[member] = parse_dat(zc.read(member).decode())
    return cache[member]


for sh in [s for s in XL.sheet_names if s.startswith("C_")]:
    d = XL.parse(sh)
    cls = int(sh.split("_")[1])
    ok_nm = ok_lb = lb_ge = tot = viol = n_sampled = 0
    viol_ctrl = n_ctrl = 0
    cand = []
    for r in d.itertuples(index=False):
        member = f"C/C{cls}/C_{cls}_{int(r.Run):03d}.dat"
        if member not in cnames:
            continue
        m, n, p = get(member)
        l = our_lb(m, p)
        tot += 1
        ok_nm += int(n == r.n and m == r.m)
        ok_lb += int(l == r.LB)
        lb_ge += int(r.LB >= l)
        if member in BEST_MK.index:
            n_sampled += 1
            viol += int(BEST_MK[member] < r.Z)
        ctrl = f"C/C{cls}/C_{cls}_{int(r.Run) + 1:03d}.dat"     # controle: deslocamento errado
        if ctrl in BEST_MK.index:
            n_ctrl += 1
            viol_ctrl += int(BEST_MK[ctrl] < r.Z)
        cand.append(dict(family="C", member=member, class_sheet=sh, Z=r.Z, LB_sheet=r.LB,
                         LB_ours=l, gap_sheet=r.Gap, n=n, m=m))
    accepted = (tot == len(d) and ok_nm == tot and lb_ge == tot and viol == 0)
    print(f"{sh}: linhas={len(d)} ligadas={tot} n,m ok={ok_nm} LB identico={ok_lb} ({100 * ok_lb / max(tot, 1):.1f}%) "
          f"LB_sheet>=ours={lb_ge} | violacoes Z>melhor makespan: {viol}/{n_sampled} (controle +1: {viol_ctrl}/{n_ctrl}) "
          f"-> {'ACEITA' if accepted else 'REJEITADA'}")
    if accepted:
        rows.extend(cand)

ref = pd.DataFrame(rows)
ref.to_csv(HERE / "iaf_reference.csv", index=False)
print(f"\niaf_reference.csv: {len(ref)} instancias ({ref.family.value_counts().to_dict()})")
print("Z < LB_ours (impossivel se Z e' viavel):", int((ref.Z < ref.LB_ours).sum()))
print("LB_sheet == LB_ours:", f"{100 * (ref.LB_sheet == ref.LB_ours).mean():.2f}%")
print("linhas com Gap>0 na planilha (otimalidade nao provada):", int((ref.gap_sheet > 0).sum()))
