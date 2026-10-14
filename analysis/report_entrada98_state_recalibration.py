"""
analysis/report_entrada98_state_recalibration.py
==================================================
Analise dos itens 1 e 2 da prioridade 0 (Entradas 97-98). Dados:
results/state_recalibration_entrada98.csv (experiment/run_state_recalibration_
entrada98.py) + results/results.csv (ig_random, ig_ql, ig_ql_flat, ig_ucb,
ils_random e ils_ucb das 6 classes later, todos N=100).

PRE-REGISTRO (congelado em 05/10/2026, ANTES de executar a bateria; os limiares
vem so' de instancias de calibracao, seeds 10000-10019, em
results/state_calibration_entrada98.json)
------------------------------------------------------------------------------
Unidade: a instancia (classe x rep, N=100 por classe, 12 classes). Desfecho:
gap_pct. Estatistica: para cada instancia, diferenca RELATIVA PADRONIZADA
d = (gap_metodo - gap_controle) / media_da_classe(gap_controle), que torna as
classes comparaveis; Wilcoxon signed-rank pareado bicaudal sobre todos os d
(1.200 instancias), diferencas nulas descartadas; effect size rank-biserial
(analysis.stats.rank_biserial_paired sobre os gaps; >0 => metodo PIOR);
mudanca relativa pooled = soma(gap_metodo - gap_controle) / soma(gap_controle).

ITEM 1 -- hipotese: o condicionamento por um estado INFORMATIVO muda o desempenho
do Q-Learning em relacao a selecao aleatoria.
  Familia IG  (Holm sobre 2, alpha 0,05): ig_ql_cal vs ig_random; ig_ql_rel vs ig_random.
  Familia ILS (Holm sobre 2, alpha 0,05): ils_ql_cal vs ils_random; ils_ql_rel vs ils_random.
  Por familia: "EFEITO DO ESTADO" se algum braco tem p_holm<0,05 e d medio<0
  (melhor que aleatorio); "EFEITO CONTRARIO" se p_holm<0,05 e d medio>0;
  "NULO" caso contrario. Secundarias (descritivas, Holm separado): braco vs
  ig_ql_flat (IG); braco vs ig_ql / ils_ql de estado padrao; por classe.
  Diagnostico obrigatorio: ocupacao dos bins de f1/f2 e estados distintos por
  corrida de cada braco (se f1 e f2 continuarem >95% num unico bin, o braco
  NAO e' um teste valido de estado informativo e isso deve ser dito).

ITEM 2 -- hipotese: a pior performance do UCB1 original e' artefato da escala da
recompensa.
  (a) ils_ucb_norm vs ils_ucb (bruto): pooled sobre 12 classes;
  (b) ils_ql (estado padrao) vs ils_ucb_norm: pooled sobre as 6 classes originais
      (N=100), bicaudal. Holm sobre (a),(b).
  Veredito: alegacao "UCB1 consistentemente pior / QL vence UCB" RETRATADA se (a)
  p_holm<0,05 com ils_ucb_norm melhor E (b) nao favoravel ao ils_ql com p_holm<0,05;
  MANTIDA se (a) nao significativa; PARCIAL caso contrario. Descritivo: ig_ucb_norm
  vs ig_ucb; ils_ucb_norm vs ils_random.

Uso: python -m analysis.report_entrada98_state_recalibration
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

ORIG6 = ["rand_20j_4m", "rand_20j_6m", "rand_30j_8m", "rand_50j_10m", "rand_75j_15m", "rand_100j_20m"]
new = pd.read_csv(ROOT / "results" / "state_recalibration_entrada98.csv")
old = pd.read_csv(ROOT / "results" / "results.csv")
both = pd.concat([new.assign(src="new"), old.assign(src="old")], ignore_index=True)
order = old.groupby("instance").n.first().sort_values().index.tolist()
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
L: list[str] = []


def emit(s=""):
    print(s)
    L.append(s)


def series(method, cls=None, prefer_new=True):
    d = both[both.method == method]
    if cls is not None:
        d = d[d.instance == cls]
    if prefer_new:
        d = d.sort_values("src").drop_duplicates(["instance", "rep"], keep="first")   # 'new' < 'old'
    return d.set_index(["instance", "rep"]).gap_pct


def paired(method, control, classes):
    a, b = [], []
    for cls in classes:
        x, y = series(method, cls), series(control, cls)
        idx = x.index.intersection(y.index)
        a.append(x.loc[idx]); b.append(y.loc[idx])
    x, y = pd.concat(a), pd.concat(b)
    cm = y.groupby(level=0).mean()
    scale = y.index.get_level_values(0).map(cm).to_numpy()
    d = (x.to_numpy() - y.to_numpy()) / np.where(scale > 0, scale, np.nan)
    ok = ~np.isnan(d) & (d != 0)
    p = float(sstats.wilcoxon(d[ok]).pvalue) if ok.sum() >= 10 else float("nan")
    return dict(n=len(x), better=int((x.to_numpy() < y.to_numpy()).sum()), worse=int((x.to_numpy() > y.to_numpy()).sum()),
                tie=int((x.to_numpy() == y.to_numpy()).sum()), mean_rel_d=float(np.nanmean(d)),
                pooled_rel_change_pct=100 * float((x.sum() - y.sum()) / y.sum()),
                rank_biserial=rank_biserial_paired(x.to_numpy(), y.to_numpy()), p_raw=p)


def family(rows, label):
    t = pd.DataFrame(rows)
    h = holm_bonferroni(t.p_raw.tolist(), alpha=0.05)
    t["p_holm"], t["reject"] = h["p_adjusted"], h["reject"]
    emit(f"\n--- {label} (mean_rel_d<0 => metodo MELHOR que o controle; rank_biserial>0 => metodo PIOR) ---")
    emit(t.to_string(index=False))
    return t


emit("=" * 100)
emit("DIAGNOSTICO DO ESTADO (QL recalibrado): ocupacao dos bins f1/f2 e estados distintos por corrida")
emit("=" * 100)
diag = new[new.method.str.contains("_ql_(cal|rel)", regex=True)].copy()
for col, name in [("f1_bins", "f1"), ("f2_bins", "f2")]:
    arr = diag[col].str.split("|", expand=True).astype(float)
    diag[f"{name}_max_bin"] = arr.max(axis=1)
dd = diag.groupby("method").agg(distinct_states=("distinct_states", "mean"), f1_max_bin_share=("f1_max_bin", "mean"),
                                f2_max_bin_share=("f2_max_bin", "mean"), runs=("rep", "count"))
emit(dd.to_string())
emit("(participacao media do bin mais ocupado; 1,0 = feature constante; estado padrao = 1,0 em f1 e f2)")

emit("\n" + "=" * 100)
emit("ITEM 1 -- PRIMARIA")
emit("=" * 100)
verdicts = {}
for fw, controls in [("IG", "ig_random"), ("ILS", "ils_random")]:
    pre = fw.lower()
    rows = []
    for arm in ("cal", "rel"):
        r = paired(f"{pre}_ql_{arm}", controls, order)
        r.update(method=f"{pre}_ql_{arm}", vs=controls)
        rows.append(r)
    t = family(rows, f"familia {fw}: Holm sobre 2")
    better = t[(t.reject) & (t.mean_rel_d < 0)]
    worse = t[(t.reject) & (t.mean_rel_d > 0)]
    verdicts[fw] = "EFEITO DO ESTADO" if len(better) else ("EFEITO CONTRARIO" if len(worse) else "NULO")
    emit(f"VEREDITO {fw}: {verdicts[fw]}")

emit("\n" + "=" * 100)
emit("ITEM 1 -- SECUNDARIAS (descritivas)")
emit("=" * 100)
sec = []
for arm in ("cal", "rel"):
    for ctrl, m_ in [("ig_ql_flat", f"ig_ql_{arm}"), ("ig_ql", f"ig_ql_{arm}")]:
        r = paired(m_, ctrl, order); r.update(method=m_, vs=ctrl); sec.append(r)
    r = paired(f"ils_ql_{arm}", "ils_ql", ORIG6); r.update(method=f"ils_ql_{arm}", vs="ils_ql (6 orig.)"); sec.append(r)
family(sec, "secundarias (Holm sobre o conjunto)")
emit("\nmedia de gap por classe e metodo:")
cols = ["ig_random", "ig_ql", "ig_ql_flat", "ig_ql_cal", "ig_ql_rel", "ils_random", "ils_ql", "ils_ql_cal", "ils_ql_rel"]
tab = pd.DataFrame({c: [series(c, cl).mean() if len(series(c, cl)) else np.nan for cl in order] for c in cols}, index=order)
emit(tab.to_string())

emit("\n" + "=" * 100)
emit("ITEM 2 -- UCB normalizado")
emit("=" * 100)
a = paired("ils_ucb_norm", "ils_ucb", order); a.update(test="(a) ils_ucb_norm vs ils_ucb (12 classes)")
b = paired("ils_ql", "ils_ucb_norm", ORIG6); b.update(test="(b) ils_ql vs ils_ucb_norm (6 orig.)")
t2 = family([a, b], "primarias do item 2 (Holm sobre 2)")
a_ok = bool(t2.iloc[0].reject and t2.iloc[0].mean_rel_d < 0)
b_fav = bool(t2.iloc[1].reject and t2.iloc[1].mean_rel_d < 0)
v2 = "RETRATADA" if (a_ok and not b_fav) else ("MANTIDA" if not t2.iloc[0].reject else "PARCIAL")
emit(f"VEREDITO item 2 (alegacao 'UCB1 consistentemente pior / QL vence UCB'): {v2}")
desc = []
for m_, c_, cl in [("ig_ucb_norm", "ig_ucb", order), ("ils_ucb_norm", "ils_random", order), ("ils_ucb", "ils_random", ORIG6)]:
    r = paired(m_, c_, cl); r.update(method=m_, vs=c_); desc.append(r)
emit("\ndescritivas:")
emit(pd.DataFrame(desc).to_string(index=False))
tab2 = pd.DataFrame({c: [series(c, cl).mean() if len(series(c, cl)) else np.nan for cl in order]
                     for c in ["ils_random", "ils_ucb", "ils_ucb_norm", "ils_ql", "ig_ucb", "ig_ucb_norm"]}, index=order)
emit("\nmedia de gap por classe:")
emit(tab2.to_string())

(ROOT / "results" / "entrada98_state_recalibration_report.txt").write_text("\n".join(L), encoding="utf-8")
print("\nsalvo em results/entrada98_state_recalibration_report.txt")
