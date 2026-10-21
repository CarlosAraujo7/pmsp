"""
analysis/report_entrada176_ilsdqn.py
======================================
Aprendizes maiores sobre acoes QUALITATIVAMENTE DIFERENTES: os quatro operadores de vizinhanca do ILS de P||Cmax (Ciclo 14, frente I2). Diario: Entrada 176 (pre-registro) e 177 (resultados).

PRE-REGISTRO (congelado ANTES de executar o ajuste do lr e o teste; hashes em results/entrada176_prereg_hashes.txt)
-------------------------------------------------------------------------------------------------------------------
Contexto: o limite restante do artigo e' que os espacos de acao testados (estrategias e tamanhos de destruicao) se parecem entre si. Os quatro operadores do ILS diferem em efeito e custo (CPU por execucao de 0,15 s a 5,3 s com laços Python), e nos dados armazenados
(results/results.csv, 780 instancias com os quatro operadores unicos) a mistura aleatoria dos quatro supera cada operador unico por classe, e o oraculo por instancia entre os quatro operadores unicos e' PIOR que a mistura aleatoria em +0,038% (inspecionado ANTES do pre-registro; so' dados armazenados,
nenhum dado novo): a folga de 'escolher um operador' e' negativa, e so' uma politica que SEQUENCIE operadores (dependente do estado) poderia superar a mistura. Aprendizes: DQN 2 x 64 (pmsp/dqn.py; lr em {0,001; 0,003; 0,01} ajustado em 30 instancias/classe de seeds 90000+) e LinUCB (alpha = 1), ambos escolhendo o operador a cada iteracao;
comparador: ILS aleatorio vetorizado a T = 1000 (results/ils_eq_fast_entrada162.csv, metodo ils_fast_eqT; mesmas instancias e seeds). Braco validado: melhor operador UNICO por classe nas instancias de ajuste. Dados novos: results/ilsdqn_run_entrada176.csv (12 classes x 100 instancias x 3 metodos).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90 (2000). Equivalencia a 0: IC90 dentro de +-0,05%.
Familia primaria (Holm sobre 2, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0; se confirmada com IC90 dentro de +-0,05 a leitura e' 'paridade', licao II.24):
  K1  dqn_ils vs rand (o DQN maior capta algo sobre a mistura aleatoria)        K2  linucb_ils vs rand
Descritivas (fora da familia): val_ils (operador unico validado) vs rand e vs dqn_ils; DQN e LinUCB vs o Q-learning tabular (ils_ql armazenado, 780 instancias em comum); razao de CPU sequencial de cada aprendiz sobre o aleatorio (ilsdqn_cpu_ratio_entrada176.json); ganho por classe.
Leitura pre-fixada: 'um aprendiz maior sobre acoes qualitativamente diferentes bate a mistura aleatoria' = K1 ou K2 confirmada e nao paridade (IC90 fora de +-0,05).
Previsoes registradas antes de rodar: K1 nao confirmada com d em [-0,03; +0,15]%; K2 nao confirmada com d em [-0,03; +0,30]%; o operador unico validado e' PIOR que a mistura aleatoria (d em [+0,02; +0,30]%); razao de CPU dqn/aleatorio em [1,0; 1,6] e linucb/aleatorio em [0,9; 1,3]; a leitura e' NAO.
Uso: python -m analysis.report_entrada176_ilsdqn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.stats import holm_bonferroni, rank_biserial_paired  # noqa: E402

R = ROOT / "results"


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def test(x, y):
    d = rel(x, y)
    nz = d[d != 0]
    p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    idx = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [5, 95])
    return dict(n=len(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), tie=int((d == 0).sum()), mean_d_pct=float(d.mean()), ci90_lo=float(lo), ci90_hi=float(hi),
                rank_biserial=rank_biserial_paired(x, y), p_raw=p)


def main() -> None:
    new = pd.read_csv(R / "ilsdqn_run_entrada176.csv")
    rnd = pd.read_csv(R / "ils_eq_fast_entrada162.csv")
    rnd = rnd[rnd.method == "ils_fast_eqT"][["instance", "rep", "makespan"]].rename(columns={"makespan": "rand"})
    w = new.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index().merge(rnd, on=["instance", "rep"])
    assert len(w) == 1200 and w[["dqn_ils", "linucb_ils", "val_ils", "rand"]].notna().all().all(), (len(w), w.isna().sum().to_dict())
    ch = json.loads((R / "ilsdqn_choice_entrada176.json").read_text())
    cpu = json.loads((R / "ilsdqn_cpu_ratio_entrada176.json").read_text())
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "APRENDIZES MAIORES SOBRE OPERADORES QUALITATIVAMENTE DIFERENTES (ILS de P||Cmax) -- Entrada 176/177", "=" * 110,
         f"lr do DQN escolhido no ajuste: {ch['lr']} ({ch['lr_mean_gap']}); operador unico validado por classe (0 = inter-troca, 1 = reinsercao, 2 = intra/permuta, 3 = bloco): {ch['val']}"]
    rows = []
    for pid, lab, a in [("K1", "dqn_ils vs aleatorio", "dqn_ils"), ("K2", "linucb_ils vs aleatorio", "linucb_ils")]:
        t = test(w[a], w["rand"])
        t.update(id=pid, comparison=lab)
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["equivalente_0.05"] = (P.ci90_lo > -0.05) & (P.ci90_hi < 0.05)
    P["leitura"] = np.where(P.confirmada & P["equivalente_0.05"], "paridade (detectavel mas equivalente a zero)", np.where(P.confirmada, "confirmada", "nao confirmada"))
    L += ["\n--- PRIMARIA (Holm sobre 2) ---", P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]].to_string(index=False)]
    sec = []
    for lab, a, b in [("val_ils vs aleatorio (operador unico validado)", "val_ils", "rand"), ("dqn_ils vs val_ils", "dqn_ils", "val_ils"), ("linucb_ils vs val_ils", "linucb_ils", "val_ils")]:
        t = test(w[a], w[b])
        sec.append(dict(comparacao=lab, mean_d_pct=t["mean_d_pct"], ci90_lo=t["ci90_lo"], ci90_hi=t["ci90_hi"], melhor=t["better"], pior=t["worse"], p_raw=t["p_raw"]))
    L += ["\n--- descritivas ---", pd.DataFrame(sec).to_string(index=False)]
    old = pd.read_csv(R / "results.csv")
    ops = ["ils_op1_only", "ils_op2_only", "ils_op3_only", "ils_op4_only"]
    o = old[old.method.isin(ops + ["ils_ql", "ils_random"])].pivot_table(index=["instance", "rep"], columns="method", values="makespan").dropna().reset_index()
    o = o.merge(w[["instance", "rep", "dqn_ils", "linucb_ils"]], on=["instance", "rep"])
    t1, t2, t3 = test(o.ils_ql, o.ils_random), test(o.dqn_ils, o.ils_random), test(o.linucb_ils, o.ils_random)
    orc = rel(o[ops].min(axis=1), o.ils_random).mean()
    L.append(f"\nSobre as {len(o)} instancias com o ILS armazenado (loop; resultados identicos aos vetorizados): Q-learning tabular vs aleatorio {t1['mean_d_pct']:+.4f}%; DQN {t2['mean_d_pct']:+.4f}%; LinUCB {t3['mean_d_pct']:+.4f}%; oraculo por instancia entre os quatro operadores unicos vs aleatorio {orc:+.4f}% (positivo: pior que a mistura)")
    cl = pd.DataFrame({"inst": w.instance, "dqn": rel(w.dqn_ils, w.rand), "linucb": rel(w.linucb_ils, w.rand), "val": rel(w.val_ils, w.rand)}).groupby("inst").mean()
    L += ["\n--- por classe (d % sobre o aleatorio) ---", cl.to_string()]
    cr = pd.DataFrame(cpu).T
    L.append("\n--- razao de CPU sobre o aleatorio (sequencial, mediana de 3) ---\n" + cr.to_string() + f"\nmedias geometricas: dqn {np.exp(np.log(cr.dqn_ils).mean()):.2f}, linucb {np.exp(np.log(cr.linucb_ils).mean()):.2f}")
    beat = bool(((P.confirmada) & (~P["equivalente_0.05"])).any())
    L.append(f"\nLeitura pre-fixada: 'um aprendiz maior sobre acoes qualitativamente diferentes bate a mistura aleatoria' (K1 ou K2 confirmada e nao paridade): {'SIM' if beat else 'NAO'}")
    text = "\n".join(L)
    print(text)
    (R / "entrada176_ilsdqn.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada176_ilsdqn_primary.csv", index=False)


if __name__ == "__main__":
    main()
