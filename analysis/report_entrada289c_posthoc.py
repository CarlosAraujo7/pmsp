"""
analysis/report_entrada289c_posthoc.py (Entrada 289c, Ciclo 55) -- POS HOC
============================================================================
O relatorio pre-registrado da Entrada 289 deu 3 de 12 previsoes. Esta analise e' POS HOC (nao ha' previsoes; escrita depois de ver o resultado) e responde a tres perguntas que o resultado levanta:
 (i)  PODER. Com 10 instancias por funcao o Wilcoxon exato tem p minimo 2/1024 = 0,00195; a familia de Holm de 168 comparacoes (D5) exige p <= 0,05/168 = 0,0003 e a de 24 (D1, D6) exige 0,00208: a regra estrita pre-registrada
      quase nao pode rejeitar. Aqui: (a) o limite do teste; (b) por aprendiz, o ganho mediano e o numero de funcoes com ganho positivo, com IC90 por instancias inteiro alem de +-0,05 SEM correcao, e o teste de postos sinalizados
      sobre os 24 ganhos por funcao (a funcao como unidade).
 (ii) DESALINHAMENTO. A acao de maior taxa de sucesso (a que o credito por sucesso premia) e' a mesma que a melhor para o erro final? Fracao das funcoes em que coincidem; participacao de cada aprendiz na acao validada e na
      acao de maior sucesso.
 (iii) O aprisionamento do Q-Learning com recompensa DENSA: participacao da acao 0 (padrao) e da acao de maior participacao por funcao.
Fontes: results/de_*_entrada289.csv, results/logs289/f*_rand.npz. Uso: python -m analysis.report_entrada289c_posthoc
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
LEARNERS = ["ql_std", "ql_rt", "ucb", "pg", "pm", "ap", "etc"]
MARGIN = 0.05


def boot_ci(d, seed=2891):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(2000, len(d)))
    return np.percentile(d[idx].mean(axis=1), [5, 95])


def main():
    tune = pd.read_csv(R / "de_tune_entrada289.csv")
    test = pd.concat([pd.read_csv(R / "de_test_base_entrada289.csv"), pd.read_csv(R / "de_test_learn_entrada289.csv")], ignore_index=True)
    val_arm = {f: int(g.groupby("method").lg.mean().idxmin()[3:]) for f, g in tune.groupby("fun")}
    inst = test.groupby(["fun", "inst", "method"]).agg(lg=("lg", "mean"), rew=("rew", "mean"), **{f"u{k}": (f"u{k}", "mean") for k in range(5)}).reset_index()
    LG = inst.pivot(index=["fun", "inst"], columns="method", values="lg")
    funs = sorted(LG.index.get_level_values(0).unique())
    L = ["=" * 112, "POS HOC: PODER, DESALINHAMENTO DA RECOMPENSA E APRISIONAMENTO COM RECOMPENSA DENSA (Entrada 289c)", "=" * 112,
         f"(i-a) limite do teste: Wilcoxon exato com 10 instancias, p minimo = {2 / 1024:.5f}; Holm sobre 24 comparacoes exige <= {0.05 / 24:.5f}; sobre 168, <= {0.05 / 168:.5f}", ""]
    # ---- (i-b) por aprendiz
    rows = []
    for m in LEARNERS:
        g, up, down, sgn = [], 0, 0, 0
        for f in funs:
            x = LG.loc[f]
            d = (x.rand - x[m]).to_numpy()
            lo, hi = boot_ci(d)
            g.append(d.mean())
            up += int(lo > MARGIN)
            down += int(hi < -MARGIN)
        g = np.array(g)
        nz = g[g != 0]
        p = float(sstats.wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
        rows.append(dict(aprendiz=m, mediana_ganho=float(np.median(g)), media_ganho=float(g.mean()), funcoes_ganho_positivo=int((g > 0).sum()), funcoes_ganho_negativo=int((g < 0).sum()),
                         IC90_acima_da_margem=up, IC90_abaixo_da_margem=down, p_postos_sinalizados=p, pior_que_aleatorio_em_mais_de_1_decada=int((g < -1).sum())))
    S = pd.DataFrame(rows)
    L += ["(i-b) ganho sobre o aleatorio (decadas; positivo = melhor); IC90 por instancias SEM correcao; teste de postos sinalizados sobre os 24 ganhos por funcao:", S.round(4).to_string(index=False), ""]
    # ---- (ii) desalinhamento
    rows = []
    for f in funs:
        z = np.load(R / "logs289" / f"f{f:02d}_rand.npz")
        A, Rw = z["A"], z["Rw"].astype(float)
        succ = np.array([Rw[A == k].mean() for k in range(5)])
        a_rew = int(np.argmax(succ))
        row = dict(fun=f, val=val_arm[f], arm_sucesso=a_rew, coincidem=val_arm[f] == a_rew, sucesso_val=float(succ[val_arm[f]]), sucesso_max=float(succ.max()))
        u = inst[inst.fun == f].groupby("method")[[f"u{k}" for k in range(5)]].mean()
        for m in LEARNERS:
            row[f"{m}_na_val"] = float(u.loc[m, f"u{val_arm[f]}"])
            row[f"{m}_na_sucesso"] = float(u.loc[m, f"u{a_rew}"])
        rows.append(row)
    D = pd.DataFrame(rows)
    L += [f"(ii) a acao de maior taxa de sucesso coincide com a acao validada (melhor erro final) em {int(D.coincidem.sum())} de 24 funcoes;",
          "     participacao media dos aprendizes (todas as funcoes): " + "; ".join(f"{m}: na validada {D[f'{m}_na_val'].mean():.3f}, na de maior sucesso {D[f'{m}_na_sucesso'].mean():.3f}" for m in LEARNERS) + "  (aleatorio: 0,200)", ""]
    # nas funcoes em que a validada rende (ganho >= 0,1) e as duas acoes divergem
    gv = {f: float((LG.loc[f].rand - LG.loc[f][f"arm{val_arm[f]}"]).mean()) for f in funs}
    sub = D[(D.fun.map(gv) >= 0.1) & (~D.coincidem)]
    L += [f"     nas {len(sub)} funcoes em que a validada rende >= 0,1 decada e a acao de maior sucesso e' OUTRA: " + "; ".join(f"{m}: na validada {sub[f'{m}_na_val'].mean():.3f}, na de maior sucesso {sub[f'{m}_na_sucesso'].mean():.3f}" for m in LEARNERS), ""]
    # ---- (iii) aprisionamento
    u0 = inst[inst.method == "ql_std"].groupby("fun").u0.mean()
    umax = {}
    for m in ("ql_std", "ql_rt"):
        u = inst[inst.method == m].groupby("fun")[[f"u{k}" for k in range(5)]].mean()
        umax[m] = float(u.max(axis=1).mean())
    L += [f"(iii) participacao da acao 0 no ql_std (todas as funcoes): {u0.mean():.3f} (min {u0.min():.3f}, max {u0.max():.3f}); participacao da acao MAIS usada: ql_std {umax['ql_std']:.3f}, ql_rt {umax['ql_rt']:.3f} (aleatorio 0,200)"]
    text = "\n".join(L)
    print(text)
    (R / "entrada289c_posthoc.txt").write_text(text + "\n", encoding="utf-8")
    S.to_csv(R / "entrada289c_learners.csv", index=False)
    D.to_csv(R / "entrada289c_alignment.csv", index=False)


if __name__ == "__main__":
    main()
