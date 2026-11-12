"""
analysis/sim_crossfit_estimator.py
===================================
Entrada 257 (Ciclo 33, versao de periodico): FORMALIZACAO e VALIDACAO NUMERICA do estimador de folga com validacao cruzada de sementes.

Modelo. Instancia x, acoes a = 1..A, makespan relativo de uma execucao Y_a(x) = mu_a(x) + e, e ~ N(0, sigma^2) independente entre execucoes. Controle aleatorio: mu_rand(x) = a esperanca do makespan do controle aleatorio (que escolhe a acao a cada iteracao; nao e' a media das acoes fixas); na simulacao, mu_rand = media_a mu_a(x), o que nao altera nenhuma conclusao (mu_rand e' uma constante em relacao aos estimadores da escolha). Folga (o que um seletor que escolhe UMA acao por instancia, conhecendo as medias, ganharia sobre o aleatorio): H(x) = mu_rand(x) - min_a mu_a(x); H = E_x H(x).
Estimadores (ganho = reducao do makespan relativo, em pontos):
  H_um  (oraculo de uma semente, 'enviesado'): mu_rand - min_a Y_a^(1). E[min_a Y_a] <= min_a E[Y_a] (a esperanca do minimo nao excede o minimo das esperancas), logo E[H_um] >= H: ENVIESADO PARA CIMA (a maldicao do vencedor); com medias iguais, o vies e' sigma E[max de A normais padrao] ~ sigma sqrt(2 ln A).
  H_cv (validacao cruzada de sementes): escolhe a_hat = argmin_a (media de k1 sementes) e avalia essa acao em k2 sementes INDEPENDENTES: mu_rand - Y_{a_hat}^(k2). Dado a_hat, E = mu_rand - mu_{a_hat} <= H: um LIMITE INFERIOR, com vies = o arrependimento esperado da escolha, E[mu_{a_hat} - min_a mu_a] >= 0 (que diminui com k1).
  Logo E[H_cv] <= H <= E[H_um] (um INTERVALO para a folga verdadeira); com medias iguais (H = 0), E[H_cv] = 0 exato e E[H_um] = sigma E[max] > 0.
Experimento numerico (Monte Carlo, 20.000 instancias por configuracao): A em {5, 15}; as medias sao mu_a = tau * z_a com z_a ~ N(0,1) (tau = dispersao dos efeitos das acoes; tau = 0: nenhuma folga verdadeira); sigma em {0,5; 1; 2} (relativo a tau = 1 como escala: razao ruido/sinal); sementes (k1, k2) = (2, 2) como no artigo; tambem o vies de H_um quando as k sementes escolhem e as mesmas avaliam. Folga verdadeira H calculada exatamente na simulacao (media de mu_rand - min mu).
Validacao empirica: nos dados do R||Cmax (Entrada 234; 4 sementes por acao e instancia) o 'melhor de k sementes avaliado nas mesmas' decresce com k e converge a H_cv: o vies de selecao vem do ruido.
Saidas: results/entrada257_sim_crossfit.csv, results/entrada257_sim_crossfit.txt. Uso: python analysis/sim_crossfit_estimator.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
rng = np.random.default_rng(257)
N = 20_000


def simulate(A, tau, sigma, k1=2, k2=2):
    mu = tau * rng.standard_normal((N, A))
    mu_rand = mu.mean(axis=1)
    true_H = float((mu_rand - mu.min(axis=1)).mean())
    # oraculo de uma semente: o minimo de uma observacao por acao
    y1 = mu + sigma * rng.standard_normal((N, A))
    H_um = float((mu_rand - y1.min(axis=1)).mean())
    # oraculo de k1 sementes escolhendo e avaliando nas mesmas
    ymean = mu + sigma / np.sqrt(k1 + k2) * rng.standard_normal((N, A))
    H_mesmas = float((mu_rand - ymean.min(axis=1)).mean())
    # validacao cruzada: escolhe em k1, avalia em k2 independentes
    sel = (mu + sigma / np.sqrt(k1) * rng.standard_normal((N, A))).argmin(axis=1)
    ev = mu[np.arange(N), sel] + sigma / np.sqrt(k2) * rng.standard_normal(N)
    H_cv = float((mu_rand - ev).mean())
    regret = float((mu[np.arange(N), sel] - mu.min(axis=1)).mean())
    return dict(A=A, tau=tau, sigma=sigma, k1=k1, k2=k2, H_true=true_H, H_um=H_um, H_mesmas_k=H_mesmas, H_cv=H_cv, regret_cv=regret, bias_um=H_um - true_H, bias_cv=H_cv - true_H)


rows = []
for A in (5, 15):
    for tau in (0.0, 0.25, 1.0):
        for sigma in (0.5, 1.0, 2.0):
            rows.append(simulate(A, tau, sigma))
D = pd.DataFrame(rows)
D.to_csv(R / "entrada257_sim_crossfit.csv", index=False)

L = ["=" * 100, "SIMULACAO DO ESTIMADOR DE FOLGA COM VALIDACAO CRUZADA (Entrada 257)", "=" * 100,
     "H_true = folga verdadeira; H_um = oraculo de uma semente; H_cv = validacao cruzada (2 + 2 sementes); bias = estimador - H_true; regret = arrependimento da escolha", D.round(4).to_string(index=False)]
# checagens das duas desigualdades da teoria em todas as configuracoes
ok_up = bool((D.H_um >= D.H_true - 0.01).all())
ok_low = bool((D.H_cv <= D.H_true + 0.01).all())
zero = D[D.tau == 0.0]
ok_zero = bool((zero.H_cv.abs() < 0.03).all() and (zero.H_um > 0.5).all())
L += ["", f"E[H_um] >= H em todas as configuracoes (tolerancia de Monte Carlo 0,01): {ok_up}",
      f"E[H_cv] <= H em todas as configuracoes (tolerancia 0,01): {ok_low}",
      f"com tau = 0 (sem folga verdadeira): |H_cv| < 0,03 e H_um > 0,5 em todas: {ok_zero}",
      f"vies medio de H_um: {D.bias_um.mean():+.3f}; vies medio de H_cv: {D.bias_cv.mean():+.3f}; arrependimento medio da escolha: {D.regret_cv.mean():.3f}"]

# validacao empirica: R||Cmax (Entrada 234), 4 sementes por acao
emp = []
for sp in ("S5", "S15"):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    d["inst"] = d.cls + "#" + d.rep.astype(str)
    a = d[d.method.str.startswith("arm")].copy()
    a["arm"] = a.method.str[3:].astype(int)
    X = a.pivot_table(index=["inst", "arm"], columns="seed_i", values="makespan")
    insts = sorted(a.inst.unique())
    M = np.stack([X.loc[(i, slice(None))].sort_index().to_numpy() for i in insts])             # (instancia, acao, semente)
    rnd = d[d.method == "rand"].groupby("inst").makespan.mean().loc[insts].to_numpy()
    rel = lambda v: 100.0 * (rnd - v) / rnd                                                 # ganho relativo ao controle aleatorio (o agente que escolhe a cada iteracao, 5 sementes), em pontos
    ix = np.arange(len(insts))
    h1 = rel(M[:, :, 0].min(axis=1)).mean()
    h_same2 = rel(M[:, :, [0, 1]].mean(axis=2).min(axis=1)).mean()
    h_same4 = rel(M.mean(axis=2).min(axis=1)).mean()
    sel = M[:, :, [0, 1]].mean(axis=2).argmin(axis=1)
    h_cv = rel(M[ix, sel][:, [2, 3]].mean(axis=1)).mean()
    emp.append(dict(space=sp, instances=len(insts), um_semente=h1, melhor_de_2_mesmas=h_same2, melhor_de_4_mesmas=h_same4, cv_2mais2=h_cv))
E = pd.DataFrame(emp)
E.to_csv(R / "entrada257_empirical_decay.csv", index=False)
L += ["", "--- dados reais (R||Cmax, Entrada 234; 4 sementes por acao e instancia; ganho sobre o controle aleatorio do artigo, pontos) ---", E.round(4).to_string(index=False),
      "o oraculo cai a medida que mais sementes sao usadas para escolher e avaliar as mesmas, e a validacao cruzada fica abaixo de todos (limite inferior): "
      + str(bool(((E.um_semente > E.melhor_de_2_mesmas) & (E.melhor_de_2_mesmas > E.melhor_de_4_mesmas) & (E.melhor_de_4_mesmas > E.cv_2mais2)).all()))]
text = "\n".join(L)
print(text)
(R / "entrada257_sim_crossfit.txt").write_text(text + "\n", encoding="utf-8")
