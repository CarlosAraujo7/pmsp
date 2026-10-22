"""
analysis/power_entrada184.py (Entrada 184, Ciclo 16, frente A1)
=================================================================
PODER E EFEITO MINIMO DETECTAVEL do banco de teste do artigo (criterio 'sensibilidade / controle positivo' da nota do CEC): o resultado 'nenhum aprendiz ganha do aleatorio a iteracoes iguais' so' vale se o teste teria detectado um ganho de politica
da ordem dos que a literatura relata. Sem execucao nova: usa as diferencas pareadas por instancia de results/agents_S{3,5,15,30}_entrada180.csv (12 comparacoes aprendiz x aleatorio a T = 1000, unidade = media das 5 sementes, 1.200 pares).
Delineamento: (i) MDE analitico a 80% de poder: (z_{1 - alfa/2} + 0,84) * EP, com EP = dp(d)/raiz(n), alfa = 0,05 e alfa = 0,05/12 (pior caso de Holm sobre 12); (ii) poder EMPIRICO por efeito plantado: reamostra instancias com reposicao (B = 500, semente 0) e planta um
ganho de delta pontos percentuais fazendo o aprendiz reduzir o makespan de UMA unidade (todas as sementes) em cada instancia com probabilidade pi = delta / media(u), u_i = 100 / makespan_aleatorio_i (o efeito de uma unidade, em %), de modo que o ganho esperado e' delta e os empates
continuam empates onde nao ha' melhora; detecta quando o Wilcoxon bilateral (empates descartados) tem p < 0,05/12 e a media do d' e' negativa; delta em {0,002; 0,005; 0,01; 0,015; 0,02; 0,03; 0,05}; (iii) CONTROLE POSITIVO REAL: a acao validada por classe (uma politica estatica escolhida com informacao de classe) contra o
aleatorio, nos mesmos 1.200 pares, deve ser detectada com folga.
PREVISOES (congeladas antes de rodar; o hash do script em results/entrada184_prereg_hashes.txt): MDE analitico a 80% (alfa = 0,05) entre 0,005 e 0,020 pp em todas as 12 comparacoes; poder empirico (alfa = 0,05/12) em delta = 0,02: >= 90% em todas as 12;
em delta = 0,01: entre 50% e 95% na mediana das 12; em delta = 0,005: <= 40% na mediana; o controle positivo real (validada vs aleatorio) tem d < -0,05% e p_holm < 1e-20 em S5 e S15.
TRANSPARENCIA: a Entrada 183 previa 'delta = 0,01 detectado em 40-80%' depois de ja' ter calculado o MDE analitico de 0,006-0,014 pp (mediana 0,009), o que e' incoerente com ele; esta e' a previsao em vigor. A leitura pre-fixada 'o banco detecta ganhos de politica de 0,02 pp ou
mais' e' SIM se a previsao de delta = 0,02 se cumprir.
Uso: python -m analysis.power_entrada184
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SPACES = ["S3", "S5", "S15", "S30"]
LEARNERS = ["ql", "ucb", "pg"]
DELTAS = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
B = 500
ALPHA = 0.05


def per_instance(space):
    d = pd.read_csv(R / f"agents_{space}_entrada180.csv")
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    return ms


def detect(dp, alpha):
    nz = dp[dp != 0]
    if len(nz) < 6:
        return False
    return bool(sstats.wilcoxon(nz).pvalue < alpha and dp.mean() < 0)


def main():
    rng = np.random.default_rng(0)
    rows, ctrl = [], []
    for sp in SPACES:
        ms = per_instance(sp)
        base = ms["rand"].to_numpy(float)
        u = 100.0 / base
        for k in LEARNERS:
            a = ms[k].to_numpy(float)
            d = 100.0 * (a - base) / base
            n = len(d)
            se = d.std(ddof=1) / np.sqrt(n)
            mde = (sstats.norm.ppf(1 - ALPHA / 2) + sstats.norm.ppf(0.8)) * se
            mde_h = (sstats.norm.ppf(1 - ALPHA / 24) + sstats.norm.ppf(0.8)) * se
            rec = dict(space=sp, learner=k, n=n, mean_d=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se), mde80=float(mde), mde80_holm12=float(mde_h))
            idx = rng.integers(0, n, size=(B, n))
            for delta in DELTAS:
                pi = min(1.0, delta / float(u.mean()))
                hit = 0
                for b in range(B):
                    j = idx[b]
                    plant = rng.random(n) < pi
                    dp = d[j] - np.where(plant, u[j], 0.0)
                    hit += detect(dp, ALPHA / 12)
                rec[f"power_{delta}"] = hit / B
            e = next((dl for dl in DELTAS if rec[f"power_{dl}"] >= 0.8), float("nan"))
            rec["mde80_empirical"] = e
            rows.append(rec)
        if sp in ("S5", "S15", "S3", "S30"):
            full = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
            val = full[full.method == "val"].groupby(["cls", "rep"]).makespan.mean().to_numpy(float)
            dv = 100.0 * (val - base) / base
            nz = dv[dv != 0]
            ctrl.append(dict(space=sp, n=len(dv), mean_d=float(dv.mean()), p_raw=float(sstats.wilcoxon(nz).pvalue), better=int((dv < 0).sum()), worse=int((dv > 0).sum())))
    P = pd.DataFrame(rows)
    C = pd.DataFrame(ctrl)
    P.to_csv(R / "entrada184_power.csv", index=False)
    C.to_csv(R / "entrada184_power_control.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    out = ["=" * 110, "PODER E EFEITO MINIMO DETECTAVEL -- Entrada 184", "=" * 110,
           "12 comparacoes aprendiz vs aleatorio a T = 1000 (1.200 pares cada); poder empirico com alfa = 0,05/12 (pior caso de Holm), B = 500", "",
           P.round(4).to_string(index=False), "",
           f"MDE80 analitico (alfa 0,05): min {P.mde80.min():.4f}, mediana {P.mde80.median():.4f}, max {P.mde80.max():.4f} pp; (alfa 0,05/12): min {P.mde80_holm12.min():.4f}, mediana {P.mde80_holm12.median():.4f}, max {P.mde80_holm12.max():.4f}",
           f"MDE80 empirico (menor delta com poder >= 80%): min {P.mde80_empirical.min()}, mediana {P.mde80_empirical.median()}, max {P.mde80_empirical.max()}",
           f"poder em delta = 0,02: min {P['power_0.02'].min():.3f}; em 0,01: mediana {P['power_0.01'].median():.3f}; em 0,005: mediana {P['power_0.005'].median():.3f}", "",
           "--- controle positivo real: acao validada por classe vs aleatorio (T = 1000) ---", C.round(5).to_string(index=False)]
    (R / "entrada184_power.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
