"""
analysis/report_entrada173_dqn.py
===================================
Aprendiz neural maior (DQN 2 x 64 com replay e rede-alvo) e espaco de acao conjunto estrategia x tamanho (Ciclo 14, frente I). Diario: Entrada 173 (pre-registro) e 175 (resultados).

PRE-REGISTRO (congelado ANTES de executar as fases de ajuste do lr e de teste; hashes em results/entrada173_prereg_hashes.txt)
--------------------------------------------------------------------------------------------------------------------------------
Limite (i) do artigo: 'o aprendiz neural era pequeno e os espacos de acao, pequenos'. Aprendiz: pmsp/dqn.py (5 testes; gradiente conferido por diferencas finitas); unico hiperparametro ajustado: lr in {0,001; 0,003; 0,01}, nas instancias de ajuste
de cada ambiente (menor razao makespan/melhor conhecido em R; menor gap medio em P5 e P15). Ambientes e dados (experiment/run_dqn_entrada173.py):
  R   LP-IG em R||Cmax, d/n in {0,01; ...; 0,20}: 700 instancias de teste (results/dpolicy_test_entrada145.csv: rand_d, fix_0..4, ql_d, ucb_d; nq_test_entrada162.csv: nq_d pequeno; dqn_test_r_entrada173.csv: dqn_big); val_tn = braco validado por (tipo, n) no ajuste.
  P5  IG em P||Cmax, d/n in {0,05; ...; 0,40}: 12 classes x 100 (results/dsel_test_entrada114.csv: igd_fix05..40, igd_random, igd_ql_def, igd_ucb; dqn_test_p5_entrada173.csv: dqn_d); val5 = melhor braco fixo por classe no ajuste (dsel_tuning_entrada114.csv).
  P15 espaco conjunto estrategia (3) x tamanho (5) = 15 acoes, as mesmas instancias (dqn_test_p15_entrada173.csv: rand15, ql15, linucb15, dqn15, val15; dqn_arms15_test_entrada173.csv: os 15 bracos fixos compilados, para o oraculo); val15 = melhor dos 15 bracos
      por classe no ajuste (compilado; o braco validado e' reexecutado em Python no teste, com a mesma semente dos aprendizes).
d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90 (2000). Equivalencia a 0: IC90 dentro de +-0,05%.
Familia primaria (Holm sobre 6, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0; se confirmada com IC90 dentro de +-0,05 a leitura e' 'paridade', regra da licao II.24):
  I1R  dqn_big vs rand_d     I2R  val_tn vs dqn_big     (R)
  I1P5 dqn_d vs igd_random   I2P5 val5 vs dqn_d         (P5)
  I1P15 dqn15 vs rand15      I2P15 val15 vs dqn15       (P15)
  I1 pergunta se a politica neural maior capta a folga; I2, se a validacao offline ainda a vence.
Descritivas (fora da familia): dqn_big vs nq_d (o tamanho da rede importa?); ql15 e linucb15 vs rand15 e vs val15; razao de captura (ganho sobre o aleatorio / ganho do braco validado) de todos os aprendizes por ambiente; folga (oraculo por instancia vs
aleatorio; em P15 o oraculo e' compilado e o aleatorio, Python: a equivalencia entre as duas implementacoes foi verificada na Entrada 171); Spearman(folga da celula, ganho do DQN) por ambiente (a lei da folga, de novo, prospectivamente); razao de CPU dqn/aleatorio; lr escolhido.
Leituras pre-fixadas:
  L1 'um aprendiz neural maior ou um espaco de acao maior muda a conclusao' = em ao menos um ambiente, I1 confirmada (e nao paridade) com captura >= 0,5 do ganho validado, OU I2 nao confirmada (o DQN empata ou vence a validacao).
  L2 'a falha do aprendizado online nao depende do tamanho do aprendiz nem do espaco de acao' = nenhum ambiente satisfaz L1 E I2 confirmada nos tres ambientes.
Previsoes registradas antes de rodar:
  R: I1R nao confirmada (d >= -0,05%) com captura <= 0,25; I2R confirmada com d em [-0,35; -0,10]%. P5: I1P5 nao confirmada (d em [-0,02; +0,15]%); I2P5 confirmada com d em [-0,20; -0,06]%. P15: folga do espaco conjunto (oraculo sobre o aleatorio) em [-0,70; -0,25]%;
  I1P15 nao confirmada com captura <= 0,25; I2P15 confirmada com d em [-0,35; -0,08]%. dqn_big vs nq_d: |d| < 0,05% (o tamanho nao importa). Spearman(folga, ganho do DQN) <= 0,3 em cada ambiente. Razao de CPU dqn/aleatorio: R <= 1,15; P5 e P15 em [1,2; 2,5]. L1 NAO; L2 SIM.
Uso: python -m analysis.report_entrada173_dqn
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
F = [0.01, 0.02, 0.05, 0.10, 0.20]
FIX5 = ["igd_fix05", "igd_fix10", "igd_fix20", "igd_fix30", "igd_fix40"]


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


def load_r():
    cols = ["type", "instance", "m", "n", "idx", "method", "makespan", "cpu_time"]
    d = pd.concat([pd.read_csv(R / "dpolicy_test_entrada145.csv", dtype={"instance": str})[cols], pd.read_csv(R / "nq_test_entrada162.csv", dtype={"instance": str})[cols],
                   pd.read_csv(R / "dqn_test_r_entrada173.csv", dtype={"instance": str})[cols]])
    w = d.pivot(index=["type", "instance", "m", "n"], columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=["type", "instance"], columns="method", values="cpu_time")
    assert len(w) == 700 and w[["rand_d", "dqn_big", "nq_d"]].notna().all().all(), (len(w), w.isna().sum().to_dict())
    tune = pd.read_csv(R / "lpig_tune_d_entrada141.csv", dtype={"instance": str})
    tune["frac"] = tune.method.str.split("@").str[1].astype(float)
    tune["ratio"] = tune.makespan / tune.best_known
    val = {(t, int(n)): F.index(float(g.groupby("frac").ratio.mean().idxmin())) for (t, n), g in tune.groupby(["type", "n"])}
    w["val"] = [r[f"fix_{val[(r['type'], int(r['n']))]}"] for _, r in w.iterrows()]
    w["oracle"] = w[[f"fix_{k}" for k in range(5)]].min(axis=1)
    w["cell"] = w["type"] + "|" + w["n"].astype(str)
    w = w.rename(columns={"rand_d": "rand", "dqn_big": "dqn"})
    cpu = cpu.rename(columns={"rand_d": "rand", "dqn_big": "dqn"})
    return w, cpu


def load_p5():
    t = pd.read_csv(R / "dsel_test_entrada114.csv")
    d = pd.concat([t, pd.read_csv(R / "dqn_test_p5_entrada173.csv")])
    w = d.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    cpu = d.pivot(index=["instance", "rep"], columns="method", values="cpu_time")
    tune = pd.read_csv(R / "dsel_tuning_entrada114.csv")
    best = {c: min(FIX5, key=lambda k: g[g.method == k].makespan.mean()) for c, g in tune.groupby("instance")}
    assert len(w) == 1200 and w[["igd_random", "dqn_d"]].notna().all().all(), (len(w), w.isna().sum().to_dict())
    w["val"] = [r[best[r["instance"]]] for _, r in w.iterrows()]
    w["oracle"] = w[FIX5].min(axis=1)
    w["cell"] = w["instance"]
    return w.rename(columns={"igd_random": "rand", "dqn_d": "dqn"}), cpu.rename(columns={"igd_random": "rand", "dqn_d": "dqn"})


def load_p15():
    t = pd.read_csv(R / "dqn_test_p15_entrada173.csv")
    w = t.pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    cpu = t.pivot(index=["instance", "rep"], columns="method", values="cpu_time")
    arms = pd.read_csv(R / "dqn_arms15_test_entrada173.csv").pivot(index=["instance", "rep"], columns="arm", values="makespan")
    w = w.merge(arms.min(axis=1).rename("oracle").reset_index(), on=["instance", "rep"])
    assert len(w) == 1200 and w[["rand15", "dqn15", "val15"]].notna().all().all(), (len(w), w.isna().sum().to_dict())
    w["cell"] = w["instance"]
    return w.rename(columns={"rand15": "rand", "dqn15": "dqn", "val15": "val"}), cpu.rename(columns={"rand15": "rand", "dqn15": "dqn"})


def main() -> None:
    envs = {"R": load_r(), "P5": load_p5(), "P15": load_p15()}
    ch = json.loads((R / "dqn_choice_entrada173.json").read_text())
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "APRENDIZ NEURAL MAIOR E ESPACO DE ACAO CONJUNTO -- Entrada 173/175", "=" * 110,
         f"lr escolhido no ajuste: R {ch['lr_r']} ({ch['lr_r_mean_ratio']}); P5 {ch['lr_p5']} ({ch['lr_p5_mean_gap']}); P15 {ch['lr_p15']} ({ch['lr_p15_mean_gap']})"]
    rows = []
    for env, (w, _) in envs.items():
        for pid, lab, a, b in [(f"I1{env}", f"{env}: dqn vs aleatorio", "dqn", "rand"), (f"I2{env}", f"{env}: validado vs dqn", "val", "dqn")]:
            t = test(w[a], w[b])
            t.update(id=pid, comparison=lab, env=env)
            rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    P["equivalente_0.05"] = (P.ci90_lo > -0.05) & (P.ci90_hi < 0.05)
    P["leitura"] = np.where(P.confirmada & P["equivalente_0.05"], "paridade (detectavel mas equivalente a zero)", np.where(P.confirmada, "confirmada", "nao confirmada"))
    L += ["\n--- PRIMARIA (Holm sobre 6; I1: dqn melhor que o aleatorio; I2: validado melhor que o dqn) ---",
          P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]].to_string(index=False)]
    Pi = P.set_index("id")

    cap_rows, hd_rows, l1 = [], [], False
    for env, (w, cpu) in envs.items():
        gain = lambda k: rel(w[k], w.rand).mean()
        gval = gain("val")
        learners = {"R": ["nq_d", "ql_d", "ucb_d", "dqn"], "P5": ["igd_ql_def", "igd_ucb", "dqn"], "P15": ["ql15", "linucb15", "dqn"]}[env]
        for k in learners:
            cap_rows.append(dict(ambiente=env, aprendiz=k, ganho_sobre_aleatorio=gain(k), captura=gain(k) / gval))
        cap_rows.append(dict(ambiente=env, aprendiz="validado", ganho_sobre_aleatorio=gval, captura=1.0))
        cap_rows.append(dict(ambiente=env, aprendiz="oraculo", ganho_sobre_aleatorio=gain("oracle"), captura=gain("oracle") / gval))
        hd = rel(w.oracle, w.rand)
        cell = pd.DataFrame({"hd": hd, "g": rel(w.dqn, w.rand), "cell": w.cell}).groupby("cell").mean()
        rho, p = sstats.spearmanr(cell.hd, cell.g)
        hd_rows.append(dict(ambiente=env, folga_media=float(hd.mean()), n_celulas=len(cell), spearman_folga_ganho_dqn=float(rho), p=float(p),
                            razao_cpu_dqn_aleatorio=float(cpu.dqn.mean() / cpu.rand.mean())))
        cap_dqn = gain("dqn") / gval
        l1 = l1 or bool((Pi.loc[f"I1{env}", "confirmada"] and not Pi.loc[f"I1{env}", "equivalente_0.05"] and cap_dqn >= 0.5) or (not Pi.loc[f"I2{env}", "confirmada"]))
    L += ["\n--- captura: ganho sobre o aleatorio (d %; negativo = melhor) e razao ao ganho do braco validado ---", pd.DataFrame(cap_rows).to_string(index=False),
          "\n--- folga (oraculo por instancia vs aleatorio, %), lei da folga (Spearman por celula) e custo ---", pd.DataFrame(hd_rows).to_string(index=False)]
    wr = envs["R"][0]
    t = test(wr.dqn, wr.nq_d)
    L.append(f"\nTamanho da rede importa? (R) dqn_big vs nq_d pequeno: d = {t['mean_d_pct']:+.4f}% (IC90 {t['ci90_lo']:+.3f}; {t['ci90_hi']:+.3f}; {t['better']}/{t['worse']}/{t['tie']}; p = {t['p_raw']:.3g})")
    w15 = envs["P15"][0]
    sec = []
    for a, b in [("ql15", "rand"), ("linucb15", "rand"), ("val", "ql15"), ("val", "linucb15"), ("val", "rand")]:
        t = test(w15[a], w15[b])
        sec.append(dict(comparacao=f"P15: {a} vs {b}", mean_d_pct=t["mean_d_pct"], ci90_lo=t["ci90_lo"], ci90_hi=t["ci90_hi"], melhor=t["better"], pior=t["worse"], p_raw=t["p_raw"]))
    L += ["\n--- P15: outros aprendizes (descritivo) ---", pd.DataFrame(sec).to_string(index=False)]
    val15 = ch["val15"]
    L.append("bracos validados por classe em P15 (0..14 = estrategia*5 + fracao; fracoes 0,05 ... 0,40): " + ", ".join(f"{c}: {a}" for c, a in val15.items()))
    l2 = bool((not l1) and all(Pi.loc[f"I2{e}", "confirmada"] for e in envs))
    L += ["\n--- LEITURAS PRE-FIXADAS ---",
          f"L1 (um aprendiz neural maior ou um espaco de acao maior muda a conclusao): {'SIM' if l1 else 'NAO'}",
          f"L2 (a falha do aprendizado online nao depende do tamanho do aprendiz nem do espaco de acao): {'SIM' if l2 else 'NAO'}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada173_dqn.txt").write_text(text, encoding="utf-8")
    P.to_csv(R / "entrada173_dqn_primary.csv", index=False)
    pd.DataFrame(cap_rows).to_csv(R / "entrada173_dqn_capture.csv", index=False)
    pd.DataFrame(hd_rows).to_csv(R / "entrada173_dqn_headroom.csv", index=False)


if __name__ == "__main__":
    main()
