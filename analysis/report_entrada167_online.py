"""
analysis/report_entrada167_online.py
======================================
Os aprendizes online de outras classes (LinUCB contextual e Q-learning neural) nos OUTROS dois ambientes (P||Cmax e flowshop), tamanho de destruicao. Ciclo 13, frente F. Diario: Entrada 167 (pre-registro) e 168 (resultados).

PRE-REGISTRO (congelado ANTES de executar a fase --run; hashes em results/entrada167_prereg_hashes.txt)
--------------------------------------------------------------------------------------------------------
Dados novos: results/online_test_P_entrada167.csv (1.200 instancias de P||Cmax x 2 aprendizes) e results/online_test_F_entrada167.csv (160 instancias de flowshop: 120 geradas + 40 classicas de Taillard, x 2 aprendizes); hiperparametros
(alpha do LinUCB, lr da rede) escolhidos nas instancias de ajuste de cada ambiente (results/online_choice_{P,F}_entrada167.json). Dados ja' existentes (mesmas instancias e sementes): P||Cmax results/dsel_test_entrada114.csv (igd_fix05..40, igd_random)
com o braco validado por classe de results/dsel_tuning_entrada114.csv (menor makespan medio entre os 5 bracos fixos, como em run_igcvls_entrada127.d_choice); flowshop results/pfsp_test_entrada152.csv e results/pfsp_taillard_entrada159.csv
(ig_fix_0..4, ig_rand) com o d validado por tamanho de results/pfsp_choice_entrada152.json. d = 100*(A-B)/B em makespan (negativo: A melhor); Wilcoxon pareado bicaudal pooled; nulos descartados; bootstrap IC90.
Familia primaria POR AMBIENTE (Holm sobre 4, alpha = 0,05; 'confirmada' = p_holm < 0,05 e d < 0):
  G1 linucb vs aleatorio ; G2 nq (Q neural) vs aleatorio ; G3 validado vs linucb ; G4 validado vs nq
Leitura pre-fixada: 'os aprendizes contextual e neural nao mudam a atribuicao nos outros ambientes' = G1 e G2 NAO confirmadas em P e em F. Descritivas: razoes de captura (ganho sobre o aleatorio / ganho do validado), oraculo, por fonte (F: geradas e Taillard), CPU.
Previsoes registradas antes de rodar: P: G1 e G2 NAO confirmadas (d >= -0,02%); G3 e G4 confirmadas com d em [-0,12; -0,04]%. F: G1 e G2 NAO confirmadas (d >= -0,05%); G3 e G4 com d em [-0,25; -0,05]% (confirmacao incerta, N = 160: ~50%).
Uso: python -m analysis.report_entrada167_online
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


def family(w, env):
    rows = []
    for pid, lab, a, b in [("G1", "linucb vs random d", "linucb", "rand"), ("G2", "neural Q vs random d", "nq", "rand"), ("G3", "validated d vs linucb", "val", "linucb"), ("G4", "validated d vs neural Q", "val", "nq")]:
        t = test(w[a], w[b])
        t.update(id=f"{env}-{pid}", comparison=f"{env}: {lab}")
        rows.append(t)
    P = pd.DataFrame(rows)
    h = holm_bonferroni(P.p_raw.tolist(), alpha=0.05)
    P["p_holm"], P["reject"] = h["p_adjusted"], h["reject"]
    P["confirmada"] = P.reject & (P.mean_d_pct < 0)
    return P


def env_P():
    t = pd.read_csv(R / "dsel_test_entrada114.csv")
    tu = pd.read_csv(R / "dsel_tuning_entrada114.csv")
    new = pd.read_csv(R / "online_test_P_entrada167.csv")
    fix = ["igd_fix05", "igd_fix10", "igd_fix20", "igd_fix30", "igd_fix40"]
    best = {c: min(fix, key=lambda k: g[g.method == k].makespan.mean()) for c, g in tu.groupby("instance")}
    w = t[t.method.isin(fix + ["igd_random"])].pivot(index=["instance", "rep"], columns="method", values="makespan").reset_index()
    nw = new.assign(instance=lambda d: "rand_" + d.n.astype(str) + "j_" + d.m.astype(str) + "m").pivot(index=["instance", "rep"], columns="learner", values="makespan").reset_index()
    w = w.merge(nw, on=["instance", "rep"])
    assert len(w) == 1200 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    w["val"] = [r[best[r["instance"]]] for _, r in w.iterrows()]
    w["rand"] = w["igd_random"]
    w["oracle"] = w[fix].min(axis=1)
    return w


def env_F():
    ch = json.loads((R / "pfsp_choice_entrada152.json").read_text())
    g = pd.read_csv(R / "pfsp_test_entrada152.csv").assign(src="gen")
    ta = pd.read_csv(R / "pfsp_taillard_entrada159.csv").assign(src="tai")
    base = pd.concat([g, ta])
    fix = [f"ig_fix_{k}" for k in range(5)]
    w = base[base.method.isin(fix + ["ig_rand"])].pivot(index=["src", "n", "m", "rep"], columns="method", values="makespan").reset_index()
    new = pd.read_csv(R / "online_test_F_entrada167.csv").pivot(index=["src", "n", "m", "rep"], columns="learner", values="makespan").reset_index()
    w = w.merge(new, on=["src", "n", "m", "rep"])
    assert len(w) == 160 and w.isna().sum().sum() == 0, (len(w), w.isna().sum().to_dict())
    w["val"] = [r[f"ig_fix_{ch['d_val'][f'{int(r.n)}x{int(r.m)}']['action']}"] for _, r in w.iterrows()]
    w["rand"] = w["ig_rand"]
    w["oracle"] = w[fix].min(axis=1)
    return w


def main() -> None:
    pd.set_option("display.width", 230)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    L = ["=" * 110, "APRENDIZES ONLINE DE OUTRAS CLASSES NOS OUTROS AMBIENTES (tamanho de destruicao) -- Entrada 167/168", "=" * 110]
    out = []
    unchanged = []
    for env, w in (("P", env_P()), ("F", env_F())):
        P = family(w, env)
        out.append(P)
        chs = json.loads((R / f"online_choice_{env}_entrada167.json").read_text())
        L.append(f"\n--- ambiente {env}: {len(w)} instancias | hiperparametros escolhidos no ajuste: LinUCB alpha = {chs['linucb']['hp']}, Q neural lr = {chs['nq']['hp']} ---")
        L.append(P[["id", "comparison", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "rank_biserial", "p_raw", "p_holm", "confirmada"]].to_string(index=False))
        gain = lambda k: rel(w[k], w.rand).mean()
        cap = lambda k: gain(k) / gain("val") if gain("val") != 0 else float("nan")
        L.append("Ganho sobre o aleatorio (d %): " + ", ".join(f"{k} {gain(k):+.3f}" for k in ["val", "linucb", "nq", "oracle"]) + " | captura (ganho/ganho validado): " + ", ".join(f"{k} {cap(k):.2f}" for k in ["linucb", "nq", "oracle"]))
        c = P.set_index("id")
        unchanged.append(bool((not c.loc[f"{env}-G1", "confirmada"]) and (not c.loc[f"{env}-G2", "confirmada"])))
        if env == "F":
            for s, g in w.groupby("src"):
                L.append(f"  fonte {s} ({len(g)}): linucb {rel(g.linucb, g.rand).mean():+.3f}, nq {rel(g.nq, g.rand).mean():+.3f}, val {rel(g.val, g.rand).mean():+.3f}")
    L.append(f"\nLeitura pre-fixada: 'os aprendizes contextual e neural nao mudam a atribuicao nos outros ambientes' (G1 e G2 nao confirmadas em P e em F): {'SIM' if all(unchanged) else 'NAO'} (P: {unchanged[0]}, F: {unchanged[1]})")
    text = "\n".join(L)
    print(text)
    (R / "entrada167_online.txt").write_text(text, encoding="utf-8")
    pd.concat(out).to_csv(R / "entrada167_online_primary.csv", index=False)


if __name__ == "__main__":
    main()
