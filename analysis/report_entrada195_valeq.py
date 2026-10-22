"""
analysis/report_entrada195_valeq.py
=====================================
Ciclo 18, frente A: a acao VALIDADA contra cada aprendiz a CPU IGUAL (revisao cega, critica 2). Diario: Entrada 194 (revisao e plano), 195 (pre-registro) e 196 (resultados). Delineamento: experiment/run_cycle18_valeq_entrada195.py.

PRE-REGISTRO (congelado ANTES de rodar a calibracao e o teste; hashes em results/entrada195_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------------------
Ate' agora 'validada vs aprendiz' foi comparada a ITERACOES iguais (T = 1000 para os dois). O artigo prefere a CPU igual como orcamento, e a acao validada tem seu proprio custo K_val (CPU da acao fixa sobre a do aleatorio a T = 1000, calibrado em sequencia). Aqui a validada roda T_v = round(1000 K_aprendiz / K_val) iteracoes
(K_aprendiz: razoes ja' calibradas), consumindo a mesma CPU que o aprendiz em T = 1000. Estatistica a das Entradas 180-190: unidade = media das 5 sementes por instancia; d = 100 (A - B) / B em makespan com A = validada a CPU igual e B = o aprendiz a T = 1000 (negativo: a validada vence); Wilcoxon pareado bilateral
(empates descartados), IC90 bootstrap (2000), Holm sobre 18 (nove aprendizes x S5 e S15) em cada bateria, equivalencia a 0 se o IC90 cabe em +-0,05%; 'a validada vence' = p_holm < 0,05 e d < 0; 'o aprendiz vence' = p_holm < 0,05 e d > 0. Baterias: sintetica (12 x 100 instancias), publicadas descoberta (2.550) e held-out (3.950).
Previsoes (9): (1) K_val em [0,2; 1,8] em todas as celulas (18 grupos publicados por espaco x 2 espacos; 24 classes sinteticas); (2) contra o Q-Learning a CPU igual a validada vence (confirmada) nos seis casos (sintetica, descoberta e held-out x S5 e S15); (3) held-out: a validada vence em >= 15 dos 18 pares;
(4) descoberta: >= 15 dos 18; (5) sintetica: >= 12 dos 18; (6) nenhum aprendiz vence a validada a CPU igual por mais que a margem (nenhum par com p_holm < 0,05 e d > +0,05) em nenhuma bateria; (7) held-out, Q-Learning em S15: d em [-0,12; -0,02]; (8) held-out, roleta ALNS em S5: d em [-0,08; 0,0];
(9) held-out, familia C grupos 1-2: d da validada contra o Q-Learning em [-0,12; 0,0] em S5 e em S15. Transparencia: um teste de fumaca em pasta temporaria (subconjunto minusculo, fora dos dados) validou o pipeline e mostrou K_val entre 0,29 e 1,44; a faixa da previsao (1) foi alargada de [0,4; 1,6] para [0,2; 1,8] por isso (custo, nao resultado). Leitura L 'a validada nunca foi pior que um aprendiz ao orcamento de CPU igual' = (6) cumprida e (2) cumprida.
Uso: python -m analysis.report_entrada195_valeq
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, geo, rel, test  # noqa: E402
from analysis.report_entrada189_published import COLS, LEARNERS  # noqa: E402

R = ROOT / "results"
EXPECT = {"syn": int(os.environ.get("C18_EXPECT_SYN", "1200")), "discovery": int(os.environ.get("C18_EXPECT_DISC", "0")), "heldout": int(os.environ.get("C18_EXPECT_HELD", "0"))}
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def inst_means(d):
    return d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()


def load_syn(sp):
    parts = []
    for f in (f"agents_{sp}_entrada180.csv", f"aos_{sp}_entrada185.csv", f"new_{sp}_entrada188.csv", f"valeq_syn_{sp}_entrada195.csv"):
        d = pd.read_csv(R / f)
        d = d[~d.method.str.startswith("arm")]
        parts.append(d.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index())
    w = parts[0]
    for q in parts[1:]:
        w = w.merge(q[[c for c in q.columns if c not in w.columns or c in ("cls", "rep")]], on=["cls", "rep"])
    return w


def load_pub(sp, setname):
    pat, vpat = (f"pub_{sp}_s*of*_entrada189.csv", f"valeq_pub_discovery_{sp}_s*of*_entrada195.csv") if setname == "discovery" else (f"pubh_{sp}_s*of*_entrada190.csv", f"valeq_pub_heldout_{sp}_s*of*_entrada195.csv")
    a = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat)))], ignore_index=True)
    b = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / vpat)))], ignore_index=True)
    wa = inst_means(a)
    wb = b.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    return wa.merge(wb, on="cls")


def main():
    L = ["=" * 110, "VALIDADA vs APRENDIZ A CPU IGUAL -- Ciclo 18, frente A", "=" * 110]
    KV = json.loads((R / "valeq_kval_entrada195.json").read_text())
    res = {}
    allk = []
    for bat in ("syn", "discovery", "heldout"):
        rows, ksum = [], []
        for sp in ("S5", "S15"):
            w = load_syn(sp) if bat == "syn" else load_pub(sp, bat)
            n_exp = EXPECT[bat]
            if n_exp:
                assert len(w) == n_exp, (bat, sp, len(w))
            if bat != "syn":
                w["fam"] = w.cls.str.split("|").str[0]
                w["grp"] = w.cls.str.split("|").str[1].astype(int)
            for k in LEARNERS:
                t = test(w[f"val_eq_{k}"], w[k]); t.update(space=sp, learner=k, id=f"Veq-{bat}-{sp}-{k}"); rows.append(t)
            kv = list(KV["syn" if bat == "syn" else "pub"][sp].values())
            ksum.append(dict(bateria=bat, space=sp, K_min=min(kv), K_geomean=geo(kv), K_max=max(kv)))
            if bat == "heldout":
                x = w[(w.fam == "C") & w.grp.isin([1, 2])]
                res[f"C12_{sp}"] = float(rel(x["val_eq_ql"], x["ql"]).mean())
        F = family(rows)
        L += [f"\n--- {bat}: validada a CPU igual vs aprendiz (Holm sobre 18; negativo: a validada vence) ---", F[COLS].to_string(index=False), f"\nK_val ({bat}): " + "; ".join(f"{r['space']} min {r['K_min']:.2f}, geomean {r['K_geomean']:.2f}, max {r['K_max']:.2f}" for r in ksum)]
        res[bat] = F
        allk += ksum
    ok = lambda b: "SIM" if b else "NAO"
    kmin = min(r["K_min"] for r in allk)
    kmax = max(r["K_max"] for r in allk)
    q = lambda bat, sp: res[bat].set_index("id").loc[f"Veq-{bat}-{sp}-ql"]
    p2 = all(bool(q(b, sp).confirmada) for b in ("syn", "discovery", "heldout") for sp in ("S5", "S15"))
    nb = {b: int(res[b].confirmada.sum()) for b in res if b in ("syn", "discovery", "heldout")}
    worse = any(bool(((r.p_holm < 0.05) and (r.mean_d_pct > 0.05))) for b in ("syn", "discovery", "heldout") for _, r in res[b].iterrows())
    dq = float(q("heldout", "S15").mean_d_pct)
    da = float(res["heldout"].set_index("id").loc["Veq-heldout-S5-alns"].mean_d_pct)
    c12 = [res["C12_S5"], res["C12_S15"]]
    L += ["\n--- PREVISOES E LEITURAS PRE-FIXADAS ---", f"(1) K_val em [0,2; 1,8] em todas as celulas: {ok(0.4 <= kmin and kmax <= 1.6)} (min {kmin:.2f}, max {kmax:.2f})", f"(2) contra o Q-Learning a CPU igual a validada vence nos seis casos: {ok(p2)}",
          f"(3) held-out: a validada vence em >= 15 dos 18 pares ({nb['heldout']}/18): {ok(nb['heldout'] >= 15)}", f"(4) descoberta: >= 15 dos 18 ({nb['discovery']}/18): {ok(nb['discovery'] >= 15)}", f"(5) sintetica: >= 12 dos 18 ({nb['syn']}/18): {ok(nb['syn'] >= 12)}",
          f"(6) nenhum aprendiz vence a validada a CPU igual por mais que a margem: {ok(not worse)}", f"(7) held-out, Q-Learning S15: d em [-0,12; -0,02]: {ok(-0.12 <= dq <= -0.02)} ({dq:.4f})", f"(8) held-out, roleta ALNS S5: d em [-0,08; 0,0]: {ok(-0.08 <= da <= 0.0)} ({da:.4f})",
          f"(9) held-out, familia C grupos 1-2: d da validada vs Q-Learning em [-0,12; 0,0] em S5 e S15: {ok(all(-0.12 <= v <= 0.0 for v in c12))} ({[round(v, 4) for v in c12]})", f"L (a validada nunca foi pior que um aprendiz a CPU igual): {ok((not worse) and p2)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada195_valeq.txt").write_text(text, encoding="utf-8")
    for b in ("syn", "discovery", "heldout"):
        res[b].to_csv(R / f"entrada195_valeq_{b}.csv", index=False)
    pd.DataFrame(allk).to_csv(R / "entrada195_valeq_k.csv", index=False)


if __name__ == "__main__":
    main()
