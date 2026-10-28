"""
analysis/report_entrada220_ga_mix.py
======================================
Ciclo 19, frente F, ESTAGIO 2: a MISTURA estatica validada (vmix: uniforme entre as tres melhores acoes unicas por classe, escolhidas nas 30 instancias de ajuste) como competidor estatico no hospedeiro populacional (experiment/run_cycle19_ga_mix_entrada220.py; docstring). Diario: Entrada 220 (resultados do estagio 1 e pre-registro do estagio 2) e 221 (resultados do estagio 2).
Estagio 1 (Entrada 219-220; 4 de 7 previsoes): a acao unica validada perde para o aleatorio em G9 (-0,057), Thompson (+0,051 e +0,047) e, em G9, UCB1 e a politica de gradiente ganham do aleatorio; a CPU igual a validada vence os dez pares so' por ser mais barata.
Estatistica: a das Entradas 180-219 (d = 100 (A - B)/B; Wilcoxon pareado; IC90 bootstrap; margem +-0,05; Holm sobre 10 por familia): Mrand vmix vs aleatorio; Mval vmix vs a validada unica; Mit vmix vs cada aprendiz a iteracoes iguais; Meq vmix a CPU igual do aprendiz vs aprendiz. 1.200 instancias, 5 sementes, G9 e G18.

PRE-REGISTRO DO ESTAGIO 2 (registrado DEPOIS de ver o estagio 1 e ANTES de rodar; selecao so' nas instancias de ajuste; hashes em results/entrada220_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
(V1) a mistura validada ganha do aleatorio nos dois espacos (Mrand confirmada, ganho > 0).  (V2) a mistura valida vence a acao unica validada em G9 e em G18 (Mval confirmada nos dois).  (V3) a mistura vence o melhor aprendiz (Thompson) a iteracoes iguais nos dois espacos (Mit confirmada).
(V4) a mistura vence os cinco aprendizes a iteracoes iguais em >= 8 dos 10 pares (Mit confirmada).  (V5) a CPU igual (K proprio) a mistura vence os cinco aprendizes em >= 9 dos 10 pares (Meq confirmada).
Leitura M ('onde as acoes sao complementares o competidor estatico certo e' uma mistura validada, e ele vence o aprendizado online') = V1, V3 e V5. Probabilidades a priori do autor: V1 80%, V2 85%, V3 60%, V4 55%, V5 65%.
Uso: python -m analysis.report_entrada220_ga_mix
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada219_ga import LEARNERS, load_test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_m(sp):
    w = load_test(sp)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"gamix_test_{sp}_s*of*_entrada220.csv")))], ignore_index=True)
    d["key"] = d.cls + "#" + d.rep.astype(str)
    m = d.groupby(["key", "method"]).makespan.mean().unstack("method").reset_index()
    return w.merge(m, on="key")


def main():
    L = ["=" * 110, "MISTURA ESTATICA VALIDADA NO HOSPEDEIRO POPULACIONAL (Entrada 220)", "=" * 110]
    W = {sp: load_m(sp) for sp in ("G9", "G18")}
    rows = {"Mrand": [], "Mval": [], "Mit": [], "Meq": []}
    for sp, w in W.items():
        t = test(w["vmix"], w["rand"]); t.update(id=f"Mrand-{sp}"); rows["Mrand"].append(t)
        t = test(w["vmix"], w["val"]); t.update(id=f"Mval-{sp}"); rows["Mval"].append(t)
        for k in LEARNERS:
            t = test(w["vmix"], w[k]); t.update(id=f"Mit-{sp}-{k}"); rows["Mit"].append(t)
            t = test(w[f"vmix_eq_{k}"], w[k]); t.update(id=f"Meq-{sp}-{k}"); rows["Meq"].append(t)
    F = {k: family(v) for k, v in rows.items()}
    for k, X in F.items():
        L += [f"\n--- {k} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    P = pd.DataFrame([dict(space=sp, vmix=float(-rel(w.vmix, w.rand).mean()), val=float(-rel(w.val, w.rand).mean()), **{k: float(-rel(w[k], w.rand).mean()) for k in LEARNERS}) for sp, w in W.items()])
    L += ["\n--- ganho sobre o aleatorio, agrupado (%, positivo: melhor) ---", P.to_string(index=False)]
    mr, mv = F["Mrand"], F["Mval"]
    v1 = bool(mr.confirmada.all())
    v2 = bool(mv.confirmada.all())
    mi = F["Mit"].set_index("id")
    v3 = bool(mi.loc[["Mit-G9-ts", "Mit-G18-ts"], "confirmada"].all())
    v4 = int(F["Mit"].confirmada.sum()) >= 8
    v5 = int(F["Meq"].confirmada.sum()) >= 9
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---", f"(V1) vmix ganha do aleatorio (confirmado) em G9 e G18: {yn(v1)} (" + ", ".join(f"{-v:+.3f}" for v in mr.mean_d_pct) + ")", f"(V2) vmix vence a validada unica nos dois: {yn(v2)} (" + ", ".join(f"{-v:+.3f}" for v in mv.mean_d_pct) + ")",
          f"(V3) vmix vence Thompson a iteracoes iguais nos dois: {yn(v3)} (" + ", ".join(f"{-mi.loc[k, 'mean_d_pct']:+.3f}" for k in ('Mit-G9-ts', 'Mit-G18-ts')) + ")", f"(V4) vmix vence os cinco aprendizes a iteracoes iguais em >= 8 dos 10: {yn(v4)} ({int(F['Mit'].confirmada.sum())}/10)",
          f"(V5) a CPU igual vmix vence os cinco em >= 9 dos 10: {yn(v5)} ({int(F['Meq'].confirmada.sum())}/10)", f"M (a mistura validada vence o aprendizado online) = V1, V3 e V5: {yn(v1 and v3 and v5)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada220_ga_mix.txt").write_text(text, encoding="utf-8")
    for k, X in F.items():
        X.to_csv(R / f"entrada220_ga_mix_{k}.csv", index=False)


if __name__ == "__main__":
    main()
