"""
analysis/report_entrada223_optq.py
====================================
Ciclo 20, frente D: o Q-LEARNING OTIMISTA (qo; tabela inicial otimista, recompensa invariante a escala, epsilon constante 0,05; ajustado por familia; experiment/run_cycle20_optq_entrada223.py, docstring) contra o aleatorio, o Q-Learning frio padrao (ql), o Q frio ajustado (qt, Entrada 203), a validada e o seletor offline, a iteracoes iguais.
Descoberta (2.550) e held-out (3.950), S5 e S15, 5 sementes; Holm sobre 4 por familia. Diario: Entrada 223 (pre-registro) e 224 (resultados).

PRE-REGISTRO (congelado ANTES do ajuste e do teste; hashes em results/entrada223_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Familias: Or qo contra o aleatorio; Oq qo contra o Q frio padrao; Ot qo contra o Q frio ajustado (qt); Ov val contra qo; Os sel contra qo.
(O1) a inicializacao otimista evita o defeito do Q aquecido: ganho medio de qo sobre o aleatorio >= 0 nos quatro pares (probabilidade a priori do autor 60%).  (O2) qo supera o Q frio padrao em media nos quatro pares e com ganho >= 0,03 em S15 held-out (50%).
(O3) a validada vence qo nos quatro pares (Ov confirmada; 85%).  (O4) o seletor vence qo nos quatro pares (Os confirmada; 95%).  (O5) qo nao supera o Q frio ajustado (qt) por mais que a margem de 0,05 em nenhum dos quatro pares (80%).
Leitura R ('um Q-Learning com otimismo e exploracao constante repara o aprendiz online': qo nao e' confirmadamente pior que a validada em >= 3 dos 4 pares) previsao NAO.
Uso: python -m analysis.report_entrada223_optq
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402
from analysis.report_entrada200_selector import SETS, load  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_o(sp, cj):
    w = load(sp, cj)
    for pat, nm in ((f"optq_{cj}_{sp}_s*of*_entrada223.csv", "qo"), (f"qvar_{sp}_{cj}_s*of*_entrada203.csv", "qvar")):
        q = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat)))], ignore_index=True)
        m = q.groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
        w = w.merge(m, on="cls")
    return w


def main():
    L = ["=" * 110, "Q-LEARNING OTIMISTA (Entrada 223)", "=" * 110]
    W = {(sp, cj): load_o(sp, cj) for sp in ("S5", "S15") for cj in SETS}
    spec = {"Or": ("qo", "rand"), "Oq": ("qo", "ql"), "Ot": ("qo", "qt"), "Ov": ("val", "qo"), "Os": ("sel", "qo")}
    fams = {k: [] for k in spec}
    for (sp, cj), w in W.items():
        for fid, (a, b) in spec.items():
            t = test(w[a], w[b]); t.update(id=f"{fid}-{sp}-{cj}"); fams[fid].append(t)
    F = {k: family(v) for k, v in fams.items()}
    for k, X in F.items():
        a, b = spec[k]
        L += [f"\n--- {k}: {a} contra {b} (Holm sobre {len(X)}) ---", X[COLS].to_string(index=False)]
    gain = lambda X: -X.mean_d_pct
    o1 = bool((gain(F["Or"]) >= 0).all())
    o2 = bool((gain(F["Oq"]) > 0).all()) and float(gain(F["Oq"]).iloc[3]) >= 0.03
    o3 = bool(F["Ov"].confirmada.all())
    o4 = bool(F["Os"].confirmada.all())
    o5 = bool((gain(F["Ot"]) <= 0.05).all())
    r = int(((F["Ov"].mean_d_pct < 0) & (F["Ov"].p_holm < 0.05)).sum()) < 3
    yn = lambda b: "SIM" if b else "NAO"
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---", f"(O1) qo >= aleatorio nos quatro: {yn(o1)} (ganho: " + ", ".join(f"{v:+.3f}" for v in gain(F['Or'])) + ")", f"(O2) qo melhor que o Q frio nos quatro e >= 0,03 em S15 held-out: {yn(o2)} (" + ", ".join(f"{v:+.3f}" for v in gain(F['Oq'])) + ")",
          f"(O3) a validada vence qo nos quatro (confirmada): {yn(o3)} (" + ", ".join(f"{-v:+.3f}" for v in F['Ov'].mean_d_pct) + ")", f"(O4) o seletor vence qo nos quatro: {yn(o4)} ({int(F['Os'].confirmada.sum())}/4)", f"(O5) qo nao supera o Q frio ajustado por mais que 0,05: {yn(o5)} (" + ", ".join(f"{v:+.3f}" for v in gain(F['Ot'])) + ")",
          f"R (um Q otimista repara o aprendiz online: validada confirmadamente melhor em < 3 dos 4 pares): {yn(r)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada223_optq.txt").write_text(text, encoding="utf-8")
    for k, X in F.items():
        X.to_csv(R / f"entrada223_optq_{k}.csv", index=False)


if __name__ == "__main__":
    main()
