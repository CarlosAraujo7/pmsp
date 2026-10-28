"""
analysis/report_entrada228_cf_fine.py
========================================
Ciclo 21, frente A: a folga do oraculo COM VALIDACAO CRUZADA nos TEMPOS FINOS p ~ U[1,10^4] (S5 e S15; 1.200 instancias, 12 classes x 100), onde a folga de uma semente e' 0,142 e 0,175 e nenhum aprendiz ganha (Entrada 185). Dados: results/fine_ESPACO_entrada185.csv (semente 0 das acoes fixas e todos os metodos) e results/cfo_fine_ESPACO_entrada228.csv (sementes 1-3, experiment/run_cycle21_cffine_entrada228.py). A acao do oraculo e' escolhida nas sementes {0,1} e avaliada nas {2,3}; limiar h* = o das Entradas 209/210 (congelado: results/entrada209_cforacle.txt).
Saida: results/entrada228_cf_fine.txt e results/entrada228_cf_fine_cells.csv. Diario: Entrada 228 (pre-registro) e 229 (resultados).

PRE-REGISTRO (congelado ANTES de qualquer saida do runner; hashes em results/entrada228_prereg_hashes.txt)
------------------------------------------------------------------------------------------------------------
Celula = classe (n,m) x espaco: 12 por espaco, 24 no total; ganho do Q = ganho do Q-Learning sobre o aleatorio a iteracoes iguais, media das cinco sementes; a regra preve 'Q ganha' se folga cf >= h*, e 'Q ganha' significa ganho > 0 (a definicao das Entradas 209/210).
(F1) a folga cf agregada fica >= h* nos dois espacos (75%).  (F2) a parcela de ruido da folga de uma semente, (h_enviesada - h_cf)/h_enviesada, e' MENOR que a das sinteticas grossas (0,497 em S5; 0,647 em S15) nos dois espacos (65%).  (F3) a parcela de ruido e' >= 0,20 nos dois espacos (85%).
(F4) a regra h* classifica corretamente >= 75% das 24 celulas (40%).  (F5) a correlacao de Spearman entre a folga cf e o ganho do Q nas 12 classes e' < 0,5 em pelo menos um espaco (60%).  (F6) o ganho agregado da acao validada por classe sobre o aleatorio e' MAIOR que a folga cf em pelo menos um espaco (a folga cf e' uma cota inferior frouxa; 65%).
Leitura C ('folga sem retorno para o aprendiz online': F1 verdadeira e |ganho agregado do Q| < 0,05 nos dois espacos): previsao SIM (75%).
Uso: python -m analysis.report_entrada228_cf_fine
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import rel  # noqa: E402
from analysis.report_entrada209_cforacle import arms_table  # noqa: E402

R = ROOT / "results"
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load_fine(sp):
    d = pd.read_csv(R / f"fine_{sp}_entrada185.csv")
    d["cls"] = d.cls + "#" + d.rep.astype(str)
    new = pd.read_csv(R / f"cfo_fine_{sp}_entrada228.csv")
    new["cls"] = new.cls + "#" + new.rep.astype(str)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method")
    w = ms.join(arms_table(d, new))
    w["cell"] = [c.split("#")[0] for c in w.index]
    return w.reset_index()


def main():
    t209 = (R / "entrada209_cforacle.txt").read_text(encoding="utf-8")
    hstar = float(re.search(r"limiar h\* escolhido na descoberta \(acuracia balanceada [\d.]+\): ([\d.]+)", t209).group(1))
    P9 = pd.read_csv(R / "entrada209_cforacle_pooled.csv").set_index(["set", "space", "sub"])
    L = ["=" * 110, "FOLGA COM VALIDACAO CRUZADA NOS TEMPOS FINOS (Entrada 228)", "=" * 110, f"limiar h* (congelado) = {hstar:.4f}"]
    pooled, cells = [], []
    for sp in ("S5", "S15"):
        w = load_fine(sp)
        hb, hc = float(-rel(w.biased, w.rand).mean()), float(-rel(w.cf, w.rand).mean())
        pooled.append(dict(space=sp, n=len(w), h_biased=hb, h_cf=hc, bias_rel=(hb - hc) / hb, bias_rel_coarse=float(P9.loc[("syn", sp, "all"), "bias_rel"]), gain_ql=float(-rel(w.ql, w.rand).mean()), gain_val=float(-rel(w.val, w.rand).mean())))
        for c, x in w.groupby("cell"):
            cells.append(dict(space=sp, cell=c, n=len(x), h_biased=float(-rel(x.biased, x.rand).mean()), h_cf=float(-rel(x.cf, x.rand).mean()), gq=float(-rel(x.ql, x.rand).mean()), gval=float(-rel(x.val, x.rand).mean())))
    Pd, C = pd.DataFrame(pooled), pd.DataFrame(cells)
    C["pred_gain"] = C.h_cf >= hstar
    C["true_gain"] = C.gq > 0
    L += ["\n--- agregado por espaco (pontos percentuais) ---", Pd.to_string(index=False), "\n--- celulas (classe x espaco) ---", C.to_string(index=False)]
    acc = float((C.pred_gain == C.true_gain).mean())
    sp_rho = {sp: sstats.spearmanr(x.h_cf, x.gq).statistic for sp, x in C.groupby("space")}
    L += [f"\nacuracia da regra h* nas {len(C)} celulas: {acc:.3f} (a fracao de celulas em que o Q ganha: {C.true_gain.mean():.3f})", "Spearman folga cf x ganho do Q por espaco: " + ", ".join(f"{k} {v:+.3f}" for k, v in sp_rho.items())]
    yn = lambda b: "SIM" if b else "NAO"
    f1 = bool((Pd.h_cf >= hstar).all())
    f2 = bool((Pd.bias_rel < Pd.bias_rel_coarse).all())
    f3 = bool((Pd.bias_rel >= 0.20).all())
    f4 = acc >= 0.75
    f5 = min(sp_rho.values()) < 0.5
    f6 = bool((Pd.gain_val > Pd.h_cf).any())
    c = f1 and bool((Pd.gain_ql.abs() < 0.05).all())
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---",
          f"(F1) folga cf agregada >= h* nos dois espacos: {yn(f1)} (" + ", ".join(f"{v:.3f}" for v in Pd.h_cf) + ")",
          f"(F2) parcela de ruido menor que a das sinteticas grossas nos dois espacos: {yn(f2)} (" + ", ".join(f"{v:.3f} vs {u:.3f}" for v, u in zip(Pd.bias_rel, Pd.bias_rel_coarse)) + ")",
          f"(F3) parcela de ruido >= 0,20 nos dois espacos: {yn(f3)}", f"(F4) acuracia da regra >= 0,75 nas 24 celulas: {yn(f4)} ({acc:.3f})",
          f"(F5) Spearman < 0,5 em pelo menos um espaco: {yn(f5)} (" + ", ".join(f"{v:+.3f}" for v in sp_rho.values()) + ")",
          f"(F6) ganho agregado da validada > folga cf em pelo menos um espaco: {yn(f6)} (" + ", ".join(f"{a:.3f} vs {b:.3f}" for a, b in zip(Pd.gain_val, Pd.h_cf)) + ")",
          f"C (folga sem retorno para o aprendiz online: F1 e |ganho do Q| < 0,05): {yn(c)} (ganho do Q " + ", ".join(f"{v:+.3f}" for v in Pd.gain_ql) + ")"]
    text = "\n".join(L)
    print(text)
    (R / "entrada228_cf_fine.txt").write_text(text, encoding="utf-8")
    C.to_csv(R / "entrada228_cf_fine_cells.csv", index=False)
    Pd.to_csv(R / "entrada228_cf_fine_pooled.csv", index=False)


if __name__ == "__main__":
    main()
