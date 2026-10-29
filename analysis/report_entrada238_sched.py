"""
analysis/report_entrada238_sched.py
=====================================
Ciclo 22, frente B: o ORACULO/COMPETIDOR DINAMICO NO TEMPO (a melhor sequencia de 3 fases de acoes do espaco S5 por classe, escolhida nas instancias de ajuste) contra a acao estatica validada, no P||Cmax sintetico ('pc') e no R||Cmax ('ur'). Desenho e dados: experiment/run_cycle22_sched_entrada238.py (125 cronogramas x 30 instancias x 2 sementes por classe no ajuste;
100 instancias de teste por classe x 5 sementes, metodo 'sch'); a acao validada, o aleatorio e os aprendizes vem dos arquivos existentes. Diario: Entrada 238 (pre-registro) e 239 (resultados). Saidas: results/entrada238_sched.txt, entrada238_sched_*.csv.

PRE-REGISTRO (congelado ANTES de qualquer saida do teste; hashes em results/entrada238_prereg_hashes.txt, que inclui as escolhas do ajuste)
------------------------------------------------------------------------------------------------------------------------------------------------
Transparencia: o AJUSTE (instancias de ajuste) ja' foi feito antes deste registro e as previsoes o levam em conta: a vantagem media do melhor cronograma sobre a melhor acao estatica nas instancias de ajuste e' de 0,010% no P||Cmax (7 das 12 classes com cronograma nao constante) e de 0,178% no R||Cmax (8 de 12; ate' 0,6% por classe, em parte ruido de selecao entre 125 candidatos).
Pergunta: ha' estrutura no TEMPO que um aprendiz que se adapta durante a execucao poderia explorar alem da melhor acao estatica da classe? Testes (Wilcoxon pareado, ties fora, IC90 por bootstrap, margem de equivalencia +-0,05; 'confirmada' = p_holm < 0,05 e o cronograma melhor; Holm sobre 2 hospedeiros em cada familia):
SV = cronograma contra a acao validada; SR = cronograma contra o aleatorio; SQ (descritiva) = cronograma contra o melhor aprendiz online do hospedeiro (o de maior ganho agregado).
(S1) no P||Cmax o cronograma NAO supera a validada alem da margem (SV nao 'confirmada' ou em paridade) (92%).  (S2) no R||Cmax tambem nao supera (45%).  (S3) a vantagem do cronograma sobre a melhor estatica e' MAIOR no ajuste que no teste nos dois hospedeiros (a maldicao do vencedor entre 125 candidatos; 65%).
(S4) o ganho agregado do cronograma sobre o aleatorio e' maior que o do melhor aprendiz online nos dois hospedeiros (90%).  (S5) o ganho agregado do cronograma supera o da validada (estimativa pontual) nos dois hospedeiros (30%).
Leitura T1 ('adaptar-se no tempo nao acrescenta nada, ao nivel da classe, alem da estatica validada'): S1 e S2 verdadeiras (previsao: SIM no P||Cmax, incerta no R||Cmax; 41%).
Uso: python -m analysis.report_entrada238_sched
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
LEARN = {"pc": ["ql", "ucb", "pg"], "ur": ["ql", "ucb", "pg", "alns", "qo"]}


def load(host):
    if host == "pc":
        d = pd.read_csv(R / "agents_S5_entrada180.csv")
    else:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "ur_test_S5_s*of*_entrada234.csv")))], ignore_index=True)
    d = d[~d.method.str.startswith("arm")]
    ref = d[d.method.isin(["rand", "val"] + LEARN[host])].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"sched_test_{host}_s*of*_entrada238.csv")))], ignore_index=True)
    sch = s.groupby(["cls", "rep"]).makespan.mean().rename("sch")
    return ref.join(sch, how="inner").reset_index()


def main():
    L = ["=" * 110, "CRONOGRAMA DE ACOES (ORACULO DINAMICO NO TEMPO) CONTRA A ACAO ESTATICA VALIDADA (Entrada 238)", "=" * 110]
    W, rows, tsv, tsr, tsq = {}, [], [], [], []
    for host in ("pc", "ur"):
        w = load(host)
        W[host] = w
        ch = json.loads((R / f"sched_choice_{host}_entrada238.json").read_text())
        g = {k: float(-rel(w[k], w.rand).mean()) for k in ["sch", "val"] + LEARN[host]}
        best = max(LEARN[host], key=lambda k: g[k])
        tune_adv = float(np.mean([v["rel_gain_vs_static_pct"] for v in ch.values()]))
        nonconst = sum(len(set(v["sched"])) > 1 for v in ch.values())
        rows.append(dict(host=host, n=len(w), g_sch=g["sch"], g_val=g["val"], best_learner=best, g_best=g[best], tune_adv_pct=tune_adv, test_adv_pct=float(-rel(w.sch, w.val).mean()), classes_nonconstant=nonconst))
        for lst, (a, b) in ((tsv, ("sch", "val")), (tsr, ("sch", "rand")), (tsq, ("sch", best))):
            t = test(w[a], w[b])
            t.update(id=f"{host}")
            lst.append(t)
        L += [f"\n--- {host}: cronogramas escolhidos por classe (fases 1-3; indices das acoes) ---", ", ".join(f"{c.split('_')[1]}x{c.split('_')[2][:-1]}:{''.join(map(str, v['sched']))}" for c, v in ch.items())]
    Sm = pd.DataFrame(rows)
    FV, FR, FQ = family(tsv), family(tsr), family(tsq)
    L += ["\n--- agregado (pontos percentuais; vantagens em % do makespan) ---", Sm.to_string(index=False), "\n--- SV: cronograma contra a acao validada (negativo: o cronograma e' melhor; Holm sobre 2) ---", FV[COLS].to_string(index=False),
          "\n--- SR: cronograma contra o aleatorio ---", FR[COLS].to_string(index=False), "\n--- SQ (descritiva): cronograma contra o melhor aprendiz online ---", FQ[COLS].to_string(index=False)]
    yn = lambda b: "SIM" if b else "NAO"
    win = lambda r: bool(r.confirmada) and not bool(r["equivalente_0.05"]) and r.mean_d_pct < 0
    s1 = not win(FV.iloc[0])
    s2 = not win(FV.iloc[1])
    s3 = bool((Sm.tune_adv_pct > Sm.test_adv_pct).all())
    s4 = bool((Sm.g_sch > Sm.g_best).all())
    s5 = bool((Sm.g_sch > Sm.g_val).all())
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---", f"(S1) P||Cmax: o cronograma nao supera a validada alem da margem: {yn(s1)} (diferenca {-FV.iloc[0].mean_d_pct:+.4f})", f"(S2) R||Cmax: idem: {yn(s2)} (diferenca {-FV.iloc[1].mean_d_pct:+.4f})",
          f"(S3) vantagem maior no ajuste que no teste nos dois: {yn(s3)} (ajuste " + ", ".join(f"{v:.3f}" for v in Sm.tune_adv_pct) + "; teste " + ", ".join(f"{v:+.3f}" for v in Sm.test_adv_pct) + ")",
          f"(S4) ganho do cronograma > o do melhor aprendiz nos dois: {yn(s4)}", f"(S5) ganho do cronograma > o da validada nos dois: {yn(s5)} (" + ", ".join(f"{a:+.3f} vs {b:+.3f}" for a, b in zip(Sm.g_sch, Sm.g_val)) + ")", f"T1 (adaptar-se no tempo nao acrescenta nada: S1 e S2): {yn(s1 and s2)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada238_sched.txt").write_text(text, encoding="utf-8")
    Sm.to_csv(R / "entrada238_sched_pooled.csv", index=False)
    FV.to_csv(R / "entrada238_sched_SV.csv", index=False)
    FR.to_csv(R / "entrada238_sched_SR.csv", index=False)


if __name__ == "__main__":
    main()
