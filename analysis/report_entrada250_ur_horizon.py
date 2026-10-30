"""
analysis/report_entrada250_ur_horizon.py
=========================================
Entrada 250 (Ciclo 28): HORIZONTE LONGO no R||Cmax (experiment/run_cycle28_horizon_ur_entrada250.py). T = 5.000, 20.000 e 100.000 iteracoes (a bateria principal da Entrada 234 e' T = 1.000), S5 e S15, 12 classes x 20 instancias (as primeiras de cada classe), 2 sementes por metodo; unidade de analise: a media das sementes por instancia (240 pares por linha). Metodos: rand, ql (padrao), qls (epsilon escalado a T), qo (otimista), alns, val (a validada em T = 1.000). Ganho = -d (pontos do makespan) sobre o aleatorio ao MESMO T. Familias por horizonte e espaco (Holm sobre os 6 horizonte-espacos dentro de cada familia): A (aprendiz contra o aleatorio; ql, qls, qo, alns), B (a validada contra o aprendiz; mesmos quatro), B' (o aprendiz contra a validada).
T = 1.000 de referencia: as mesmas 240 instancias na bateria da Entrada 234 (5 sementes).

PRE-REGISTRO (congelado ANTES de rodar o teste; hashes em results/entrada250_prereg_hashes.txt; um piloto em instancias de AJUSTE so' mediu o tempo)
-----------------------------------------------------------------------------------------------------------------------------------------------
(H1) O ganho da acao validada sobre o aleatorio cai com o horizonte nos dois espacos (S5 e S15: ganho(T=100.000) < ganho(T=1.000)). (65%)
(H2) Em todos os horizontes e nos dois espacos, a validada bate o Q-Learning padrao (ql) e o escalado (qls) (B, confirmada) (12 de 12). (85%)
(H3) O ganho do Q-Learning padrao sobre o aleatorio no S15 em T = 100.000 e' maior que em T = 1.000 (ele aprende com o horizonte). (70%)
(H4) Mesmo em T = 100.000, o melhor entre ql e qls ganha no maximo 50% do que a validada ganha, nos dois espacos. (65%)
(H5) O epsilon escalado (qls) nao supera o padrao (ql) alem da margem (0,05) em nenhum dos seis horizonte-espacos. (50%)
(H6) Nenhum aprendiz (ql, qls, qo, alns) bate a validada alem da margem (B', confirmada e nao equivalente) em nenhum horizonte e espaco. (88%)
Uso: python -m analysis.report_entrada250_ur_horizon
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
SPACES, HORIZONS = ("S5", "S15"), (5000, 20000, 100000)
LEARNERS = ["ql", "qls", "qo", "alns"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")


def load(sp, T):
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_hz_{sp}_T{T}_s*of*_entrada250.csv")))], ignore_index=True)
    return c.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()


def load_base(sp, keys):
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    m = m[m.method.isin(["rand", "ql", "qo", "alns", "val"])]
    w = m.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    return keys.merge(w, on=["cls", "rep"], how="inner")


def beats(F):
    return (F.confirmada) & (~F["equivalente_0.05"])


def main():
    W = {(sp, T): load(sp, T) for sp in SPACES for T in HORIZONS}
    base = {sp: load_base(sp, W[(sp, HORIZONS[0])][["cls", "rep"]]) for sp in SPACES}
    T = {k: [] for k in "ABC"}
    gains = []
    for sp in SPACES:
        b = base[sp]
        gains.append(dict(space=sp, T=1000, n=len(b), **{f"g_{k}": float(-rel(b[k], b.rand).mean()) for k in ["val", "ql", "qo", "alns"]}, g_qls=float("nan")))
        for t in HORIZONS:
            w = W[(sp, t)]
            gains.append(dict(space=sp, T=t, n=len(w), **{f"g_{k}": float(-rel(w[k], w.rand).mean()) for k in ["val", "ql", "qls", "qo", "alns"]}))
            for k in LEARNERS:
                a = test(w[k], w.rand); a.update(id=f"A-{sp}-T{t}-{k}"); T["A"].append(a)
                bb = test(w.val, w[k]); bb.update(id=f"B-{sp}-T{t}-{k}"); T["B"].append(bb)
                r = test(w[k], w.val); r.update(id=f"B'-{sp}-T{t}-{k}"); T["C"].append(r)
    G = pd.DataFrame(gains)
    # Holm dentro de cada familia e aprendiz sobre os seis horizonte-espacos
    F = {}
    for fam, rows in T.items():
        parts = []
        for k in LEARNERS:
            sub = [r for r in rows if r["id"].endswith(f"-{k}")]
            parts.append(family(sub))
        F[fam] = pd.concat(parts, ignore_index=True)
    L = ["=" * 110, "HORIZONTE LONGO NO R||Cmax (Entrada 250)", "=" * 110,
         "--- ganhos sobre o aleatorio ao mesmo T (pontos); T = 1.000: as mesmas instancias na bateria da Entrada 234 ---", G.to_string(index=False)]
    for fam, lab in (("A", "A: aprendiz contra o aleatorio"), ("B", "B: validada contra o aprendiz"), ("C", "B': aprendiz contra a validada")):
        L += ["", f"--- Familia {lab} (Holm sobre os 6 horizonte-espacos, por aprendiz) ---", F[fam][COLS].to_string(index=False)]
    g = G.set_index(["space", "T"])
    h1 = bool(all(g.loc[(sp, 100000), "g_val"] < g.loc[(sp, 1000), "g_val"] for sp in SPACES))
    qq = F["B"][F["B"].id.str.endswith(("-ql", "-qls"))]
    h2n = int(qq.confirmada.sum())
    h3 = bool(g.loc[("S15", 100000), "g_ql"] > g.loc[("S15", 1000), "g_ql"])
    h4 = {sp: float(max(g.loc[(sp, 100000), "g_ql"], g.loc[(sp, 100000), "g_qls"]) / g.loc[(sp, 100000), "g_val"]) for sp in SPACES}
    d = {}
    for sp in SPACES:
        for t in HORIZONS:
            w = W[(sp, t)]
            d[(sp, t)] = float(-rel(w.qls, w.ql).mean())
    h5 = bool(all(v <= 0.05 for v in d.values()))
    h6n = int(beats(F["C"]).sum())
    yn = lambda b: "SIM" if b else "NAO"
    L += ["", "--- previsoes ---",
          f"(H1) o ganho da validada cai de T = 1.000 para T = 100.000 nos dois espacos: {yn(h1)} (S5 {g.loc[('S5', 1000), 'g_val']:.3f} -> {g.loc[('S5', 100000), 'g_val']:.3f}; S15 {g.loc[('S15', 1000), 'g_val']:.3f} -> {g.loc[('S15', 100000), 'g_val']:.3f})",
          f"(H2) a validada bate ql e qls (confirmada): {h2n} de 12 (previsto 12): {yn(h2n == 12)}",
          f"(H3) ganho do ql no S15, T = 100.000 contra T = 1.000: {g.loc[('S15', 100000), 'g_ql']:.3f} contra {g.loc[('S15', 1000), 'g_ql']:.3f}: {yn(h3)}",
          f"(H4) melhor entre ql e qls / ganho da validada em T = 100.000: S5 {h4['S5']:.3f}, S15 {h4['S15']:.3f} (previsto <= 0,5): {yn(all(v <= 0.5 for v in h4.values()))}",
          f"(H5) qls supera ql alem da margem: {({f'{k[0]}-T{k[1]}': round(v, 3) for k, v in d.items()})} (previsto nenhum > 0,05): {yn(h5)}",
          f"(H6) aprendizes que batem a validada alem da margem: {h6n} de 24 (previsto 0): {yn(h6n == 0)}"]
    pred = [h1, h2n == 12, h3, all(v <= 0.5 for v in h4.values()), h5, h6n == 0]
    L.append(f"previsoes acertadas: {sum(pred)} de {len(pred)}")
    text = "\n".join(L)
    print(text)
    (R / "entrada250_ur_horizon.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([F["A"], F["B"]]).to_csv(R / "entrada250_ur_horizon_tests.csv", index=False)
    G.to_csv(R / "entrada250_ur_horizon_gains.csv", index=False)


if __name__ == "__main__":
    main()
