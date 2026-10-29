"""
analysis/report_entrada244_irace.py
=====================================
Ciclo 24: O IRACE COMO CONFIGURADOR (R 4.6.1, pacote irace 4.5, configuracoes padrao exceto maxExperiments e a semente). Desenho e dados: experiment/run_cycle24_irace_entrada244.py (alvo em Python/numba; o irace fala com ele por soquete) e experiment/irace_target_entrada244.R.
 'cfg5' / 'cfg15': o irace ajusta a CONFIGURACAO ESTATICA do IG por classe (tamanho de destruicao d/n continuo em [0,05; 0,40]; em cfg15 tambem a estrategia {0, 1, 2}), nas 30 instancias de ajuste; 'q5' / 'q15': o irace ajusta, por classe, alfa, gama, o decaimento e o piso do epsilon do Q-Learning de estados.
 Orcamento = bm x o custo de ajuste da acao validada na mesma classe e espaco (pc: 150/450 execucoes; ur: 300/900; para q, no minimo 180, o piso exigido pelo irace com quatro parametros), bm em {1, 3, 10}; o principal e' bm = 1. Teste: a melhor configuracao por classe nas 100 instancias de teste x 5 sementes (T = 1000) contra o aleatorio ('rand'), a acao validada ('val') e o Q-Learning padrao ('ql') dos arquivos
 existentes (agents_S5/S15_entrada180.csv; ur_test_S5/S15_*_entrada234.csv). Hospedeiros: pc (P||Cmax sintetico) e ur (R||Cmax). Diario: Entrada 244 (pre-registro) e 245 (resultados). Saidas: results/entrada244_irace.txt, entrada244_irace_*.csv.

PRE-REGISTRO (congelado ANTES do ajuste completo e de qualquer saida do teste; hashes em results/entrada244_prereg_hashes.txt)
-----------------------------------------------------------------------------------------------------------------------------
Testes (Wilcoxon pareado, ties fora, IC90 por bootstrap, margem de equivalencia +-0,05; Holm dentro de cada familia de 4 testes = host x espaco). Ganho = -d em pontos percentuais. Familias: FA = cfg contra o aleatorio; FB = a acao validada contra cfg; FC = q contra o aleatorio; FD = a acao validada contra q; FE = q contra o Q-Learning padrao. 'Bate alem da margem' = p_holm < 0,05, primeiro braco melhor
e o IC90 fora de +-0,05 (nao 'equivalente').
Transparencia: o codigo foi depurado em 2 classes de AJUSTE (20x4 e 100x20) com bm = 1, sem tocar nas instancias de teste; isso mostrou duas coisas: o irace exige no minimo 180 experimentos com quatro parametros (o piso do orcamento de q) e as configuracoes de q escolhidas nos pilotos tem piso de epsilon alto (0,10-0,49), isto e, perto de escolha aleatoria. As previsoes levam isso em conta.
(I1) cfg (bm = 1) bate o aleatorio alem da margem nos quatro pares host x espaco (FA) (60%).  (I2) cfg (bm = 1) nao bate a acao validada alem da margem em nenhum dos quatro pares (85%).  (I3) a acao validada bate cfg (bm = 1) alem da margem em pelo menos um dos quatro pares (60%).
(I4) q (bm = 1) bate o Q-Learning padrao (FE, confirmada) nos dois espacos do R||Cmax (85%).  (I5) q (bm = 1) nao bate o aleatorio alem da margem em nenhum espaco do P||Cmax (80%).  (I6) a acao validada bate q (bm = 1) (FD, confirmada) nos quatro pares (85%).
(I7) no R||Cmax o ganho de q (bm = 1) sobre o aleatorio e' menor que 25% do ganho da acao validada em cada espaco (70%).  (I8) com bm = 10, cfg bate a acao validada alem da margem em pelo menos um dos quatro pares (40%).
Leitura R6 ('um configurador estabelecido nao muda o quadro: a configuracao estatica do irace nao supera a validada e o Q-Learning ajustado pelo irace continua atras dela'): I2 e I6.
Uso: python -m analysis.report_entrada244_irace
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
HOSTS = ("pc", "ur")
SPACES = ("S5", "S15")
BMS = (1, 3, 10)
IRT = {h: pd.read_csv(R / f"irace_test_{h}_entrada244.csv") for h in HOSTS}


def load(host, sp, bm):
    if host == "pc":
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
    else:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    d = d[~d.method.str.startswith("arm")]
    ref = d[d.method.isin(["rand", "val", "ql"])].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    t = IRT[host]
    t = t[t.space == sp]
    pa = t.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    k = sp[1:]
    pa = pa.rename(columns={f"cfg{k}_b{bm}": "cfg", f"q{k}_b{bm}": "q"})[["cfg", "q"]]
    return ref.join(pa, how="inner").reset_index()


def analyse(bm):
    W, rows, T = {}, [], {k: [] for k in "ABCDE"}
    for host in HOSTS:
        for sp in SPACES:
            w = load(host, sp, bm)
            W[(host, sp)] = w
            rows.append(dict(host=host, space=sp, bm=bm, n=len(w), **{f"g_{k}": float(-rel(w[k], w.rand).mean()) for k in ["val", "ql", "cfg", "q"]}))
            for key, a, b in (("A", "cfg", "rand"), ("B", "val", "cfg"), ("C", "q", "rand"), ("D", "val", "q"), ("E", "q", "ql")):
                t = test(w[a], w[b]); t.update(id=f"{key}-{host}-{sp}-b{bm}"); T[key].append(t)
    Pd = pd.DataFrame(rows)
    F = {k: family(v) for k, v in T.items()}
    # o sentido contrario (cfg melhor que a validada): teste simetrico, Holm sobre 4
    rev = family([dict(test(W[(h, s)].cfg, W[(h, s)].val), id=f"B'-{h}-{s}-b{bm}") for h in HOSTS for s in SPACES])
    return Pd, F, rev


def beats(Fm):
    return (Fm.confirmada) & (~Fm["equivalente_0.05"]) & (Fm.mean_d_pct < 0)


def main():
    L = ["=" * 110, "O IRACE COMO CONFIGURADOR (Entrada 244)", "=" * 110]
    res = {bm: analyse(bm) for bm in BMS}
    for bm in BMS:
        Pd, F, rev = res[bm]
        L += [f"\n########## orcamento bm = {bm} x o custo de ajuste da acao validada " + ("(PRINCIPAL)" if bm == 1 else "(sensibilidade)") + " ##########", "--- ganhos sobre o aleatorio (pontos percentuais) ---", Pd.to_string(index=False)]
        for key, nm in (("A", "cfg contra o aleatorio"), ("B", "acao validada contra cfg (negativo: a validada e' melhor)"), ("C", "q contra o aleatorio"), ("D", "acao validada contra q"), ("E", "q contra o Q-Learning padrao")):
            L += [f"\n--- F{key}: {nm} (Holm sobre 4) ---", F[key][COLS].to_string(index=False)]
        L += [f"\n--- F{chr(66)}': cfg contra a acao validada (negativo: cfg e' melhor; Holm sobre 4) ---", rev[COLS].to_string(index=False)]
    Pd1, F1, rev1 = res[1]
    Pd10, F10, rev10 = res[10]
    yn = lambda b: "SIM" if b else "NAO"
    i1 = bool(beats(F1["A"]).all())
    i2 = not bool(beats(rev1).any())
    i3 = bool(beats(F1["B"]).any())
    ur = Pd1[Pd1.host == "ur"].set_index("space")
    i4 = bool(F1["E"][F1["E"].id.str.contains("-ur-")].pipe(lambda X: X.confirmada.all()))
    i5 = not bool(beats(F1["C"][F1["C"].id.str.contains("-pc-")]).any())
    i6 = bool(F1["D"].confirmada.all())
    i7 = bool((ur.g_q < 0.25 * ur.g_val).all())
    i8 = bool(beats(rev10).any())
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS (bm = 1, exceto I8) ---",
          f"(I1) cfg bate o aleatorio alem da margem nos 4 pares: {yn(i1)} ({int(beats(F1['A']).sum())}/4)",
          f"(I2) cfg nao bate a validada alem da margem em nenhum par: {yn(i2)} ({int(beats(rev1).sum())} pares em que bate)",
          f"(I3) a validada bate cfg alem da margem em pelo menos um par: {yn(i3)} ({int(beats(F1['B']).sum())}/4)",
          f"(I4) q bate o Q padrao (confirmada) nos 2 espacos do R||Cmax: {yn(i4)} ({int(F1['E'][F1['E'].id.str.contains('-ur-')].confirmada.sum())}/2)",
          f"(I5) q nao bate o aleatorio alem da margem no P||Cmax: {yn(i5)} ({int(beats(F1['C'][F1['C'].id.str.contains('-pc-')]).sum())} pares em que bate)",
          f"(I6) a validada bate q (confirmada) nos 4 pares: {yn(i6)} ({int(F1['D'].confirmada.sum())}/4)",
          f"(I7) R||Cmax: ganho de q < 25% do ganho da validada em cada espaco: {yn(i7)} (" + ", ".join(f"{a:.3f}/{b:.3f}={a / b:.2f}" for a, b in zip(ur.g_q, ur.g_val)) + ")",
          f"(I8) com bm = 10, cfg bate a validada alem da margem em pelo menos um par: {yn(i8)} ({int(beats(rev10).sum())}/4)",
          f"R6 (o irace nao muda o quadro: I2 e I6): {yn(i2 and i6)}"]
    # descritivo: o que o irace escolheu
    L += ["\n--- o que o irace escolheu (bm = 1) ---"]
    for host in HOSTS:
        tj = [json.loads(x) for x in (R / f"irace_tune_{host}_entrada244.jsonl").read_text().splitlines()]
        td = pd.DataFrame([dict(kind=r["kind"], bm=r["bm"], ci=r["ci"], n_exp=r["n_exp"], secs=r["secs"], **r["best"]) for r in tj])
        q = td[(td.bm == 1) & td.kind.str.startswith("q")]
        c = td[(td.bm == 1) & td.kind.str.startswith("cfg")]
        L += [f"{host}: execucoes do IG no ajuste por job: mediana {td.n_exp.median():.0f}; irace-Q (bm=1) piso do epsilon mediano {q['floor'].median():.3f}, alfa mediano {q.alpha.median():.3f}, gama mediano {q.gamma.median():.3f}, decaimento mediano {q.decay.median():.4f}; "
              f"cfg (bm=1) d/n mediano {c.dfrac.median():.3f} (minimo {c.dfrac.min():.3f}, maximo {c.dfrac.max():.3f}); estrategia != 0 em cfg15: {int((c[c.kind == 'cfg15'].get('strat', pd.Series(dtype=float)) != 0).sum())} de 12"]
    text = "\n".join(L)
    print(text)
    (R / "entrada244_irace.txt").write_text(text, encoding="utf-8")
    pd.concat([res[b][0] for b in BMS]).to_csv(R / "entrada244_irace_pooled.csv", index=False)
    for key in "ABCDE":
        pd.concat([res[b][1][key] for b in BMS]).to_csv(R / f"entrada244_irace_F{key}.csv", index=False)
    pd.concat([res[b][2] for b in BMS]).to_csv(R / "entrada244_irace_FBrev.csv", index=False)


if __name__ == "__main__":
    main()
