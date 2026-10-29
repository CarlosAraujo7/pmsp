"""
analysis/report_entrada242_pubaos.py
======================================
Ciclo 23, frente A: DOIS PROJETOS DE AOS DA LITERATURA -- o Q-Learning episodico de estado binario do IG para o flowshop ('iq'; Karimi-Mamaghan et al. 2023 e a extensao arXiv 2408.14864) e o FRRMAB ('fr'; Fialho et al. 2010) -- no protocolo, no P||Cmax sintetico ('pc') e no R||Cmax ('ur'), espacos S5 e S15, T = 1000. Desenho e dados:
experiment/run_cycle23_pubaos_entrada242.py (hiperparametros ajustados nas instancias de ajuste: os dos artigos nao estao disponiveis por inteiro; 100 instancias de teste por classe x 5 sementes); o aleatorio, a acao validada e os demais aprendizes vem dos arquivos existentes
(agents_S5/S15_entrada180.csv; ur_test_S5/S15_*_entrada234.csv). Diario: Entrada 242 (pre-registro) e 243 (resultados). Saidas: results/entrada242_pubaos.txt, entrada242_pubaos_*.csv.

PRE-REGISTRO (congelado ANTES de qualquer saida do teste; hashes em results/entrada242_prereg_hashes.txt, que inclui as escolhas do ajuste)
-----------------------------------------------------------------------------------------------------------------------------------------
Testes (Wilcoxon pareado, ties fora, IC90 por bootstrap, margem de equivalencia +-0,05; 'confirmada' = p_holm < 0,05 e o primeiro braco melhor): A = aprendiz contra o aleatorio a iteracoes iguais (Holm sobre 8 = 2 aprendizes x 2 espacos x 2 hospedeiros); B = acao validada contra o aprendiz (Holm sobre 8). Ganho = -d (pontos percentuais).
Transparencia: o ajuste (instancias de AJUSTE) ja' foi feito antes deste registro e as previsoes o levam em conta: a lacuna media ao limite inferior dos melhores ajustes e' de 0,27-0,33% (iq) e 0,24-0,26% (fr) no P||Cmax e de 17,5-18,9% no R||Cmax, valores parecidos com os do Q otimista ajustado (17,4-19,0%).
(A1) no P||Cmax nenhum dos dois projetos bate o aleatorio por mais que a margem (confirmada e nao paridade) em nenhum espaco (85%).  (A2) no R||Cmax pelo menos um dos dois bate o aleatorio por mais que a margem em pelo menos um espaco (50%).  (A3) a acao validada vence os dois projetos (confirmada) nos quatro pares hospedeiro x espaco, isto e', nas 8 comparacoes (88%).
(A4) no R||Cmax o 'iq' tem ganho maior que o do Q-Learning padrao (-1,06 em S5; -0,48 em S15) nos dois espacos (85%).  (A5) no R||Cmax o 'fr' tem ganho >= o da roleta ALNS em S15 (+0,148) (35%).  (A6) no R||Cmax o maior ganho de um projeto da literatura e' menor que 25% do ganho da validada em cada espaco (80%).
Leitura R5 ('os projetos de AOS da literatura tambem nao fecham a distancia para a acao estatica validada'): A3 e A6.
Uso: python -m analysis.report_entrada242_pubaos
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.report_entrada180_agents import family, rel, test  # noqa: E402

R = ROOT / "results"
COLS = ["id", "n", "better", "worse", "tie", "mean_d_pct", "ci90_lo", "ci90_hi", "p_raw", "p_holm", "confirmada", "equivalente_0.05", "leitura"]
pd.set_option("display.width", 230)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
AG = ["iq", "fr"]


def load(host, sp):
    if host == "pc":
        d = pd.read_csv(R / f"agents_{sp}_entrada180.csv")
        d["cls"] = d.cls
    else:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / f"ur_test_{sp}_s*of*_entrada234.csv")))], ignore_index=True)
    d = d[~d.method.str.startswith("arm")]
    ref = d[d.method.isin(["rand", "val", "ql"] + (["alns", "qo"] if host == "ur" else []))].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    t = pd.read_csv(R / f"pubaos_test_{host}_{sp}_entrada242.csv")
    pa = t.groupby(["cls", "rep", "method"]).makespan.mean().unstack("method")
    return ref.join(pa, how="inner").reset_index()


def main():
    L = ["=" * 110, "DOIS PROJETOS DE AOS DA LITERATURA NO PROTOCOLO (Entrada 242)", "=" * 110]
    W, rows, ta, tb = {}, [], [], []
    for host in ("pc", "ur"):
        for sp in ("S5", "S15"):
            w = load(host, sp)
            W[(host, sp)] = w
            g = {k: float(-rel(w[k], w.rand).mean()) for k in ["val", "ql"] + AG + (["alns", "qo"] if host == "ur" else [])}
            rows.append(dict(host=host, space=sp, n=len(w), **{f"g_{k}": v for k, v in g.items()}))
            for k in AG:
                t = test(w[k], w.rand); t.update(id=f"A-{host}-{sp}-{k}"); ta.append(t)
                t = test(w.val, w[k]); t.update(id=f"B-{host}-{sp}-{k}"); tb.append(t)
    Pd = pd.DataFrame(rows)
    FA, FB = family(ta), family(tb)
    L += ["\n--- ganhos sobre o aleatorio (pontos percentuais) ---", Pd.to_string(index=False), "\n--- A: aprendiz contra o aleatorio (negativo: o aprendiz e' melhor; Holm sobre 8) ---", FA[COLS].to_string(index=False), "\n--- B: acao validada contra o aprendiz (negativo: a validada e' melhor; Holm sobre 8) ---", FB[COLS].to_string(index=False)]
    yn = lambda b: "SIM" if b else "NAO"
    win = lambda X: ((X.confirmada) & (~X["equivalente_0.05"]) & (X.mean_d_pct < 0))
    a1 = not bool(win(FA[FA.id.str.contains("-pc-")]).any())
    a2 = bool(win(FA[FA.id.str.contains("-ur-")]).any())
    a3 = bool(FB.confirmada.all())
    ur = Pd[Pd.host == "ur"].set_index("space")
    a4 = bool((ur.g_iq > ur.g_ql).all())
    a5 = bool(ur.loc["S15", "g_fr"] >= ur.loc["S15", "g_alns"])
    a6 = bool(((ur[["g_iq", "g_fr"]].max(axis=1)) < 0.25 * ur.g_val).all())
    L += ["\n--- PREVISOES E LEITURA PRE-FIXADAS ---", f"(A1) P||Cmax: nenhum dos dois bate o aleatorio por mais que a margem: {yn(a1)}", f"(A2) R||Cmax: algum dos dois bate o aleatorio por mais que a margem: {yn(a2)}", f"(A3) a validada vence os dois nas 8 comparacoes (confirmada): {yn(a3)} ({int(FB.confirmada.sum())}/8)",
          f"(A4) R||Cmax: o iq supera o Q padrao nos dois espacos: {yn(a4)} (iq " + ", ".join(f"{v:+.3f}" for v in ur.g_iq) + "; ql " + ", ".join(f"{v:+.3f}" for v in ur.g_ql) + ")", f"(A5) R||Cmax: o fr >= a roleta ALNS em S15: {yn(a5)} (" + f"{ur.loc['S15', 'g_fr']:+.3f} vs {ur.loc['S15', 'g_alns']:+.3f})",
          f"(A6) R||Cmax: o maior ganho de um projeto da literatura < 25% do ganho da validada em cada espaco: {yn(a6)} (" + ", ".join(f"{m / v:.2f}" for m, v in zip(ur[['g_iq', 'g_fr']].max(axis=1), ur.g_val)) + ")", f"R5 (os projetos da literatura nao fecham a distancia: A3 e A6): {yn(a3 and a6)}"]
    text = "\n".join(L)
    print(text)
    (R / "entrada242_pubaos.txt").write_text(text, encoding="utf-8")
    Pd.to_csv(R / "entrada242_pubaos_pooled.csv", index=False)
    FA.to_csv(R / "entrada242_pubaos_A.csv", index=False)
    FB.to_csv(R / "entrada242_pubaos_B.csv", index=False)


if __name__ == "__main__":
    main()
