"""
analysis/pretest_vnr_entrada198.py (Entrada 198, Ciclo 18, frente C)
=======================================================================
O PRE-TESTE de folga/ruido (pmsp.budgetfair.validated_to_noise): VNR = (ganho da acao validada, por validacao cruzada, sobre o aleatorio) / (ruido da execucao aleatoria), estimado SO' em instancias de AJUSTE. Pergunta: o VNR de uma celula (classe ou familia x tercil de n)
prediz se o Q-Learning online GANHA do aleatorio (a iteracoes iguais) nessa celula?
Esta rotina (a) calcula o VNR de cada celula a partir das sondas (results/pretest_*_entrada198.csv), (b) junta o desfecho de teste (ganho g_Q do Q-Learning sobre o aleatorio, %), (c) CALIBRA o limiar tau nas celulas sinteticas e de descoberta (maximiza a acuracia balanceada de 'g_Q > 0'; ponto medio entre vizinhos),
(d) descreve as celulas held-out (que o autor ja' tinha visto numa exploracao: NAO sao validacao cega) e (e) CONGELA tau e as regras em results/pretest_tau_entrada198.json. A validacao confirmatoria e' fora da amostra (familia F publicada, flowshop): analysis/report_entrada199_pretest_outofsample.py.
Funcoes reutilizaveis: cell_vnr(df), evaluate(vnr, gq, tau). Uso: python -m analysis.pretest_vnr_entrada198
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pmsp import budgetfair as bf  # noqa: E402

R = ROOT / "results"


def cell_vnr(d: pd.DataFrame) -> dict:
    """d: sondas de UMA celula (colunas rep, method, seed_i, makespan; metodos 'rand' e 'arm<k>'). Devolve noise, v_hat, vnr (pmsp.budgetfair.validated_to_noise)."""
    d = d.assign(inst=d.cls.astype(str) + "#" + d.rep.astype(str))      # inst_id das publicadas reinicia por classe: a chave da instancia e' (cls, rep)
    reps = sorted(d.inst.unique())
    r = d[d.method == "rand"].pivot(index="inst", columns="seed_i", values="makespan").loc[reps].to_numpy()
    arms = sorted((m for m in d.method.unique() if m.startswith("arm")), key=lambda s: int(s[3:]))
    a = np.stack([d[d.method == m].pivot(index="inst", columns="seed_i", values="makespan").loc[reps].to_numpy() for m in arms], axis=1)
    out = bf.validated_to_noise(r, a)
    out["n_inst"] = len(reps)
    return out


def evaluate(vnr, gq, tau) -> dict:
    """AUC de VNR para 'g_Q > 0', acuracia balanceada em tau e Spearman (celulas com vnr definido)."""
    x = pd.DataFrame(dict(vnr=vnr, gq=gq)).dropna()
    y = (x.gq > 0).to_numpy()
    out = dict(n=len(x), frac_pays=float(y.mean()))
    if y.all() or (~y).all():
        out.update(auc=float("nan"), bacc=float("nan"))
    else:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, x.vnr))
        pred = (x.vnr >= tau).to_numpy()
        out["bacc"] = float(0.5 * ((pred & y).sum() / y.sum() + (~pred & ~y).sum() / (~y).sum()))
    out["acc"] = float(((x.vnr >= tau).to_numpy() == y).mean())
    r = sstats.spearmanr(x.vnr, x.gq)
    out["spearman"] = float(r.statistic)
    return out


def calibrate_tau(vnr, gq) -> float:
    x = pd.DataFrame(dict(vnr=vnr, gq=gq)).dropna().sort_values("vnr")
    y = (x.gq > 0).to_numpy()
    v = x.vnr.to_numpy()
    best, bt = -1.0, None
    for i in range(len(v) - 1):
        t = 0.5 * (v[i] + v[i + 1])
        pred = v >= t
        bacc = 0.5 * ((pred & y).sum() / max(y.sum(), 1) + (~pred & ~y).sum() / max((~y).sum(), 1))
        if bacc > best:
            best, bt = bacc, t
    return float(bt)


def outcomes_syn(space):
    d = pd.read_csv(R / f"agents_{space}_entrada180.csv")
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "rep", "method"]).makespan.mean().unstack("method").reset_index()
    ms["g"] = -100 * (ms.ql - ms.rand) / ms.rand
    return ms.groupby("cls").g.mean()


def outcomes_pub(space, pat):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / pat.format(sp=space))))], ignore_index=True)
    ms = d[~d.method.str.startswith("arm")].groupby(["cls", "method"]).makespan.mean().unstack("method").reset_index()
    ms["g"] = -100 * (ms.ql - ms.rand) / ms.rand
    ms["cell"] = ms.cls.str.split("|").str[:2].str.join("|")
    return ms.groupby("cell").g.mean()


def main():
    rows = []
    for sp in ("S5", "S15"):
        d = pd.read_csv(R / f"pretest_syn_{sp}_entrada198.csv")
        og = outcomes_syn(sp)
        for cls, x in d.groupby("cls"):
            rows.append(dict(set="synthetic", space=sp, cell=cls, **cell_vnr(x), gq=float(og[cls])))
        d = pd.read_csv(R / f"pretest_pub_{sp}_entrada198.csv")
        d["cell"] = d.cls.str.split("|").str[:2].str.join("|")
        for nm, pat in (("discovery", "pub_{sp}_s*of*_entrada189.csv"), ("heldout", "pubh_{sp}_s*of*_entrada190.csv")):
            og = outcomes_pub(sp, pat)
            for cell, x in d.groupby("cell"):
                rows.append(dict(set=nm, space=sp, cell=cell, **cell_vnr(x), gq=float(og[cell])))
    C = pd.DataFrame(rows)
    cal = C[C.set.isin(["synthetic", "discovery"])]
    tau = calibrate_tau(cal.vnr, cal.gq)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    L = ["=" * 100, "PRE-TESTE VNR: calibracao nas celulas sinteticas e de descoberta (Entrada 198)", "=" * 100, C.to_string(index=False), f"\ntau (acuracia balanceada maxima em {len(cal)} celulas de calibracao): {tau:.3f}"]
    summ = {}
    for nm, sub in (("calibracao (sinteticas + descoberta)", cal), ("sinteticas", C[C.set == "synthetic"]), ("descoberta", C[C.set == "discovery"]), ("held-out (descritivo, nao cego)", C[C.set == "heldout"])):
        e = evaluate(sub.vnr, sub.gq, tau)
        summ[nm] = e
        L.append(f"{nm:42s} n={e['n']:3d} frac(Q ganha)={e['frac_pays']:.2f}  AUC={e['auc']:.3f}  acc={e['acc']:.3f}  acc.balanc.={e['bacc']:.3f}  Spearman(VNR, g_Q)={e['spearman']:+.3f}")
    text = "\n".join(L)
    print(text)
    (R / "pretest_vnr_entrada198.txt").write_text(text, encoding="utf-8")
    C.to_csv(R / "pretest_vnr_cells_entrada198.csv", index=False)
    (R / "pretest_tau_entrada198.json").write_text(json.dumps(dict(tau=tau, n_calibration_cells=int(len(cal)), calibration_summary=summ["calibracao (sinteticas + descoberta)"], rule="o seletor online (Q-Learning) paga na celula se VNR >= tau; VNR = ganho validado em validacao cruzada / ruido do aleatorio, estimado so' em instancias de ajuste"), indent=1))


if __name__ == "__main__":
    main()
