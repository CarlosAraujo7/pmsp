"""
experiment/run_cycle22_urselector_entrada240.py
=================================================
Entrada 240 (Ciclo 22, frente C): o SELETOR OFFLINE condicionado a instancia no hospedeiro de MAQUINAS NAO RELACIONADAS (R||Cmax; Entradas 234/235), a pergunta que as revisoes cegas #6-#7 fizeram sobre a justica do orcamento: no hospedeiro com folga real, o que um aprendizado OFFLINE (cujo custo de treino e' cobrado) captura que o online nao captura,
e o que ele acrescenta ao conhecimento de CLASSE (a acao validada e a tabela de consulta por classe)? Delineamento (o das Entradas 200/205 em escala menor): treino em 1.200 instancias NOVAS (100 por classe, sementes 40.000.000 + 1000 ci + rep, disjuntas das de ajuste 30.000.000 e de teste 20.000.000), cada acao fixa x 3 sementes (T = 1000, espacos S5 e S15),
atributos de pmsp/ur_features.py; ExtraTreesRegressor (300 arvores, folha minima 10, max_features 0,5, semente 0) multi-saida sobre o desempenho relativo centrado por instancia de cada acao; politica = argmin da predicao. Teste: as 1.200 instancias da Entrada 234, 5 sementes: 'sel' (a acao do seletor) e 'cls' (a acao de melhor makespan medio na CLASSE nas instancias de treino, uma tabela de consulta).
Nota de codigo: a primeira versao de fit() alinhava mal as linhas dos atributos e dos rotulos (o merge mantinha ordens diferentes); foi detectado porque o seletor perdia 2,4 pontos para a tabela de classe feita dos MESMOS rotulos, corrigido e o modelo ajustado de novo, e o teste refeito; as previsoes (registradas antes da primeira execucao) nao mudaram.
Fases (retomaveis): --label ESPACO | --fit | --run ESPACO. Saidas: results/urselector_train_ESPACO_entrada240.csv, urselector_feat_entrada240.csv, urselector_model_ESPACO_entrada240.joblib, urselector_class_ESPACO_entrada240.json, results/urselector_test_ESPACO_entrada240.csv. Pre-registro: analysis/report_entrada240_urselector.py (docstring; hashes em results/entrada240_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle22_ur_entrada234 as UR  # noqa: E402
from experiment.run_agents_numba_entrada180 import CLASSES, arm_table  # noqa: E402
from pmsp import ur_features as UF  # noqa: E402
from pmsp import ur_numba_agents as UA  # noqa: E402
from pmsp.instance_ur import constructive_ur, lower_bound_ur, makespan_ur, random_ur  # noqa: E402

R = ROOT / "results"
N_TRAIN = 100
SPACES = ["S5", "S15"]


def train_instance(n, m, rep, ci):
    P = random_ur(n, m, 40_000_000 + 1000 * ci + rep)
    a0, od = constructive_ur(P)
    return P, a0, od, lower_bound_ur(P)


def feat_of(P, a0, lb):
    return UF.features(P, lb, makespan_ur(P, a0))


def run_arm(space, P, a0, m, od, lb, n, arm, seed):
    st, d, ls = arm_table(space, n)
    best, _, ev, used, _ = UA.ur_agent_core(P, a0, m, st, d, ls, 1000, UA.AGENT_FIXED, arm, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, UA.HP_DEFAULT, 0)
    return int(best)


def label(space):
    UR.warm()
    A = len(arm_table(space, 100)[0])
    out = R / f"urselector_train_{space}_entrada240.csv"
    feat_out = R / "urselector_feat_entrada240.csv"
    write_feat = not feat_out.exists()
    rows, frows = [], []
    t0 = time.perf_counter()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(N_TRAIN):
            P, a0, od, lb = train_instance(n, m, rep, ci)
            if write_feat:
                frows.append([cls, rep] + UF.features(P, lb, makespan_ur(P, a0)).tolist())
            for a in range(A):
                for s in range(3):
                    rows.append((space, cls, rep, a, s, run_arm(space, P, a0, m, od, lb, n, a, 400_000_000 + 1000 * ci + rep + 50_000 * s + 7_919 * a + 20_000_000 * SPACES.index(space))))
        print(f"[urselector label {space}] {cls} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["space", "cls", "rep", "arm", "seed_i", "makespan"])
        w.writerows(rows)
    if write_feat:
        with open(feat_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cls", "rep"] + UF.NAMES)
            w.writerows(frows)


def fit():
    import joblib
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor
    F = pd.read_csv(R / "urselector_feat_entrada240.csv")
    for sp in SPACES:
        d = pd.read_csv(R / f"urselector_train_{sp}_entrada240.csv")
        m = d.groupby(["cls", "rep", "arm"]).makespan.mean().unstack("arm").reset_index()
        arms = [c for c in m.columns if c not in ("cls", "rep")]
        X = m[["cls", "rep"]].merge(F, on=["cls", "rep"], how="left")      # na ordem de m: atributos e rotulos ALINHADOS
        assert len(X) == len(m) and X[UF.NAMES].notna().all().all()
        mk = m[arms].to_numpy(float)
        mean = mk.mean(axis=1, keepdims=True)
        y = 100.0 * (mk - mean) / mean
        mdl = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=0, n_jobs=-1).fit(X[UF.NAMES].to_numpy(), y)
        joblib.dump(mdl, R / f"urselector_model_{sp}_entrada240.joblib")
        cls_tab = {c: int(g[arms].mean().idxmin()) for c, g in m.groupby("cls")}
        (R / f"urselector_class_{sp}_entrada240.json").write_text(json.dumps(cls_tab, indent=1))
        print(sp, "modelo ajustado; tabela de classe:", cls_tab)


def run(space):
    import joblib
    import pandas as pd
    mdl = joblib.load(R / f"urselector_model_{space}_entrada240.joblib")
    cls_tab = json.loads((R / f"urselector_class_{space}_entrada240.json").read_text())
    UR.warm()
    rows = []
    t0 = time.perf_counter()
    for ci, (n, m) in enumerate(CLASSES):
        cls = f"rand_{n}j_{m}m"
        for rep in range(UR.N_TEST):
            P, a0, od, lb = UR.make(n, m, rep, ci, False)
            x = UF.features(P, lb, makespan_ur(P, a0)).reshape(1, -1)
            sel = int(np.argmin(mdl.predict(x)[0]))
            for name, arm in (("sel", sel), ("cls", int(cls_tab[cls]))):
                for s in range(5):
                    rows.append((space, cls, rep, name, s, arm, run_arm(space, P, a0, m, od, lb, n, arm, 500_000_000 + 1000 * ci + rep + 50_000 * s + 7_919 * (name == "cls") + 20_000_000 * SPACES.index(space))))
        print(f"[urselector run {space}] {cls} ({time.perf_counter() - t0:.0f}s)", flush=True)
    with open(R / f"urselector_test_{space}_entrada240.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["space", "cls", "rep", "method", "seed_i", "arm", "makespan"])
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    if a.label:
        label(a.label)
    if a.fit:
        fit()
    if a.run:
        run(a.run)
