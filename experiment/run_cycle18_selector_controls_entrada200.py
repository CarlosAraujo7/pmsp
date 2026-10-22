"""
experiment/run_cycle18_selector_controls_entrada200.py
========================================================
Entrada 200, ESTAGIO 2 (registrado depois de ver o estagio 1 e ANTES de rodar estes controles): o seletor offline usou 5.720 instancias de treino e a validada estatica, so' 525; e o seletor roda ~15% mais avaliacoes que a validada. Controles que isolam o que e' CONDICIONAR NA INSTANCIA:
  valbig    a acao estatica por (familia, tercil de n) (os mesmos grupos e limiares da validada) escolhida nas MESMAS 5.720 instancias de treino do seletor (menor desempenho relativo centrado medio);
  valclass  a acao estatica por classe (familia, subpasta, m, n) escolhida nas mesmas 5.720 (uma tabela de consulta; cai para valbig se a classe nao tem treino);
  e a versao a AVALIACOES IGUAIS (sufixo E) de val, valbig e valclass: T' = round(1000 E_sel / E_c), onde E_sel e' a media (5 sementes) de avaliacoes do seletor na instancia e E_c a media do controle com T = 1000 na mesma instancia (a acao fixa nao tem custo de decisao, entao avaliacoes iguais = CPU igual).
Metodos gravados: valbig, valclass, valE, valbigE, valclassE (5 sementes cada; valE roda a acao validada da Entrada 189). Saidas: results/selctl_ESPACO_CONJUNTO_s*of*_entrada200.csv. Fases: --prepare | --run ESPACO CONJUNTO --shard i --nshards n.
Pre-registro: analysis/report_entrada200_controls.py (docstring, congelado antes de rodar; hashes em results/entrada200_controls_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle17_replication_entrada190 as RH  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402

R = ROOT / "results"
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals", "arm"]


def manifest_class(path):
    return {r["member"]: (r["family"], r["subfolder"], int(r["m"]), int(r["n"])) for r in csv.DictReader(open(path, newline=""))}


def prepare():
    """Tabelas estaticas aprendidas nas instancias de treino: valbig por (familia, grupo) e valclass por classe, por espaco."""
    thr = json.loads(P.groups_path().read_text())["thresholds"]
    cls_of = manifest_class(S.M_TRAIN)
    out = {}
    for sp in ("S5", "S15"):
        X, Y, fam = S.load_train(sp)
        members = list(Y.index)
        cl = [cls_of[mb] for mb in members]
        n = [c[3] for c in cl]
        grp = [(c[0], P.group_of(c[0], c[3], thr)) for c in cl]
        Yn = Y.to_numpy()
        big, klass = {}, {}
        for key in sorted(set(grp)):
            idx = [i for i, g in enumerate(grp) if g == key]
            big[f"{key[0]}|{key[1]}"] = int(Yn[idx].mean(axis=0).argmin())
        for key in sorted(set(cl)):
            idx = [i for i, g in enumerate(cl) if g == key]
            klass["|".join(map(str, key))] = dict(arm=int(Yn[idx].mean(axis=0).argmin()), n=len(idx))
        out[sp] = dict(valbig=big, valclass=klass)
    (R / "selctl_tables_entrada200.json").write_text(json.dumps(out, indent=1))
    print({sp: dict(valbig=out[sp]["valbig"], n_classes=len(out[sp]["valclass"])) for sp in out})


def run_space(space, conj, shard, nshards):
    tabs = json.loads((R / "selctl_tables_entrada200.json").read_text())[space]
    thr = json.loads(P.groups_path().read_text())["thresholds"]
    recs, info = S.test_records(conj)
    man = P.M_TEST if conj == "discovery" else RH.M_HOLD
    cls_of = manifest_class(man)
    ch = json.loads(P.choice_path(space).read_text())
    sel = {}
    for f in glob.glob(str(R / f"sel_{space}_{conj}_s*of*_entrada200.csv")):
        for r in csv.DictReader(open(f, newline="")):
            sel.setdefault(r["cls"], []).append(float(r["evals"]))
    out = R / f"selctl_{space}_{conj}_s{shard}of{nshards}_entrada200.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    nfall = 0
    for i, rec in enumerate(recs[shard::nshards]):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        lb, a0, od = P.prepare_inst(rec)
        key = "|".join(map(str, cls_of[rec["member"]]))
        a_big = tabs["valbig"][f"{fam}|{g}"]
        if key in tabs["valclass"]:
            a_cls = tabs["valclass"][key]["arm"]
        else:
            a_cls, nfall = a_big, nfall + 1
        arms = {"val": ch[fam]["val"][str(g)], "valbig": a_big, "valclass": a_cls}
        e_sel = float(np.mean(sel[cls]))
        rows = []
        for nm, a in arms.items():
            r1 = [P.run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], nm + "_ctl", s)) for s in range(S.N_SEED_TEST)]
            if nm != "val":
                for s, (mk, ev, _) in enumerate(r1):
                    rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=nm, seed_i=s, makespan=mk, lb=lb, T=1000, evals=ev, arm=a))
            Tq = max(50, int(round(1000 * e_sel / float(np.mean([x[1] for x in r1])))))
            for s in range(S.N_SEED_TEST):
                mk, ev, _ = P.run(space, rec, lb, a0, od, H.AGENT_FIXED, a, 0.1, 0.1, P.seed_of(space, "test", fam, rec["inst_id"], nm + "_ctlE", s), T=Tq)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=nm + "E", seed_i=s, makespan=mk, lb=lb, T=Tq, evals=ev, arm=a))
        S._append(out, FIELDS, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[selctl {space} {conj} s{shard}] {i} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[selctl {space} {conj} s{shard}] concluido ({time.perf_counter() - t0:.0f}s) | classes sem treino (valclass cai para valbig): {nfall}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", nargs=2, default=None, metavar=("ESPACO", "CONJUNTO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
