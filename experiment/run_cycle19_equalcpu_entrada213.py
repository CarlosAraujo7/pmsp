"""
experiment/run_cycle19_equalcpu_entrada213.py
===============================================
Entrada 213 (Ciclo 19, frente C): o seletor offline e o Q-Learning aquecido a CPU IGUAL, cada um com o SEU PROPRIO K (critica 1 da revisao cega #2: o orcamento de CPU igual foi aplicado so' a validada). K de uma acao escolhida pelo seletor = CPU da execucao com a acao do seletor / CPU do aleatorio, medida em SEQUENCIA (time.process_time, janelas de 1 s,
duas instancias medianas de n por grupo (familia, tercil de n) do conjunto de AJUSTE); idem para o Q aquecido (wq, tabela da Entrada 211). K do Q-Learning frio e da roleta: results/pub_cpu_ratio_entrada189.json; K da validada: results/valeq_kval_entrada195.json.
O braco A recebe T_A = round(1000 K_B / K_A) iteracoes (piso 50) contra o braco B com T = 1000. Metodos novos gravados (5 sementes): sel_eq_ql, sel_eq_alns, sel_eq_wq, sel_eq_val (o seletor contra ql, alns, wq e a validada), wq_eq_ql, wq_eq_alns, wq_eq_val (o Q aquecido contra ql, alns e a validada). T = 1000 e' o do braco B.
Fases (retomaveis; a calibracao SOZINHA na maquina): --calibrate | --run CONJUNTO ESPACO --shard i --nshards n. Saidas: results/eqcpu_k_entrada213.json, results/eqcpu_CONJUNTO_ESPACO_s*of*_entrada213.csv. Pre-registro: analysis/report_entrada213_equalcpu.py (docstring; hashes em results/entrada213_prereg_hashes.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import run_cycle17_published_entrada189 as P  # noqa: E402
from experiment import run_cycle18_selector_entrada200 as S  # noqa: E402
from experiment import run_cycle19_warmq_entrada211 as W  # noqa: E402
from experiment.run_agents_numba_entrada180 import arm_table  # noqa: E402
from pmsp import ig_numba_agents3 as H  # noqa: E402
from pmsp import pcmax_features as PF  # noqa: E402

R = ROOT / "results"
FIELDS = ["space", "cls", "rep", "method", "seed_i", "makespan", "lb", "T", "evals"]


def k_path():
    return R / "eqcpu_k_entrada213.json"


def sel_arm(model, rec, lb):
    return int(model.predict(PF.features(rec["p"], rec["m"], lb).reshape(1, -1))[0].argmin())


def run_fixed(space, rec, lb, a0, od, arm, seed, T):
    st, d, ls = arm_table(space, rec["n"])
    best, _, ev, used, _ = H.ig_agent_core3(rec["p"], a0, rec["m"], st, d, ls, T, H.AGENT_FIXED, arm, 0.1, 0.1, seed, od, lb, 0, 0, 0.0)
    return int(best), float(ev)


def calibrate():
    import joblib
    info = json.loads(P.groups_path().read_text())
    cal = P.load_set(P.M_CAL)
    P.warm()
    res = {}
    for sp in ("S5", "S15"):
        model = joblib.load(R / f"selector_model_{sp}_entrada200.joblib")
        Q0 = np.load(W.q_path(sp))
        res[sp] = {}
        for fam in P.FAMS:
            for g in range(3):
                mem = [r for r in cal if r["family"] == fam and info["calib"][r["member"]][1] == g]
                med = float(np.median([r["n"] for r in mem]))
                pick = sorted(mem, key=lambda r: (abs(r["n"] - med), r["inst_id"]))[:2]
                ratios = {"sel": [], "wq": []}
                for rec in pick:
                    lb, a0, od = P.prepare_inst(rec)
                    a = sel_arm(model, rec, lb)

                    def cpu(fn):
                        reps, t0 = 0, time.process_time()
                        while time.process_time() - t0 < 1.0:
                            fn(reps + 7)
                            reps += 1
                        return (time.process_time() - t0) / reps
                    base = cpu(lambda sd: P.run(sp, rec, lb, a0, od, H.AGENT_RANDOM, 0, 0.1, 0.1, sd))
                    ratios["sel"].append(cpu(lambda sd: run_fixed(sp, rec, lb, a0, od, a, sd, 1000)) / base)
                    ratios["wq"].append(cpu(lambda sd: W.run5(sp, rec, lb, a0, od, sd, W.HP_WQ, Q0, 0.05)) / base)
                res[sp][f"{fam}|{g}"] = {k: float(np.median(v)) for k, v in ratios.items()}
                print(f"[eqcpu-calib {sp}] {fam} grupo {g} (n ~ {med:.0f}): sel {res[sp][f'{fam}|{g}']['sel']:.2f}, wq {res[sp][f'{fam}|{g}']['wq']:.2f}", flush=True)
    k_path().write_text(json.dumps(res, indent=1))


def _append(out, rows, new):
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_space(conj, space, shard, nshards):
    import joblib
    model = joblib.load(R / f"selector_model_{space}_entrada200.joblib")
    Q0 = np.load(W.q_path(space))
    K = json.loads(k_path().read_text())[space]
    Kb = json.loads((R / "pub_cpu_ratio_entrada189.json").read_text())[space]
    Kv = json.loads((R / "valeq_kval_entrada195.json").read_text())["pub"][space]
    recs, info = S.test_records(conj)
    recs = recs[shard::nshards]
    out = R / f"eqcpu_{conj}_{space}_s{shard}of{nshards}_entrada213.csv"
    done = set()
    if out.exists():
        done = {r["cls"] for r in csv.DictReader(open(out, newline=""))}
    P.warm()
    new = not out.exists()
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        fam, g = info[rec["member"]]
        cls = f"{fam}|{g}|{rec['member']}"
        if cls in done:
            continue
        key = f"{fam}|{g}"
        lb, a0, od = P.prepare_inst(rec)
        a = sel_arm(model, rec, lb)
        Ksel, Kwq, Kql, Kal, Kval = K[key]["sel"], K[key]["wq"], Kb[key]["ql"], Kb[key]["alns"], Kv[key]
        rows = []
        Tof = lambda kb, ka: max(50, int(round(1000 * kb / ka)))
        for meth, who, kb in (("sel_eq_ql", "sel", Kql), ("sel_eq_alns", "sel", Kal), ("sel_eq_wq", "sel", Kwq), ("sel_eq_val", "sel", Kval), ("wq_eq_ql", "wq", Kql), ("wq_eq_alns", "wq", Kal), ("wq_eq_val", "wq", Kval)):
            T = Tof(kb, Ksel if who == "sel" else Kwq)
            for s in range(S.N_SEED_TEST):
                sd = P.seed_of(space, "test", fam, rec["inst_id"], meth, s)
                if who == "sel":
                    mk, ev = run_fixed(space, rec, lb, a0, od, a, sd, T)
                else:
                    mk, ev, _ = W.run5(space, rec, lb, a0, od, sd, W.HP_WQ, Q0, 0.05, T=T)
                rows.append(dict(space=space, cls=cls, rep=rec["inst_id"], method=meth, seed_i=s, makespan=mk, lb=lb, T=T, evals=ev))
        _append(out, rows, new)
        new = False
        if i % 200 == 0:
            print(f"[eqcpu {conj} {space} s{shard}] {i}/{len(recs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[eqcpu {conj} {space} s{shard}] concluido ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", nargs=2, default=None, metavar=("CONJUNTO", "ESPACO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
    if a.run:
        run_space(a.run[0], a.run[1], a.shard, a.nshards)
