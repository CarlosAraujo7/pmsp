"""
experiment/run_cycle54_logs_entrada287.py (Entrada 287, Ciclo 54)
===================================================================
REGISTRO POR DECISAO do controle aleatorio nas 69 celulas do artigo do CEC (24 de P||Cmax sintetico, 18 de P||Cmax publicado held-out, 21 do benchmark padrao de R||Cmax, 6 das familias de R||Cmax).
So' gera dados: nenhuma analise nem previsao aqui (a analise e as previsoes ficam em analysis/report_entrada287_learnability.py e sao congeladas antes de abrir os registros).

Cada execucao e' o IG do hospedeiro da celula com o agente ALEATORIO (agent = 1) e o mesmo espaco de acoes da celula (S5, S15, B5, B15 ou B30), T = 1000, pelos nucleos com registro pmsp/ig_logged.py e pmsp/ur_logged.py
(equivalencia bit a bit com os originais: tests/test_logged_cores_entrada287.py). Para cada decisao t: acao, estado (0..80), recompensa BRUTA max(melhora do makespan, 0). Por execucao: makespan inicial, makespan final, limite inferior, tamanho (n, m).
Instancias: ate' 100 por celula (as primeiras das sinteticas e das familias, por identificador nas publicadas, todas as do benchmark padrao), 5 sementes por instancia (a escala da bateria principal; as execucoes custam no maximo 0,15 s).
Saida: results/logs287/<celula>.npz (comprimido; NAO vai para o git: e' regeneravel e determinista) e results/logs287_manifest_entrada287.csv (celula, execucoes, md5 do arquivo).
Uso: python -m experiment.run_cycle54_logs_entrada287 --list | --pilot | --all --workers 10 | --cell "S5|rand_20j_4m"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R = ROOT / "results"
LOGS = R / "logs287"
T = 1000
HP = np.array([0.1, 0.9, 0.995, 0.05, 0.0])
N_INST = {"sintetico": 100, "publicado": 100, "familias": 100, "padrao": 100}
N_SEED = 5
SEED0 = 61_000_000
TYPES = ["U1a100", "U10a100", "U100a120", "U100a200", "U1000a1100", "JobsCorre", "MaqCorre"]
PADRAO_SPACES = ("B5", "B15", "B30")
PCMAX_SPACES = ("S5", "S15")


def fname(cell: str) -> Path:
    return LOGS / (cell.replace("|", "_") + ".npz")


def list_cells() -> list[tuple[str, str]]:
    """(colecao, celula) das 69 celulas; os nomes sao os de analysis/cells_instance_tables.py."""
    from experiment import run_agents_numba_entrada180 as E
    from experiment import run_cycle30_ur_external_entrada254 as E2
    from experiment import run_cycle17_replication_entrada190 as RP
    out = []
    for sp in PADRAO_SPACES:
        out += [("padrao", f"{sp}|{ty}") for ty in TYPES]
    for sp in PCMAX_SPACES:
        out += [("familias", f"{sp}|{f}") for f in dict.fromkeys(c[0] for c in E2.CLASSES)]
    for sp in PCMAX_SPACES:
        out += [("sintetico", f"{sp}|rand_{n}j_{m}m") for (n, m) in E.CLASSES]
    info = json.loads(RP.groups_path().read_text())
    groups = sorted({(fam, g) for fam, g in info["test"].values()})
    for sp in PCMAX_SPACES:
        out += [("publicado", f"{sp}|{fam}|{g}") for fam, g in groups]
    return out


def run_ig(p, a0, m, od, lb, st, d, ls, seed):
    from pmsp import ig_logged as IL
    return IL.ig_agent_core_logged(p, a0, m, st, d, ls, T, 1, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, len(st))), 1.0, 1)


def run_ur(P, a0, m, od, lb, st, d, ls, seed):
    from pmsp import ur_logged as UL
    return UL.ur_agent_core_logged(P, a0, m, st, d, ls, T, 1, 0, 0.1, 0.1, seed, od, lb, 0, 0, 0.0, HP, 0, np.zeros((81, len(st))), 1.0, 1)


def instances_of(coll: str, cell: str, pilot: bool):
    """Gera (chave, X, a0, m, od, lb, st, d, ls, nucleo) para as instancias escolhidas da celula."""
    k = 2 if pilot else None
    if coll == "sintetico":
        from experiment import run_agents_numba_entrada180 as E
        sp, cl = cell.split("|")
        n, m = (int(v) for v in cl.replace("rand_", "").replace("j", "").replace("m", "").split("_"))
        st, d, ls = E.arm_table(sp, n)
        for rep in range(k or N_INST[coll]):
            inst, p, a0, od = E.make(n, m, rep, False)
            yield (f"rep{rep}", p, a0, m, od, float(inst.lower_bound), st, d, ls, run_ig)
    elif coll == "familias":
        from experiment import run_cycle30_ur_external_entrada254 as E2
        from experiment.run_agents_numba_entrada180 import arm_table
        sp, fam = cell.split("|")
        cls_ix = [i for i, c in enumerate(E2.CLASSES) if c[0] == fam]
        per = max(1, N_INST[coll] // len(cls_ix))
        for ci in cls_ix:
            f, n, m = E2.CLASSES[ci]
            st, d, ls = arm_table(sp, n)
            for rep in range(min(per, E2.N_TEST) if not pilot else 1):
                P, a0, od, lb = E2.make(ci, rep, False)
                yield (f"c{ci}r{rep}", P, a0, m, od, float(lb), st, d, ls, run_ur)
                if pilot:
                    break
            if pilot:
                break
    elif coll == "publicado":
        from experiment import run_cycle17_published_entrada189 as P
        from experiment import run_cycle17_replication_entrada190 as RP
        from experiment.run_agents_numba_entrada180 import arm_table
        sp, fam, g = cell.split("|")
        info = json.loads(RP.groups_path().read_text())
        recs = [r for r in P.load_set(RP.M_HOLD) if tuple(info["test"][r["member"]]) == (fam, int(g))]
        recs = sorted(recs, key=lambda r: str(r["inst_id"]))[: (k or N_INST[coll])]
        for rec in recs:
            lb, a0, od = P.prepare_inst(rec)
            st, d, ls = arm_table(sp, rec["n"])
            yield (f"{rec['member']}", rec["p"], a0, int(rec["m"]), od, float(lb), st, d, ls, run_ig)
    else:
        from experiment import run_cycle32_fanjul_entrada256 as F
        sp, ty = cell.split("|")
        man = F.manifest()
        man = man[(man.role == "test") & (man["type"] == ty)].sort_values("ri")
        step = max(1, len(man) // N_INST[coll])
        man = man.iloc[::step].head(k or N_INST[coll])
        for _, r in man.iterrows():
            Pm, a0, od, lb = F.load(r)
            st, d, ls = F.arm_table_b(sp, int(r.n))
            yield (f"{ty}{int(r.instance)}", Pm, a0, int(r.m), od, float(lb), st, d, ls, run_ur)


def do_cell(args):
    coll, cell, ci, pilot = args
    out = fname(cell)
    if out.exists() and not pilot:
        return cell, 0, "existe", 0.0
    t0 = time.perf_counter()
    A, S, Rw, C0, BEST, LB, NN, MM, INST, SEEDS = [], [], [], [], [], [], [], [], [], []
    arm_d = arm_strat = arm_ls = None
    for ii, (key, X, a0, m, od, lb, st, d, ls, core) in enumerate(instances_of(coll, cell, pilot)):
        arm_d, arm_strat, arm_ls = d, st, ls
        for s in range(1 if pilot else N_SEED):
            seed = SEED0 + 100_000 * ci + 100 * ii + s
            res = core(X, a0, m, od, lb, st, d, ls, seed)
            best, la, lsate, lc, lr = int(res[0]), res[6], res[7], res[8], res[9]
            A.append(la.astype(np.int8)); S.append(lsate.astype(np.int8)); Rw.append(lr.astype(np.float32))
            C0.append(float(lc[0])); BEST.append(best); LB.append(lb); NN.append(len(X)); MM.append(m); INST.append(key); SEEDS.append(seed)
    if pilot:
        return cell, len(A), "piloto", time.perf_counter() - t0
    LOGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, A=np.array(A), S=np.array(S), Rw=np.array(Rw), C0=np.array(C0), best=np.array(BEST), lb=np.array(LB), n=np.array(NN), m=np.array(MM),
                        inst=np.array(INST), seed=np.array(SEEDS), arm_d=np.asarray(arm_d), arm_strat=np.asarray(arm_strat), arm_ls=np.asarray(arm_ls), coll=np.array([coll]))
    return cell, len(A), "ok", time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cells = list_cells()
    if a.list:
        print(len(cells), "celulas")
        for c in cells:
            print(*c)
        return
    jobs = [(coll, cell, i, a.pilot) for i, (coll, cell) in enumerate(cells) if a.cell is None or cell == a.cell]
    if a.pilot:                                   # uma celula de cada colecao e espaco, para medir o tempo
        seen, sel = set(), []
        for j in jobs:
            key = (j[0], j[1].split("|")[0])
            if key not in seen:
                seen.add(key); sel.append(j)
        jobs = sel
    t0 = time.perf_counter()
    rows = []
    with Pool(a.workers) as pool:
        for cell, n, status, dt in pool.imap_unordered(do_cell, jobs):
            print(f"[logs287] {cell}: {status}, {n} execucoes, {dt:.0f}s (total {time.perf_counter() - t0:.0f}s)", flush=True)
    if a.all:
        for coll, cell in cells:
            f = fname(cell)
            if f.exists():
                rows.append(f"{cell},{coll},{hashlib.md5(f.read_bytes()).hexdigest()}")
        (R / "logs287_manifest_entrada287.csv").write_text("cell,coll,md5\n" + "\n".join(rows) + "\n", encoding="utf-8")
        print("manifesto:", len(rows), "arquivos")


if __name__ == "__main__":
    main()
