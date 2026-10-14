"""
experiment/compare_selectors_fixed_perturb.py
======================================
Compara estrategias de selecao de operador (ils_random, ils_ucb, ils_ql,
ils_rf) sob um regime de perturbacao corrigido (perturbation_threshold
menor -- achado do grid search, Entrada 22 do diario), para checar se a
diferenciacao entre metodos adaptativos e nao-adaptativos reaparece em
escala grande uma vez que o mecanismo de perturbacao funciona de
verdade (ao contrario do regime original, onde quase nada melhorava e
todo metodo empatava com LPT).

Uso:
    python3 -u experiment/compare_selectors_fixed_perturb.py N M REP_START REP_END \
        METHODS_CSV THRESHOLD N_PERTURB OUT_SUFFIX [MAX_ITER]

Salva/atualiza incrementalmente em
results/compare_selectors_{N}j_{M}m_th{THRESHOLD}_np{N_PERTURB}_{OUT_SUFFIX}.json
"""
from __future__ import annotations
import sys
import json
import os

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.ils import ils_run
from experiment.runner import make_ils_selector
from experiment.config import BASE_SEED, STAGNATION_WINDOW


def run_one(method, n, m, rep, threshold, n_perturb, max_iter):
    seed = BASE_SEED + rep
    inst = PMSPInstance.random(n, m, seed=seed)
    selector = make_ils_selector(method, seed)
    res = ils_run(inst, selector, max_iter=max_iter,
                   perturbation_threshold=threshold, n_perturb=n_perturb,
                   seed=seed, stagnation_window=STAGNATION_WINDOW)
    return {"gap_pct": res["gap_pct"], "n_improvements": res["n_improvements"]}


def main():
    n, m, rep_start, rep_end, methods_csv, threshold, n_perturb, out_suffix = sys.argv[1:9]
    n, m, rep_start, rep_end = int(n), int(m), int(rep_start), int(rep_end)
    threshold, n_perturb = int(threshold), int(n_perturb)
    max_iter = int(sys.argv[9]) if len(sys.argv) > 9 else 3000
    methods = methods_csv.split(",")

    out_path = f"results/compare_selectors_{n}j_{m}m_th{threshold}_np{n_perturb}_{out_suffix}.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)

    for method in methods:
        out.setdefault(method, {})
        for rep in range(rep_start, rep_end):
            if str(rep) in out[method]:
                continue
            out[method][str(rep)] = run_one(method, n, m, rep, threshold, n_perturb, max_iter)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  {method} concluido", flush=True)

    print(f"Salvo em {out_path}", flush=True)


if __name__ == "__main__":
    main()
