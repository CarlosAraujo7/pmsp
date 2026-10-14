"""
experiment/grid_search_perturb.py
======================================
Busca sistematica conjunta de (n_perturb, perturbation_threshold) --
Entrada 21/22 do diario: testa se uma janela de recuperacao maior
(perturbation_threshold) permite que uma perturbacao mais forte
(n_perturb) valha a pena em instancias grandes (150+ tarefas), em vez
de se comportar como reinicio semi-aleatorio (hipotese da Entrada 21).

Usa RandomSelector para isolar o mecanismo de perturbacao do efeito de
selecao de operador -- a pergunta desta etapa e "existe ALGUMA
combinacao de (n_perturb, threshold) que destrava a busca nesta
escala", antes de comparar estrategias de selecao nessa combinacao
(etapa 2, script separado).

Uso:
    python3 -u experiment/grid_search_perturb.py N M REP_START REP_END OUT_SUFFIX \
        [MAX_ITER] [N_PERTURB_CSV] [THRESHOLD_CSV]

Salva/atualiza incrementalmente em
results/grid_perturb_{N}j_{M}m_{OUT_SUFFIX}.json (merge com o que ja
existir no arquivo, para permitir rodar em lotes sem perder progresso).
"""
from __future__ import annotations
import sys
import json
import os
import time

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.ils import ils_run
from pmsp.selectors import RandomSelector
from experiment.config import BASE_SEED, STAGNATION_WINDOW

DEFAULT_N_PERTURB_GRID = [3, 6, 12, 22]
DEFAULT_THRESHOLD_GRID = [50, 100, 200, 400]


def run_one(n, m, rep, n_perturb, threshold, max_iter):
    seed = BASE_SEED + rep
    inst = PMSPInstance.random(n, m, seed=seed)
    selector = RandomSelector(seed=seed)
    res = ils_run(inst, selector, max_iter=max_iter,
                   perturbation_threshold=threshold, n_perturb=n_perturb,
                   seed=seed, stagnation_window=STAGNATION_WINDOW)
    return {"gap_pct": res["gap_pct"], "n_improvements": res["n_improvements"]}


def main():
    n, m, rep_start, rep_end, out_suffix = sys.argv[1:6]
    n, m, rep_start, rep_end = int(n), int(m), int(rep_start), int(rep_end)
    max_iter = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
    np_grid = [int(x) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else DEFAULT_N_PERTURB_GRID
    th_grid = [int(x) for x in sys.argv[8].split(",")] if len(sys.argv) > 8 else DEFAULT_THRESHOLD_GRID

    out_path = f"results/grid_perturb_{n}j_{m}m_{out_suffix}.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)

    t0 = time.time()
    for np_ in np_grid:
        for th in th_grid:
            key = f"np{np_}_th{th}"
            out.setdefault(key, {})
            for rep in range(rep_start, rep_end):
                if str(rep) in out[key]:
                    continue
                out[key][str(rep)] = run_one(n, m, rep, np_, th, max_iter)
            print(f"  {key} concluido ({time.time()-t0:.1f}s acumulado)", flush=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    print(f"Salvo em {out_path}", flush=True)


if __name__ == "__main__":
    main()
