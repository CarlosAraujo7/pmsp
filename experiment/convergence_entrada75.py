# -*- coding: utf-8 -*-
"""
experiment/convergence_entrada75.py
=====================================
Gera dados e figura de convergencia (GAP% best-so-far vs. iteracao) para
ig_ql vs. ils_ql vs. sa_li_perkins -- item do Item 8.1 de
ANALISE_E_PLANO_EJOR.md ("os graficos de convergencia ig_ql vs. SOTA"),
Entrada 75.

Por que um script avulso (nao parte de experiment/runner.py):
run_single()/run_all() NUNCA passam track_history=True (todos os 4
run_*() de pmsp/ suportam o parametro -- ig_run, ils_run, sa_run,
bco_run -- mas o runner de producao nao o usa, para nao gastar memoria
guardando historico em uma bateria de milhares de tarefas). Reexecutar
a bateria inteira so para isto desperdicaria tempo de computo; este
script roda 1 replica por tamanho de instancia (inst_id=0, mesma
seed=BASE_SEED+0 da bateria real), reusando a MESMA logica de
construcao de instancia/seletor de experiment/runner.py
(PMSPInstance.random, make_ils_selector) para que a curva reflita
fielmente o protocolo experimental real, e nao um substituto ad-hoc.

Tamanhos escolhidos (3 paineis; Carlos aprovou incluir uma classe
grande em 08/09/2026):
    (20,  4)  -- menor classe da bateria principal publicada
                 (INSTANCE_CONFIGS[0] em experiment/config.py).
    (100, 20) -- maior classe da bateria principal publicada
                 ("muito grande", INSTANCE_CONFIGS[-1]).
    (300, 20) -- uma das "4 classes grandes" (150-300 tarefas) da
                 Entrada 70/Task #101 (ver experiment/config.py,
                 comentario de MAX_ITER_BY_SIZE, e
                 run_large_missing_methods_entrada68.py). Escolhida
                 entre as 4 (as outras sao (150,15), (150,30),
                 (250,25)) porque tem o MENOR orcamento em
                 MAX_ITER_BY_SIZE (2000, o dobro do MAX_ITER generico
                 usado aqui) -- o proprio comentario de config.py
                 registra que (300,20) "plato quase de imediato", o
                 que a torna rapida de gerar e visualmente clara para
                 uma figura ilustrativa de convergencia.

Orcamento: max_iter=CFG.MAX_ITER (1000) uniforme nos 3 tamanhos -- o
MESMO orcamento-base usado por ig_ql/ils_ql/sa_li_perkins na bateria
principal (nenhum dos 3 pertence a familia "_hi" de orcamento
estendido, entao a curva reflete o regime real usado no artigo, nao um
orcamento arbitrario so para o grafico).

Metodos: ig_ql (metodo headline do artigo), ils_ql (melhor variante
ILS baseada em RL -- comparador direto da familia F1) e sa_li_perkins
(um dos 3 baselines SOTA -- comparador direto da familia F2). bco fica
de fora deste primeiro corte por simplicidade -- seu 'history' tem
granularidade por iteracao EXTERNA da colonia (nao por avaliacao
individual), semantica ja conferida em pmsp/bco.py mas nao explorada
aqui; ver diario_pesquisa.md Entrada 75 se for adicionado depois.
multifit/lpt ficam de fora por serem construtivos (uma unica solucao,
sem historico iterativo).

ACHADO da Entrada 75 (por isso esta versao existe -- v2): sa_run() tem
uma regra de PARADA ANTECIPADA legitima e documentada (pmsp/sa.py: para
se best_makespan <= lb*p_target, p_target=1.01 por padrao -- fidelidade
ao artigo original de Li & Perkins, nao um bug). Isso significa que
r["history"] para sa_li_perkins pode ter MENOS de max_iter pontos --
inclusive ZERO pontos, se a propria solucao LPT inicial ja satisfizer
o alvo (caso real observado aqui em (300,20): LPT ja comeca com GAP=
0.25%, abaixo do alvo de ~1%, entao o loop de sa_run() nunca executa
seu corpo nem uma vez). O "melhor-ate-agora" e, por definicao, CONSTANTE
a partir do ponto de parada (nenhuma busca adicional ocorreu), entao
_history_to_gap() abaixo estende (forward-fill) a serie ate max_iter
pontos usando o ultimo valor conhecido (ou o makespan inicial, se a
serie for vazia) -- isso NAO e um artificio, e o valor correto de
"melhor makespan usando ate t iteracoes de busca" para qualquer t. A
figura (analysis.plots.plot_convergence, tambem atualizada na Entrada
75) marca visualmente o trecho estendido com linha tracejada, para
nao sugerir que 1000 iteracoes de busca real aconteceram quando na
verdade a busca parou antes.

Saida:
    results/convergence/convergence_entrada75.json  (dados brutos)
    results/figures/plots/fig_convergence.png        (figura, via
                                                        analysis.plots.
                                                        plot_convergence,
                                                        mesmo diretorio
                                                        de generate_all_plots)

Uso: python3 experiment/convergence_entrada75.py
(rodar da raiz do repositorio; sem argumentos)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from pmsp.instance import PMSPInstance
from pmsp.bounds import gap_percent
from pmsp.ig import ig_run
from pmsp.ils import ils_run
from pmsp.sa import sa_run
from experiment.runner import make_ils_selector
from experiment import config as CFG

# As "4 classes grandes" completas (Entrada 70/Task #101) sao (150,15),
# (150,30), (250,25) e (300,20) -- ver docstring acima para a
# justificativa de usar (300,20) como representante aqui.
SIZES = [(20, 4), (100, 20), (300, 20)]
INST_ID = 0
MAX_ITER = CFG.MAX_ITER


def _history_to_gap(history, lb, pad_to=None, fallback_makespan=None):
    """Converte uma lista de makespans (best-so-far) em GAP% ponto a
    ponto, usando a mesma funcao gap_percent() de pmsp/bounds.py que o
    proprio pipeline de producao usa para o gap_pct final.

    Se pad_to for fornecido e len(history) < pad_to, estende a lista
    repetindo o ULTIMO valor conhecido ate completar pad_to pontos.
    NAO e um artificio: para metodos com parada antecipada (ex.:
    sa_li_perkins -- ver docstring do modulo acima), o "melhor-ate-
    agora" e, por definicao, constante a partir do ponto de parada
    (nenhuma busca adicional ocorreu), entao o valor correto de
    "melhor-ate-agora na iteracao t" para qualquer t <= pad_to e
    exatamente o ultimo valor registrado -- ou, se a parada ocorreu
    ANTES da primeira iteracao (history vazio), o proprio
    fallback_makespan (o makespan da solucao inicial, que e o
    resultado final de sa_run() nesse caso).
    """
    if not history:
        if fallback_makespan is None:
            return []
        history = [fallback_makespan]
    if pad_to is not None and len(history) < pad_to:
        history = list(history) + [history[-1]] * (pad_to - len(history))
    return [gap_percent(h, lb) for h in history]


def run_one_size(n: int, m: int) -> dict:
    """Roda ig_ql, ils_ql e sa_li_perkins com track_history=True em UMA
    instancia (n, m), reusando exatamente a construcao de instancia e
    de seletor do protocolo de producao (experiment/runner.py)."""
    seed = CFG.BASE_SEED + INST_ID
    inst_name = f"rand_{n}j_{m}m"
    inst = PMSPInstance.random(n=n, m=m, seed=seed, name=inst_name)
    lb = inst.lower_bound

    curves = {}

    # --- ig_ql -----------------------------------------------------------
    selector = make_ils_selector("ig_ql", seed)
    t0 = time.perf_counter()
    r = ig_run(
        instance=inst,
        selector=selector,
        max_iter=MAX_ITER,
        seed=seed,
        stagnation_window=CFG.STAGNATION_WINDOW,
        track_history=True,
    )
    curves["ig_ql"] = {
        "history_gap": _history_to_gap(r["history"], lb, pad_to=MAX_ITER,
                                        fallback_makespan=r["makespan"]),
        "n_iters_ran": len(r["history"]),
        "cpu_time": time.perf_counter() - t0,
        "final_gap_pct": r["gap_pct"],
    }
    print(f"  [{inst_name}] ig_ql        OK  "
          f"gap_final={r['gap_pct']:.2f}%  n_iters_ran={len(r['history'])}  "
          f"cpu={curves['ig_ql']['cpu_time']:.1f}s")

    # --- ils_ql ------------------------------------------------------------
    selector = make_ils_selector("ils_ql", seed)
    t0 = time.perf_counter()
    r = ils_run(
        instance=inst,
        selector=selector,
        max_iter=MAX_ITER,
        perturbation_threshold=CFG.PERTURBATION_THRESHOLD,
        n_perturb=CFG.N_PERTURB,
        seed=seed,
        stagnation_window=CFG.STAGNATION_WINDOW,
        track_history=True,
    )
    curves["ils_ql"] = {
        "history_gap": _history_to_gap(r["history"], lb, pad_to=MAX_ITER,
                                        fallback_makespan=r["makespan"]),
        "n_iters_ran": len(r["history"]),
        "cpu_time": time.perf_counter() - t0,
        "final_gap_pct": r["gap_pct"],
    }
    print(f"  [{inst_name}] ils_ql       OK  "
          f"gap_final={r['gap_pct']:.2f}%  n_iters_ran={len(r['history'])}  "
          f"cpu={curves['ils_ql']['cpu_time']:.1f}s")

    # --- sa_li_perkins (sem selector) --------------------------------------
    # Unico dos 3 com parada antecipada real (ver docstring do modulo) --
    # n_iters_ran pode ser < MAX_ITER, inclusive 0.
    t0 = time.perf_counter()
    r = sa_run(inst, max_iter=MAX_ITER, seed=seed, track_history=True)
    n_ran = len(r["history"])
    curves["sa_li_perkins"] = {
        "history_gap": _history_to_gap(r["history"], lb, pad_to=MAX_ITER,
                                        fallback_makespan=r["makespan"]),
        "n_iters_ran": n_ran,
        "cpu_time": time.perf_counter() - t0,
        "final_gap_pct": r["gap_pct"],
    }
    flag = " [PARADA ANTECIPADA]" if n_ran < MAX_ITER else ""
    print(f"  [{inst_name}] sa_li_perkins OK  "
          f"gap_final={r['gap_pct']:.2f}%  n_iters_ran={n_ran}{flag}  "
          f"cpu={curves['sa_li_perkins']['cpu_time']:.1f}s")

    return {
        "n": n, "m": m, "lb": int(lb), "seed": seed, "max_iter": MAX_ITER,
        "inst_id": INST_ID, "curves": curves,
    }


def main():
    print(f"[config] tamanhos: {SIZES}")
    print(f"[config] max_iter={MAX_ITER} (CFG.MAX_ITER, uniforme)")
    print(f"[config] inst_id={INST_ID} (seed={CFG.BASE_SEED + INST_ID})")
    print()

    all_data = []
    for n, m in SIZES:
        print(f"Rodando n={n}, m={m}...")
        all_data.append(run_one_size(n, m))

    out_dir = Path("results/convergence")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "convergence_entrada75.json"
    out_path.write_text(json.dumps(all_data, indent=2))
    print(f"\nDados salvos em {out_path}")

    from analysis.plots import plot_convergence
    fig_path = plot_convergence(all_data, Path("results/figures/plots"))
    print(f"Figura salva em {fig_path}")


if __name__ == "__main__":
    main()
