"""
experiment/run_new_classes_entrada68.py
=========================================
Roda do ZERO as 2 classes sinteticas novas do plano experimental
(PLANO_EXPERIMENTAL_EXTENSIVO.md) que nunca chegaram a ser executadas:
rand_40j_5m (n=40, m=5) e rand_60j_20m (n=60, m=20). Ver
diario_pesquisa.md, Entrada 68 (02/09/2026): das 6 classes novas
planejadas, so 4 (150/15, 150/30, 250/25, 300/20) tinham dados: estas
2 tinham ZERO linhas em results/results.csv.

Roda 21 dos 23 metodos do projeto -- TODOS exceto ils_ql_hi e
ils_rand_hi. Estes 2 dependem de CFG.MAX_ITER_BY_SIZE, que so tem
entradas para as 6 classes originais (20/4 a 100/20) -- (40,5) e
(60,20) NAO tem entrada la, entao cairiam no max_iter generico (1000),
o que descaracterizaria o proprio "orcamento estendido" que o metodo
representa (nas classes originais chega a 4x o generico). Mesma
decisao e mesmo motivo do script irmao
(run_large_missing_methods_entrada68.py) para as 4 classes grandes --
ver o docstring de la para mais detalhes. Fica como pendencia
separada, nao bloqueia este script.

Configuracao (diferente do script irmao das 4 classes grandes -- ali
usa perturbation_threshold=15, ajuste especifico pra escala 150-300
tarefas validado na Entrada 22/48; aqui, (40,5) e (60,20) tem porte
comparavel as 6 classes ORIGINAIS -- 20 a 100 tarefas -- entao NAO ha
razao para importar o ajuste das classes grandes. Usa o
perturbation_threshold PADRAO do projeto, sem monkeypatch):
  - CFG.PERTURBATION_THRESHOLD: default (50, inalterado)
  - max_iter = CFG.MAX_ITER (1000, generico -- MAX_ITER_BY_SIZE nao
    tem entrada pra (40,5)/(60,20); mesmo criterio ja usado nas 4
    classes grandes, que tambem nao tem entrada la)
  - n_instances = 100 (segue o precedente das 4 classes grandes da
    Entrada 48, acima do N=50 do plano original)
  - append=True direto em results/results.csv -- run_all() ja tem
    guarda anti-duplicata embutida, interromper e retomar e seguro.

Custo estimado (proxy: cpu_time medio dos mesmos 21 metodos nas
classes originais de porte mais proximo, 30/8 e 50/10 -- NAO e uma
extrapolacao empirica direta como nos outros 2 scripts, e so uma
referencia de mesma ordem de grandeza, porque (40,5)/(60,20) nunca
foram executadas antes): ~4.200 tarefas totais (2 classes x 100
instancias x 21 metodos), ~0,3-0,5h estimado (run_all() e sequencial).
Estimativa grosseira -- cpu real pode variar bastante, sobretudo pro
metodo ils_rf (o mais caro no proxy, ~2,8s/tarefa).

Uso: python3 experiment/run_new_classes_entrada68.py
(rodar da raiz do repositorio; sem argumentos)
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from experiment.runner import run_all
from experiment import config as CFG

METHODS = [
    "lpt", "multifit", "sa_li_perkins", "bco",
    "ils_random", "ils_ucb", "ils_ql", "ils_rf",
    "ils_ap", "ils_op1_only", "ils_op2_only", "ils_op3_only",
    "ils_op4_only", "ils_pm", "ils_ql_ws", "ils_sarsa",
    "ig_random", "ig_ucb", "ig_ucb_ctx", "ig_ql", "ig_ql_flat",
]

INSTANCE_CONFIGS = [(40, 5), (60, 20)]


def main():
    assert len(METHODS) == 21, f"esperado 21 metodos, tem {len(METHODS)}"
    print(f"[config] metodos ({len(METHODS)}): {METHODS}")
    print(f"[config] classes: {INSTANCE_CONFIGS}")
    print(f"[config] perturbation_threshold={CFG.PERTURBATION_THRESHOLD} "
          f"(padrao do projeto, sem monkeypatch)")
    print(f"[config] max_iter={CFG.MAX_ITER} (generico)")
    print("[config] EXCLUIDOS desta rodada: ils_ql_hi, ils_rand_hi "
          "(dependem de MAX_ITER_BY_SIZE ainda nao derivado para "
          "estas classes -- ver docstring deste arquivo)")
    print()

    out_path = run_all(
        methods=METHODS,
        instance_configs=INSTANCE_CONFIGS,
        n_instances=100,
        append=True,
        save_qtables=True,
        verbose=True,
    )
    print(f"\nConcluido -> {out_path}")


if __name__ == "__main__":
    main()
