"""
experiment/run_ils_hi_new_classes_entrada70.py
=================================================
Roda ils_ql_hi e ils_rand_hi -- os 2 metodos que dependem de
CFG.MAX_ITER_BY_SIZE -- nas 6 classes novas/grandes que tinham ficado
de fora dos 2 scripts irmaos da Entrada 68
(run_large_missing_methods_entrada68.py e run_new_classes_entrada68.py)
justamente por essa dependencia ainda nao estar derivada para essas
classes. Ver diario_pesquisa.md, Entrada 70 (05/09/2026), Task #101.

CFG.MAX_ITER_BY_SIZE agora tem entrada para as 6 classes (adicionado
em experiment/config.py em 05/09/2026, Entrada 70):
    (40,   5): 8000
    (60,  20): 8000
    (150, 15): 8000
    (150, 30): 8000
    (250, 25): 4000
    (300, 20): 2000

IMPORTANTE -- honestidade metodologica (ver diario, Entrada 70 para
os detalhes completos): essas entradas NAO foram derivadas do
criterio original do comentario de config.py ("~50% dos 81 estados
com 3+ visitas"). Esse criterio foi re-medido diretamente nesta
mesma investigacao e NAO se confirma nem para as classes ORIGINAIS
(ex.: (100,20) no seu proprio max_iter ja configurado, 4000, atinge
so 30/81 = 37% dos estados com 3+ visitas, e ja esta em platou --
32000 iteracoes, 8x mais, nao muda nada). As 6 entradas novas foram
escolhidas por "ponto de joelho" (onde o ganho de cobertura passa a
ser claramente decrescente frente ao custo) em curvas reais medidas
por classe (1 seed cada, nao multi-seed). Diagnostico completo em
results/diag_max_iter_by_size_entrada70.json. Ver tambem o comentario
atualizado em experiment/config.py.

Este script roda em 2 blocos sequenciais, com CFG.PERTURBATION_THRESHOLD
diferente por bloco -- mesma logica de particionamento ja usada nos 2
scripts irmaos da Entrada 68:

  Bloco 1 -- 4 classes "grandes" (150 a 300 tarefas):
             perturbation_threshold=15 (monkeypatch, mesmo ajuste da
             Entrada 22/48, ja usado pros outros 8 metodos dessas
             classes em run_large_missing_methods_entrada68.py).

  Bloco 2 -- 2 classes "medias novas" (40,5) e (60,20):
             perturbation_threshold=PADRAO do projeto (50, sem
             monkeypatch) -- mesma decisao de
             run_new_classes_entrada68.py, por essas 2 classes terem
             porte comparavel as 6 classes ORIGINAIS (20 a 100
             tarefas), nao as 4 grandes.

append=True direto em results/results.csv nos 2 blocos -- run_all()
ja tem guarda anti-duplicata embutida (existing_keys por
(instance,method,rep)), entao interromper (Ctrl+C) e rodar de novo e
seguro, continua de onde parou. Os 2 blocos escrevem no MESMO
arquivo -- NAO rode este script em paralelo com outro processo que
tambem escreva em results/results.csv (ex.: nao rodar ao mesmo tempo
que outro run_all() avulso); pode rodar em paralelo com a bateria
externa (results_external_symmetry2025.csv, arquivo diferente).

Custo estimado: 6 classes x 100 instancias x 2 metodos = 1.200
tarefas, com max_iter bem maior que o generico (2000 a 8000 vs 1000)
-- custo por tarefa proporcional ao max_iter usado. Estimativa
GROUNDED no smoke test real deste proprio script (N=1 instancia,
Entrada 70, 05/09/2026): cpu_time somado das 6 classes x 2 metodos
= ~87,9s por instancia (150/15=18,9s; 150/30=19,5s; 250/25=25,4s;
300/20=16,2s; 40/5=2,8s; 60/20=5,1s). Para n_instances=100 (producao):
~8.790s = ~2,4h no total (run_all() e sequencial, sem --workers).
cpu real pode variar por hardware e carga da maquina no momento.

Uso: python3 experiment/run_ils_hi_new_classes_entrada70.py
(rodar da raiz do repositorio; sem argumentos)
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from experiment.runner import run_all
from experiment import config as CFG

METHODS = ["ils_ql_hi", "ils_rand_hi"]

GRANDES_CONFIGS = [(150, 15), (150, 30), (250, 25), (300, 20)]
MEDIAS_CONFIGS = [(40, 5), (60, 20)]


def main():
    todas = GRANDES_CONFIGS + MEDIAS_CONFIGS
    faltando = [c for c in todas if c not in CFG.MAX_ITER_BY_SIZE]
    assert not faltando, (
        f"classes sem entrada em CFG.MAX_ITER_BY_SIZE: {faltando} -- "
        f"verifique experiment/config.py antes de rodar"
    )

    print(f"[config] metodos: {METHODS}")
    print(f"[config] MAX_ITER_BY_SIZE das 6 classes: "
          f"{ {c: CFG.MAX_ITER_BY_SIZE[c] for c in todas} }")
    print()

    # --- Bloco 1: 4 classes grandes, perturbation_threshold=15 ---
    CFG.PERTURBATION_THRESHOLD = 15
    print(f"[bloco 1/2] classes grandes {GRANDES_CONFIGS}, "
          f"perturbation_threshold={CFG.PERTURBATION_THRESHOLD} (monkeypatch)")
    out_path = run_all(
        methods=METHODS,
        instance_configs=GRANDES_CONFIGS,
        n_instances=100,
        append=True,
        save_qtables=True,
        verbose=True,
    )
    print(f"[bloco 1/2] concluido -> {out_path}\n")

    # --- Bloco 2: 2 classes medias novas, perturbation_threshold padrao ---
    CFG.PERTURBATION_THRESHOLD = 50
    print(f"[bloco 2/2] classes medias {MEDIAS_CONFIGS}, "
          f"perturbation_threshold={CFG.PERTURBATION_THRESHOLD} "
          f"(padrao do projeto, sem monkeypatch)")
    out_path = run_all(
        methods=METHODS,
        instance_configs=MEDIAS_CONFIGS,
        n_instances=100,
        append=True,
        save_qtables=True,
        verbose=True,
    )
    print(f"[bloco 2/2] concluido -> {out_path}\n")

    print("Concluido: ils_ql_hi e ils_rand_hi rodados nas 6 classes "
          "novas/grandes, cada uma com seu MAX_ITER_BY_SIZE proprio "
          "(Entrada 70, Task #101).")


if __name__ == "__main__":
    main()
