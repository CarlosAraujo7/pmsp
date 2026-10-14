"""
experiment/run_large_missing_methods_entrada68.py
====================================================
Completa os metodos que faltam nas 4 classes grandes (150-300 tarefas)
ja parcialmente rodadas na Entrada 48. Ver diario_pesquisa.md, Entrada
68 (02/09/2026): revisao de cobertura encontrou que essas 4 classes
tinham so 13 dos 23 metodos do projeto (a bateria "nucleo" definida em
Entrada 48 -- lpt+multifit+sa+bco+4 ils_* baseline+5 ig_*), faltando
10: a familia de ablacao (ils_pm, ils_ap, ils_sarsa, ils_op1_only a
ils_op4_only), e as 3 variantes de orcamento estendido/warm-start
(ils_ql_hi, ils_rand_hi, ils_ql_ws).

Deste 10, este script roda 8 -- TODOS exceto ils_ql_hi e ils_rand_hi:

  - ils_ql_ws NAO tem, na verdade, nenhuma dependencia de tamanho de
    instancia (verificado em experiment/runner.py: pretrain_qtable()
    sempre treina em CFG.WS_SOURCE_SIZES=[(20,4)], INDEPENDENTE das
    classes-alvo passadas a run_all(); o proprio docstring de
    pretrain_qtable ja justifica isso -- "os features do estado sao
    normalizados e independentes de escala, tornando a transferencia
    de Q-table entre tamanhos de instancia valida"). A Entrada 48
    tinha agrupado ils_ql_ws junto com ils_ql_hi/ils_rand_hi como "so
    dependem de MAX_ITER_BY_SIZE" -- na leitura cuidadosa do codigo
    (nesta revisao) isso nao e exato: so ils_ql_hi e ils_rand_hi
    JOGAM com CFG.MAX_ITER_BY_SIZE.get((n,m), max_iter) dentro de
    run_all() -- ils_ql_ws usa o max_iter comum, igual a todo mundo.
    Por isso ils_ql_ws ENTRA nesta lista, corrigindo esse detalhe.

  - ils_ql_hi e ils_rand_hi FICAM DE FORA -- estes sim dependem de
    CFG.MAX_ITER_BY_SIZE, que so tem entradas para as 6 classes
    originais (20/4 a 100/20). Rodar com o max_iter generico (1000)
    para eles especificamente nao teria o mesmo significado do
    "orcamento estendido" que o metodo representa nas classes
    originais (la, MAX_ITER_BY_SIZE chega a 4000 -- 4x o generico).
    Extrapolar um valor sem re-derivar do criterio original (~50% dos
    81 estados do agente QL visitados 3+ vezes, ver config.py) seria
    menos honesto que simplesmente excluir e documentar -- exatamente
    a mesma decisao, pela mesma razao, que a Entrada 48 ja tinha
    tomado para estas 4 classes. Fica como pendencia separada (nao
    bloqueia este script) -- se Carlos quiser esses 2 metodos aqui
    tambem, precisa primeiro decidir/derivar um MAX_ITER_BY_SIZE para
    (150,15), (150,30), (250,25), (300,20).

Configuracao usada (IDENTICA a Entrada 48, para ficar comparavel com
os 13 metodos ja rodados nestas mesmas 4 classes):
  - CFG.PERTURBATION_THRESHOLD = 15 (monkeypatch so neste processo --
    o mesmo ajuste que experiment/run_large_core_battery_entrada48.py
    ja usou para ils_random/ils_ucb/ils_ql/ils_rf nestas classes;
    valido pra TODOS os 8 metodos daqui porque todos sao da familia
    ils_* / ils_run() por baixo -- confirmado em make_ils_selector()).
  - max_iter = CFG.MAX_ITER (1000, o generico -- mesmo valor usado
    para os 13 metodos ja existentes nestas 4 classes, EXCETO
    ils_ql_hi/ils_rand_hi que la nao entraram).
  - n_instances = 100 (mesmo N das classes grandes ja rodadas --
    acima do N=50 do plano original, mas seguindo o precedente ja
    estabelecido na Entrada 48 para estas mesmas 4 classes).
  - append=True direto em results/results.csv -- run_all() ja tem
    guarda anti-duplicata embutida (existing_keys por
    (instance,method,rep)), entao interromper (Ctrl+C) e rodar de
    novo e seguro, continua de onde parou.

Custo estimado (extrapolado do cpu_time medio dos mesmos 8 metodos
nas classes originais 75/15 e 100/20, escalado pela razao observada
grandes/originais nos 4 metodos ils_* ja compartilhados -- media
~1.73x): ~3.200 tarefas totais (4 classes x 100 instancias x 8
metodos), ~0,9h estimado (run_all() e sequencial, nao paralelo).
Estimativa grosseira -- cpu real pode variar por hardware.

Uso: python3 experiment/run_large_missing_methods_entrada68.py
(rodar da raiz do repositorio; sem argumentos)
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from experiment.runner import run_all
from experiment import config as CFG

# Ver docstring acima -- mesmo ajuste da Entrada 48 para estas 4
# classes. So afeta este processo (nao altera config.py em disco).
CFG.PERTURBATION_THRESHOLD = 15

METHODS = [
    "ils_ap", "ils_op1_only", "ils_op2_only", "ils_op3_only",
    "ils_op4_only", "ils_pm", "ils_ql_ws", "ils_sarsa",
]

INSTANCE_CONFIGS = [(150, 15), (150, 30), (250, 25), (300, 20)]


def main():
    print(f"[config] metodos: {METHODS}")
    print(f"[config] classes: {INSTANCE_CONFIGS}")
    print(f"[config] perturbation_threshold={CFG.PERTURBATION_THRESHOLD} "
          f"(monkeypatch, so neste processo)")
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
