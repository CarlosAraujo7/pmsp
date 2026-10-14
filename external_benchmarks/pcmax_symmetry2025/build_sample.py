"""
build_sample.py
================
Constroi uma amostra estratificada e determinista das 42.530 instancias
do benchmark Symmetry 2025 (B+C+E+F) -- rodar tudo nao e viavel (ver
diario_pesquisa.md, Entrada 51: nenhuma bateria deste projeto passou
de alguns milhares de linhas ate hoje). Gera manifest.csv, que
run_external_battery.py le para saber exatamente o que rodar.

Heuristica de amostragem (simples, determinista, SEM curadoria manual
de classes -- ajuste as 2 constantes abaixo e rode de novo se quiser
uma amostra maior/menor):

  1. Agrupa as instancias de cada familia por (subfolder, m, n) --
     cada grupo corresponde aproximadamente a uma "regra de geracao"
     do artigo (Tabelas 9-12), as vezes bundlando 2+ distribuicoes
     diferentes que geram o mesmo (m,n) (isso acontece nas familias
     E e B; ver load_instance_array/list_instances em loader.py).
  2. Ordena os grupos por n crescente e seleciona ate
     MAX_CLASSES_PER_FAMILY grupos, com indices IGUALMENTE ESPACADOS
     ao longo da lista ordenada -- cobre do menor ao maior n
     disponivel na familia, sem escolha manual.
  3. Dentro de cada grupo selecionado, pega INSTANCES_PER_CLASS
     instancias com indices tambem igualmente espacados (ordenadas
     por nome de arquivo) -- em vez de so as primeiras N, isso evita
     ficar preso a uma unica distribuicao quando o grupo bundla mais
     de uma (ver passo 1) e ainda assim e 100% determinista/reproduzivel.

O manifest.csv gerado pode ser editado a mao antes de rodar
run_external_battery.py -- por exemplo pra remover uma familia,
adicionar uma classe especifica de interesse, ou tirar linhas de
instancias muito grandes (a familia F, por sinal, tem uma classe com
n=1000 -- bem maior que qualquer coisa ja rodada no projeto, vai
demorar mais que o resto).

--- Expansao de 02/09/2026 (Entrada 68) ------------------------------
Carlos identificou (veja diario_pesquisa.md, Entrada 68) que a amostra
de 10 classes/familia cobria uma fatia pequena das classes de geracao
DISTINTAS disponiveis: B tem 270 grupos (subfolder,m,n) distintos (so
10 amostrados = 3,7%), C tem 125 (8,0%), E tem 55 (18,2%), F tem 26
(38,5% -- e por isso a familia mais bem coberta antes desta expansao).
MAX_CLASSES_PER_FAMILY sobe de 10 para 50 (INSTANCES_PER_CLASS
permanece 40 -- o gargalo identificado foi cobertura de CLASSES
distintas, nao tamanho de amostra dentro de cada classe ja
selecionada). Cobertura resultante: B 50/270 (18,5%), C 50/125 (40%),
E 50/55 (90,9%), F 26/26 (100%, capped -- F so tem 26 grupos ao todo)
-- 6.780 instancias no total (antes: 1.500).

IMPORTANTE: a selecao de classes agora usa cumulative_indices() em vez
de evenly_spaced_indices() puro, GARANTINDO que as classes ja
escolhidas pela amostra anterior (PREVIOUS_MAX_CLASSES_PER_FAMILY=10)
continuam presentes na amostra nova -- evenly_spaced_indices(total,10)
e evenly_spaced_indices(total,50) NAO produzem conjuntos aninhados em
geral (o espacamento e recalculado do zero a cada chamada), entao uma
troca ingenua de MAX_CLASSES_PER_FAMILY poderia derrubar do manifest
classes que ja tem resultados computados em
results_external_symmetry2025.csv (nao seria um bug -- os resultados
antigos continuariam validos e no CSV -- mas ficariam "orfaos", fora
do manifest.csv oficial, e o proximo re-run desperdicaria trabalho ja
feito ao trocar de classe em vez de so adicionar novas). Ver
cumulative_indices() abaixo.

Se quiser ampliar de novo no futuro, atualize PREVIOUS_MAX_CLASSES_PER_FAMILY
para o valor atual (50) e MAX_CLASSES_PER_FAMILY para o novo alvo, e
faca backup de build_sample.py/manifest.csv antes (mesmo padrao:
*_pre_<novo_total>_backup_<data>.*).
-----------------------------------------------------------------------

Uso: python3 build_sample.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import list_instances, FAMILIES  # noqa: E402

# ---- ajuste aqui se quiser uma amostra maior/menor ----
# Ate 02/09/2026: 10 classes x 40 instancias x 4 familias = 1.500
# instancias (F ficava em 300 porque seus grupos tem so 30 instancias
# cada -- capped, ver diario_pesquisa.md, Entrada 54). Antes: 8x20=640.
#
# 02/09/2026 (Entrada 68): sobe para 50 classes/familia -- cobertura
# de classes distintas por familia era muito baixa (B 3,7%, C 8,0%,
# E 18,2%, F 38,5%). Resultado: 6.780 instancias (B 2000 = 50/270,
# C 2000 = 50/125, E 2000 = 50/55 [quase todas], F 780 = 26/26 [TODAS,
# F so tem 26 grupos ao todo]).
MAX_CLASSES_PER_FAMILY = 50
INSTANCES_PER_CLASS = 40

# Valor de MAX_CLASSES_PER_FAMILY usado para gerar o manifest.csv
# ANTERIOR (o de 1.500 linhas, Entrada 54/57) -- usado por
# cumulative_indices() para garantir que a amostra so CRESCE, nunca
# troca classes ja escolhidas por outras. NAO mude isto ao reajustar
# MAX_CLASSES_PER_FAMILY -- atualize-o para o valor QUE ACABOU DE SER
# SUBSTITUIDO, na proxima vez que ampliar a amostra de novo.
PREVIOUS_MAX_CLASSES_PER_FAMILY = 10
# ---------------------------------------------------------

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.csv"


def evenly_spaced_indices(total: int, k: int) -> list[int]:
    """k indices em [0, total) igualmente espacados, sempre incluindo
    o primeiro e o ultimo quando k >= 2. Deterministico."""
    if total <= k:
        return list(range(total))
    if k <= 1:
        return [0]
    step = (total - 1) / (k - 1)
    return sorted(set(round(i * step) for i in range(k)))


def cumulative_indices(total: int, old_k: int, new_k: int) -> list[int]:
    """Como evenly_spaced_indices(total, new_k), mas GARANTE que todo
    indice ja escolhido por evenly_spaced_indices(total, old_k)
    continua presente no resultado -- a amostra so cresce, nunca troca
    classes ja escolhidas (e portanto ja possivelmente executadas em
    run_external_battery.py) por outras. Os indices extras (new_k -
    old_k deles) sao escolhidos com o mesmo criterio de espacamento
    uniforme, mas dentro do conjunto remanescente (nao escolhido
    ainda) -- cobertura um pouco menos perfeitamente uniforme do que
    um recalculo do zero, mas sem orfaos no manifest.

    Se new_k <= old_k, comporta-se como evenly_spaced_indices(total,
    new_k) normal (reducao/igualdade nao tem o problema de orfaos)."""
    if new_k <= old_k:
        return evenly_spaced_indices(total, new_k)
    base = set(evenly_spaced_indices(total, old_k))
    target = min(new_k, total)
    if len(base) >= target:
        return sorted(base)
    remaining_idx = [i for i in range(total) if i not in base]
    n_extra = target - len(base)
    extra_positions = evenly_spaced_indices(len(remaining_idx), n_extra)
    extra = {remaining_idx[p] for p in extra_positions}
    return sorted(base | extra)


def main():
    rows = []
    for family in FAMILIES:
        print(f"[{family}] listando instancias do .zip...")
        refs = list_instances(family)
        groups = defaultdict(list)
        for r in refs:
            groups[(r.subfolder, r.m, r.n)].append(r)

        keys_sorted = sorted(groups.keys(), key=lambda k: k[2])  # por n
        chosen_idx = cumulative_indices(
            len(keys_sorted), PREVIOUS_MAX_CLASSES_PER_FAMILY, MAX_CLASSES_PER_FAMILY)
        chosen_keys = [keys_sorted[i] for i in chosen_idx]

        print(f"[{family}] {len(refs)} instancias totais, "
              f"{len(keys_sorted)} grupos (subfolder,m,n) distintos, "
              f"{len(chosen_keys)} selecionados "
              f"(n de {chosen_keys[0][2]} a {chosen_keys[-1][2]})")

        for (subfolder, m, n) in chosen_keys:
            group_refs_sorted = sorted(groups[(subfolder, m, n)], key=lambda r: r.member)
            pick_idx = evenly_spaced_indices(len(group_refs_sorted), INSTANCES_PER_CLASS)
            for inst_id, gi in enumerate(pick_idx):
                ref = group_refs_sorted[gi]
                rows.append({
                    "family": ref.family,
                    "member": ref.member,
                    "subfolder": ref.subfolder,
                    "m": ref.m,
                    "n": ref.n,
                    "inst_id": inst_id,
                })

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["family", "member", "subfolder", "m", "n", "inst_id"])
        writer.writeheader()
        writer.writerows(rows)

    by_fam = defaultdict(int)
    for r in rows:
        by_fam[r["family"]] += 1
    print(f"\nTotal: {len(rows)} instancias selecionadas -> {MANIFEST_PATH}")
    print("Por familia:", dict(by_fam))
    print("Revise/edite manifest.csv a mao se quiser, antes de rodar "
          "run_external_battery.py.")


if __name__ == "__main__":
    main()
