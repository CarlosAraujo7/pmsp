"""
pmsp/ig_numba.py (Entrada 170, Ciclo 14)
==========================================
Iterated Greedy de P||Cmax e a busca local do IG hibrido (inter-troca + reinsercao, ate' 2 passes) COMPILADOS com numba, com um CONTADOR DE AVALIACOES (unidade de orcamento independente de implementacao).
Motivacao (critica de revisor, limite (iv) do artigo): a razao de CPU K entre o IG com e sem busca local mudou de 9,6 (lacos Python) para 1,8 (vetorizada) e o '93% do ganho e orcamento' virou 30%; as conclusoes 'a CPU igual'
dependem da eficiencia da implementacao. Aqui os DOIS metodos rodam compilados (o custo do interpretador some dos dois lados) e cada execucao conta as avaliacoes que fez.
Unidade de trabalho ('avaliacao'): (i) cada candidato de movimento cujo novo makespan e' calculado na busca local (inter-troca: |A| x (n - |A|); reinsercao: |A| x (m - 1), A = tarefas da maquina gargalo) e (ii) cada comparacao de maquina na
reconstrucao gulosa (m por tarefa reinserida). Nao ha pesos entre os dois tipos (suposicao declarada).
Equivalencia com a implementacao de referencia (pmsp/ig.py + pmsp/operators_fast.py):
  * a busca local e' EXATAMENTE equivalente: mesma ordem de varredura (maquina gargalo, tarefas na ordem de machine_jobs, outras maquinas em ordem crescente), mesmo desempate (primeiro maximo estrito de delta > 0), mesmas listas
    machine_jobs (remocao preserva a ordem, insercao no fim); testado em tests/test_ig_numba.py;
  * a reconstrucao gulosa dada a lista de tarefas removidas e' exatamente equivalente (ordem decrescente de p, estavel);
  * os sorteios (estrategias de destruicao 0 e 2, escolha da estrategia) usam o gerador do numba, que NAO reproduz o do numpy: as trajetorias individuais diferem das da referencia e a equivalencia e' ESTATISTICA
    (experiment/run_numba_entrada170.py --verify).
Estrategias: 0 aleatoria uniforme, 1 as d de maior p (ordem de numpy.argsort(p)[::-1], passada pronta), 2 concentrada em uma maquina aleatoria; strat = -1 sorteia uma das tres a cada iteracao (ig_random).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _argmax(a):
    k = 0
    for i in range(1, len(a)):
        if a[i] > a[k]:
            k = i
    return k


@njit(cache=True)
def _argmin(a):
    k = 0
    for i in range(1, len(a)):
        if a[i] < a[k]:
            k = i
    return k


@njit(cache=True)
def init_state(assign, p, m):
    """loads, machine_jobs (matriz m x n, preenchida ate' cnt[k]) e cnt a partir de assign, na ordem dos indices (como PMSPSolution._rebuild_from_assignment)."""
    n = len(assign)
    loads = np.zeros(m, dtype=np.int64)
    mj = np.zeros((m, n), dtype=np.int32)
    cnt = np.zeros(m, dtype=np.int64)
    for j in range(n):
        k = assign[j]
        loads[k] += p[j]
        mj[k, cnt[k]] = j
        cnt[k] += 1
    return loads, mj, cnt


@njit(cache=True)
def _remove(mj, cnt, k, job):
    c = cnt[k]
    pos = 0
    while mj[k, pos] != job:
        pos += 1
    for q in range(pos, c - 1):
        mj[k, q] = mj[k, q + 1]
    cnt[k] = c - 1


@njit(cache=True)
def _append(mj, cnt, k, job):
    mj[k, cnt[k]] = job
    cnt[k] += 1


@njit(cache=True)
def move_job(assign, loads, mj, cnt, p, j, tgt):
    src = assign[j]
    if src == tgt:
        return
    loads[src] -= p[j]
    loads[tgt] += p[j]
    _remove(mj, cnt, src, j)
    _append(mj, cnt, tgt, j)
    assign[j] = tgt


@njit(cache=True)
def swap_jobs(assign, loads, mj, cnt, p, ja, jb):
    """Troca inter-maquinas (como PMSPSolution.swap_jobs com maquinas distintas)."""
    ma = assign[ja]
    mb = assign[jb]
    loads[ma] += p[jb] - p[ja]
    loads[mb] += p[ja] - p[jb]
    _remove(mj, cnt, ma, ja)
    _append(mj, cnt, ma, jb)
    _remove(mj, cnt, mb, jb)
    _append(mj, cnt, mb, ja)
    assign[ja] = mb
    assign[jb] = ma


@njit(cache=True)
def _other_max(loads, a):
    """other[b] = maior carga entre as maquinas k != a, k != b (0 se nao houver), como operators_fast._other_max."""
    m = len(loads)
    out = np.zeros(m, dtype=np.int64)
    i1 = -1
    m1 = -1
    for k in range(m):
        if k != a and (i1 < 0 or loads[k] > m1):
            i1 = k
            m1 = loads[k]
    if i1 < 0:
        return out
    m2 = 0
    for k in range(m):
        if k != a and k != i1 and loads[k] > m2:
            m2 = loads[k]
    for k in range(m):
        out[k] = m1
    out[i1] = m2
    return out


@njit(cache=True)
def op1_best(assign, loads, mj, cnt, p):
    """Melhor troca entre a maquina gargalo e as demais; retorna (melhora, avaliacoes)."""
    m = len(loads)
    a = _argmax(loads)
    old = loads[a]
    if cnt[a] == 0:
        return 0, 0
    other = _other_max(loads, a)
    la = loads[a]
    best = 0
    bja = -1
    bjb = -1
    ev = 0
    for ia in range(cnt[a]):
        ja = mj[a, ia]
        pa = p[ja]
        for b in range(m):
            if b == a:
                continue
            lb = loads[b]
            om = other[b]
            for ib in range(cnt[b]):
                jb = mj[b, ib]
                pb = p[jb]
                nc = max(la - pa + pb, lb - pb + pa, om)
                ev += 1
                if old - nc > best:
                    best = old - nc
                    bja = ja
                    bjb = jb
    if best <= 0:
        return 0, ev
    swap_jobs(assign, loads, mj, cnt, p, bja, bjb)
    return best, ev


@njit(cache=True)
def op2_best(assign, loads, mj, cnt, p):
    """Melhor reinsercao de uma tarefa da maquina gargalo; retorna (melhora, avaliacoes)."""
    m = len(loads)
    a = _argmax(loads)
    old = loads[a]
    if cnt[a] == 0 or m < 2:
        return 0, 0
    other = _other_max(loads, a)
    la = loads[a]
    best = 0
    bj = -1
    bt = -1
    ev = 0
    for ia in range(cnt[a]):
        j = mj[a, ia]
        pj = p[j]
        for b in range(m):
            if b == a:
                continue
            nc = max(la - pj, loads[b] + pj, other[b])
            ev += 1
            if old - nc > best:
                best = old - nc
                bj = j
                bt = b
    if best <= 0:
        return 0, ev
    move_job(assign, loads, mj, cnt, p, bj, bt)
    return best, ev


@njit(cache=True)
def local_search(assign, loads, mj, cnt, p):
    """make_ls_fast(passes=2): ate' 2 passes de (inter-troca; se nao melhorou, reinsercao); para quando as duas nao melhoram. Retorna as avaliacoes."""
    ev = 0
    for _ in range(2):
        r1, e1 = op1_best(assign, loads, mj, cnt, p)
        ev += e1
        if r1 <= 0:
            r2, e2 = op2_best(assign, loads, mj, cnt, p)
            ev += e2
            if r2 <= 0:
                break
    return ev


@njit(cache=True)
def reconstruct(assign, loads, mj, cnt, p, removed, d):
    """Reconstrucao gulosa de apply_destroy_reconstruct: desconta as removidas, ordena por p decrescente (estavel) e poe cada uma na maquina de menor carga. Retorna as avaliacoes (d x m)."""
    m = len(loads)
    tracker = loads.copy()
    for i in range(d):
        j = removed[i]
        tracker[assign[j]] -= p[j]
    order = removed[:d].copy()
    for i in range(1, d):                       # insercao estavel por -p (empates: ordem de remocao)
        x = order[i]
        q = i - 1
        while q >= 0 and p[order[q]] < p[x]:
            order[q + 1] = order[q]
            q -= 1
        order[q + 1] = x
    for i in range(d):
        j = order[i]
        k = _argmin(tracker)
        tracker[k] += p[j]
        move_job(assign, loads, mj, cnt, p, j, k)
    return d * m


@njit(cache=True)
def _destroy(strat, assign, p, d, order_desc, perm, removed, m):
    n = len(assign)
    if strat == 0:                              # d distintas ao acaso (Fisher-Yates parcial sobre 'perm', persistente)
        for i in range(d):
            r = i + np.random.randint(0, n - i)
            t = perm[i]
            perm[i] = perm[r]
            perm[r] = t
            removed[i] = perm[i]
    elif strat == 1:
        for i in range(d):
            removed[i] = order_desc[i]
    else:                                       # concentrada: maquina aleatoria; completa com tarefas de outras maquinas se tiver menos de d
        mach = np.random.randint(0, m)
        on = np.empty(n, dtype=np.int64)
        off = np.empty(n, dtype=np.int64)
        c_on = 0
        c_off = 0
        for j in range(n):
            if assign[j] == mach:
                on[c_on] = j
                c_on += 1
            else:
                off[c_off] = j
                c_off += 1
        if c_on >= d:
            for i in range(d):
                r = i + np.random.randint(0, c_on - i)
                t = on[i]
                on[i] = on[r]
                on[r] = t
                removed[i] = on[i]
        else:
            for i in range(c_on):
                removed[i] = on[i]
            extra = min(d - c_on, c_off)
            for i in range(extra):
                r = i + np.random.randint(0, c_off - i)
                t = off[i]
                off[i] = off[r]
                off[r] = t
                removed[c_on + i] = off[i]


@njit(cache=True)
def ig_core(p, assign0, m, d, strat, T, eval_budget, ls_on, seed, order_desc, w_rec):
    """Um IG (aceita se nao piora; partida em assign0). Para apos T iteracoes OU, se eval_budget > 0, quando as avaliacoes acumuladas atingem eval_budget (verificado antes de cada iteracao).
    Retorna (melhor makespan, n_melhorias, avaliacoes, iteracoes feitas, atribuicao final). Como so' se aceita se nao piora, o makespan da atribuicao final e' o melhor makespan (invariante testada)."""
    np.random.seed(seed)
    n = len(p)
    assign = assign0.astype(np.int64).copy()
    loads, mj, cnt = init_state(assign, p, m)
    c_assign = assign.copy()
    c_loads = loads.copy()
    c_mj = mj.copy()
    c_cnt = cnt.copy()
    perm = np.arange(n)
    removed = np.empty(n, dtype=np.int64)
    best = loads.max()
    n_imp = 0
    ev = 0.0
    done = 0
    for t in range(T):
        if eval_budget > 0 and ev >= eval_budget:
            break
        done += 1
        s = strat if strat >= 0 else np.random.randint(0, 3)
        for k in range(m):                      # copia da solucao corrente (como sol.copy())
            c_loads[k] = loads[k]
            c_cnt[k] = cnt[k]
            for q in range(cnt[k]):
                c_mj[k, q] = mj[k, q]
        for j in range(n):
            c_assign[j] = assign[j]
        _destroy(s, c_assign, p, d, order_desc, perm, removed, m)
        ev += w_rec * reconstruct(c_assign, c_loads, c_mj, c_cnt, p, removed, d)
        if ls_on:
            ev += local_search(c_assign, c_loads, c_mj, c_cnt, p)
        cm = c_loads.max()
        if cm <= loads.max():
            for k in range(m):
                loads[k] = c_loads[k]
                cnt[k] = c_cnt[k]
                for q in range(c_cnt[k]):
                    mj[k, q] = c_mj[k, q]
            for j in range(n):
                assign[j] = c_assign[j]
            if cm < best:
                best = cm
                n_imp += 1
    return best, n_imp, ev, done, assign
