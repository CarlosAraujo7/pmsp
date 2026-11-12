"""
experiment/train_neural_entrada259.py (Entrada 259, Ciclo 34, frente 4)
Treino OFFLINE, por estrategias evolutivas (OpenAI-ES: perturbacoes antiteticas, formatacao de aptidao por postos, Adam), da politica neural de pmsp/ur_numba_neural.py sobre o IG do benchmark de Fanjul-Peyro e Ruiz.
Aptidao de um candidato = media, sobre um lote de instancias de AJUSTE (280 no total; nunca as de teste), da razao makespan / melhor conhecido de uma execucao de T = 1.000 iteracoes (menor e' melhor); o lote e as sementes sao
comuns a todos os candidatos de uma geracao. Duas variantes: 'full' (caracteristicas do estado e da instancia) e 'const' (so' o vies: uma MISTURA constante de acoes treinada com o mesmo orcamento; controle).
Custo de treino registrado (execucoes do IG): geracoes x candidatos x lote. Saidas: results/nn_theta_{espaco}_{variante}_entrada259.npy, results/nn_train_{espaco}_{variante}_entrada259.csv.
Uso: python -m experiment.train_neural_entrada259 --space B30 --variant full --gens 120 --workers 12
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R = ROOT / "results"
H = 16
T = 1000
_INST = None


def _init(space):
    global _INST, _F, _NN
    from experiment import run_cycle32_fanjul_entrada256 as F
    from pmsp import ur_numba_neural as NN
    _F, _NN = F, NN
    man = F.manifest()
    man = man[man.role == "tune"].reset_index(drop=True)
    _INST = []
    for _, r in man.iterrows():
        P, a0, od, lb = F.load(r)
        st, d, ls = F.arm_table_b(space, int(r.n))
        _INST.append((P, a0, int(r.m), od, lb, float(r.best_known), st, d, ls))


def _eval(args):
    theta, A, use_feat, idxs, seed_base = args
    W1, b1, W2, b2 = _NN.unpack(theta, A, H)
    tot = 0.0
    for i in idxs:
        P, a0, m, od, lb, bks, st, d, ls = _INST[i]
        best, _, _ = _NN.ur_neural_core(P, a0, m, st, d, ls, T, W1, b1, W2, b2, use_feat, seed_base + 7919 * int(i), od, lb)
        tot += best / bks
    return tot / len(idxs)


def main(space, variant, gens, workers, K, batch, seed):
    from pmsp import ur_numba_neural as NN
    A = {"B5": 5, "B15": 15, "B30": 30}[space]
    use_feat = 1 if variant == "full" else 0
    D = NN.n_params(A, H)
    rng = np.random.default_rng(seed)
    theta = np.zeros(D)
    i = 0
    theta[i:i + H * NN.NF] = rng.normal(0, 0.3, H * NN.NF)
    i += H * NN.NF + H
    theta[i:i + A * H] = rng.normal(0, 0.1, A * H)
    sigma, lr = 0.15, 0.05
    m1, m2 = np.zeros(D), np.zeros(D)
    t0 = time.perf_counter()
    log, best_theta, best_mon = [], theta.copy(), 1e9
    mon_idx = np.random.default_rng(999).choice(280, 40, replace=False)
    with mp.Pool(workers, initializer=_init, initargs=(space,)) as pool:
        base = pool.apply(_eval, ((theta, A, use_feat, mon_idx, 5_000_000),))
        print(f"[nn {space} {variant}] razao inicial no lote de monitoramento: {base:.4f}", flush=True)
        for g in range(gens):
            idxs = rng.choice(280, batch, replace=False)
            sb = 1_000_000 + 100 * g
            eps = rng.standard_normal((K, D))
            tasks = [(theta + s * sigma * eps[k], A, use_feat, idxs, sb) for k in range(K) for s in (1.0, -1.0)]
            fit = np.array(pool.map(_eval, tasks, chunksize=1)).reshape(K, 2)       # menor e' melhor
            allf = fit.ravel()
            rk = np.empty(2 * K)
            rk[np.argsort(allf)] = np.arange(2 * K)
            u = (rk / (2 * K - 1) - 0.5).reshape(K, 2)       # posto 0 (melhor) -> -0,5
            adv = -(u[:, 0] - u[:, 1])                       # + quando o candidato +eps e' MELHOR que o -eps
            grad = (adv[:, None] * eps).sum(axis=0) / (2 * K * sigma)
            m1 = 0.9 * m1 + 0.1 * grad
            m2 = 0.999 * m2 + 0.001 * grad ** 2
            theta = theta + lr * (m1 / (1 - 0.9 ** (g + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (g + 1))) + 1e-8)
            row = dict(gen=g, fit_mean=float(allf.mean()), fit_best=float(allf.min()))
            if (g + 1) % 10 == 0 or g == gens - 1:
                mon = pool.apply(_eval, ((theta, A, use_feat, mon_idx, 5_000_000),))
                row["monitor"] = float(mon)
                if mon < best_mon:
                    best_mon, best_theta = mon, theta.copy()
                print(f"[nn {space} {variant}] gen {g + 1}/{gens} aptidao media {allf.mean():.4f} monitor {mon:.4f} (melhor {best_mon:.4f}) {time.perf_counter() - t0:.0f}s", flush=True)
            log.append(row)
    np.save(R / f"nn_theta_{space}_{variant}_entrada259.npy", best_theta)
    pd.DataFrame(log).assign(runs_per_gen=2 * K * batch).to_csv(R / f"nn_train_{space}_{variant}_entrada259.csv", index=False)
    print(f"[nn {space} {variant}] concluido: {gens * 2 * K * batch} execucoes do IG de treino; razao no monitor {best_mon:.4f} (inicial {base:.4f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.space, a.variant, a.gens, a.workers, a.K, a.batch, a.seed)
