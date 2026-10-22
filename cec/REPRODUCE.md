# Reproducing the IEEE CEC 2027 paper

Everything in `cec/cec2027_paper.tex` is recomputed from the files in `results/` by scripts in this repository; no number in a table is typed by hand, and every number in the prose is checked by `analysis/audit_cec_claims.py`.

## 0. Environment
Python 3.12 with `numpy`, `pandas`, `scipy`, `matplotlib`, `numba` (see the repository README), and a LaTeX installation with `IEEEtran` (MiKTeX or TeX Live). On Windows with Git Bash: `export PATH` as in `cec/build.sh`.

```bash
python -m pytest -q                    # 1,2xx unit and regression tests, ~20 s
python -m analysis.verify_prereg_hashes   # code frozen before each battery still matches its md5 hashes
```

## 1. From the results to the paper (minutes)
```bash
python -m analysis.make_table_cec      # Tables of the paper -> cec/tab_*.tex
python -m analysis.make_fig_cec        # Figures 1 and 2 -> cec/fig_cec.pdf, cec/fig_cec2.pdf
python -m analysis.report_entrada185_cycle16 long    # (also: aos, plant, fine) -> results/entrada185_*.txt/csv
python -m analysis.make_fig_cec3       # Figure 3 -> cec/fig_cec3.pdf
python -m analysis.audit_cec_claims    # every number of the prose recomputed and matched (exits 1 on any mismatch)
bash cec/build.sh                      # draft with names and red notes
bash cec/build.sh anon                 # double-anonymous version for submission (no names, e-mail, repository link)
```

## 2. Re-running the experiments (hours; one heavy job at a time, the CPU-ratio calibrations alone on a quiet machine)
| Result in the paper | Entrada | Command (from the repository root) |
|---|---|---|
| IG vs ILS, vectorized, equal CPU | 162-164 | `experiment/run_ils_fast_entrada162.py` (see its docstring) |
| Local search budget by implementation | 124, 129, 162, 170 | `experiment/run_igls_*`, `experiment/run_numba_entrada170.py` |
| Compiled agents, 3 to 30 actions | 180 | `bash experiment/pipeline_agents_entrada180.sh` |
| Power and minimum detectable effect | 184 | `python -m analysis.power_entrada184` |
| Classic AOS, planted reward, long horizon, finer times | 185 | `bash experiment/pipeline_cycle16_entrada185.sh` (the longest step is `--long-run 100000`, about 2 h) |
| DQN, ILS operators, LinUCB, neural Q | 167, 173, 176 | `experiment/run_online_*`, `run_dqn_entrada173.py`, `run_ils_dqn_entrada176.py` |
| Literature audit | 159, 172 | `analysis/litaudit_*` |

Each battery's hypotheses, decision rules and predictions are in the docstring of its `analysis/report_entrada*.py` and in the dated entry of `diario_pesquisa.md`; the md5 hashes of the code frozen before the run are in `results/entradaNNN_prereg_hashes.txt`.

## 3. The protocol as a library
`pmsp/budgetfair.py` (tests in `tests/test_budgetfair.py`) exposes the protocol independently of the scheduling code: paired test with an equivalence margin, Holm, sequential cost-ratio calibration, equal-CPU iteration counts, budget share, the decision/cost split of a learner's gain, the analytic minimum detectable effect and the planted-effect power.
