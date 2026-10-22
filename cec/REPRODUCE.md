# Reproducing the IEEE CEC 2027 paper

Everything in `cec/cec2027_paper.tex` is recomputed from the files in `results/` by scripts in this repository; no number in a table is typed by hand, and every number in the prose is checked by `analysis/audit_cec_claims.py`.

## 0. Environment
Python 3.12 with `numpy`, `pandas`, `scipy`, `matplotlib`, `numba` (see the repository README), and a LaTeX installation with `IEEEtran` (MiKTeX or TeX Live). On Windows with Git Bash: `export PATH` as in `cec/build.sh`.

```bash
python -m pytest -q                    # 1,386 unit and regression tests, a few minutes
python -m analysis.verify_prereg_hashes   # code frozen before each battery still matches its md5 hashes
```

## 1. From the results to the paper (minutes)
```bash
python -m analysis.make_table_cec      # Tables of the paper -> cec/tab_*.tex
python -m analysis.published_cells     # cells and subsets of the published-instance batches -> results/entrada192_published_*.csv
python -m analysis.make_fig_cec        # Figure 1 -> cec/fig_cec2.pdf
python -m analysis.report_entrada185_cycle16 long    # (also: aos, plant, fine) -> results/entrada185_*.txt/csv
python -m analysis.report_entrada188_new; python -m analysis.report_entrada189_published; python -m analysis.report_entrada190_replication
python -m analysis.report_entrada195_valeq; python -m analysis.report_entrada197_pfsp; python -m analysis.report_entrada199_pretest_outofsample
python -m analysis.report_entrada200_selector; python -m analysis.report_entrada200_controls; python -m analysis.report_entrada203_qvariants; python -m analysis.report_entrada205_pfsp_selector
python -m analysis.make_table_cec; python -m analysis.make_table_cec2   # Tables I-VII
python -m analysis.make_fig_cec4       # Figure 1 -> cec/fig_cec4.pdf
python -m analysis.audit_cec_claims; python -m analysis.audit_cec_claims2    # every number of the prose recomputed and matched (exit 1 on any mismatch)
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
| Thompson, DMAB, online racing | 188 | `bash experiment/pipeline_cycle17_new_entrada188.sh` (about 1 h) |
| Published instances, discovery batch (compiled agents) | 189 | `bash experiment/pipeline_cycle17_published_entrada189.sh` (about 11 min); needs the benchmark zips in `external_benchmarks/pcmax_symmetry2025/` |
| Published instances, held-out replication (nothing retuned) | 190 | `bash experiment/pipeline_cycle17_replication_entrada190.sh` (about 5 min) |
| Validated action at equal CPU | 195 | `bash experiment/pipeline_cycle18_valeq_entrada195.sh` (about 5 min) |
| Pre-test VNR: probes, calibration of tau, family F out of sample | 198, 199 | `python -m experiment.run_cycle18_pretest_entrada198 --syn S5` (and S15, `--pub`), `python -m analysis.pretest_vnr_entrada198`; `bash experiment/pipeline_cycle18_famF_entrada199.sh` after `--probe S5/S15` (a few minutes) |
| Offline selector (P||Cmax) and its controls | 200 | `bash experiment/pipeline_cycle18_selector_label_entrada200.sh` (7 min), `--fit`, `bash experiment/pipeline_cycle18_selector_test_entrada200.sh`, `--prepare` and `bash experiment/pipeline_cycle18_selector_controls_entrada200.sh` (minutes) |
| Tuned and plateau-reward Q-Learning | 203 | `bash experiment/pipeline_cycle18_qvariants_entrada203.sh` (about 9 min) |
| Permutation flowshop (battery) | 197 | `bash experiment/pipeline_cycle18_pfsp_entrada197.sh` (about 1.5 h; the cost-ratio calibration phase must run alone) |
| Offline selector for the flowshop | 205 | `bash experiment/pipeline_cycle18_pfsp_selector_label_entrada205.sh` (about 45 min), then `bash experiment/pipeline_cycle18_pfsp_selector_test_entrada205.sh` |
| DQN, ILS operators, LinUCB, neural Q | 167, 173, 176 | `experiment/run_online_*`, `run_dqn_entrada173.py`, `run_ils_dqn_entrada176.py` |
| Literature audit | 159, 172 | `analysis/litaudit_*` |

Each battery's hypotheses, decision rules and predictions are in the docstring of its `analysis/report_entrada*.py` and in the dated entry of `diario_pesquisa.md`; the md5 hashes of the code frozen before the run are in `results/entradaNNN_prereg_hashes.txt`.

## 3. The protocol as a library
`pmsp/budgetfair.py` (tests in `tests/test_budgetfair.py`) exposes the protocol independently of the scheduling code: paired test with an equivalence margin, Holm, sequential cost-ratio calibration, equal-CPU iteration counts, budget share, the decision/cost split of a learner's gain, the analytic minimum detectable effect and the planted-effect power.
