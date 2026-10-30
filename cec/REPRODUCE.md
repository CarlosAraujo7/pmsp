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
python -m analysis.make_fig_cec6       # Figure 1 -> cec/fig_cec6.pdf
python -m analysis.report_entrada211_warmq; python -m analysis.report_entrada212_warmq_tb; python -m analysis.report_entrada213_equalcpu; python -m analysis.report_entrada215_robustness; python -m analysis.report_entrada217_longc; python -m analysis.report_entrada219_ga; python -m analysis.report_entrada220_ga_mix
python -m analysis.report_entrada222_threshold   # post hoc: base rate, balanced accuracy and leave-one-host-out of the threshold
python -m analysis.report_entrada223_optq        # optimistic Q-Learning (after the pipeline below)
python -m analysis.report_entrada227_cfvariants     # post hoc: estimator variants, cluster-bootstrap intervals of the rule
python -m analysis.report_entrada228_cf_fine       # cross-fitted headroom at finer times (after the pipeline below)
python -m analysis.report_entrada234_ur             # unrelated-machines host (after the pipeline below)
python -m analysis.report_entrada238_sched         # time-varying schedule competitor (after the pipeline below)
python -m analysis.report_entrada240_urselector     # offline selector on unrelated machines (after the commands below)
python -m analysis.report_entrada242_pubaos         # two published AOS designs (after the commands below)
python -m analysis.report_entrada244_irace          # irace as configurator (after the commands below; needs R 4.6 + irace 4.5)
python -m analysis.report_entrada248_ur_eqcpu       # equal CPU time on unrelated machines (after the commands below)
python -m analysis.report_entrada249_closedloop     # closed-loop (state-dependent) oracles (after the commands below)
python -m analysis.report_entrada250_ur_horizon     # long horizons on unrelated machines (after the commands below)
python -m analysis.report_entrada247_global         # global Benjamini-Yekutieli control over every registered test (run last; replaces the global part of report_entrada215_robustness, which re-reads its own output)
python -m analysis.audit_cec_claims; python -m analysis.audit_cec_claims2; python -m analysis.audit_cec_claims3    # every number of the prose recomputed and matched (exit 1 on any mismatch)
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
| Cross-fitted oracle (3 new seeds per fixed action on every test instance) | 209 | `bash experiment/pipeline_cycle19_cforacle_entrada209.sh` (about 1 h; heavy: flowshop arms), then `python -m analysis.report_entrada209_cforacle` |
| Warm-started Q-Learning (pre-train + test, then random tie-breaking) | 211, 212 | `bash experiment/pipeline_cycle19_warmq_entrada211.sh` (about 6 min), `bash experiment/pipeline_cycle19_warmq_tb_entrada212.sh` (about 4 min) |
| Selector and warm Q at equal CPU (own K) | 213 | `bash experiment/pipeline_cycle19_equalcpu_entrada213.sh` (the cost-ratio calibration must run alone; about 15 min) |
| Long horizon in family C (T = 5,000 and 20,000) | 217 | `bash experiment/pipeline_cycle19_longc_entrada217.sh` (about 10 min) |
| Genetic-algorithm host and its validated mixture | 219, 220 | `bash experiment/pipeline_cycle19_ga_stage1_entrada219.sh`, `bash experiment/pipeline_cycle19_ga_stage2_entrada219.sh` (calibration alone), `bash experiment/pipeline_cycle19_ga_mix_entrada220.sh` (about 25 min in all) |
| Optimistic Q-Learning, tuned per family (Q0 x alpha grid, constant exploration, scale-free reward) | 223 | `bash experiment/pipeline_cycle20_optq_entrada223.sh` (about 5 min) |
| Cross-fitted oracle at finer times (3 new seeds per fixed action, S5 and S15) | 228 | `bash experiment/pipeline_cycle21_cffine_entrada228.sh` (about 4 min) |
| Unrelated-machines host (R||Cmax): tuning, optimistic-Q tuning, test | 234 | `python -m experiment.run_cycle22_ur_entrada234 --tune S5; ... --tune S15; ... --tuneqo S5; ... --tuneqo S15`, then `bash experiment/pipeline_cycle22_ur_entrada234.sh` (about 5 min) |
| Time-varying schedule competitor (125 three-phase schedules per class; P||Cmax and R||Cmax) | 238 | `python -m experiment.run_cycle22_sched_entrada238 --tune pc; ... --tune ur`, then `bash experiment/pipeline_cycle22_sched_entrada238.sh` (about 15 s) |
| Offline selector on unrelated machines (1,200 training instances) | 240 | `python -m experiment.run_cycle22_urselector_entrada240 --label S5; ... --label S15; ... --fit; ... --run S5; ... --run S15` (about 10 min) |
| Two published AOS designs (tune on tuning instances, then test) | 242 | `for h in pc ur; do for sp in S5 S15; do python -m experiment.run_cycle23_pubaos_entrada242 --tune $h $sp; done; done; bash experiment/pipeline_cycle23_pubaos_entrada242.sh` (about 5 min) |
| irace as configurator of the static action and of Q-Learning (R 4.6.1 + irace 4.5 from CRAN; set `IRACE_RSCRIPT` and `R_LIBS_USER`) | 244 | `bash experiment/pipeline_cycle24_irace_entrada244.sh` (tuning, about 11 min), record the md5 of `results/irace_tune_*_entrada244.jsonl`, then `bash experiment/pipeline_cycle24_irace_test_entrada244.sh` (about 3 min) |
| Equal CPU time on unrelated machines (cost ratios K calibrated alone on the machine, then the test) | 248 | `python -m experiment.run_cycle26_eqcpu_ur_entrada248 --calibrate` (about 6 min, nothing else running), record the md5 of `results/ur_eqcpu_k_entrada248.json`, then `bash experiment/pipeline_cycle26_eqcpu_ur_entrada248.sh` (about 3 min) |
| Closed-loop oracles: myopic lookahead and rollout over the validated action (P||Cmax and R||Cmax; not budget-fair) | 249 | `python -m pytest tests/test_closedloop_entrada249.py`, record the md5 of the code, then `bash experiment/pipeline_cycle27_closedloop_entrada249.sh` (about 40 min on 6 processes) |
| Long horizons on unrelated machines (T = 5,000, 20,000, 100,000; Q-Learning with exploration scaled to T) | 250 | `python -m pytest tests/test_horizon_ur_entrada250.py`, record the md5 of the code, then `bash experiment/pipeline_cycle28_horizon_ur_entrada250.sh` (about 14 min on 6 processes) |
| Statistical robustness (Pratt, permutation, global BY) | 215 | `python -m analysis.report_entrada215_robustness` (run after the other batteries) |
| DQN, ILS operators, LinUCB, neural Q | 167, 173, 176 | `experiment/run_online_*`, `run_dqn_entrada173.py`, `run_ils_dqn_entrada176.py` |
| Literature audit | 159, 172 | `analysis/litaudit_*` |

Each battery's hypotheses, decision rules and predictions are in the docstring of its `analysis/report_entrada*.py` and in the dated entry of `diario_pesquisa.md`; the md5 hashes of the code frozen before the run are in `results/entradaNNN_prereg_hashes.txt`.

## 3. The protocol as a library
`pmsp/budgetfair.py` (tests in `tests/test_budgetfair.py`) exposes the protocol independently of the scheduling code: paired test with an equivalence margin, Holm, sequential cost-ratio calibration, equal-CPU iteration counts, budget share, the decision/cost split of a learner's gain, the analytic minimum detectable effect and the planted-effect power.
