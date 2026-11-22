# Reproducing the journal manuscript (`journal/`)

Every number quoted in `journal/*.tex` is recomputed from the files in `results/` by `analysis/audit_journal_claims.py`;
the experiments themselves can be re-run with the pipelines below. Heavy runs are sharded over 12 processes (12 logical cores).

## 1. Check the paper against the stored results (about 3 minutes, no experiment is re-run)

```bash
pip install -r requirements.txt          # Python 3.12; numba 0.68, numpy, pandas, scipy
python -m pytest tests -q                # 1,648 tests (about 1 minute on an idle machine with a warm numba cache; about 4 minutes cold)
python -m analysis.audit_journal_claims  # 448 checks: numbers, text snippets, registered-prediction counters
python -m analysis.audit_journal_short   # every number of the condensed version occurs in the audited full text
python -m analysis.audit_cec_claims4     # numbers of the CEC correction (random-tie Q-Learning)
python -m analysis.audit_entradas286_290 # 153 checks: the numbers of the diary for Cycles 53-55 (estimands, learnability logs, order test, horizon, differential evolution)
python -m analysis.audit_cec_claims && python -m analysis.audit_cec_claims2 && python -m analysis.audit_cec_claims3   # 530 checks for the 8-page version
python analysis/make_table_journal.py && python analysis/make_fig_journal_granularity.py   # tables and the figure of the manuscript
bash journal/build.sh                    # journal/main.pdf, the full manuscript (MiKTeX or TeX Live; elsarticle)
bash journal/build_short.sh              # journal/main_short.pdf, the condensed version
```

The md5 hashes recorded before each battery are in `results/entradaNNN_prereg_hashes.txt`; check one with
`md5sum -c results/entrada266_prereg_hashes.txt`. They were computed on the Windows working copy: on another platform
the line-ending conversion of git can change the hash of a text file, so compare after `git config core.autocrlf false`.

## 2. Where each battery lives

| Entry | What | Pipeline / runner | Report (with the registered predictions) | Section |
|---|---|---|---|---|
| 256 | standard R\|\|Cmax benchmark (700 test instances) | `experiment/pipeline_cycle32_fanjul_entrada256.sh` | `analysis/report_entrada256_fanjul.py` | 5.2 |
| 258 | why the standard Q-Learning loses (index-0 lock-in), long horizon T=10^4 | `experiment/run_cycle34_mechanism_entrada258.py`, `pipeline_cycle34_longT_entrada258.sh` | `report_entrada258_mechanism*.py`, `report_entrada258L_longT.py` | 4.5, 5.4 |
| 259, 262 | neural policy trained offline (evolution strategies) and its test | `pipeline_cycle36_neural_entrada259.sh` | `report_entrada262_neural.py` | 5.5 |
| 260, 261, 263, 264 | order-invariant Q-Learning on every host | `run_cycle35_*`, `run_cycle37_*`, `run_cycle38_*` | `report_entrada260..264_*.py` | 4.5 |
| 266 | value of offline knowledge by granularity | `pipeline_cycle39_arms_entrada266.sh` | `report_entrada266_granularity.py` (+ post hoc `report_entrada266b_posthoc.py`) | 5.6 |
| 271, 272 | optimistic Q-Learning warm-started with an offline prior, and the control without learning | `pipeline_cycle43_warmqo_entrada271.sh`, `pipeline_cycle44_frozenqo_entrada272.sh` | `report_entrada271_warmqo.py`, `report_entrada272_frozenqo.py` | 5.6 |
| 267, 270 | second domain: ILS for the TSP (generated and TSPLIB instances) | `pipeline_cycle40_tsp_entrada267.sh`, `pipeline_cycle42_tsplib_entrada270.sh` | `report_entrada267_tsp.py`, `report_entrada270_tsplib.py` | 5.7 |
| 274, 275 | the warm-start finding replicated in the TSP host (synthetic and TSPLIB instances) | `pipeline_cycle45_warmtsp_entrada274.sh` (hashes, then both runs) | `report_entrada274_warmtsp.py`, `report_entrada275_warmtsplib.py` | 5.7 |
| 268 | equal CPU time on the standard benchmark | `pipeline_cycle41_eqcpu_fanjul_entrada268.sh` | `report_entrada268_eqcpu_fanjul.py` | 5.3 |
| 277, 278 | price of offline knowledge (tuning cost, learning curve, equal total compute); ceiling and margin of the synthetic classes | none (existing data) | `report_entrada277_cost.py`, `report_entrada278_ceiling.py` | 5.4, 4.2 |
| 281 | scarce tuning data: static action, warm-started learner and frozen prior with the same k = 1, 2, 3, 5 tuning instances | `pipeline_cycle49_scarce_entrada281.sh` (hashes, then the run) | `report_entrada281_scarce.py` | 5.4 |
| 282, 282b | the protocol on a third-party implementation: ALNS for the CVRP with the `alns` library (needs `pip install alns[mabwiser] vrplib py7zr`; X instances from CVRPLIB) | `pipeline_cycle50_cvrp_entrada282.sh` (hashes, tuning, choice, test, X) | `report_entrada282_cvrp.py`, `report_entrada282b_xposthoc.py` | 5.5 |
| 283, 283b | reanalysis for the CEC revision, no new runs: capture ratio with class-level bootstrap, margin sensitivity, the order of the actions in one table, first-action equivalence, class table (283b: the 69 CEC cells, post hoc) | none (reads stored results; hashes in `results/entrada283_prereg_hashes.txt`) | `report_entrada283_reanalysis.py`, `report_entrada283b_cec69.py` | 4.5 |
| 284 | ablation of the IG's components, no learner: greedy or random reconstruction x short descent, with the ILS-style variant | `pipeline_cycle51_ablation_entrada284.sh` (run `--calibrate` of `run_cycle51_ablation_entrada284.py` first, on a quiet machine; hashes in `results/entrada284_prereg_hashes.txt`) | `report_entrada284_ablation.py` | 4.1 |
| 285, 285b | the ablation with a full descent: the classical ILS (random reinsertion + descent to a local optimum) against the plain IG; 285b: the ILS with the best fixed kick size of each class | `pipeline_cycle52_fulldescent_entrada285.sh`, `pipeline_cycle52b_fixedsize_entrada285b.sh` (run `--calibrate` of each runner first, on a quiet machine; hashes in `results/entrada285_prereg_hashes.txt`, `entrada285b_prereg_hashes.txt`) | `report_entrada285_fulldescent.py`, `report_entrada285b_fixedsize.py` | 4.1 |
| 286, 286b | estimands, inference and coupling on stored data, no new runs: capture ratio with cluster bootstrap and Fieller interval, the coupling of the rank correlation through the shared random arm (instance halves), the strict decision rule and TOST parity, unlabelled pools of classes (286b, post hoc: the choice rule of the single static action) | none (reads stored results; hashes in `results/entrada286_prereg_hashes.txt`) | `report_entrada286_estimands.py`, `report_entrada286b_pool_sensitivity.py` (loader `cells_instance_tables.py`) | -- |
| 287, 287b | per-decision logs of the random control in the 69 CEC cells and the learnability model (information scarcity, win-stay capture, oblivious-bandit digital twin; the pre-registered gate FAILED); 287b post hoc | `experiment/run_cycle54_logs_entrada287.py --all` (logs in `results/logs287/`, not tracked; `results/logs287_manifest_entrada287.csv` holds their md5; the logged cores `pmsp/ig_logged.py`, `pmsp/ur_logged.py` come from `python -m pmsp.make_logged_cores`) | `report_entrada287_learnability.py`, `report_entrada287b_posthoc.py` (model in `analysis/learnability_model.py`) | -- |
| 288 | order-invariance permutation test (`pmsp/order_test.py`) and the zero-row accounting of the standard Q-Learning, 19 cells, 12 orders of the action list | `experiment/run_cycle54b_order_entrada288.py` | `report_entrada288_order.py` | -- |
| 289, 289b | the protocol on differential evolution with strategy selection (BBOB, D = 10, five strategies, success-rate reward, dense signal); 289b: the digital twin's numeric predictions, frozen before the learner stage | `experiment/run_cycle55_de_entrada289.py --stage tune`, `--stage base`, then `analysis/predict_entrada289_twin.py`, then `--stage learn` (needs `cma==4.6.0`; `pmsp/de_host.py`) | `report_entrada289_de.py` (twin in `analysis/learnability_dense.py`) | -- |
| 290 | horizon dose-response on the standard R||Cmax benchmark (T = 1,000, 4,000, 16,000; the first run, with a flawed validated action, is kept in `results/horizon290_run1_flawed/` and not counted) | `experiment/run_cycle54c_horizon_entrada290.py` | `report_entrada290_horizon.py` | -- |
| 286-290 | numeric audit of the entries above against the diary | `analysis/audit_entradas286_290.py`; all steps in order: `experiment/pipeline_cycle53_55_entradas286_290.sh` | -- | -- |
| 279, 280, 280b | Q-Learning with ties broken at random and permuted lists; the same in the 85 cells of the order diagnostic; post hoc test against the validated action | `pipeline_cycle47_ties_entrada279.sh`, `pipeline_cycle48_tiesall_entrada280.sh` (hashes, then the runs) | `report_entrada279_ties.py`, `report_entrada280_tiesall.py`, `report_entrada280b_headline.py` | 4.5 |
| 265, 269, 273, 276 | global Benjamini-Yekutieli control | none (reads all test files) | `report_entrada265_global.py`, `report_entrada269_global.py`, `report_entrada273_global.py`, `report_entrada276_global.py` | 6 |

The registered predictions of every battery (the `PRE-REGISTRO` block of each report) are compiled in one document by
`python analysis/make_osf_registry.py` (`entrega/OSF_REGISTRO_PREVISOES.md`). The research diary is `diario_pesquisa.md`.

## 3. Re-running a battery

Run the pipeline of the table above from the repository root, for example `bash experiment/pipeline_cycle39_arms_entrada266.sh`,
then the report with `python -m analysis.report_entrada266_granularity`. Approximate wall-clock times on a 12-core laptop:
Entry 266 about 12 minutes; Entry 267 about 15 minutes; Entry 270 about 3 minutes; Entry 268 about an hour of sequential
cost-ratio calibration (it needs an idle machine) plus 12 minutes; Entry 258 (T=10^4) about 1 hour for the test after the tuning;
the neural training of Entry 259 several hours. The pipelines freeze the md5 hashes between tuning and test, so
do not edit the hashed files while a pipeline runs. Reports refuse to run on incomplete data.

## 4. Data and provenance

- Synthetic instances are regenerated from their seeds (`pmsp/generators.py`, `pmsp/instance_ur*.py`, `pmsp/instance_tsp.py`).
- `external_benchmarks/rcmax_fanjul_ruiz/`: the standard R||Cmax benchmark of Fanjul-Peyro and Ruiz (2010), MIT-licensed copy of the
  Fraunhofer-Chalmers repository `RCmax`; `external_benchmarks/pcmax_symmetry2025/`: the P||Cmax benchmark reviewed by Ostojic et al. (2025).
- `external_benchmarks/tsplib/`: 37 EUC_2D instances of the TSPLIB (Reinelt, 1991) with 200 <= n <= 2103 from a public GitHub mirror
  (`pdrozdowski/TSPLib.Net`, copy of TSPLIB95; 816,923 bytes) and the optimal values of the TSPLIB file `solutions`
  (`mastqe/tsplib`); `download_tsplib.py` and `manifest_tsplib.json` record the source, size and md5. The files are read as text and never executed.
- irace runs (Entries 244, 255) need R 4.6.1 with irace 4.5; every other battery is Python only.
