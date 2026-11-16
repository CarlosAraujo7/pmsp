# Reproducing the journal manuscript (`journal/`)

Every number quoted in `journal/*.tex` is recomputed from the files in `results/` by `analysis/audit_journal_claims.py`;
the experiments themselves can be re-run with the pipelines below. Heavy runs are sharded over 12 processes (12 logical cores).

## 1. Check the paper against the stored results (about 3 minutes, no experiment is re-run)

```bash
pip install -r requirements.txt          # Python 3.12; numba 0.68, numpy, pandas, scipy
python -m pytest tests -q                # 1,585 tests, about 4 minutes
python -m analysis.audit_journal_claims  # 422 checks: numbers, text snippets, registered-prediction counters
python -m analysis.audit_journal_short   # every number of the condensed version occurs in the audited full text
python -m analysis.audit_cec_claims4     # numbers of the CEC correction (random-tie Q-Learning)
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
