# PMSP Learnheuristics: is it the mechanism or the policy?

Code, data and paper source for a study that decomposes the gains of
Q-Learning-guided metaheuristics for the identical-parallel-machine
makespan problem (P||Cmax) into the contribution of the **search mechanism**
(Iterated Greedy, `ig_ql`; Iterated Local Search, `ils_ql`) and of the
**learned selection policy**. Companion repository for the manuscript in
`paper/learnheuristic_pmsp.tex` ("Mechanism or policy? Budget-fair attribution
of the gains of learning-guided metaheuristics in scheduling"), a working draft
whose recommended venue is the *Journal of Heuristics* (`DECISAO_SUBMISSAO.md`),
with its supplementary material in `paper/supplementary_material.tex`. Text in
red in the PDFs marks what is still to be improved (`\draftmarkstrue` in the
preambles; set `\draftmarksfalse` to remove it).

## Repository structure

- `pmsp/` -- core package: instance/solution representation (`instance.py`,
  `solution.py`), constructive heuristic (`constructive.py`), operators and
  selectors (`operators.py`, `selectors.py`, `fixed_selector.py`), the tabular
  Q-Learning agent (`qlearning.py`), the ILS and IG search engines (`ils.py`,
  `ig.py`; both accept injected `reward_fn`/`state_fn` and, for IG,
  `initial_solution`), the adaptive dispatch rule (`apriori_select.py`), the
  hybrid/racing prototypes (`hybrid.py`, `racing.py`), the reimplemented
  published baselines (`multifit.py`, `sa.py`, `bco.py`, plus `ldm.py`, the
  Largest Differencing Method; fidelity to each original publication is
  documented in each module's docstring and cross-checked in `tests/`),
  `seeding.py` (best of LPT/MULTIFIT/LDM as IG start), `rewards.py` (dense
  reward used in the signal test), `generators.py` (six processing-time
  distributions) and `exact.py` (certified lower bound from the linear
  relaxation of the arc-flow model, validated against brute force and
  against the optima of the iAF method).
- `experiment/` -- driver (`runner.py`, `config.py`) and one script per
  extension, named after the diary entry it supports. Pre-specified batteries
  of the final study: `run_seeded_baselines_entrada102.py` (LDM, seeded IG),
  `run_signal_headroom_entrada104.py` (headroom, dense reward, pre-training),
  `run_hyperparam_sensitivity_entrada105.py`, `run_timing_entrada107.py`,
  `run_new_distributions_entrada108.py`, `run_seed_variance_entrada109.py`,
  `run_equal_cpu_entrada110.py`, `run_dsel_entrada114.py` (destruction size as the action space),
  `run_external_confirm2_entrada115.py` (second confirmatory test), `run_ils_seeded_entrada117.py`,
  `run_seed_variance_all_entrada117.py`, `run_unrelated_entrada118.py` (R||Cmax: tuning, calibration, test, exact),
  `run_budget_law_entrada121.py`, `run_dsel_budget_entrada121.py`, `run_unrelated_budget_entrada121.py`,
  `run_igcv_seed_entrada121.py` (tuned destruction size + strong start), `run_unrelated_law_entrada124.py`,
  `run_igls_p_entrada124.py`, `run_igcvls_entrada127.py` (recipe with local search; equal CPU),
  `run_unrelated_large_entrada127.py` (R||Cmax 1000 x 50 vs MILP), `run_igls_eq_entrada129.py` (hybrid at equal CPU),
  `run_unrelated_policy_entrada133.py` (policy at 1000 x 50, equal CPU), `run_unrelated_valid_entrada135.py`
  (offline validation vs policy at 1000 x 50), `run_rcmax_benchmark_entrada137.py` (external validation on the standard R||Cmax
  benchmark of Fanjul-Peyro and Ruiz; data in `external_benchmarks/rcmax_fanjul_ruiz/`, not redistributed), `run_recipe_budget_entrada139.py` (quality vs budget of the recipe),
  `run_lpig_benchmark_entrada141.py`, `run_lpig_large_entrada143.py` and `run_lpigsr_benchmark_entrada149.py` (IG started from the linear relaxation of R||Cmax, `pmsp/unrelated_lp.py`, followed by a restricted MILP, `pmsp/unrelated_srmilp.py`), `run_lprc_benchmark_entrada156.py` (LP-reduced-cost candidates for the restricted MILP; `pmsp/window_bandits.py` and `pilot_window_bandits_entrada155.py` are the pilot of window-aware bandits), `run_pfsp_entrada152.py` (third problem: permutation flowshop, `pmsp/pfsp.py`), `run_pfsp_taillard_entrada159.py` (replication on Taillard's classic instances, regenerated from the published generator by `external_benchmarks/pfsp_taillard/generate_taillard.py`), `run_ctx_entrada159.py` (learners of another class for the destruction size: offline instance-conditional selector and the contextual bandit `pmsp/linucb.py`), `run_igls_fast_entrada162.py` and `run_ils_fast_entrada162.py` (sensitivity of the equal-CPU conclusions to the efficiency of the implementation: vectorized local search and ILS operators `pmsp/operators_fast.py`, results identical to the loop versions), `run_neural_entrada162.py` (neural Q-learner `pmsp/neuralq.py` and offline MLP selector), `run_online_replic_entrada167.py` (LinUCB and neural Q in the other two environments), `run_dpolicy_entrada145.py` (prospective test of the
  headroom law with the destruction size as the action; `diag_dpolicy_reward_entrada147.py` is its post hoc diagnostic); earlier ones: `run_state_recalibration_entrada98.py`,
  `run_granularity_factorial_entrada98.py`, `run_sota_sensitivity_entrada90.py`.
  The R||Cmax model (instance, constructive heuristic, IG with 4 destruction strategies and local search, ILS,
  LP bound, MILP) is in `pmsp/unrelated.py` and `pmsp/unrelated_exact.py`.
- `analysis/` -- post-processing: statistics (`stats.py`: Wilcoxon,
  Holm-Bonferroni, paired rank-biserial effect size, bootstrap intervals), LaTeX
  table generators (`make_table_*.py`, `make_tables_*.py`), figures
  (`make_figure_pareto.py`, `plots.py`), the **pre-specified analysis scripts**
  (`report_entradaNNN_*.py`) and the audits (`audit_entrada*.py`;
  `audit_entrada113_final_claims.py`, `audit_entrada126_final_claims.py`, `audit_entrada127_cycle4_claims.py` and
  `audit_entrada133_cycle5_claims.py` recompute the numbers
  stated in the manuscript from the CSV files and compare them with the text).
- `data/` -- synthetic instance generation and the 180 materialised
  instances of the six original classes; the 12 classes of the study regenerate
  each instance from its seed, `PMSPInstance.random(n, m, seed=BASE_SEED + rep)`.
- `external_benchmarks/pcmax_symmetry2025/` -- external validation on the public
  benchmark of Ostojic et al. (2025); see its own `README.md` and the section
  "External benchmark data" below.
- `results/` -- `results.csv` (synthetic battery, 23 methods x 12 classes), the
  CSV/TXT outputs of every later experiment, and `entradaNNN_prereg_hashes.txt`
  (md5 hashes of the code and data frozen **before** each pre-specified analysis).
- `tests/` -- 1,110 unit and regression tests (`pytest`, ~15 s).
- `paper/` -- the manuscript and its supplement (`elsarticle`), figures, tables
  generated by `analysis/make_table*`, and `build.sh` (compiles both, in the
  right order, and reports pages, errors and undefined references).
- `diario_pesquisa.md` -- the research diary (numbered entries, in Portuguese):
  the dated, append-only record of decisions, hypotheses and results. It is the
  provenance of the pre-specifications (it is a dated, hash-stamped record,
  **not** a public registry).
- `ANALISE_E_PLANO_EJOR.md`, `ENQUADRAMENTO_EJOR.md`, `AVALIACAO_CRITICA_05102026.md`,
  `LEI_GERAL_PESQUISA.md` -- strategy, framing, the critical self-assessment
  that drove the revision, and the author's methodological ground rules.
- `archive/` -- superseded material kept for the record (dated backups, one-off
  patch scripts, build logs); ignored by git.

## Environment

Python 3.12 (developed and run on Windows 11; the code is platform independent).
`requirements.txt` pins the versions in which every result was produced:

```
pip install -r requirements.txt
```

The certified bound uses the HiGHS solver shipped with SciPy
(`scipy.optimize.linprog`, `method="highs-ipm"`); no commercial solver is needed.
LaTeX: MiKTeX or TeX Live with `xelatex` and `bibtex`
(`elsarticle.cls` and `elsarticle-harv.bst` are in `paper/`).

## Reproducing the results

1. Test suite: `python -m pytest tests` (1,110 tests, ~15 s), including
   hand-computed regression cases for every reimplemented baseline, brute-force
   checks of the certified bound, determinism checks of the new hooks, and the
   arbitrary-precision overflow fix for the external instances.
2. Synthetic battery: `python experiment/runner.py` (see `experiment/config.py`).
   Each later experiment is a single command, e.g.
   `python -m experiment.run_seeded_baselines_entrada102 --workers 12`;
   external instances: `cd external_benchmarks/pcmax_symmetry2025 &&
   python run_external_battery.py --methods ldm,ig_ql_seed,ig_random_seed
   --output results_external_seeded_entrada102.csv`.
3. Pre-specified analyses: `python -m analysis.report_entrada102_seeded_baselines`
   (and `..._entrada104_signal_headroom`, `..._entrada105_hyperparam_sensitivity`,
   `..._entrada108_new_distributions`, `..._entrada109_seed_variance`,
   `..._entrada110_equal_cpu`; earlier: `report_entrada90_*`, `report_entrada94_*`,
   `report_entrada98_*`). Before re-running any of them, check the frozen files
   against `results/entradaNNN_prereg_hashes.txt` (`md5sum -c` style);
   amendments made before results were read are listed at the end of each file.
4. Tables and figures: `python -m analysis.make_table_stronger_baselines`,
   `analysis.make_table_policy_robustness`, `analysis.make_tables_supplement_robust`,
   `analysis.make_figure_pareto` (and the older `make_table_*`).
4b. Cycle 5-7 tables: `python -m analysis.make_tables_cycle5` (needs the `report_entrada127_*`, `report_entrada129_*`, `report_entrada133_*`, `report_entrada135_*`
   outputs and `analysis.explore_entrada134_policy_by_dist`, a post hoc analysis).
5. Audit: `python -m analysis.audit_entrada113_final_claims`, `analysis.audit_entrada126_final_claims`, `analysis.audit_entrada127_cycle4_claims` and
   `analysis.audit_entrada133_cycle5_claims` (225 checks covering Cycles 5-11).
6. Paper: `bash paper/build.sh` (main -> supplement -> main; main ~29 pages, supplement ~45). The main paper is
   `paper/short/*.tex`; the moved material is generated by `python paper/make_s_moved.py` into `paper/s_moved.tex`; the
   46-page version `paper/learnheuristic_pmsp_long.tex` is a historical record, superseded in places
   (`paper/LONG_VERSION_SUPERSEDED.md`). Venue decision: `DECISAO_SUBMISSAO.md`; trajectory and lessons:
   `TRAJETORIA_E_LICOES.md`; reusable methodology: `PLAYBOOK_METODOLOGIA_CLAUDE_CODE.md`.
6b. Cycles 12-14 (Entradas 158-175; each batch has its pre-registration in the docstring of its `analysis/report_entradaNNN_*.py`,
   hashes in `results/entradaNNN_prereg_hashes.txt`): Taillard replication (`run_pfsp_taillard_entrada159.py`), LinUCB and
   instance-conditional learners (`run_ctx_entrada159.py`), the literature audit (`analysis/litaudit_*`, rounds 1 and 2), vectorized local search
   and ILS (`pmsp/operators_fast.py`; `run_igls_fast_entrada162.py`, `run_ils_fast_entrada162.py`), neural learners
   (`pmsp/neuralq.py`, `pmsp/dqn.py`; `run_neural_entrada162.py`, `run_dqn_entrada173.py`), online replication and larger offline training sets
   (`run_online_replic_entrada167.py`, `run_reserve_arms_entrada168.py`), learners over the operators of the ILS (`run_ils_dqn_entrada176.py`),
   the vectorized-ILS timing and equal-CPU repeats (`run_timing_ils_fast_entrada174.py`, `run_equal_cpu_fast_entrada178.py`), and the compiled IG with an evaluation counter
   (`pmsp/ig_numba.py`, needs `numba`; `run_numba_entrada170.py`, `analysis/report_entrada170_numba.py`). Independent processes
   (`--shard i --nshards N`, then `--merge`) instead of a process pool. Compiled learners and larger action spaces (`pmsp/ig_numba_agents.py`, `run_agents_numba_entrada180.py`). Pre-registration hashes can be checked ignoring line endings with `python -m analysis.verify_prereg_hashes [pattern]` (entries for core modules that evolved after a freeze report `MUDOU`, as expected). Audits: `python -m analysis.audit_entrada159_claims`,
   `analysis.audit_entrada162_claims`, `analysis.audit_entrada171_claims`, `analysis.audit_short_numbers`.
7. Certified bound for new instances:
   `from pmsp.exact import lp_lower_bound; lp_lower_bound(p, m, upper_bound)`
   returns `(LB, proven_optimal, seconds)`.

Randomness: every run is deterministic given its seed (`numpy.random.default_rng`);
re-executing `ig_ql` on 850 instances reproduced every stored makespan exactly,
and the new hooks reproduce the stored `ig_ql_cal` makespans (checked by
`run_signal_headroom_entrada104 --check` and `run_hyperparam_sensitivity_entrada105 --check`).

## External benchmark data

The external-validation section evaluates all methods on the public instance set
released with:

> Ostojic, D., Ramljak, D., Urosevic, A., Jolovic, M., Draskovic, R.,
> Kakka, J., Jaksic Kruger, T., Davidovic, T. (2025). Systematic
> Literature Review of Optimization Algorithms for P||Cmax Problem.
> *Symmetry*, 17(2), 178.

The four raw instance families (B, C, E, F) are **not** duplicated in this
repository; download them from the source named in
`external_benchmarks/pcmax_symmetry2025/README.md` and place the `.zip` files
alongside `loader.py`, which reads them directly. `manifest.csv` (tracked)
lists the stratified sample used in the paper (6,780 instances: B 2,000,
C 2,000, E 2,000, F 780), and `manifest_holdout_entrada90.csv` lists the
4,160 further instances reserved for the pre-specified confirmatory test
(Entradas 90-91). `results_external_symmetry2025.csv` (89,648 rows) and the
other CSV files in that folder are this study's own output, not third-party
data. Two hygiene notes: 4 manifest members have all 13 methods recorded twice
(52 `(member, method)` pairs: the same file ran under two instance indices
after a manifest regeneration), and 112 family-B instances from an older
manifest are present in the CSV but outside the current manifest. The analysis
scripts drop duplicates by `(member, method)` and restrict to the manifest; no
stored row was altered.

## Author

Carlos Vinicio Dantas Araujo, M.Sc. student, Computer Science,
Universidade Estadual do Ceara (UECE), Brazil.

## License and citation

Code: MIT. Result files and generated tables/figures: CC BY 4.0 (see `LICENSE`).
The external benchmark instances remain under their original terms.
Citation metadata: `CITATION.cff` (repository URL and DOI to be filled in
when the repository is made public).

## Generative AI disclosure

Portions of this codebase, the research diary, and the manuscript were
drafted with the assistance of a generative AI tool (Claude, Anthropic),
under the author's direction and review; see the paper's own
"Declaration of Generative AI and AI-Assisted Technologies" section for the
statement accompanying the submitted manuscript.
