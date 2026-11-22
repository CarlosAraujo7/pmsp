# Revisor 1 (computação evolutiva, hiper-heurísticas, seleção adaptativa de operadores)

Modelo: opus. Notas: content_novelty 56, teto 67, borderline (inclinado a weak reject). Não usou busca na web.

---

**Reviewer 1 report: IEEE CEC 2027, "Budget-Fair Attribution of the Gains of Learning-Guided Iterated Greedy: Where Does Online Learning Pay?"**

(No web search was used. The citations below are from my own knowledge of the field. Items I am less sure of are marked "(unverified)".)

**A. Claimed contributions**

C1. A "budget-fair attribution protocol". It has six parts: uniform random choice over the same actions at a matched budget; equal CPU time calibrated through a cost ratio K, which splits the gain into a decision effect and a cost effect; a static action (or mixture) validated offline on separate tuning instances; a cross-fitted oracle headroom; a power check with a positive control; and statistics fixed beforehand (Holm correction plus a ±0.05 pp equivalence margin). The protocol is released as a library.
C2. RQ1: in P||Cmax, the IG's advantage over a classical ILS comes from greedy reconstruction, not from the learner. Learners add at most 0.04 pp.
C3. RQ2: across about ten online selectors (QL, UCB1, policy gradient, PM, AP, DMAB, TS, racing, LinUCB, DQN), no learner beats random choice beyond the margin at equal iterations. The offline-validated static action beats them, with 2 exceptions in 69 cells.
C4. RQ3: zero-initialised Q-Learning with first-index tie-breaking repeats the first listed action, so its "wins" track list order. With random tie-breaking it captures about 16% of the static action's gain, in step with it (Spearman 0.78). An offline instance-conditioned selector captures more, but this is mostly class-level knowledge.
C5. Hosts used as checks: a flowshop IG, a steady-state GA, R||Cmax, and the third-party `alns` CVRP.

**B. Novelty audit**

**C1 (protocol). Novelty 2/5.**
Every component is established practice:
- Random-selection baselines are the standard reference in selection hyper-heuristics. Simple Random is the baseline in the HyFlex/CHeSC 2011 framework (Ochoa et al., EvoCOP 2012) and in Burke et al. (JORS 2013).
- Equal-CPU comparisons are standard practice (Johnson 2002, "A theoretician's guide…"; Bartz-Beielstein et al. 2020, cited).
- Component ablation: Fawcett & Hoos, J. Heuristics 2016; Bezerra, López-Ibáñez & Stützle on component-wise AOS/MOEA design (unverified as to exact venue).
- Tuned static versus control: Karafotias, Hoogendoorn & Eiben (TEVC 2015, cited) make exactly this point. The DAC literature always reports the best static configuration as the baseline (Biedenkapp et al., ECAI 2020; Adriaensen et al., JAIR 2022; DACBench, Eimer et al., IJCAI 2021).
- The single-best/oracle gap closure is ASlib practice (Bischl et al. 2016).
- Equivalence testing: Lakens 2017.

What is new is the combination. The most original pieces are the decision-effect/cost-effect split of equal-CPU gains, and cross-fitting the oracle to remove the optimistic bias of "best of one seed". Both are useful but small. As packaged practice, the protocol does not stand on its own as a CEC contribution.

**C2 (greedy reconstruction explains the IG). Novelty 2/5.**
Ruiz & Stützle (EJOR 2007) and Stützle & Ruiz's Iterated Greedy chapter (Handbook of Heuristics, 2018; unverified as to edition) frame IG as an ILS whose perturbation is destruction plus greedy construction. That the constructive step drives quality is the field's received view. The clean factorial on P||Cmax (Table III: G vs R with three descent levels) is a nice confirmation, but on a single problem whose greedy LPT-like insertion is near-optimal by construction. It is also partly a statement about how weak the chosen ILS is: it uses a single random-reinsertion kick, as the authors admit in Limitation (ii).

**C3 (learners ≈ random choice; tuned static wins). Novelty 2/5. This is the main overlap issue.**
"A random choice among the operators is as good as the learned selection" has been shown many times:
- Bilgin, Özcan & Korkmaz (PATAT 2006) and Özcan, Bilgin & Korkmaz (Intelligent Data Analysis 2008, "A comprehensive analysis of hyper-heuristics") found that the move-acceptance criterion matters more than the selection mechanism, and that Simple Random is competitive.
- CHeSC 2011 analyses repeatedly found simple-random-based methods hard to beat on several domains.
- Alanazi & Lehre (CEC 2014, cited) proved that on LeadingOnes, classical learning mechanisms (random gradient, greedy, permutation) give no asymptotic gain over Simple Random. Lissovoi, Oliveto & Warwicker (AAAI 2019 / ECJ 2020; unverified as to exact venue) and Doerr et al. (GECCO 2018) showed that the learning period must be tuned for learning to pay at all.
- Fialho et al. (AMAI 2010, cited) and Da Costa et al. (GECCO 2008) show that AOS gains depend on operator heterogeneity: with near-equivalent operators, the adaptive and uniform strategies coincide.
- Karafotias et al. (2014/2015) report that RL-based generic parameter control rarely beats a tuned static configuration.

The new elements here are the scale of the null (around ten learners, 69 cells, explicit equivalence margins and power) and the comparison against a validated static action. That is a solid negative replication, not a discovery. The paper cites [11], [13], [1] but does not tell the reader that its central null result is anticipated by this hyper-heuristics literature. That framing has to change.

**C4 (tie-rule lock-in in Q-Learning). Novelty 2/5 as an insight, 3/5 as a methodological warning.**
Breaking argmax ties at random is textbook advice (Sutton & Barto, 2018, cited). Zero-initialised Q with non-negative rewards making the first arm absorbing is a known pitfall. What is publishable is the audit: published learning-guided IG studies may report gains that track list order, and the effect is large on R||Cmax (55–73% of choices go to the first arm; reversing the list moves the lock-in; ordering the validated action first recovers 81–86% of its gain). The authors themselves cannot verify that [9] has this artefact ("takes the argmax without stating a tie rule"). The claim about the literature is therefore speculative. The 16% capture with Spearman 0.78 is a modest new empirical regularity.

**C5 (hosts). Novelty 3/5.**
Running the protocol on the third-party `alns` library with its own selectors is the most persuasive generality check: 0 of 40 selector-class pairs beat the validated operator. It is the right kind of evidence.

**C. Soundness audit**

**Strengths**
- Pre-specified decision rules with Holm correction and equivalence margins, plus a power analysis. The minimum detectable gain is 0.006–0.014 pp, and the planted-gain power is 58–99.8% at 0.02.
- Positive controls (a planted reward bonus is detected), which address the "the pipeline cannot detect learning" objection.
- Held-out confirmation on 3,950 published instances, where all five confirmatory tests held.
- A third-party host, irace as a competitor, and re-implemented published AOS designs.
- An explicit K-decomposition showing that equal-CPU "wins" are cost effects (QL K = 0.50–0.60).

**Threats**

1. **Ceiling and headroom.** In P||Cmax, random choice reaches the bound in 7–96% of runs. The validated action gains only 0.003–0.24 pp, and the cross-fitted headroom is negative in four classes (Table I). A null result for learners where there is almost nothing to learn is close to uninformative. The authors partly answer this with R||Cmax, where headroom is 1.5–2.8 pp. There the learners still fail (QL −1.06/−0.48), which is the stronger part of the evidence. But the R||Cmax learners also include configurations "with our untuned settings", and their capture is ≤18%.
2. **The action spaces are weak tests of AOS.** Operator spaces are 3–30 actions that are largely redundant (destruction sizes on one scale). AOS is known to pay when operators differ strongly and their usefulness changes over the run (Fialho; DMAB on dynamic operator quality). The 81-state Q-table is visited about 2.5 times per cell in T = 1000. That is a test of tabular RL under a hopeless sample budget, not of online learning in general. The conclusion "gains come from knowledge acquired offline, not from learning during the run" overreaches beyond this regime.
3. **Horizon.** T up to 10^5 is mentioned only briefly (within 0.006 pp). Non-stationarity, the main rationale for online control, is never shown to exist in these hosts. Without a measurement of whether the best action changes over the run, "online learning does not pay" is unsurprising.
4. **Tuning asymmetry.** The validated action uses 600 tuning runs per class (11.4 times the test run). Learner hyperparameters were tuned on the same tuning instances, but on a small grid (e.g. three learning rates). Is the comparison symmetric in tuning budget? irace on learner hyperparameters, with credit assignment, window size and the PM/AP p_min and β, would close this.
5. **Capture ratio.** The denominator can be near zero or negative (Table I headroom −0.026). The pooled ratio of 16% and the per-collection ratios are unstable, and the rank correlation is driven by between-collection scale differences: Spearman 0.78 overall but 0.49–0.94 within collections.
6. **Garden of forking paths.** By the authors' own admission, the batteries were sequential and earlier results shaped later hypotheses. "Registered" means an internal dated log, not a public registry. The capture and correlation analyses are exploratory, but the abstract presents the 16% and the 0.78 as findings.
7. **Generality of the abstract claims.** Claim (1) holds for one problem and one ILS. Claim (3) holds for one implementation family. The last sentence of the abstract is a general causal claim that the design does not carry.

**D. Scores and recommendation**

- Content and novelty: 56 / 100
- Relevance: 4. Originality: 2. Soundness: 4. Evaluation: 3. Significance: 2.
- Ceiling: 67
- Recommendation: borderline, leaning weak reject.

The work is careful, honest and statistically above the norm for CEC. But its central null result replicates a well-known hyper-heuristic and AOS finding, in a regime chosen so that learning has little to find.

**Three things that would move my score up by 10 points**
1. Re-centre on the hyper-heuristics literature, and add a regime where online learning should pay: heterogeneous operators whose usefulness changes over the run, measured by phase. Show whether the protocol then detects a positive learner effect. A positive control on a real problem, not a planted reward.
2. Add a formal model, a bandit with near-equivalent arms, that predicts when capture goes to zero, and test the prediction.
3. Run the standard EC baselines with standard settings and tuning parity: PM, AP, DMAB, FRRMAB, Extreme-Value AOS, and HyFlex-style Simple Random paired with different acceptance criteria. Ideally include one HyFlex domain run against the CHeSC results.

**E. How to improve the content**

*METHOD proposals*

M1. **Non-stationarity diagnostic.** Estimate, per phase of the run (budget quartiles), the best fixed action on tuning instances. Report a "switching gain": the gain of the best phase-wise static schedule over the best single static action.
- Why it matters: it separates "no learnable dynamics" from "learners fail". It is the obvious question an AOS reviewer asks.
- Effort: 3–5 days. Risk: low; either result is informative.
- Points: +4.

M2. **Sample-complexity-aware learner.** Use a learner matched to the information available: a contextual bandit with pooled features, or Q-Learning with function approximation over the four features, warm-started across instances within a run sequence (lifelong or online-across-instances).
- Why it matters: it addresses the critique that the 81-state table with 2.5 visits per cell is a strawman.
- Effort: 1 week. Risk: medium; it may still fail, which would strengthen the thesis.
- Points: +3.

M3. **Tuning-parity estimator.** Give learners the same configuration budget as the validated action (irace over learner hyperparameters, 600 runs per class). Report the gain as a function of tuning budget for both, as a curve.
- Why it matters: it removes the asymmetry objection and generalises "the price of offline knowledge".
- Effort: 1 week of compute. Risk: low.
- Points: +3.

M4. **Order-invariance test as a formal diagnostic.** Use the distribution of a learner's gain over random permutations of the action list, with a permutation test statistic. A significant order effect flags tie or initialisation artefacts. Package it in the library.
- Effort: 2 days. Points: +1.5.

*FORMAL proposals*

F1. **Capture bound for near-equivalent arms.**
- Definition: let the validated action have gap Δ over the mean arm and let per-step reward noise be σ. For any learner with regret R(T) in a K-armed stochastic bandit, capture ≤ 1 − R(T)/(TΔ_eff), where Δ_eff accounts for the ε-optimal arms.
- Proposition: with the lower bound R(T) = Ω(Σ σ²/Δ_i · log T) (Lai–Robbins), when Δ is close to σ/√T capture tends to 0.
- Test: check the prediction against the measured Δ and σ per cell. This turns the observed ~16% capture and the Spearman correlation into a prediction.
- Effort: 1 week. Points: +4.

F2. **Identification of the decision and cost effects.**
- Define the equal-CPU gain G_cpu = G_dec + G_cost, with G_cost = f(round(1000K)) − f(1000), where f is the improvement-versus-iterations curve of random choice.
- State the conditions (monotone f, K measured independently, additive noise) under which the decomposition is unbiased. Give a delta-method or bootstrap variance for K.
- Effort: 3 days. Points: +2.

F3. **Absorbing-arm lemma.**
- Lemma: zero-initialised Q-Learning with rewards in [0, ∞), max(·,0) truncation, ε-greedy and first-index tie-breaking chooses arm 1 with probability ≥ (1−ε_t) + ε_t/K at every step until some other arm receives a strictly positive reward under exploration. Its expected first-arm share is then a closed-form function of the ε schedule and of p_i = P(reward > 0 | arm i).
- Test: check the formula against the 55–73% observed share. This raises C4 from an anecdote to a result.
- Effort: 3–4 days. Points: +3.

F4. **Estimand framework.** Recast "capture" as a ratio estimand θ = E[G_learner]/E[G_static] within potential outcomes. Report Fieller intervals, or better, report the difference E[G_static − G_learner] as the primary estimand with the ratio as secondary. This also answers Q4. Points: +1.5.

*EVIDENCE proposals*

E1. **A HyFlex domain, or several.** Use the HyFlex domains (e.g. SAT, bin packing, flowshop) with Simple Random plus several acceptance criteria versus PM, AP, DMAB, FRRMAB and Q-Learning, plus a validated static low-level heuristic.
- Why it matters: it is the community benchmark and connects directly to the prior art. The Java framework means a re-implementation would be needed, or a Python port of one domain.
- Effort: 1–2 weeks. Risk: medium; learners may win on heterogeneous low-level heuristics, which would qualify the thesis but make the paper far more interesting.
- Points: +5.

E2. **A designed regime where learning must pay.** Use a synthetic host in which the best operator switches with the phase or with the instance features: a fitness-landscape mix, or a problem with heterogeneous operator families (e.g. CVRP with intra-route and inter-route operators whose value changes as the search converges). Show that the protocol attributes a positive gain to learners there.
- Why it matters: this is the positive control at the level of the problem, and it makes the null result elsewhere interpretable.
- Effort: 1 week. Risk: low. Points: +5.

E3. **A stronger ILS and a second problem for RQ1.** Use an ILS with tuned kick strength and ejection-chain moves for P||Cmax, and repeat the G/R factorial on the flowshop (IG versus ILS with an insertion kick).
- Why it matters: it tests whether "reconstruction explains the IG advantage" generalises beyond a problem where greedy insertion is near-optimal.
- Effort: 4–6 days. Risk: medium; it may weaken C2. Points: +3.

E4. **Longer horizons and real-time budgets on R||Cmax**, where headroom exists: T = 10^4–10^5 with learning-curve plots of gain against T for each learner. Effort: 3 days of compute. Points: +2.

*What to cut to make room:* the Python-loop and vectorised-implementation narrative (Section III-D and the K = 9.6/1.8/2.0 trail), most of the GA host, and the published-instance "reading and qualification" story, which can move to a supplement.

*The best version of this paper.* I would re-centre it as "When can an online operator selector beat a validated static choice? A predictive account, tested." Lead with the formal capture bound (F1) and the absorbing-arm lemma (F3). Use the protocol as the measurement tool, not as the headline. Show three regimes: (a) near-ceiling or redundant operators (P||Cmax, flowshop), where the theory predicts capture ≈ 0 and the data agree; (b) real headroom with stationary operator values (R||Cmax, `alns`), where a validated static choice wins and the learners capture a predictable fraction; (c) a designed non-stationary or heterogeneous regime, where learners do pay and the protocol detects it. The tie-rule artefact becomes a short cautionary section with a lemma and an order-permutation test. The claim about the IG's greedy reconstruction becomes a supporting ablation. Headline: "Online selection pays only when operator values differ by more than the per-decision noise and change during the run. In standard learning-guided IG hosts neither holds, so the validated static action wins." Frame it explicitly as extending Özcan et al., Alanazi & Lehre, and Karafotias et al.

**F. Answers to the authors' questions**

**Q1. Is the protocol of CEC-grade novelty?** No. Each component is known good practice: random baseline (HyFlex Simple Random), ablation (Fawcett & Hoos 2016), equal-time comparison, a tuned static baseline (Karafotias et al. 2015; DAC papers), and the oracle gap (ASlib). The delta is (i) the decision-effect/cost-effect decomposition through K, (ii) the cross-fitted oracle headroom, and (iii) combining these with equivalence margins and positive controls in one released package. That is a methodology-checklist contribution, valuable but not the headline. Formalising the decomposition (F2) would raise it.

**Q2. Is the Q-Learning lock-in a general insight?** As stated, it is closer to a bug-level observation: random argmax tie-breaking is textbook. It becomes publishable if (a) you prove the first-arm share (F3), (b) you show that a published result reproduces the artefact, and (c) you provide an order-permutation test. The order experiment is convincing for your implementation: reversing the list moves the lock-in, and putting the validated action first recovers 81–86% of its gain. It is not evidence about others' implementations.

**Q3. Is "greedy reconstruction, not the learner, explains the IG advantage" general?** It is mostly expected from the IG/ILS literature, and on P||Cmax greedy LPT-like insertion is near-optimal. It would be convincing with a strong, tuned ILS and replication on a problem where greedy construction is weak (flowshop with tardiness, or CVRP with a regret-insertion versus random-insertion factorial). It would be unconvincing if a well-tuned ILS closed the gap.

**Q4. Is the capture ratio sound?** It is fragile. The denominator is near zero or negative in several cells, the ratio has no finite-variance guarantee, and pooling mixes scales. Use the difference E[G_static − G_learner] as the primary estimand, and the ratio only where the denominator is significantly positive, with Fieller or bootstrap-ratio intervals. In bandit terms, normalised regret relative to the best fixed arm is the standard, well-defined estimand. Report it against T.

**Q5. Does "gains come from knowledge acquired offline" follow?** Only for the regime tested: near-ceiling or stationary hosts, redundant operator sets, T ≈ 10^3, and tabular learners with very few state visits. Online learning has a real chance with heterogeneous operators, non-stationary operator value, long runs, or learning across instances within a deployment. A negative result there would be highly informative and would shift my score substantially. A negative result only where headroom is near zero is not.

SCORES: content_novelty=56; relevance=4; originality=2; soundness=4; evaluation=3; significance=2; ceiling=67; recommendation=borderline; confidence=4
