# Revisor 4 (teoria de aprendizado online, bandidos e aprendizado por reforço, seleção de algoritmos)

Modelo: opus. Notas: content_novelty 60, teto 74, borderline (inclinado a weak accept para a trilha de metodologia).

---

**Reviewer 4: theory of online learning, bandits and RL, algorithm selection, sample complexity**

**A. Claimed contributions**

C1. A "budget-fair attribution protocol" for learning-guided metaheuristics. It has six parts: a uniform-random control at the same iterations; equal CPU time via a measured cost ratio K, which splits a learner's gain into a decision effect and a cost effect; a static action validated offline as the competitor; cross-fitted oracle headroom; power analysis with a planted positive control; and pre-specified equivalence testing. It is released as a library.
C2. In IG for P||Cmax, the greedy reconstruction (not the learner) explains the advantage over a classical ILS (Table III). How much of a local search's gain is only budget depends on the implementation (93%, 30%, 28%).
C3. Across 69 cells and several hosts (flowshop, GA, R||Cmax, a third-party ALNS for the CVRP), online learners add at most about 0.04 pp over random choice. The validated static action is beaten in only 2 of 69 cells.
C4. Standard Q-Learning (zero-initialised table, first-index argmax, non-negative reward) locks in on the first listed action. With ties broken at random it captures about 16% of the validated gain (rank correlation 0.78). An offline instance-conditioned selector captures 79-84% of the one-seed headroom, mostly through class-level knowledge.

**B. Novelty audit**

**C1 (protocol). Novelty 2/5.** Each part is established practice:
- Random-selection baselines are standard in hyper-heuristics (HyFlex, Ochoa et al., EvoCOP 2012) and AOS (Fialho et al., AMAI 2010; Da Costa et al., GECCO 2008).
- Equal-CPU comparison and the "best fixed configuration" competitor come from benchmarking guidance (Bartz-Beielstein et al., 2020) and algorithm configuration (irace, López-Ibáñez et al., 2016).
- The single-best and virtual-best-solver gap is AS-lib's standard metric (Bischl et al., AIJ 2016; Kerschke et al., ECJ 2019).
- Equivalence testing follows Lakens (2017).

The genuine delta is small but useful. It has three parts:
- (i) The explicit split of an equal-CPU gain into a decision effect and a cost effect via K. This is the most original element: it shows that "wins at equal CPU" can come purely from a learner drifting toward cheap actions.
- (ii) Cross-fitting the oracle so that the max-over-arms winner's curse is removed. This is a standard debiasing idea, applied correctly.
- (iii) Bundling the parts as a mandatory checklist.

It is a methodological consolidation, not a new method. It stands as a supporting contribution, not a headline one.

**C2 (structure, not policy). Novelty 2.5/5.** That IG's strength lies in greedy reinsertion is folklore since Ruiz and Stützle (EJOR 2007). What is new is a controlled ablation on P||Cmax (Table III: G0 vs R0 +1.73 pp, which is more than 40 times the learner's 0.039). It is solid but host-specific.

**C3 (online learning adds about nothing over uniform choice). Novelty 3/5.** "Simple random or choice-function selection is competitive" appears repeatedly in the HyFlex and selection hyper-heuristic literature. Runtime analyses show that simple random mixing is often near-optimal for selection hyper-heuristics (Alanazi and Lehre, CEC 2014, cited; Lissovoi, Oliveto and Warwicker, AAAI 2019 / ECJ 2020). The new parts are the scale and the controls: 69 cells, 9 learners, a third-party ALNS host, power checks and a planted positive control. A well-powered, well-controlled null is valuable.

**C4 (Q-Learning lock-in; offline beats online). Novelty 2.5/5.**
- Lock-in under zero initialisation, greedy argmax with deterministic tie-breaking, and non-negative rewards is well understood in RL. Sutton and Barto (2018) explicitly specify random tie-breaking, and optimistic versus pessimistic initialisation is textbook. The empirical relevance comes from showing that published learning-guided IGs may have this property (the authors could not verify [9]) and that the "order of actions" experiment moves the gain (Fig. 1c: listing the validated action first recovers 81-86%). That moves it beyond a bug report into a reproducibility warning.
- "Offline per-class knowledge beats online learning" is the standard per-instance algorithm-selection finding (Rice 1976; AS-lib). The control showing that a per-class lookup table matches the ExtraTrees selector is honest and useful.

**C. Soundness audit**

**Strengths.**
- The experimental hygiene is unusually strong: paired designs, Holm correction, bootstrap intervals, equivalence margins, minimum detectable effect at 80% power of 0.006-0.014 pp, and a planted reward bonus showing that the learners *can* pick up a signal (1.5-13.7 times the uniform rate).
- The cost-effect decomposition genuinely explains the equal-CPU "wins" (decision effect -0.003/-0.004 pp, cost effect +0.062/+0.053 pp).
- The third-party ALNS host answers the "it's your code" objection.
- The random-tie re-run and the permutation-per-run control show that the authors corrected their own earlier interpretation.

**Threats and gaps.**

1. **Ceiling and low headroom confound the headline.** On the synthetic P||Cmax classes the cross-fitted headroom is 0.159 pp at best and negative in 4 of 12 classes (Table I). The certified bound is reached in 75% of instances, and one makespan unit is about 0.38%. A learner cannot gain what isn't there. The only host with real headroom, R||Cmax (cross-fitted 0.43 and 1.60), is where online learners *do* gain something (optimistic Q-Learning +0.17, ALNS roulette +0.15, random-tie Q-Learning in all six published cells, up to 25%). The general claim "gains come from knowledge acquired offline, not from learning during the run" is therefore really a claim about low-headroom, short-horizon regimes. The abstract should say so.

2. **There is no model of why learning fails, so the null result is not predictive.** At T=1000, with K=5-30 arms, about 200 pulls per arm (2.5 visits per state-action cell for the 81-state Q-table), sparse non-negative truncated rewards and between-arm gaps of order 0.01-0.1 pp, a bandit learner is statistically expected to lose most of the best arm's advantage to exploration. Best-arm identification needs roughly σ²/Δ²·log(K/δ) pulls per arm (Even-Dar, Mannor and Mansour, JMLR 2006; Audibert and Bubeck, COLT 2010; Mannor and Tsitsiklis, JMLR 2004 for lower bounds). The paper never estimates the per-arm reward means, variances or nonzero rates from its own logs. So we cannot tell whether a 16% capture is near the information-theoretic ceiling (learning is impossible here) or far below it (these learners are poor). This is the central gap from my perspective.

3. **The reward does not match the objective, and rewards are non-stationary.** The reward is the per-iteration improvement truncated at zero, but the estimand is the final best makespan. Improvement rewards decay as the search converges, so arms that are pulled early collect most of the reward. That is a non-stationary, rested-bandit setting in which UCB1 and fixed-ε Q-Learning are mis-specified (Garivier and Moulines, ALT 2011; Besbes, Gur and Zeevi, NeurIPS 2014). It also mixes "arm quality" with "time of pull". Discounted Thompson sampling and DMAB are included, but there is no analysis of whether the arm ranking itself is stable over the run. If the best arm changes during the run, the "validated static action" is not the right comparator either.

4. **"Capture" is a ratio of two noisy differences.** When the denominator is near zero (most synthetic cells), the ratio is unstable. The pooled 16% and the rank correlation of 0.78 partly reflect a scale effect: cells with larger validated gains offer larger absolute gains to everyone. A regression through the origin with a cell-level random effect, or a Fieller interval, is needed. The correlation across 69 cells mixes collections with different scales. The within-collection intervals (0.49-0.94) are more honest and much wider.

5. **The equal-CPU arm depends on K measured on one laptop, sequentially.** The decomposition is sound in principle, but it is implementation-specific (the authors acknowledge this).

6. **Forking paths.** The batteries were sequential and the capture and correlation analyses are exploratory. The registration is an internal log. For RQ3 this matters: the "16%" number appeared after the order artefact was found.

7. **Learner coverage.** Q-Learning with γ=0.9 over 81 states and 1000 steps is an odd choice: the transition structure is weakly controllable, so it is effectively a noisy contextual bandit with too many contexts. Missing from the comparison: algorithms suited to heavy-tailed or sparse rewards (median-of-means or robust UCB, Bubeck, Cesa-Bianchi and Lugosi, IEEE TIT 2013), explicit best-arm identification followed by commitment (explore-then-commit), and successive halving across the run. These are exactly the learners theory would recommend at T≈10³.

**Claims in the abstract the evidence does not carry.**
- "Gains come from knowledge acquired offline, not from learning during the run." This holds only in the tested regime. R||Cmax and the ALNS (23-61% capture) are counter-examples to its unqualified form.
- "A detail of the standard Q-Learning, not learning, can produce apparent wins" is shown for this implementation. Whether it explains published wins is speculation, since [9] could not be checked.

**D. Scores and recommendation**

- Content and novelty: 60
- Relevance 4; originality 2; soundness 4; evaluation 4; significance 3
- Ceiling: 74
- Recommendation: borderline (leaning weak accept for the methodology track)

The three things that would raise my score by 10 points:
1. A calibrated bandit model, fitted to the logged per-arm reward distributions, that *predicts* the observed capture across cells, together with a lower bound showing that capture is limited at T≈10³ for these gaps and noise levels.
2. A headroom-by-horizon phase diagram (simulated and real) showing where online learning starts to pay.
3. Restating the headline as a conditional, learnability-based claim, with the capture estimand formalised.

**E. How to improve the content (ranked by gain per unit of effort)**

**New FORMAL content**

**F1. A bandit model of operator selection and a capture bound. Effort 4-6 days; risk low; +5.**
Model each cell as a K-armed stochastic bandit over T steps. Arm a gives a reward X_a that is zero with probability 1-p_a and otherwise drawn from a heavy-tailed positive distribution with mean μ_a, possibly decaying with t. The terminal gain of a policy is approximately proportional to Σ_t μ_{A_t}.

Candidate proposition: for any policy, E[capture] ≤ 1 − c·min(1, K·σ²/(T·Δ_min²)), where Δ_min is the gap between the best arm and uniform. The proof would use the standard change-of-measure argument (Lattimore and Szepesvári, *Bandit Algorithms*, 2020, chapters 15-16). Consequently, capture tends to 0 when T·Δ²/σ² ≪ K. Plug in the logged estimates: with p small, σ²/Δ² will likely be in the thousands, which predicts a capture of about 0.1-0.2. This turns the empirical null into a predicted one.

**F2. Formal definition of the estimand and an identification result. Effort 2 days; risk low; +3.**
Define:
- the uniform-baseline value V(U);
- the best-static value V* = max_a V(a), estimated by cross-fitting;
- the policy-regret capture κ(π) = (V(π) − V(U)) / (V* − V(U)).

Show that the cross-fitted estimate of V* is a lower bound on V* (selection on independent seeds gives E[V(â)] ≤ V*). Show that the naive one-seed estimate is an upper bound, so the true κ is bracketed (see Q4). State the decomposition V_CPU(π) − V_CPU(U) = decision effect + cost effect as an identity under a linear time-to-quality assumption, and say when it fails (non-linear anytime curves).

**F3. A formal statement of the zero-initialisation lock-in. Effort 1-2 days; risk very low; +2.**
Proposition: with Q₀ ≡ 0, rewards r ≥ 0, first-index argmax, and ε_t → 0 geometrically with Σε_t < ∞ after some time, the probability that the greedy action equals action 1 forever is at least ∏(1−ε_t(K−1)/K)·P(r₁>0 on its first greedy pulls). In words, the policy is absorbed in the first action whose estimate becomes positive, with probability bounded away from zero, independent of the arm means. With random tie-breaking, the absorption probability is uniform over arms with positive first rewards, which predicts capture ≈ (fraction of arms with nonzero success rate weighted by μ). That turns Q2 from a bug note into a characterised failure mode with a testable prediction (Fig. 1c percentages).

**F4 (optional). A no-free-lunch-style remark. +1.**
Within a class, if the arm ranking varies across instances with no observable context, no online learner can exceed the per-class static oracle by more than the within-run information about the instance. Formalise this as the gap between the per-instance oracle and the per-class oracle, bounded by the mutual information between the early-run reward trace and the instance's best arm.

**New METHOD**

**M1. A "learnability diagnostic" per cell. Effort 2-3 days; risk low; +4.**
From the logs of the uniform control, estimate per-arm success rates, mean improvement and variance, and the time profile. Compute the effective signal-to-noise index S = T·Δ²/(K·σ²) and the predicted capture under F1. Report it in every cell and plot observed capture against S. If the points line up, you have a predictive tool that practitioners can run *before* adding a learner. This is a real method contribution and the natural complement to the protocol.

**M2. A calibrated bandit simulator ("digital twin"). Effort 3-4 days; risk medium (the model could fail on non-stationarity, but that failure is itself informative); +4.**
Resample logged reward traces per arm and time window, then run all nine learners on the simulator. Compare simulated and real capture per cell. Agreement would mean the metaheuristic context adds nothing beyond a bandit. Disagreement would quantify state effects and non-stationarity.

**M3. A capture estimator with valid inference. Effort 2 days; risk low; +2.**
Replace the pooled ratio with a hierarchical estimator: the learner gain regressed on the cross-fitted validated gain, with cell random effects, and Fieller or bootstrap intervals for the slope. Report the slope (the capture) separately for each collection.

**M4. Theory-recommended learners as positive controls. Effort 2-3 days; +2.**
Add explore-then-commit or successive halving (spend m pulls per arm, then commit), robust UCB with median-of-means, and sliding-window UCB. If even explore-then-commit with the oracle-tuned m does not beat uniform choice at T=1000, the null is much stronger. If it does, the paper gets a constructive recommendation.

**New EVIDENCE**

**E1. A headroom-by-horizon phase diagram. Effort 4-5 days; risk medium; +4.**
On R||Cmax (the only host with real headroom), vary T over {10³, 10⁴, 10⁵} with a fixed CPU budget per iteration, and vary headroom by changing the time-correlation structure. Map capture over (T, headroom). Theory predicts that capture rises with T·Δ². Finding the boundary answers Q5 directly.

**E2. Planted-structure simulations inside the real host. Effort 2 days; +2.**
Add synthetic per-arm reward shifts of known size Δ, with optional instance dependence, to the real IG, and measure capture as a function of Δ. This extends the existing positive control into a calibration curve.

**E3. Non-stationarity measurement. Effort 1-2 days; +2.**
Estimate the arm ranking in time windows of the run (early, middle, late). If the best arm changes, test a time-scheduled static policy (best arm per phase, validated offline). This is the proper static competitor in a non-stationary setting and may beat the single action.

**E4. A host built so that state matters. Effort 4 days; risk high; +2.**
Design an IG variant whose best destruction size provably depends on the stagnation state (for example, large kicks are useful only after stagnation). This tests whether the 81-state learners can exploit context at all. A negative result there would be informative.

**What would have to go to make room:** compress Table IV into the text, drop the irace and FRRMAB paragraphs to one sentence, and move the GA and flowshop details to a supplement.

**The best version of this paper.** Re-centre it on one question: *when is online operator selection learnable?* Keep the protocol, but as the measurement instrument rather than the headline. Present a bandit model calibrated from the logs that predicts the observed captures, plus a lower bound showing that at T≈10³ with gaps of 0.01-0.1 pp and sparse rewards no learner can capture much of the best static arm's gain. Then show the empirical phase diagram on R||Cmax and the ALNS, where learning starts to pay. Keep the Q-Learning lock-in as a formal proposition with the order experiment as its verification. The headline would read: "Online learning in IG/ALNS is information-limited: its gain is predicted by T·Δ²/σ², which is small in typical benchmarks, so offline knowledge dominates unless the horizon or the headroom is large." Cut most of the per-cell number lists and the ILS-implementation side story (keep one line on the decision versus cost decomposition).

**F. Answers to the authors' questions**

**Q1.** Mostly known good practice. The real delta is (a) the decision-effect versus cost-effect decomposition at equal CPU time and (b) the cross-fitted oracle as a conservative headroom estimate, combined as a mandatory checklist. That is useful but not CEC-grade novelty on its own. A predictive learnability diagnostic (M1) would make it one.

**Q2.** As stated, it is a known RL pitfall: deterministic argmax with zero initialisation and non-negative rewards. It becomes publishable because it is shown to produce *apparent* wins in exactly the kind of design used in recent learning-guided IG papers, and because of the order experiment. The order experiment is convincing (81-86% recovered by reordering). Formalise the absorption result (F3) and, ideally, re-run one published design to show that its reported win is order-dependent.

**Q3.** It is convincing for IG on P||Cmax: Table III is a clean component ablation. Its generality is unknown. The ILS baseline perturbs weakly, so "IG beats ILS" partly measures perturbation strength. It would be convincing if replicated on the flowshop (NEH reconstruction versus an ILS with tuned kicks) and if the best-kick ILS were compared at equal CPU time in every class. It would be unconvincing if a stronger ILS (for example, an ejection-chain or larger-kick variant) closed the gap.

**Q4.** It is sound in spirit but ill-conditioned when the denominator is near zero, and the denominator is itself an estimate (lower-biased by cross-fitting, upper-biased in the one-seed version). A better estimand is the policy-regret ratio κ(π) = (V(π)−V(U))/(V*−V(U)), estimated by a hierarchical slope with valid intervals. The natural framework is stochastic bandits with regret against the best fixed arm (or against the best context-to-arm map for contextual policies), and its regret bounds give a predicted κ.

**Q5.** It holds only in the regime tested: short horizon, low headroom, near-ceiling instances, small gaps relative to reward noise. R||Cmax and the ALNS already show non-trivial online capture (25-61%). Online learning has a real chance with long horizons (T ≥ 10⁴-10⁵ effective decisions), large and stable arm gaps, non-stationarity that a static choice cannot track, or instance-dependent best arms without offline training data. A negative result there would be highly informative, because theory would predict success.

SCORES: content_novelty=60; relevance=4; originality=2; soundness=4; evaluation=4; significance=3; ceiling=74; recommendation=borderline; confidence=4
