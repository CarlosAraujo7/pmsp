# Revisor 2 (metodologia experimental e inferência estatística)

Modelo: opus. Notas: content_novelty 60, teto 73, borderline (inclinado a weak accept para o CEC pela amplitude e pela análise de poder).

---

**Review report: "Budget-Fair Attribution of the Gains of Learning-Guided Iterated Greedy: Where Does Online Learning Pay?" (CEC 2027, Reviewer 2: experimental methodology and statistical inference)**

**A. Claimed contributions**

C1. A "budget-fair attribution protocol" for learning-guided metaheuristics. It has six parts: uniform random choice at a matched budget, equal CPU time through a measured cost ratio K, a static action validated offline, cross-fitted oracle headroom, power analysis and positive controls, and statistics fixed in advance (Wilcoxon with Holm correction, bootstrap intervals, a ±0.05 pp equivalence margin). It is released as a library.
C2. In P||Cmax, the IG's advantage over ILS comes from greedy reconstruction, not from the learner. The share of a local search's gain that is only budget depends on the implementation.
C3. Online learners do not beat random choice (at most 0.04 pp) or the validated static action (beaten in 2 of 69 cells). This holds across four further hosts, including a third-party ALNS.
C4. Standard Q-Learning (zero-initialised table, ties broken by first index, non-negative reward) locks onto the action listed first. With random tie-breaking it captures about 16% of the validated gain, with rank correlation 0.78. An offline-trained selector captures 79-84% of the one-seed headroom.

**B. Novelty audit**

*C1, the protocol.* Most components are established practice.
- Random-choice baselines in hyper-heuristics: the "simple random" heuristics in HyFlex; Alanazi and Lehre, CEC 2014.
- Equal-CPU comparisons and their pitfalls: Johnson, "A theoretician's guide to the experimental analysis of algorithms", 2002; Bartz-Beielstein et al., 2020.
- Tuned static baselines against parameter control: Karafotias, Hoogendoorn and Eiben, IEEE TEVC 2015.
- Single-best versus oracle gap closure: ASlib, Bischl et al., AIJ 2016.
- Component ablation: Fawcett and Hoos, "Analysing differences between algorithm configurations through ablation", J. Heuristics 2016.
- Equivalence testing: Lakens, 2017.
- Power and sample size in algorithm comparison: Campelo and Takahashi, J. Heuristics 2019.

The closest prior art for the substantive question is Turkeš, Sörensen and Hvattum, "Meta-analysis of metaheuristics: quantifying the effect of adaptiveness in adaptive large neighborhood search", EJOR 2021. They found that ALNS's adaptive layer contributes only marginally. The paper does not cite this, and it must.

What is new: the combination, in particular the explicit split of equal-CPU gain into a decision effect and a cost effect; cross-fitting to debias the oracle; and positive controls (planted reward bonuses) that check the pipeline can detect learning. Novelty 2.5/5. As a framework it does not stand on its own, because none of its quantities is formally defined as an estimand (Section E).

*C2, reconstruction rather than learner.* Ruiz and Stützle (2007) and later IG studies already stress that greedy reinsertion is central to IG. What is new is a clean 2×3 factorial (reconstruction × descent depth) on P||Cmax with the action randomised. Novelty 2/5. It is sound but narrow: one ILS family, implemented by the authors.

*C3, offline beats online.* This is consistent with Turkeš et al. (2021) and with the long-standing parameter-control-versus-tuning debate. What is new is the breadth (five hosts, nine learners, a third-party library) and the power analysis that makes the null results informative. Novelty 3/5. It stands as a careful negative result, but only within the tested regime (see Section C).

*C4, order lock-in.* Optimistic initialisation and random tie-breaking are textbook advice (Sutton and Barto, 2018). The first-index bias of argmax is well known to practitioners. What is new is showing that this artefact can produce apparent AOS wins in a scheduling setting, with a convincing order experiment (reversed list, validated action first, permuted per run, random ties). The 16% capture and the 0.78 correlation are exploratory. Novelty 3/5 as a cautionary methodological finding. It is not a general insight unless the paper shows that a published result is affected.

**C. Soundness audit**

*Strongest parts.*
- Power is computed and planted gains are recovered (58-99.8% power at 0.02 pp). This turns "no difference" into informative bounds, which is rare in this literature.
- Positive controls on the learners.
- A two-stage confirmatory design on the published instances (discovery on 2,550, five confirmatory hypotheses on 3,950 held-out, all held).
- The headroom estimator is cross-fitted.
- A third-party host (alns 7.0.0) answers the "your code" objection.
- The R||Cmax host has real headroom (1.5-2.8 pp one-seed). There, learners still lose to the validated action in all ten learner-space pairs. This is the strongest evidence in the paper, much stronger than the near-ceiling synthetic P||Cmax.

*Serious threats.*

1. **The ceiling decides the synthetic verdicts.** Random choice reaches the bound in 7-96% of runs. Validated gains are 0.003-0.24 pp, and the cross-fitted headroom is negative in four classes. The equivalence margin of ±0.05 pp is about 40% of the oracle headroom of 0.12 pp. So "parity" on the synthetic classes is compatible with a learner capturing a large share of everything there is to capture. An absolute margin applied across hosts whose gain scales differ by two orders of magnitude (0.01 pp versus 2 pp versus CVRP gaps) is not a coherent decision rule. The margin should be relative, for example a fraction of the cross-fitted headroom of each cell, or standardised per host.

2. **The definition of "beyond the margin" is lenient and mixes estimands.** A difference counts as "beyond the margin" if it is Holm-significant (Wilcoxon) and its bootstrap 90% interval for the mean is "not inside ±0.05". The paper admits "its point estimate may be smaller". The standard superiority-by-margin criterion is a lower bound above +0.05. Under the current rule, a gain of 0.02 with interval [0.004, 0.06] counts as beyond. Also, Wilcoxon tests a pseudo-median shift while the interval and the margin concern the mean. The test and the interval target different estimands. Parity is defined only for Holm-significant differences inside the margin, which leaves non-significant differences inside the margin unclassified. Use TOST on the mean throughout, or a single paired-bootstrap or t-based framework.

3. **The capture ratio is an ill-conditioned ratio estimator.** The denominator (the validated gain) is near zero in many cells and negative in some. "Pooled capture 16%" does not say whether it is a ratio of sums or a mean of ratios. These differ greatly when denominators vary from 0.003 to 2 pp. No interval accounts for denominator uncertainty (Fieller or delta method). Table V uses a different denominator (the one-seed oracle), so "capture" means two different things in the paper.

4. **Mathematical coupling and pseudo-replication inflate the 0.78 correlation.** Both the learner's gain and the validated action's gain are differences from the same random-choice arm on the same instances and seeds. The noise in that shared arm enters both with the same sign, which induces positive covariance. The 69 cells are also not independent: the same 12 synthetic classes appear in two action spaces, and the 7 benchmark types in three. Cells also pool collections whose gain scales differ by orders of magnitude, so a correlation across collections is largely driven by scale. The class-level interval (0.57-0.89) and the within-collection values (0.49-0.94) partly address this, but the coupling remains. Counts such as "25 of 69 cells" double-count units.

5. **The headroom concept leaves out the thing online learning is for.** Every oracle in the paper is static per instance (the best fixed action). An online learner's distinctive potential is adaptation within the run: changing the action as the search moves from descent to stagnation. The paper never estimates a dynamic oracle, for example the best two-phase or k-phase schedule, or a rollout policy. Without it, "there is nothing to learn beyond what a static action gives" is assumed, not shown. This is the main gap behind the claim "Gains come from knowledge acquired offline".

6. **The learning signal is weak by design.** With 81 states and T = 1000, each Q-table cell is visited about 2.5 times. Reward differences between actions are of order 0.01 pp per improvement, against substantial noise. A simple bandit sample-complexity argument says per-step differences of this size cannot be identified within T = 1000 for K ≥ 5. The negative result for stateful learners is therefore close to guaranteed, not a discovery. The T = 10^5 check is run on the near-ceiling host only.

7. **Causal language.** "The IG's advantage comes from its greedy reconstruction" is identified only through the G-versus-R contrasts in one host, and the full IG-versus-ILS comparison also differs in acceptance, restart and perturbation (the authors admit the ILS perturbs weakly). The decision/cost decomposition is order-dependent: it measures the decision effect at equal iterations first and the cost effect as the remainder. It also assumes performance is roughly linear in iterations between T and round(1000K). Calling the equal-CPU gain from cheap actions "not learning" is a framing choice. Spending budget on cheap actions is a legitimate policy. Here it is an artefact of list order, which is the paper's point, but the general rule needs care.

8. **Pre-registration and multiplicity.** "Registered" means a hash-stamped internal log, and the batteries were sequential (acknowledged). Holm correction is applied "over each family", and the families are defined after the fact. With dozens of batteries, the family-wise error control across the paper is unclear.

*Abstract claims not carried by the evidence.*
- "Gains come from knowledge acquired offline, not from learning during the run." This holds for the hosts, horizons and learners tested, in largely near-ceiling regimes. In the CVRP, online selectors do capture 23-61% where any selector gains, which is not nil.
- "The IG's advantage … comes from its greedy reconstruction." This is shown against one authors' ILS family in one host.
- "In step with it (rank correlation 0.78)." This is exploratory and inflated by coupling.

**D. Score block and recommendation**

- Content and novelty: 60
- Relevance 4/5; originality 3/5; soundness 3/5; evaluation 4/5; significance 3/5
- Ceiling: 73
- Recommendation: borderline (leaning weak accept for CEC thanks to the breadth and power analysis)
- Confidence: 4/5

The three things that would move my score up by 10 points:
- (i) Formal estimands with a relative equivalence margin and a coherent single inferential framework (E-F1 plus E-M2).
- (ii) A dynamic-oracle headroom estimate, with a host where it is large, to test whether online adaptivity has anything to exploit (E-M1 plus E-E1).
- (iii) Reproduce one published Q-Learning AOS result ([9] or [36]) and show whether lock-in explains it (E-E3).

**E. How to improve the content**

*Methods.*

M1. **A dynamic-oracle headroom estimator.**
- (i) Estimate the best k-phase static schedule per instance (switch points on a grid; k = 2 or 3; one seed set for selection and fresh seeds for evaluation, as in the existing cross-fitting). Report dynamic minus static headroom.
- (ii) It separates "nothing to adapt to" from "learners fail to adapt". Without it, the central claim is untestable.
- (iii) 4-6 days; the compiled code makes this feasible (K²·grid runs per instance on a subset).
- (iv) Risk: the dynamic headroom may be about zero. That would still strengthen the thesis considerably.
- (v) +5.

M2. **A hierarchical estimator for gain and capture, with proper intervals.**
- (i) A mixed model or hierarchical bootstrap (instances within classes within collections). Capture is defined as a ratio of expectations with Fieller or delta-method intervals. Seeds are split so that the random-arm replicates used for the numerator and the denominator are independent, which removes the coupling.
- (ii) It fixes threats 3 and 4 directly.
- (iii) 3 days of reanalysis plus some reruns of the random arm.
- (iv) Risk: the correlation may drop noticeably, weakening C4's "in step" claim.
- (v) +3.

M3. **An order-invariance test as a general diagnostic.**
- (i) For any selector, run R random action orderings × S seeds and test whether the order component of the variance exceeds zero (a random-effects ANOVA or permutation test). Package it in the library.
- (ii) It turns the Q-Learning anecdote into a reusable tool reviewers can demand of any AOS paper.
- (iii) 2 days.
- (iv) Low risk.
- (v) +2.

M4 (optional). **Anytime comparison** (area under the makespan-versus-CPU-time curve) instead of a single equal-CPU point. This removes the linearity assumption behind K. +1.

*Formal content.*

F1. **Potential-outcome estimands and an oracle sandwich.** Define Y_i(π, b) as the makespan of instance i under policy π at budget b, with seeds as the randomisation. Then:
- τ_L = E_i E_ω[Y_i(U, b) − Y_i(π_L, b)];
- θ* = E_i[max_a E_ω(Y_i(U) − Y_i(a))];
- capture = τ_L / τ_S as a ratio of expectations.

Proposition: under exchangeable seeds, E[cross-fitted headroom] ≤ θ* ≤ E[one-seed headroom], with the upper gap equal to the winner's-curse bias, bounded by a function of σ·sqrt(2 log K / s) (an extreme-value bound on the maximum of K noisy means; s is the number of seeds). This makes the existing upper and lower bounds into theorems and gives a principled equivalence margin, for example δ = 0.1·θ̂*. +3.

F2. **Shapley attribution of improvements.** Treat the factors {reconstruction, descent, acceptance and restart, selection policy, budget (iterations-to-time)} as players. The value function is the mean gain of the configuration against the ILS baseline, on a full 2^k factorial. Exact Shapley values replace the order-dependent decision/cost decomposition and the one-factor ablations. For the two-player case (policy, budget), the Shapley decision effect is ½[(G(π, T) − G(U, T)) + (G(π, T′) − G(U, T′))]. Compare with Fawcett and Hoos (2016) and with Fréchette et al., "Using the Shapley value to analyze algorithm portfolios", AAAI 2016. +3.

F3. **A learnability bound explaining the negative result.** Model the per-step reward of action a as having mean μ_a and variance σ_a² (with many zeros). Proposition: any selector with T pulls over K actions suffers regret against the best static action of Ω(sqrt(KT)·σ) in the minimax sense. Hence the expected gain over uniform choice is bounded above by (Δ_max − c·σ·sqrt(K/T))⁺, where Δ_max is the largest gap between action means. Plug in the empirical Δ and σ per class to predict where online learning cannot pay. Also give a lock-in lemma: with zero initialisation, non-negative rewards that are zero with probability ≥ p in stagnation, and first-index ties, the greedy action converges to the lowest index among actions whose Q-values stay tied at zero, with a bound on the probability. A piecewise-stationary extension, with dynamic regret against the schedule oracle of M1, gives the regime where online learning can win. +4 (the most valuable formal addition).

*Evidence.*

E1. **A regime designed to favour online learning.**
- (i) Hosts where action value is non-stationary or instances are heterogeneous within a "class": mixed-generator pools without class labels, long horizons on large instances (n ≥ 500), or dynamic machine availability. Pre-register the prediction from F3.
- (ii) A negative result there would be genuinely informative. A positive one would define where online learning pays, which is the title question.
- (iii) 5-8 days.
- (iv) Risk: online learning wins. That refines the thesis rather than undercutting it.
- (v) +5.

E2. **Online learning across instances versus offline training on the same data.** Let the learner keep its table or weights across a stream of the training instances, then deploy. This equalises information between the online and offline arms. Currently, offline methods see 5,720 instances while online learners start cold on each instance, which confounds "online versus offline" with "data versus no data". 3-4 days. +3.

E3. **Replicate a published Q-Learning AOS claim.** Reproduce [9] (Q-Learning in IG for PFSP, on the Taillard instances, with their operators and rewards as far as described). Run its original order, reversed order and random ties. This is the only way to support the implication that published wins may be order artefacts. 4-6 days. Risk: the published method may break ties differently, in which case report that. +4.

E4. **A HyFlex cross-domain check** with existing selection hyper-heuristics against the validated static heuristic. 3 days. +2.

*What to cut to make room in 8 pages:* the irace and FRRMAB paragraph, some of the "repairing the online learner" variants, the three-strategy rows of Table IV, and part of the synthetic-class table (keep the ceiling statistics).

*The best version of this paper.* Re-centre it on one formally defined question: what fraction of the attainable improvement, static and dynamic, does online selection capture, at matched budget, once structure and tuning are accounted for? Define the estimands (F1), decompose the IG-versus-ILS and learner gains with Shapley values (F2), and give a learnability bound (F3) that predicts where online learning cannot pay. Then test the prediction on near-ceiling hosts (existing), a high-headroom host (R||Cmax, existing), and a host built to make online adaptation pay (E1). The order lock-in becomes a diagnostic section with a reusable test (M3) and one published replication (E3). The headline would be: "online learners capture what their signal-to-noise permits; in typical AOS settings this is a small fraction of what offline validation captures, and apparent wins can be ordering artefacts". That is a falsifiable, quantified claim rather than a collection of null results.

**F. Answers to the authors' questions**

Q1. As defined, it is mostly a principled bundle of known good practice. The real delta is three things: the decision/cost decomposition at equal CPU, cross-fitted oracle bounds, and planted-effect positive controls for learners. To be CEC-grade novel, it needs formal estimands and an attribution theory (F1, F2), and it must engage with Turkeš et al. (EJOR 2021) and Fawcett and Hoos (2016).

Q2. The mechanism (zero initialisation, non-negative reward, first-index argmax) is textbook-adjacent, almost bug-level. Its consequence for published AOS claims is publishable if demonstrated on a published method (E3) and turned into a test (M3). The order experiment on R||Cmax is convincing: the lock-in moves with the list, putting the validated action first recovers 81-86% of its gain, and random ties remove it. Note that "permuted once per run is not order-invariant" is correct and well handled.

Q3. It holds for P||Cmax against the authors' ILS family, where it is well identified by the G-versus-R contrasts at matched descent (+1.73 pp). Generality is unknown. What would make it convincing: the same factorial on the flowshop and on R||Cmax, against a strong published ILS (with its own perturbation and acceptance), with Shapley values over all differing components. What would make it unconvincing: the gap shrinking once the ILS uses a structured perturbation or the acceptance rules are equalised.

Q4. It is sound in spirit but ill-conditioned as used. It is a ratio of noisy differences with near-zero or negative denominators, it is pooled ambiguously, it has two different denominators across the paper, and it is coupled through the shared random arm. A better estimand is the ratio of expectations τ_L/θ*, with θ* bracketed by the cross-fitted and one-seed estimates (F1), estimated hierarchically with Fieller intervals. Alternatively, report the absolute regret against the validated action with a relative margin. The natural framework is potential outcomes, with seeds as randomisation and instances as the population.

Q5. It holds only in the tested regime: near-ceiling synthetic classes, T = 1000, tabular learners whose state cells are visited about 2.5 times, per-class tuning data available, and only static oracles measured. Online learning has a real chance where action values are non-stationary within the run, classes are heterogeneous and unlabelled, horizons are long relative to K, and tuning data are scarce or shifted. A negative result there, with a dynamic oracle showing headroom that learners fail to capture, would be highly informative. A negative result where the dynamic headroom is zero would show the question is moot there.

SCORES: content_novelty=60; relevance=4; originality=3; soundness=3; evaluation=4; significance=3; ceiling=73; recommendation=borderline; confidence=4
