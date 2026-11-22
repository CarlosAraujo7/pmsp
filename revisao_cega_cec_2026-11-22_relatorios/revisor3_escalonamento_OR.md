# Revisor 3 (escalonamento e roteamento, metaheurísticas de pesquisa operacional)

Modelo: opus. Notas: content_novelty 57, teto 70, borderline (inclinado a weak reject para o CEC como está).

---

**Reviewer 3 report: "Budget-Fair Attribution of the Gains of Learning-Guided Iterated Greedy: Where Does Online Learning Pay?"**

Lens: scheduling and routing metaheuristics and their benchmarking.

---

**A. Claimed contributions**

C1. A "budget-fair attribution protocol" for learning-guided metaheuristics. It has six parts: a random-choice control at matched iterations; equal CPU time, with the gap split into a decision effect and a cost effect; a static action validated offline as the competitor; cross-fitted oracle headroom; a power check with a planted positive control; and pre-specified tests with an equivalence margin. It is released as a library.

C2. On P||Cmax, the IG's advantage over ILS comes from its greedy reconstruction, not from the learner. The share of a local search's gain that is only budget depends on the implementation (93/30/28%).

C3. The standard Q-Learning (zero-initialised table, first-index tie rule) locks onto the action listed first. With ties broken at random, it captures about 16% of the validated action's gain, with rank correlation 0.78.

C4. Gains come from offline knowledge: a validated action or mixture, or an offline ExtraTrees selector (0.79-0.84 of the one-seed headroom). This is checked on a flowshop, a genetic algorithm, R||Cmax and the third-party `alns` CVRP.

---

**B. Novelty audit**

**C1, the protocol.** Each component is established practice:
- random-choice ablation of the selection layer;
- equal-CPU comparison (Bartz-Beielstein et al., 2020, which the paper cites);
- a tuned static configuration as the baseline (irace, López-Ibáñez et al., 2016);
- the single-best-solver vs oracle framing from algorithm selection (Rice; ASlib, Bischl et al., 2016);
- equivalence testing (Lakens, 2017).

The closest prior art is missing from the related work. Turkeš, Sörensen and Hvattum (EJOR 2021, "Meta-analysis of operator selection in adaptive large neighborhood search") pooled published ALNS studies and found that the adaptive layer adds little over random or static operator use. That is essentially the paper's thesis, shown on the routing side. Santini, Ropke and Hvattum (J. Heuristics, 2018) made a similar point about which ALNS components matter. On the parameter-control side, the literature has argued for years that a tuned static setting must be the baseline (Karafotis et al., 2015, which the paper cites).

What is new: the decomposition of the equal-CPU gain into decision and cost effects; the cross-fitted headroom as a debiased oracle; and putting all of these together with a planted positive control. These are useful refinements.

Novelty: 2.5/5. As a methodology contribution it stands only with a formal estimand (see E).

**C2, reconstruction beats the learner.** For P||Cmax with U[1,100] times and n/m from 3 to 15, it is classical that LPT-style greedy assignment is almost optimal. LPT is asymptotically optimal for such distributions (Frenk and Rinnooy Kan, 1987, Math. OR (unverified)). The authors' own arc-flow bound proves 75.3% of the instances optimal. So a component that is essentially "re-insert by LPT onto the least-loaded machine" naturally dominates a weak-perturbation ILS. That is expected, not a discovery.

The 93/30/28% "budget share depends on implementation" result is a sound caution, but it is a known general caveat of time-based comparisons.

Novelty: 2/5. It is a correct ablation but does not stand alone.

**C3, the Q-Learning first-index lock-in.** Sutton and Barto's own pseudo-code specifies random tie-breaking. That a zero-initialised Q table with non-negative, mostly zero rewards and argmax-first tie-breaking repeats action 0 is a direct consequence of the implementation.

The novel and useful part is empirical and sociological:
- the lock-in can masquerade as learning, through the cost effect, when the first-listed action is also the cheapest;
- reversing the list moves the gain (Fig. 1c);
- a published design (ref. [9]) may be exposed to this.

Novelty: 3/5 as a cautionary finding; it is not a general insight. It stands as a section, not as a paper.

**C4, offline beats online.** This is consistent with algorithm-selection and per-instance configuration results (offline selectors usually close more of the single-best-to-oracle gap than online adaptation in short runs), and with the Turkeš et al. ALNS meta-analysis.

What is new: a controlled, multi-host measurement of capture and the third-party ALNS check.

Novelty: 3/5. Its significance is bounded by the regime, as discussed below.

---

**C. Soundness audit**

**Strongest parts**
- The design is careful: paired tests; Holm correction; an equivalence margin; and cross-fitting to remove the upward bias of the one-seed oracle (the observation that the validated action can exceed the cross-fitted headroom in 41/60 cells is honest).
- The planted-reward positive control shows the learners can detect a signal when one exists.
- The order-of-actions intervention is a genuine causal manipulation. Reversing the list moves the lock-in to the new first action, at 63-72% of choices.
- The third-party `alns` check addresses the "your own code" objection.
- The R||Cmax host with real headroom (one-seed 1.5-2.8 points, cross-fitted 0.43-1.60) is the most informative experiment in the paper. There the learners still lose to the validated action at every horizon.

**Main threats (my lens)**

1. **Ceiling effect dominates the main battery.** In Table I, random choice is already at the bound in 7-96% of runs and, on average, 0.04-1.24 makespan units above it. The validated gain is 0.01-0.62 units, and the cross-fitted headroom is negative in four of the twelve classes. With integer times, one makespan unit is a median 0.38%. A margin of ±0.05% is therefore about an eighth of a unit, so most comparisons are about whether an instance lands one unit above or at the optimum.
   - In this regime no selector can matter. The claim "the learner's own contribution is nil to modest" is close to forced by the host.
   - The paper partly acknowledges this (Limitation ii), but the abstract's headline numbers ("adds at most 0.04", "beat it in 2 of 69 cells") are drawn mostly from these near-solved cells.
   - P||Cmax at these sizes is effectively solved exactly by arc-flow. A practitioner would not run an IG here at all.

2. **Weak or short-horizon hosts.**
   - The flowshop runs T=1000 iterations with cross-fitted headroom of 0.012-0.025. The Ruiz-Stützle IG at that budget is already close to the best known on Taillard, so again there is little to learn.
   - The CVRP ALNS reaches gaps of 2.0-2.5% on X instances with n ≤ 200. Hybrid genetic search (Vidal's HGS-CVRP) reaches roughly 0.2-0.5% on the same family. So the third-party host is far from state of the art. Its 16 operators are uniform random destroy/repair variants, where one strong operator plausibly dominates throughout the run, which is a favourable setting for a static choice.
   - None of the hosts has operator usefulness that changes across the run. That non-stationarity is the classical motivation for online AOS (Fialho et al.; DMAB).

3. **The learning problem is structurally hopeless as posed.**
   - With 81 states, five actions and T=1000, each Q cell is visited 2.5 times (as the paper states).
   - The reward is the improvement truncated at zero. Near the ceiling it is almost always zero, which also causes the lock-in.
   - A stateless bandit gets roughly 200 pulls per arm of a sparse, heavy-tailed reward.
   - The negative result on online learning therefore reflects sample complexity under this reward design and horizon more than a property of "online learning" in metaheuristics. The T=10^5 check is reported only on P||Cmax, which is at the ceiling, and on R||Cmax, where it is briefly mentioned with no capture numbers.

4. **The capture ratio is ill-conditioned.** It is a ratio of two noisy differences, and the denominator is near zero in many cells (the validated gain is 0.003-0.011 points in several synthetic classes). The pooled 16% hides this; the paper reports 0.06 in the synthetic cells against 0.28 in the published ones. The "rank correlation 0.78" between the learner's gain and the validated gain is partly mechanical: both share the random-choice reference arm, so its noise is shared, and both scale with between-cell heterogeneity.

5. **The comparison is asymmetric in information.**
   - The validated action uses the class label and 30 tuning instances; the online learner starts cold on every instance.
   - The offline selector's training cost is not charged. The "price" analysis does charge tuning (11.4 times the test run) and finds that it pays off only beyond about 70-130 test instances per class.
   - So the fair statement is conditional: "offline knowledge wins when the deployment amortises tuning". The abstract instead states an unconditional "gains come from knowledge acquired offline".

6. **Generality of C2.** The ILS used has one weak perturbation and operators on the bottleneck machine only. Against a stronger ILS, such as one with ejection chains or a VNS with multi-exchange, the 0.6-0.8 point gap may shrink. The paper's own R2* control (best kick size) still leaves 0.45-0.57 points, but the kick size was chosen on the test set.

7. **Pre-registration.** The hypotheses were logged internally, not in a public registry, and the batteries were sequential (Limitation iii). The "registered" labels carry limited evidential weight. That is acceptable, but it should not be emphasised.

**Abstract claims not carried by the evidence**
- The closing sentence, "Gains come from knowledge acquired offline, not from learning during the run", needs to be qualified as "in short-horizon runs on hosts with stationary operator value and low headroom".
- "No selector of the library did [beat it] in 40 comparisons" holds for one ALNS of moderate strength.

---

**D. Score block and recommendation**

- Content/novelty: 57
- Relevance: 4; originality: 2; soundness: 3; evaluation: 3; significance: 2
- Ceiling: 70
- Recommendation: borderline, leaning weak reject for CEC as is.

The three things that would move my score up by 10 points:
1. A host or regime with large, non-stationary headroom and long horizons where online learning has a real chance, with the protocol applied there. Either result would be informative.
2. A formal estimand for capture or regret with proper inference, replacing the ratio.
3. Positioning against Turkeš et al. (2021) and the algorithm-selection literature, with C1 sharpened to its true delta: the decision/cost decomposition and the cross-fitted headroom.

---

**E. How to improve the content**

Ranked by expected points per unit of effort.

**METHOD proposals**

**M1. Replace capture with a regret-gap estimand.**
- (i) For each cell, define G_L = E[f(random) − f(L)] and G_S = E[f(random) − f(S)], and report the difference Δ = G_S − G_L directly (the "offline premium"). Estimate it with cross-fitting and a hierarchical model (instances nested in cells) or a cluster bootstrap. Report capture only as a secondary Fieller-type interval, and only where G_S is bounded away from zero.
- (ii) This removes ratio instability and the shared-reference artefact in the 0.78 correlation.
- (iii) 2-3 days, re-analysis only.
- (iv) Low risk. It may show capture is undefined in half the synthetic cells, which is itself honest.
- (v) +3.

**M2. A headroom-conditioned analysis.**
- (i) Regress the learner gain on the cross-fitted headroom across all cells and hosts, with a measurement-error correction, since headroom is estimated. Report the slope as "the fraction of available headroom an online learner converts", with a confidence band.
- (ii) It turns many small near-ceiling results into one curve that answers "does online learning matter when there is room?".
- (iii) 2 days.
- (iv) The slope may be indistinguishable from zero, but that is informative.
- (v) +3.

**M3. A non-stationarity diagnostic.**
- (i) Estimate, from logged runs, how much the best action changes across run phases: split the run into deciles and compute the per-decile oracle. Define a "switching headroom": the gain of the best piecewise-constant schedule over the best constant action, cross-fitted.
- (ii) Online learning can only beat an offline static choice through switching headroom (or within-class instance heterogeneity). Measuring it tells whether the negative result is about the learners or about the hosts.
- (iii) 4-5 days.
- (iv) If the switching headroom is about zero everywhere, the thesis gets stronger but narrower.
- (v) +4.

**M4 (bonus). A decision/cost decomposition as a general identity.** Present it as an estimator usable with any learner, with variance formulas; that is the cleanest methodological delta. (+2)

**FORMAL proposals**

**F1. When a static action cannot be beaten.** Definition: a host is *stationary-dominated* if there is an action a* such that, for every search state s reachable under any policy, E[r(s, a*)] ≥ E[r(s, a)] − ε. Proposition: in an ε-stationary-dominated host, any online policy's gain over a* is at most εT, while its loss is at least the cumulative cost of exploration (Ω of the sum over a ≠ a* of log T divided by the suboptimality gap Δ_a, by Lai-Robbins). Hence the validated action weakly dominates up to εT.
(+3, 3-4 days.)

**F2. The lock-in of greedy tabular Q under first-index ties.** Proposition: with Q initialised to 0, rewards r ≥ 0, and P(r = 0 | a₀) = p₀, the greedy (ε = 0) policy with first-index tie-breaking selects a₀ until it receives its first positive reward, after which Q(·, a₀) > 0 and a₀ remains argmax. Under decaying ε, the expected share of choices on a₀ is at least 1 − Σ_t ε_t/T. That gives a closed-form prediction of the 55-73% lock-in, which you can check against Fig. 1c.
(+2, 2 days.)

**F3. Sample complexity for the state-based learner.** Bound the number of visits per (state, action) cell needed to separate actions whose mean improvement differs by δ, given the zero-inflated reward with P(r > 0) = q. A Bernstein-type argument gives roughly n ≳ (σ²/δ² + 1/(qδ)) log(1/α) per cell. Plug in the measured q, δ, and 81×5 cells, and show that T=1000 is orders of magnitude short.
- This turns the negative result into a predictable consequence and tells practitioners what T is needed.
- (+3, 3 days.)

**F4 (bonus).** A simulation model (a synthetic MDP with a tunable switching headroom and reward sparsity) in which the protocol's conclusions can be swept across regimes. (+2)

**EVIDENCE proposals**

**E1. A host with real headroom and non-stationary operator value.** Use the CVRP or VRPTW with an ALNS that is competitive (gap below 1% on X instances with n of 200-1000, or Solomon/Homberger), using long runs of 10^5 or more iterations. Alternatively use the Fanjul-Peyro and Ruiz R||Cmax at large n×m, where the gap to the best known is substantial, or flowshop total-flowtime / makespan at Taillard 200×20 and 500×20 or VRF large instances with longer budgets.
- (ii) This gives online learning a fair chance, and a negative result there would carry weight.
- (iii) 7-10 days on one workstation.
- (iv) Online learning may win, which is good for the paper: it would yield a "where it pays" map.
- (v) +5.

**E2. Within-class heterogeneity regime.** Mix instance classes randomly within a "class" (heterogeneous test sets) so that the static per-class action is weaker. Measure the online learner vs the static action vs the offline per-instance selector.
- (ii) Practitioners rarely have clean classes.
- (iii) 3 days.
- (iv) Low risk.
- (v) +3.

**E3. A stronger static and online baseline set.**
- (i) A stronger ILS (VNS with multi-exchange or ejection chains) for C2.
- (ii) DMAB/Page-Hinkley and sliding-window UCB at long horizons.
- (iii) A reward redesign that is not truncated at zero, such as rank-based credit or the extreme-value credit of Fialho et al.
- (iv) Reporting how far the best host is from the best known solutions (and from the solvers reviewed in Ostojić et al., 2025), so that the absolute quality of the hosts is stated.
- Effort 4-5 days; risk is low; +3.

**E4 (bonus).** Report the R||Cmax horizon sweep (T up to 10^5) with full capture numbers. Repeat the price analysis there (+1).

**What to cut to make room.** Remove the Python-loop / vectorised / compiled K series, keeping one sentence; the genetic-algorithm host (weak and partly ad hoc: the mixture was specified after the single action lost); and part of the published-instance P||Cmax "reading and qualification" narrative.

**The best version of this paper.** Re-centre it on a regime map: online operator selection pays only when the host has (a) switching headroom (non-stationary best operator) and (b) enough horizon for the learner's sample complexity. Measure both quantities with cross-fitted estimators on hosts that span the axis: solved P||Cmax as the low-headroom anchor; R||Cmax, large flowshops and a competitive long-run ALNS for the CVRP or VRPTW as the high-headroom hosts.

The headline claim becomes: "online learners convert at most x% of the available switching headroom per 10^k iterations, so in near-ceiling or stationary hosts a validated static choice dominates. We give the diagnostic to tell which regime you are in." The Q-Learning lock-in becomes a short cautionary section with its closed-form prediction. C2 shrinks to an ablation paragraph.

---

**F. Answers to the authors' questions**

**Q1. Is the protocol CEC-grade novel?** Mostly it is assembled good practice: random ablation, equal CPU, tuned static baseline, oracle gap from algorithm selection, equivalence tests. The delta is (a) the decision/cost decomposition of the equal-CPU gain, (b) the cross-fitted oracle headroom as a debiased reference, and (c) bundling a planted positive control into the pipeline. That delta is real but small. Without a formal estimand and a demonstration that the protocol changes a published conclusion beyond ref. [9], it is not CEC-grade on its own.

**Q2. Is the first-index lock-in general or bug-level?** The mechanism is bug-level and known: random tie-breaking is in the textbook. It becomes publishable as a cautionary result because:
- it produces apparent wins at equal CPU when the first action is cheap;
- the list-reversal intervention is convincing causal evidence (the lock-in moves to the new first action, 63-72%; listing the validated action first recovers 81-86% of its gain);
- published designs may be exposed to it.

Add a closed-form prediction (F2) and an audit of the exposure in published code. Keep it as a section, not a headline.

**Q3. How general is "greedy reconstruction explains the IG's advantage over ILS"?** For P||Cmax with uniform times it is close to expected: LPT reinsertion is near-optimal and the ILS used perturbs weakly. To convince me:
- replicate it on R||Cmax and on the flowshop (the NEH-based reconstruction vs a random reinsertion);
- use a strong ILS (multi-exchange, VNS);
- show the effect holds at larger n/m and with correlated times.

It would be unconvincing if a well-tuned ILS closes the gap at equal CPU time.

**Q4. Is the capture ratio sound?** It is unstable: the denominator is near zero in many cells, and the shared random reference induces spurious correlation. A better estimand is the offline premium Δ = G_S − G_L (or the regret relative to a cross-fitted per-instance oracle), with cluster-robust inference. A natural framework is algorithm selection's single-best-to-oracle gap closed, combined with bandit regret (stochastic or piecewise-stationary). Report capture only with Fieller intervals where G_S is clearly positive.

**Q5. Does "offline, not online" follow?** Only in the regime tested:
- near-ceiling hosts;
- short horizons (T=1000);
- stationary operator value;
- zero-truncated sparse reward;
- tabular learners with 2.5 visits per cell.

Online learning has a real chance with high headroom, non-stationary operator usefulness, long runs, heterogeneous instances without class labels, and no tuning budget: for example large VRPTW with a competitive long-run ALNS. A negative result there would be informative. A negative result in the current regime is close to forced by the host.

---

SCORES: content_novelty=57; relevance=4; originality=2; soundness=3; evaluation=3; significance=2; ceiling=70; recommendation=borderline; confidence=4
