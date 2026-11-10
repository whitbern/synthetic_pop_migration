# Maximum-entropy design (methods draft)

Draft text for the Method section. Code: `src/synthpop/maxent.py` (the
solver) and `src/synthpop/design_maxent.py` (the design on the Sprint 2
inputs). Verification: `tests/test_maxent.py`, `tests/test_design_maxent.py`.

## Why this design is in the study

The planted attribute (education) appears in no control margin. A design can
learn about it only from the biased sample, and nothing in the inputs says how
education relates to origin, sex and age in the flow: the joint is not
identified. Every design therefore makes an assumption about that joint,
usually without saying so.

Maximum entropy makes the assumption explicit. Among all tables that meet the
constraints it takes the one that adds no structure beyond them (Jaynes 1957;
Wu et al. 2018). An attribute that no constraint links to a control comes out
independent of it. So its behaviour on education is known before it runs:
among adults, synthetic education is independent of origin, sex and age band
(pooled), or of sex and age band given origin (separate). That makes it the
baseline for the uncontrolled-attribute test in Sprint 4. If IPF or the
hierarchical design recovers an association between education and the
controls that maximum entropy cannot, the difference has a mechanism (what
the seed carries), not just a number.

## Specification

**Table.** Attributes origin *o* (5 levels), sex *s* (2), age band *a* (8:
0–4, 5–17, 18–24, 25–34, 35–44, 45–54, 55–64, 65+) and EDATTAIN *e* (5,
including NIU). The synthetic population is a nonnegative table
*p*(*o*, *s*, *a*, *e*) on a support *S* of admissible cells.

**Support.** Structural zeros from the EDATTAIN coding alone: NIU (*e* = 0)
only in the 0–4 band, since the ACS leaves `SCHL` missing only under age 3;
and nothing above "less than primary" in the 0–4 band. Nothing else is ruled
out.

**Education strata.** The age bands are coarsened to three education
universes, *g*(*a*) ∈ {under 5, 5–17, 18+}. Education varies with being a
child for reasons that are definitional, not substantive; the strata keep
the design from giving toddlers degrees while leaving adult education free of
any assumed age pattern.

**Constraints.** All linear equalities on *p*:

1. Each control margin, exactly: Σ *p* over the cells of each origin, each
   origin × sex, and each origin × age band equals the manufactured margin.
2. Education within strata. Let *N*(*g*) be the number of people in stratum
   *g* implied by the control margins, and *q*(*e* | *g*) the share of sample
   members in *g* with education *e*. Then Σ_{*o*,*s*,*a*: *g*(*a*)=*g*}
   *p*(*o*, *s*, *a*, *e*) = *N*(*g*) *q*(*e* | *g*).
   - **Pooled:** *q* is taken over all origins.
   - **Separate:** *q*(*e* | *o*, *g*) and *N*(*o*, *g*) are taken within each
     origin. Where an origin has no sample member in a stratum, the pooled
     share is used and the cell is reported.
   - A stratum with no sample member at all is left out of constraint 2, so
     education there is at maximum entropy over its admissible levels. This
     is reported too. (In practice only the under-5 stratum can be empty,
     because the sample is tilted against minors.)

**Objective.** Maximise the entropy H(*p*) = −Σ *p* log *p* (of *p* normalised)
subject to 1 and 2 on *S*. The solution is log-linear,

  log *p*(*x*) = Σ_k λ_k[*G_k*(*x*)] for *x* ∈ *S*,

with one parameter per group *G_k*(*x*) of each constraint *k*. For the
pooled design this factorises as *p*(*o*, *s*, *a*, *e*) = *p̂*(*o*, *s*, *a*)
· *q*(*e* | *g*(*a*)), where *p̂* is the maximum-entropy fit to the controls
alone (which equals IPF from a uniform seed).

**Algorithm.** Cyclic proportional scaling (generalised IPF; Darroch &
Ratcliff 1972) from the uniform table on *S*: for each constraint in turn,
multiply every cell of each group by target / current group total. The
sequence converges to the I-projection of the uniform distribution onto the
constraint set, which is the maximum-entropy table (Csiszár 1975).

**Stopping rule.** Stop when max over constraints and groups of
|group total − target| / total ≤ 10⁻¹⁰, or fail after 5,000 cycles
(`MaxEntConvergenceError`; the table is never returned as if converged).
Before fitting, the solver rejects constraints whose totals differ, two
constraints that fix the same block of cells to different totals, and
positive targets on groups with no admissible cell. With these constraints
the fit converges in one cycle, because the education constraint nests
within the age-band margin.

**Individuals.** A synthetic population of size *N* (the margin total) is
drawn as *N* independent draws from *p* / *N* (multinomial), seeded from the
run seed, the SES strength and the variant. Wu et al. likewise sample
individuals from the fitted distribution. The control margins then hold in
expectation, not exactly; the table itself meets them exactly.

## Relation to Wu et al. (2018)

Wu et al. fit a maximum-entropy joint to marginal constraints from aggregate
tables and to constraints drawn from a sample, then sample individuals. This
design uses the same principle with the study's inputs. Differences to state
in the write-up: the table is dense (the attribute space is 400 cells, so the
scalable approximations in their paper are not needed); the sample enters
only through within-stratum education shares, not through any joint of the
controls, so the comparison isolates the uncontrolled attribute; and the
structural zeros come from the coding, not from the sample. [Check this
paragraph against the paper's Sections 3–4 before submission.]

## Verification

Analytic cases in `tests/test_maxent.py`, each with a known answer and no
data:

| # | Property | Test(s) |
|---|----------|---------|
| 1 | One-way margins only give the product of the margins, in one cycle | `test_one_way_margins_give_product_of_margins` |
| 2 | Overlapping margins give the same table as IPF from a uniform seed | `test_overlapping_margins_match_ipf_from_uniform_seed` |
| 3 | Every constraint holds, including coarsened margins and partial coverage | `test_grouped_margin_over_coarsened_axis_holds`, `test_grouped_margin_can_leave_levels_out` |
| 4 | It is the maximum: no constraint-preserving perturbation raises entropy, and the entropy gradient is orthogonal to the feasible set (KKT) | `test_no_feasible_perturbation_increases_entropy` |
| 5 | Log-linear form; interactions no constraint carries are exactly absent | `test_solution_is_log_linear_in_the_constraint_groups`, `test_uncontrolled_interactions_are_exactly_absent` |
| 6 | An unconstrained attribute is independent (pooled) or conditionally independent given origin (separate) | `test_attribute_constrained_only_by_its_own_margin_is_independent`, `test_attribute_constrained_within_origin_is_independent_given_origin` |
| 7 | A planted table of max-entropy form is recovered; a planted sex × education association is not (negative control) | `test_planted_max_entropy_table_is_recovered`, `test_planted_uncontrolled_association_is_not_recovered` |
| 8 | Structural zeros stay exactly zero; inconsistent or infeasible constraints fail loudly | `test_structural_zeros_stay_zero`, `test_positive_target_on_structural_zero_is_rejected`, `test_different_totals_fail`, `test_grouped_margin_that_disagrees_with_a_finer_one_fails`, `test_zero_pattern_that_forces_unequal_totals_is_caught_before_fitting`, `test_infeasible_constraints_fail_instead_of_returning_garbage` |
| 9 | Order and duplication of constraints do not matter; no iteration when already met; KL to the solution shrinks every cycle | `test_order_and_duplication_of_constraints_do_not_matter`, `test_constraints_already_met_by_uniform_need_no_iteration`, `test_distance_to_the_solution_shrinks_every_cycle` |

On simulated Sprint 2 inputs (`tests/test_design_maxent.py`): the controls
hold exactly; the pooled design reproduces the sample's education shares in
each stratum and has zero mutual information between adult education and the
controls; the separate design has zero conditional mutual information given
origin and keeps the origin differences; the simulated flow's
education–origin association is present in the truth and absent from the
pooled fit.

## Assumptions added

- Within each education stratum, education is independent of the controls
  (pooled) or of sex and age band given origin (separate). This is the
  assumption the design exists to make visible.
- The sample's within-stratum education shares are taken at face value,
  unweighted for its (unknown) inclusion propensities. The sample's SES tilt
  therefore passes straight into the synthetic education mix.
- The education strata (under 5, 5–17, 18+) are definitional.

## References to add

- Wu, H., Ning, Y., Chakraborty, P., Vreeken, J., Tatti, N., Ramakrishnan, N. (2018). Generating realistic synthetic population datasets. *ACM TKDD* 12(4). doi:10.1145/3182383 [verify author list]
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review* 106(4).
- Darroch, J. N., Ratcliff, D. (1972). Generalized iterative scaling for log-linear models. *Annals of Mathematical Statistics* 43(5).
- Csiszár, I. (1975). I-divergence geometry of probability distributions and minimization problems. *Annals of Probability* 3(1).
