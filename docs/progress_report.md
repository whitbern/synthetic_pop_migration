# Progress report (15 November 2026)

Paper 1, V&V course. Compiled from the build plan, the `docs/` write-ups,
the run log and merged PRs #1–#16. Points the repository could not settle were confirmed by Allison on 15 Nov.

## 1. What I completed since the proposal

- **Generators, written from scratch and verified.** IPF with strict
  stopping and loud failure on inconsistent margins (PR #1), and the
  maximum-entropy synthesizer after Wu et al. (2018) (PR #4). Both have
  analytic test suites with known answers (6 IPF properties, 9 max-entropy
  properties).
- **Planted truth and selection mechanism (PRs #2–#3).** ACS 2020–2024
  5-year PUMS, five origin groups by place of birth (Venezuela, Haiti, Cuba,
  Ecuador, Cameroon), 131,925 person records, frozen with a SHA-256 hash.
  `SCHL` collapsed to the four `EDATTAIN` categories and withheld from all
  margins, so education is a survey-only attribute. A log-linear selection
  function, calibrated exactly to the SNM Jan–Sep 2022 Darién composition
  (origin shares, 74.5% male, mean age 31, 0.5% aged 65+), run at five SES
  selection strengths γ = −4, −2, 0, +2, +4. Each γ yields margins and a
  biased ~1,369-record sample.
- **Five design families, all run on ACS except the last:**
  1. Pooled IPF (PR #5)
  2. Separate IPF, per origin (PR #5)
  3. Max entropy, pooled and separate (PR #4; run 12 Nov)
  4. Hierarchical partial pooling with empirical-Bayes κ: shared κ, per-origin
     κ (`eb_origin`), and shrinkage toward the origin's own mix
     (`shrink_to = origin`) (PRs #10–#13)
  5. Pooled IPF + origin × education margin (`pooled_edu`, the RAKEYZ
     analogue; PR #16). **Run underway now; results pending.**
- **Crossover experiment (PRs #5–#8).** Thin group oversampled from the flow
  and subsampled 5–640, 200 replicates per size, for Cameroon and Cuba, under
  two metrics (adult joint SRMSE, education TVD), plotted against the group's
  distance from the pool.
- **Scoring (PR #9).** Within-origin adult SRMSE, university-share error,
  education TVD, and mutual information between education and age band / sex.
- **UNHCR BPM and SNM checks (11–13 Nov).** PAY missingness, roster linkage,
  the two survey instruments, origin mix vs SNM, the education/age/sex
  reality check, and the plausibility check of the planted truth against SNM.
  Aggregate scripts are in the project's `scripts/` folder; microdata is
  never committed.
- **Statistical-matching literature check (PRs #14–#15).** Reading note,
  positioning, a drafted related-work paragraph, and 24 BibTeX entries with a
  verification status for each.
- **Run log** (`runs/run_log.jsonl`) and **assumptions lists** in each design
  doc, kept from the first run.

## 2. Important changes to the original plan

| Change | Why |
| --- | --- |
| The hierarchical (partial-pooling) design moved from "future work" into Paper 1 as a third design family | It was cheap to build on the IPF code, and the shared-κ collapse turned out to be a finding in itself |
| A `pooled_edu` design was added | The statistical-matching reading (Singh et al. 1993, via D'Orazio et al. 2006) suggested taking only the origin–education association from a thin origin's records |
| "Origin as an attribute" in a single IPF is not pooling | With nested controls it is mathematically identical to separate IPF, so pooling has to happen in the seed (origin-collapsed seed) |
| Separate-IPF failures are recorded, not patched | The plan's own rule; an age-band merge exists only as a reported option and was not needed |
| The crossover oversamples the thin group first, then subsamples down | So the sample sizes can go well above the group's size in the Sprint 2 sample |
| The crossover is read against heterogeneity (pooled-limit TVD), with Cuba added beside Cameroon | To show the crossover tracks distance from the pool, not γ or group size |
| The paper's claim is reframed | Statistical matching already measured what conditional independence costs in general (Rodgers 1984; Rässler 2004). The paper claims the synthesis setting: each design is a different conditional independence, measured against a planted truth |
| Origin counts cited to Servicio Nacional de Migración, not SENAFRONT | That is the source of the tables actually used |
| The UNHCR BPM got three defined roles (reality check, plausibility check, bridge to Paper 2) | Decided 11 Nov. It is still not the seed |
| No household-level design | The BPM has no household roster, so travel groups cannot be reconstructed. The proposal did not promise a household-level analysis, so this is not a departure from it |
| Fréchet-bounds design deferred to the journal version | Tight for the dominant origin and vacuous for thin ones, and it cannot cover the γ selection error |
| Two-level prior for the hierarchical design skipped | Agreed 15 Nov |
| Adult age-shape recalibration deferred | It is a robustness option and blocks nothing |

## 3. Current status of the model implementation

- Python package `src/synthpop/`, about 115 tests across 8 test files,
  CI on Python 3.10 and 3.13.
- Built and run on ACS: IPF pooled and separate, max entropy pooled and
  separate, hierarchical (shared κ, `eb_origin`, `shrink_to = origin`), the
  crossover experiment and the heterogeneity analysis.
- Built, being run now: `pooled_edu`.
- Not built: replicate-sample uncertainty intervals and coverage (see §7),
  Jensen–Shannon divergence on the joint, the pooled IPUMS International
  truth for the robustness replication (only the `EDATTAIN` collapse is in
  place). The IPUMS International extracts are downloaded.
- The male-share target is 0.745 (SNM Cuadro 002). The Sprint 2 doc's stale
  0.715 is corrected in this PR.

## 4. V&V activities completed or underway

**Code verification (done).** Analytic cases with known answers, run without
microdata:

- IPF: odds-ratio preservation, uniform seed gives the product of margins,
  structural zeros stay zero, inconsistent margins fail loudly, idempotence,
  planted three-way recovery.
- Max entropy: product-of-margins, agreement with IPF from a uniform seed,
  constraints hold, KKT optimality, log-linear form, independence of an
  unconstrained attribute, planted recovery with a negative control, loud
  failure, order invariance, monotone convergence.
- Selection function: closed-form one-feature case, rescaling invariance,
  exact calibration on a simulated truth, loud failure on unreachable targets,
  sampler properties, hash check on the frozen truth.
- Designs: controls hold exactly, pooled borrows the whole sample's pattern,
  separate uses only its own slice, thin groups fail loudly and unpatched,
  hierarchical κ limits (κ = 0 is separate, κ → ∞ is the target).

**Convergence verification (partly done).** IPF records its deviation after
every cycle (`IPFResult.history`) and the design runs log iterations and max
deviation. The comparison of a real ACS trace against Ye et al. (2022)'s
prediction has not been written up. Allison has looked at the per-cycle
deviations, but has not yet compared them with Ye et al.'s prediction.

**Validation against a planted truth (done).** Every design is scored within
each origin against the frozen truth: adult SRMSE, university-share error,
education TVD, and MI(education; age) and MI(education; sex). The
max-entropy independence prediction (MI exactly 0) is confirmed on ACS, which
turns the uncontrolled-attribute metric into a test.

**External plausibility and reality checks (done).** The planted truth
against SNM (calibration targets and untargeted shares), and the ACS seed and
truth against the UNHCR BPM on adult age, sex and education.

**Uncertainty calibration / coverage (not started).** Planned in Sprint 4
(intervals from replicate samples, checked against the planted truth overall,
for thin groups, and for the survey-only attribute). The crossover's 200
replicates give spread but not intervals.

## 5. Preliminary results

- **Pooling erases between-origin education gaps.** Pooled IPF and pooled max
  entropy overstate the adult university share of Haiti, Cuba and Ecuador by
  about 14–25 points and understate Cameroon's by 4–11 points.
- **Separate IPF breaks on thin groups.** It fitted only Venezuela at every γ.
  Cameroon and Cuba failed at every γ, always on empty age bands (0–4, 5–17,
  65+).
- **Crossover.** On the adult joint (SRMSE), separate beats pooled for Cameroon
  at about 120–290 sample members, 9–36 times the group's size in the
  Sprint 2 sample, and for Cuba at 69–143. Across both groups and all γ the
  crossover falls as distance from the pool rises (Spearman ρ = −0.92, ten
  points). Cameroon and Cuba lie on one curve with a log–log slope of about
  −2.3, *consistent with* an inverse-square bias–variance law (too few points
  to estimate one). On education alone, separate wins much earlier, wherever it
  can be fitted.
- **Fitting is binding.** Separate estimation becomes usable (≤ 5% failures)
  at several times the size where it becomes better: Cuba at 400–640,
  Cameroon only at 640 and only at γ 0 and +4.
- **Max entropy behaves as predicted.** Education comes out exactly
  independent of what is not constrained (MI = 0). Separate max entropy has
  the lowest SRMSE for Haiti, Cuba and Ecuador at every γ, and the highest for
  Venezuela, because it loses the age–education association.
- **Hierarchical.** A single shared κ collapses to pooled IPF (κ ≥ 1,108),
  because Venezuela is most of the pool. Per-origin κ lands near the oracle κ
  (≈ 10) and cuts pooled IPF's university bias (Haiti from +22 to +6..+21).
  Shrinking toward the origin's own mix is the only design good for Venezuela
  and close to max entropy for the thin origins, but fails for Cameroon, whose
  own mix rests on about 8 adults.
- **Headline: the direction of borrowing matters more than the amount.**
  Borrowing across origins keeps the dominant origin's bias; borrowing across
  cells within an origin removes it but flattens the age pattern.
- **`pooled_edu`: pending** (running now).

## 6. Problems, limitations and surprises

- **The ACS migrants are far more educated than the Darién flow**, mostly
  Venezuela: adult university share 36.5–50.1% in the planted flow vs 14.6% in
  the BPM. The ACS between-origin gaps are absent from the real flow (BPM
  Haitians are not less educated than Venezuelans), so the per-origin bias
  numbers are ACS-specific, not predictions for the Darién. γ = −4 is the
  closest case.
- **The planted flow has the wrong age shape.** Mean age and 65+ match, but
  minors are 22.8–23.9% vs SNM's 14.2%, and 17.5–18.9% of adults are 50+ vs
  4.6% in the BPM. A mean-age target cannot pin the shape.
- **Surprise: "origin in the seed" IPF is the separate design**, not a pooled
  one, under nested controls.
- **Surprise: the shared empirical-Bayes κ pools exactly where pooling is
  least wanted**, a property of the estimator under an unbalanced sample.
- **A post-hoc tuning choice:** the minimum of 5 informative cells for
  per-origin κ was chosen after seeing Cameroon's run; it must be reported as
  such.
- **Separate IPF overstates the age–education association for thin origins**
  (Haiti 0.11–0.19 nats vs truth 0.018–0.026), the plug-in MI's small-sample
  bias.
- **Survivor bias** in the education-only crossover: at small *n*, separate
  fits only in the 1–8% of draws that cover every age band.
- **UNHCR BPM data quirks:** PAY is missing by design (asked only if ABUSE = Yes
  in Oct 2022–Apr 2023), there is no household roster, two instruments split
  the file by time (education exists only in the second), Cameroon is not a
  listed nationality, and Cuba is over-represented relative to SNM.
- **External validity:** the truth is settled US residents, observed years
  after migration, not people in transit. The design comparison is internally
  valid; no number should be read as a statement about who crosses.
- **Data terms:** UNHCR BPM and IPUMS microdata cannot be redistributed;
  aggregates only in the appendix.

## 7. What remains before final submission

Must do:

1. Record `pooled_edu` ACS results in `docs/sprint4_pooled_edu.md`.
2. Uncertainty calibration and coverage, or an explicit decision to cut it.
   The plan's fallback ("cut the battery to SRMSE plus coverage") keeps
   coverage, so cutting it needs a stated reason in the paper.
3. Write up the convergence trace against Ye et al. (2022) (the plan's one
   "do not cut" item).
4. Consolidated results table: design × metric × origin × γ, and the UNHCR /
   SNM aggregates table for the appendix.
5. Turn the per-sprint assumptions lists into one Method subsection, and write
   the "why these V&V methods are appropriate" argument.
6. Write the Introduction, Background, Discussion, Conclusion and Abstract;
   assemble everything in the `acmart` template (set up locally, not in the
   repository).
7. References: resolve open flags (Wu et al. 2018 author list, Marella &
   Pfeffermann venue, Beckman journal page numbers, D'Orazio Ch. 1 pages, the
   two `[citation]` marks for mean age 31 and 0.5% 65+), and add the dataset
   and software citations (ACS PUMS, UNHCR BPM, SNM, IPUMS if used, NumPy and
   other libraries with versions).

Optional, if time allows: the IPUMS International robustness replication,
the adult age-shape recalibration, Jensen–Shannon divergence on the joint.

## 8. What I learned (draft, for you to rewrite in your own words)

These are candidates drawn from the work; keep the ones that are true for you.

- **Verification first pays off.** The analytic tests caught design-level
  facts (origin-in-seed IPF ≡ separate) that would have been invisible on ACS
  output, where every table "looks fine".
- **A planted truth makes an unanswerable question answerable.** With no
  enumerated Darién population, error can only be measured against a
  population we built ourselves, and its limits have to be measured too
  (the BPM reality check).
- **Calibrating to moments does not calibrate shape.** Matching mean age gave
  the right mean and the wrong distribution.
- **Failure is data.** Recording separate-IPF failures instead of patching
  them showed that fitting, not error, is the binding constraint for thin
  groups.
- **Estimators have politics.** An empirical-Bayes κ that looks neutral pools
  toward the dominant group when the sample is unbalanced.
- **Another field had the theory.** Statistical matching had already named the
  identification problem; reading it changed the paper's claim from "first
  measurement" to "measurement in this setting".
- **What "better" means depends on the target.** The crossover moves by an
  order of magnitude between the joint and the education marginal.
