# Pooled and separate IPF, and the crossover (Sprint 3 draft)

Draft text for the Method section, plus first results. Code:
`src/synthpop/design_ipf.py` (the two designs), `src/synthpop/crossover.py`
(the thin-group experiment and plot), `src/synthpop/scoring.py` (first-pass
scores). Verification: `tests/test_design_ipf.py`.

**Status.** Everything below runs end to end on simulated data. The ACS
numbers are not in yet: the tables marked *[ACS run]* are filled from
`python -m synthpop.design_ipf` and `python -m synthpop.crossover` on the
real Sprint 2 inputs. The simulated numbers quoted here only show what the
output looks like; they are not results.

## The two designs as implemented

Both designs fit a table over origin *o* (5 levels) × sex *s* (2) × age band
*a* (8) × EDATTAIN *e* (5) by iterative proportional fitting (Deming &
Stephan 1940), using the Sprint 1 IPF code unchanged (tolerance 1e-10 on the
largest margin gap relative to the total, at most 1,000 cycles, strict). The
targets are the manufactured control margins: origin, origin × sex,
origin × age band. Education is in no margin. The seed is the biased sample
of 1,369, cross-classified. So all that a design says about education comes
from the seed: IPF rescales each (*o*, *s*, *a*) cell as a whole, and the
education distribution inside the cell is the seed's.

- **Pooled.** One IPF run over all origins at once, with origin carried as an
  attribute of the table. Every origin's slice of the seed is the whole
  sample's *s* × *a* × *e* table. Within each origin, sex and age cell, the
  synthetic education distribution is therefore the whole sample's for that
  sex and age band. Origins differ in education only through their sex and
  age mix.
- **Separate.** One IPF run per origin group, each with that group's own
  margins (its row of origin × sex and origin × age band) and its own slice of
  the sample as the seed. Within each cell, the synthetic education
  distribution is the group's own sample distribution. Nothing is shared.

These are the IPF counterparts of the max-entropy `pooled` and `separate`
variants (education shares over all origins vs within origin), so the four
cells of the design grid line up.

### Departure 1: "origin as an attribute" is not enough to pool

The plan describes pooled IPF as one run with origin as an attribute. Taken
literally, with the full sample (origin included) as the seed, that run gives
**exactly the separate design**. Every control margin is nested within origin
(origin, origin × sex, origin × age band), so the IPF updates never mix
origins, and the fit factorizes into one independent fit per origin.
`test_origin_in_the_seed_is_the_separate_design` checks this to 1e-8. Pooling
has to happen in the seed. The pooled design here collapses origin out of the
seed, which is the standard pooled-seed construction and matches the
max-entropy pooled variant. The literal version is kept in the code as
`joint`, for verification only.

This belongs in the paper. It says that with nested controls, the choice
between pooled and separate is entirely a choice about the seed. A margin
that crosses origins (for example, sex × age band without origin) would
break the factorization, and the two runs would differ even with the same
seed.

### Departure 2: failures are recorded, not fixed

IPF can only rescale, so a seed zero stays zero. When a group's sample slice
has nobody in a sex or age-band cell that its margins ask for, no table meets
the margins. The Sprint 1 code raises `InconsistentMarginsError` before
fitting. The design records the failure and the empty cells (for example,
`age_band=65+`), leaves that group's slice of the table undefined (NaN in
memory, `null` in `table.json`), draws no synthetic people for it, and scores
it as missing. Other groups are not affected. Non-convergence is recorded the
same way (`IPFConvergenceError`, the zero pattern passes the simple checks
but no table fits). It happened once in the simulated crossover (one of
800 separate fits, at n = 40).

The pooled design fails all-or-nothing: if the *whole* sample has nobody in a
cell some origin's margins ask for, the single run fails.

When a group does fit, the separate design still cannot put anyone in the
adult education cells its slice of the sample lacks. Those are the false
zeros of Auld, Mohammadian & Wies (2009), and they are counted
(`false_zeros`: adult cells zero in the synthetic table and positive in the
truth).

### Departure 3: an optional, reported age-band merge

Under-fives are rare in the truth. The ACS truth is people *born* in the
origin countries, so young children are mostly recent arrivals, and the
selection function controls only mean age and the 65+ share. In the
simulated runs the flow had 0.3% under 5. At γ = +2 and +4, the 1,369-person
sample (tilted against minors) had nobody under 5, so **pooled IPF failed
for every origin**. That failure has nothing to do with thin groups; the
age-band grid is finer than the sample can support.

So both CLIs take one optional, reported parameter:
`"ipf": {"drop_age_edges": [5]}` merges 0–4 into 5–17 (0–17) in the margins,
the seed and the scoring truth alike. It is global: it applies to every
origin and both designs, never merges origins, and does not depend on which
group is thin. That is the opposite of Auld's adaptive aggregation, which
would merge exactly the tail origins this study is about. The default is off
(strict). Recommendation: run once strict and report what fails, then run
with `[5]` for the comparison. Adult scores are unaffected, because all the
merged bands are under 18. Merging 65+ into 55–64 (`[5, 65]`) is available
too, but should be reported as a further coarsening if it is used.

## Scores (first pass; Sprint 4 builds the battery)

Every design that fits meets the control margins exactly, so differences sit
in education and its joint with sex and age band. Scores are computed
**within each origin group**, on **adults** (age bands 18+, since "less than
primary" among children mostly means not finished yet), against the flow
(the planted truth, read only for scoring):

- **SRMSE** over the group's adult *s* × *a* × *e* cells:
  √(mean (*x* − *t*)²) / mean *t*.
- **TVD** between the synthetic and true adult education distributions.
- **University-share error**, synthetic minus true.
- **False zeros**, as above.

`python -m synthpop.design_ipf` also scores the max-entropy tables, if
`design_maxent` has been run and no age bands were merged, so all four
designs print in one table.

*[ACS run: SRMSE and university-share error by origin, pooled vs separate,
γ = −4 … +4; which groups the separate design could not fit, and the empty
cells it reported.]*

## The crossover experiment

**Question.** At what sample size does separate estimation of a group stop
losing to pooled estimation?

**Procedure** (for each γ):

1. Keep every other origin's members of the Sprint 2 sample as they are.
2. For the thin group (Cameroon by default, the smallest group in the flow),
   draw Efraimidis–Spirakis keys over **all its members in the flow**, with the
   Sprint 2 tilt (SES +1, minor −1). The *n* smallest keys are a tilted sample
   of size *n*, and the samples are nested, so the curves share random
   numbers.
3. At each size *n* ∈ {5, 10, 20, 40, 80, 160, 320, 640} (capped at the
   group's flow count), replace the group's sample members with that draw.
   Fit both designs, and score the thin group.
4. Repeat for 50 independent key draws.

**Locating the crossover.** For each replicate where separate fits, take the
paired difference (separate SRMSE − pooled SRMSE, same sample). The crossover
is the size where the median paired difference turns from positive (pooled
better) to negative, interpolated on log *n*, after the last size at which
pooled is still better. Separate's failure rate is reported beside it, not
folded into it: a run that cannot fit has no error to compare. Separate's
error curve is therefore conditional on fitting. That makes separate look
better than it is at small *n*, which is one more reason to report the
failure rate.

### Departure 4: oversample first, then subsample down

The plan says to subsample the thin group down in steps. At the Sprint 2
shares, Cameroon is about 0.7% of the flow, so roughly 9 people in a sample
of 1,369. Subsampling down from 9 cannot show a crossover, because separate
estimation never wins that low. Step 2 therefore draws the group from the
flow (about 920 Cameroonians) as if the survey had oversampled it, then
subsamples down. The Sprint 2 size is marked on the plot.

Two side effects to state in the write-up:

- **Pooled moves too.** The thin group's members are part of the pooled seed,
  so pooled improves slightly as *n* grows.
- **The top of the grid is a large share of the group.** At *n* = 640 the
  sample is about 70% of the group's flow members. Separate's error there is
  shrunk by the finite-population effect, which a real survey of a
  ~1,000-person group would also have. The crossover sits well below that in
  every simulated run where it was found.

### First results

*[ACS run: crossover n by γ, the plot `data/sprint3/crossover.png`, and
separate's failure rate at the Sprint 2 sample size.]*

Simulated dry run (three groups; the thin group is 1.5% of the flow, about
900 people, with 20–24 in the Sprint 2 sample; 20 replicates; `drop_age_edges
= [5]`). It is shown only to make the output concrete:

| γ | −4 | −2 | 0 | +2 | +4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Crossover *n* | ≈ 390 | not reached by 640 | ≈ 541 | ≈ 376 | ≈ 203 |

At the Sprint 2 size, separate failed in 70–100% of draws. All but one failure
was an empty age band, most often 65+ or 55–64; none involved education. In other words, a thin
group's separate fit usually breaks on the control margins before
education's false zeros ever matter, and that is worth saying in the
results.

## Verification (`tests/test_design_ipf.py`)

| Property | Test |
| --- | --- |
| Both designs meet every control margin exactly | `test_controls_hold_exactly` |
| Pooled: education within each origin × sex × age cell equals the whole sample's for that sex × age | `test_pooled_borrows_the_whole_sample_education_pattern` |
| Separate: equals the origin's own sample distribution, and differs from pooled | `test_separate_uses_each_origins_own_slice` |
| Origin in the seed + nested controls = separate | `test_origin_in_the_seed_is_the_separate_design` |
| A thin group with an empty control cell fails loudly, lists the cell, stays undefined; other groups unaffected | `test_thin_group_fails_loudly_and_is_not_patched` |
| Seed zeros cannot be filled (false zeros counted, university share biased down) | `test_separate_cannot_fill_seed_zeros` |
| Merged age-band margins equal margins of the merged flow | `test_coarsened_margins_equal_margins_of_the_coarsened_flow` |
| The nested-sample key function reproduces `draw_survey_sample` exactly | `test_draw_survey_sample_is_the_smallest_keys` |
| Crossover locator on hand-made curves; experiment end to end | `test_crossover_locator`, `test_crossover_experiment_runs_and_separate_stops_failing` |

## Assumptions added this sprint

- The thin group's oversample uses the same tilt as the main sample. The
  survey is assumed to reach the group the same way it reaches everyone else,
  only more often.
- Within-group scores on adults only. Children's education is scored in
  Sprint 4 only if it turns out to matter.
- The crossover is defined on SRMSE of the adult sex × age × education
  table. A different metric (TVD on education alone, university share) can
  move it, so Sprint 4 should report it under at least two.

## To read now that it breaks

Auld, Mohammadian & Wies (2009) on false-zero cells, and Auld & Mohammadian
(2010) on multi-level control. The aggregation threshold is
`drop_age_edges`, reported in every run's log entry.
