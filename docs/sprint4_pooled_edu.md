# Pooled IPF with an origin × education margin (the RAKEYZ analogue)

Code: `src/synthpop/design_ipf.py` (variant `pooled_edu`). Verification:
`tests/test_design_ipf.py` (the `pooled_edu` tests). Run with the other IPF
designs: `python -m synthpop.design_ipf configs/sprint2.json`; outputs go to
`data/sprint2/ses_<g>/ipf_pooled_edu/`.

## Why this design

The statistical-matching reading (`docs/statistical_matching_reading.md`,
*Positioning*, item 3) reports a finding of Singh et al. (1993), via D'Orazio,
Di Zio and Scanu (2006, pp. 94–95): with auxiliary micro-data, estimators do
better when they use only its (Y, Z) part, even when it covers all of
(X, Y, Z). With X = (age, sex), Y = origin and Z = education, a thin origin's
own sample records play the role of the auxiliary data. The finding says to
take from them only the origin–education association, not the full
origin × sex × age × education table. RAKEYZ (Singh et al.) does this in
matching by raking to an auxiliary (Y, Z) table.

The existing designs sit on either side of that:

| Design | Takes from the origin's own records | Age–education association |
| --- | --- | --- |
| Pooled IPF | nothing | the pool's |
| Separate IPF | the whole origin × sex × age × education table | the origin's own (noisy; fails on empty cells) |
| Separate max entropy | education within each age stratum | none (zero by construction) |
| **Pooled IPF + origin × education (`pooled_edu`)** | **education within each age stratum** | **the pool's, within the stratum** |

So `pooled_edu` should match separate max entropy on the education level of
each origin (university share, TVD) and pooled IPF on the within-stratum
age–education and sex–education association. On the ACS runs, separate max
entropy was the best design for thin origins on SRMSE and university share
but had zero age–education MI. This design tests whether the missing
association can be had back without paying the separate-IPF price.

## What it does

1. Seed: the pooled seed (origin collapsed), exactly as in pooled IPF.
2. Targets: the three controls (origin, origin × sex, origin × age band),
   plus origin × education stratum × EDATTAIN. The stratum sizes come from
   the origin × age band control, so they match it exactly. The shares within
   each origin × stratum are that origin's own weighted sample shares. The
   strata are those of the max-entropy designs (`maxent.strata_edges`,
   default under 5, 5–17, 18+).
3. Where an origin has nobody in the sample in a stratum, the pooled shares
   are used there and the fallback is reported. This is the same rule as
   separate max entropy.
4. Fitting is plain IPF on an augmented table that has an extra stratum axis,
   zero off each age band's own stratum, so the grouped target is an ordinary
   margin. The stratum axis is summed out afterwards.

**Failures.** As with pooled IPF, the fit fails only if the *whole* sample
has an empty control cell. On the ACS inputs pooled IPF never failed, so
`pooled_edu` should not either. A group whose own records lack a whole
stratum falls back rather than fails.

**What it cannot fix.** The origin × education target carries the thin
sample's noise (Cameroon has 8–13 sample records) and the γ selection bias, exactly
as separate max entropy does. Within a stratum it also inherits the pool's
age and sex pattern of education, which is wrong to the extent that the
origins differ in that pattern.

## Results on the ACS inputs

[ACS run: per-γ tables of adult SRMSE, university-share error, MI(education;
age band) and MI(education; sex) against the flow, with ipf pooled, ipf
separate, maxent separate, hierarchical and hierarchical_own for comparison;
fallbacks reported; failures (expected none).]

## Verification (`tests/test_design_ipf.py`)

On simulated inputs:

- every control margin and the origin × stratum × education target hold
  exactly;
- each origin's education shares within each stratum equal its own weighted
  sample shares (and differ from pooled IPF's);
- the within-origin age–education MI is positive (separate max entropy
  gives zero);
- a thin group that separate IPF cannot fit is fitted, and its adult
  university share equals its own sample's;
- an origin with no sample records in a stratum falls back to the pooled
  shares, and this is reported.
