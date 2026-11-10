# Simulation design (Sprint 2 draft)

Draft text for the Method section, written while the code is fresh. Numbers in
brackets are filled in from the run log once the real extract has been run.

## Planted truth

The reference population is the ACS 2020–2024 5-year PUMS person file,
restricted to people born in the study's origin groups (place of birth,
`POBP`) and linked to the household file on `SERIALNO` for household income.
The extract is frozen once, as a gzipped CSV with a SHA-256 hash recorded in a
manifest and in the run log. Every later step reads the frozen file, never the
raw extract. Each record keeps its person weight `PWGTP` and the 80 replicate
weights `PWGTP1`–`PWGTP80` for design-based variance in Sprint 4.

Recodes: sex (male indicator), age in years and in eight bands (0–4, 5–17,
18–24, 25–34, 35–44, 45–54, 55–64, 65+), and an SES proxy equal to the
weighted rank (0–1) of `ADJINC`-adjusted household income within the truth.
Educational attainment `SCHL` is collapsed to the four IPUMS International
`EDATTAIN` categories now, before any design runs, so the robustness
replication on the pooled IPUMS International truth measures the same outcome:
less than primary (`SCHL` 1–8), primary (9–15), secondary (16–20), university
(21–24), with people under 3 not in universe.

## Selection function

Truth record *i*, with weight *w_i*, enters the flow with propensity

  *p_i* ∝ exp(γ · (s_i − ½) + θ · f_i),

where *s_i* is the SES rank and *f_i* holds the calibrated features: an
indicator for each origin group except the reference, male, age, and an
indicator for age 65+. The flow is the truth reweighted by *w_i p_i*.

γ is the **selection strength**, set by hand: the log ratio of propensity
between the top and bottom of the income distribution with the other features
held fixed. θ is **calibrated** so that the reweighted truth reproduces the
target composition exactly:

| Target | Value | Source |
| --- | --- | --- |
| Origin shares | Venezuela 0.840, Haiti 0.067, Ecuador 0.052, Cuba 0.034, Cameroon 0.007 | Servicio Nacional de Migración (Panama), irregular transit through Darién by nationality (Cuadro 003), Jan–Sep 2022: 151,582 entries, 71.05% Venezuelan; renormalised to the five study groups (84.6% of the flow) |
| Male share | 0.715 | midpoint of the 68–75% range |
| Mean age | 31.0 years | |
| Share aged 65+ | 0.005 | "almost no elderly" |

Calibration is exponential tilting: among all reweightings of the
(γ-tilted) truth that hit the targets, it picks the one closest to it in
Kullback–Leibler divergence. Its dual, log Σ *w_i* exp(η_i) − θ·t, is convex,
so Newton's method with a backtracking line search finds the unique solution
when the targets lie inside what the truth can reach. Targets outside that
range (a boundary value such as 0% elderly, or targets that cannot hold
together) raise an error instead of returning a flow that silently misses; run
with `strict=False`, the gap is reported. Propensities are scaled so the largest
is 1; only their ratios affect the flow's composition.

The flow is drawn as a finite population of [151,582] people: each is a copy
of a truth record chosen with probability proportional to *w_i p_i*. Its
realised composition differs from the targets by sampling error only, which
the pipeline reports against the stated tolerance (±0.5 share points, ±0.25
years of mean age).

**Calibration is underdetermined.** The targets fix θ given γ, but every γ
gives a flow that matches them. The flows differ in what the targets do not
control. On the simulated test truth, moving γ from −2 to +2 moves the
university share of the flow from about 4% to 16% with the composition targets
unchanged. Results are therefore reported across a range of γ, [−2, −1, 0, 1,
2], not at one setting.

## Manufactured inputs

**Margins** (standing in for SENAFRONT) are exact counts from the flow: origin,
origin × sex, and origin × age band. They never include education. The code
refuses any control margin over `SCHL` or `EDATTAIN`.

**Sample** (standing in for the BPM): [1,369] people drawn from the flow
without replacement, with propensity exp(λ·x_j) using the
Efraimidis–Spirakis weighted-sampling scheme. The default tilt is λ = +1 on
the SES rank and −1 on being a minor; the designs are never told these
propensities. The sample carries education, so it is the only route by which
any design can learn about the survey-only attribute.

## Verification

Run without microdata (`tests/test_selection.py`, `tests/test_truth.py`):

- One binary feature has a closed-form solution, θ = logit(t) − logit(base
  share), and the solver matches it to 1e-9.
- Targets equal to the truth's own composition give uniform selection in zero
  iterations.
- Calibration is invariant to rescaling the weights.
- On a simulated three-origin truth, every γ in {−2, 0, 2} reproduces all
  targets to 1e-8, and the education mix moves monotonically with γ.
- Boundary and jointly unreachable targets fail loudly.
- Margins are exact, mutually consistent, and IPF fits them from a flat seed.
- The sampler is without replacement, reproducible from its seed, biased in
  the stated direction, and draws with probability *w_j*/Σ*w* when *n* = 1.
- The frozen truth file is byte-identical across writes and fails its hash
  check if altered.

## Assumptions added this sprint

- Selection acts on age, sex, origin and household income only. Nothing else
  (health, networks, prior migration) enters.
- Selection is log-linear in those features. Age enters linearly plus an
  elderly indicator, so the age distribution's shape beyond its mean and its
  65+ share is inherited from the truth.
- Household income rank, pooled across origins, is an adequate SES proxy. It
  ranks every child by their household.
- An associate's degree counts as "secondary completed" in the EDATTAIN
  collapse.
- The BPM-like sample's bias is a stand-in: λ is chosen, not estimated. Sprint 2
  takes no position on the BPM's real inclusion mechanism.

## External validity (limitation, draft)

The flow matches the Darién composition only on the moments it was calibrated
to. Its joint behaviour, how education relates to age, sex and origin among
those who cross, comes from ACS foreign-born residents of the United States,
tilted by an SES selection whose strength is unknown. That is a population of
people who already arrived and settled, observed years after migration, not
people in transit. The design comparison is internally valid under this
construction (every design faces the same truth), but no estimate from the
synthetic flow should be read as a statement about who crosses the Darién.
Reporting across a range of γ bounds how much the conclusions depend on the
one part of the selection the targets cannot identify.
