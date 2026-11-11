# synthetic_pop_migration

Population synthesis designs for migrant populations, with verification and
validation (V&V) tests.

Paper 1 compares synthesis designs (IPF, maximum entropy) on ACS 5-year PUMS
with a planted ground truth. This repository holds the code; microdata is never
committed (see `.gitignore`).

## Layout

```
src/synthpop/ipf.py        iterative proportional fitting on N-way tables
src/synthpop/maxent.py     maximum-entropy tables under linear (margin) constraints
src/synthpop/design_maxent.py  the max-entropy design on the Sprint 2 inputs
src/synthpop/design_ipf.py     pooled and separate IPF on the Sprint 2 inputs
src/synthpop/crossover.py      thin-group crossover experiment and plot
src/synthpop/scoring.py        within-origin scores against the planted truth
src/synthpop/truth.py      load, recode (SCHL -> EDATTAIN) and freeze the ACS truth
src/synthpop/selection.py  calibrated selection function, flow, biased survey sample
src/synthpop/margins.py    manufactured control margins (never over education)
src/synthpop/runlog.py     JSON-lines run log
src/synthpop/sprint2.py    Sprint 2 pipeline (python -m synthpop.sprint2)
configs/                   example config; copy and fill in
docs/                      method write-ups drafted alongside the code
tests/                     analytic and simulated-data verification (no microdata)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## IPF

```python
import numpy as np
from synthpop import ipf

seed = np.ones((3, 4))
result = ipf(seed, [((0,), [10, 20, 30]), ((1,), [15, 15, 15, 15])])
result.table, result.iterations, result.converged
```

Targets are `(axes, target)` pairs, so one-way, two-way and higher margins all
work. Before fitting, `ipf` rejects margins whose totals differ, overlapping
margins that disagree on shared axes, and positive targets over structural
zeros (`InconsistentMarginsError`). If fitting still stalls it raises
`IPFConvergenceError` instead of returning an unconverged table (pass
`strict=False` to get the table back with `converged=False`).

## Code verification tests

| # | Property | Test(s) |
|---|----------|---------|
| 1 | Odds ratios of the seed are preserved | `test_two_way_odds_ratios_preserved`, `test_three_way_conditional_odds_ratios_preserved` |
| 2 | Uniform seed gives the product of the margins | `test_uniform_seed_gives_product_of_margins` |
| 3 | Structural zeros stay exactly zero | `test_structural_zeros_stay_zero`, `test_positive_target_on_structural_zero_is_rejected` |
| 4 | Inconsistent margins fail loudly | `test_different_totals_fail`, `test_overlapping_margins_that_disagree_fail`, `test_infeasible_zero_pattern_fails_instead_of_returning_garbage` |
| 5 | Idempotence | `test_refitting_a_fitted_table_changes_nothing`, `test_seed_that_already_fits_is_returned_unchanged` |
| 6 | A planted three-way table is recovered | `test_planted_three_way_table_is_recovered` (plus a uniform-seed negative control) |

## Sprint 2: ground truth and the selection mechanism

```bash
cp configs/sprint2.example.json configs/sprint2.json   # fill in POBP codes and origin shares
python -m synthpop.sprint2 freeze configs/sprint2.json # raw PUMS -> data/truth/*.csv.gz + manifest
python -m synthpop.sprint2 build  configs/sprint2.json # flow, margins, sample per SES strength
```

`build` writes, for each SES selection strength, `data/sprint2/ses_<g>/`
holding `flow.csv.gz` (the planted truth), `margins.json` (origin, origin x
sex, origin x age band; no education), `sample.csv.gz` (the biased ~1,369
record seed, which does carry education) and `selection.json` (coefficients,
calibration gaps, compositions). Every run is appended to `runs/run_log.jsonl`.

The method is written up in [docs/sprint2_simulation_design.md](docs/sprint2_simulation_design.md).

## Maximum-entropy design

The second generator (after Wu et al. 2018): control margins as constraints,
education constrained only to the sample's distribution within education
strata (under 5, 5–17, 18+), maximum entropy over everything else. Adult
education therefore comes out independent of origin, sex and age band
(`pooled`) or of sex and age band given origin (`separate`): the baseline for
the uncontrolled-attribute test in Sprint 4. After `sprint2 build`:

```bash
python -m synthpop.design_maxent configs/sprint2.json
```

For each SES strength this writes `data/sprint2/ses_<g>/maxent_<variant>/`
holding `table.json` (the fitted origin × sex × age band × EDATTAIN table),
`population.csv.gz` (synthetic people drawn from it) and `report.json`, logs
the run, and prints adult university shares by origin for the flow, the
sample and each variant. Spec and verification table:
[docs/maxent_design.md](docs/maxent_design.md).

## Sprint 3: pooled vs separate IPF and the crossover

After `sprint2 build` (PowerShell, one command per line):

```powershell
pip install -e ".[data,plot]"
python -m synthpop.design_ipf configs/sprint2.json
python -m synthpop.crossover configs/sprint2.json
```

`design_ipf` writes `data/sprint2/ses_<g>/ipf_pooled/` and `ipf_separate/`
(`table.json`, `population.csv.gz`, `report.json`) and prints adult SRMSE and
university-share error by origin, plus any group the separate design could not
fit and the empty control cells that stopped it. `crossover` subsamples the
thin group (Cameroon by default) through sizes 5 to 640, 50 replicates per size,
and writes `data/sprint3/crossover_runs.csv`, `crossover_summary.json` and
`crossover.png`. Optional config keys: `"ipf": {"drop_age_edges": [5]}` merges
0–4 into 5–17 (a reported, global aggregation), and `"crossover": {"origin",
"sizes", "replicates", "output_dir"}`. Write-up:
[docs/sprint3_ipf_designs.md](docs/sprint3_ipf_designs.md).

```python
from synthpop import maxent, margin, grouped_margin
shape = (3, 4)
res = maxent(shape, [margin(shape, (0,), [10, 20, 30]), margin(shape, (1,), [15, 15, 15, 15])])
```
