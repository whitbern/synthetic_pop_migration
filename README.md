# synthetic_pop_migration

Population synthesis designs for migrant populations, with verification and
validation (V&V) tests.

Paper 1 compares synthesis designs (IPF, maximum entropy) on ACS 5-year PUMS
with a planted ground truth. This repository holds the code; microdata is never
committed (see `.gitignore`).

## Layout

```
src/synthpop/ipf.py        iterative proportional fitting on N-way tables
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
