"""The maximum-entropy design on simulated Sprint 2 inputs (no microdata).

Simulated truth -> calibrated flow -> margins + biased sample -> max entropy.
Checks that the design meets every control margin, reproduces the sample's
education shares within each stratum, and leaves adult education independent
of the controls (pooled) or of sex and age given origin (separate).
"""

import json

import numpy as np
import pytest

from simulated import ORIGINS, simulated_truth
from synthpop import marginal
from synthpop.design_maxent import (
    conditional_mutual_information,
    fit_maxent_design,
    load_margins,
    strata_of_age_bands,
)
from synthpop.margins import crosstab, manufacture_margins
from synthpop.selection import DarienTargets, draw_survey_sample, realize_flow, select_flow
from synthpop.truth import AGE_BANDS


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(11)
    truth = simulated_truth()
    sel = select_flow(truth, ORIGINS, DarienTargets(origin_shares={"A": 0.71, "B": 0.22, "C": 0.07}), ses_strength=2.0)
    flow, _ = realize_flow(truth, sel.flow_weight, 30_000, rng)
    flow["weight"] = np.ones(len(flow["age"]))
    margins = manufacture_margins(flow, len(ORIGINS))
    tilt = {"ses": (flow["ses"] - 0.5, 1.0), "minor": ((flow["age"] < 18).astype(float), -1.0)}
    sample, _ = draw_survey_sample(flow, 1369, tilt, rng)
    return flow, margins, sample


def adult_axis_slice(table, strata):
    return table[:, :, strata == strata.max(), :]


def test_strata_map_age_bands():
    np.testing.assert_array_equal(strata_of_age_bands((0, 5, 18)), [0, 1, 2, 2, 2, 2, 2, 2])
    with pytest.raises(ValueError):
        strata_of_age_bands((0, 6, 18))


@pytest.mark.parametrize("by", [(), ("origin",)])
def test_controls_hold_exactly(inputs, by):
    _, margins, sample = inputs
    d = fit_maxent_design(margins, sample, by=by)
    assert d.fit.converged
    assert d.variables == ("origin", "male", "age_band", "edattain")
    ctrl = d.table.sum(axis=3)
    for axes, target in margins.margins:
        np.testing.assert_allclose(marginal(ctrl, axes), target, atol=1e-6)


def test_pooled_reproduces_sample_shares_and_is_independent(inputs):
    _, margins, sample = inputs
    d = fit_maxent_design(margins, sample)
    strata = d.strata
    checked = 0
    for s in range(3):
        syn = d.table[:, :, strata == s, :].sum(axis=(0, 1, 2))
        smp = crosstab(sample, ["age_band", "edattain"], {"age_band": len(AGE_BANDS), "edattain": 5})[strata == s].sum(axis=0)
        if smp.sum() == 0:
            assert s in d.unconstrained_strata
            continue
        checked += 1
        np.testing.assert_allclose(syn / syn.sum(), smp / smp.sum(), atol=1e-9)
    assert checked >= 2
    adults = adult_axis_slice(d.table, strata)
    assert conditional_mutual_information(adults, 3, (0, 1, 2)) == pytest.approx(0.0, abs=1e-10)


def test_separate_is_independent_given_origin_and_keeps_origin_differences(inputs):
    _, margins, sample = inputs
    d = fit_maxent_design(margins, sample, by=("origin",))
    adults = adult_axis_slice(d.table, d.strata)
    assert conditional_mutual_information(adults, 3, (1, 2), (0,)) == pytest.approx(0.0, abs=1e-10)
    assert conditional_mutual_information(adults, 3, (0,)) > 1e-4


def test_children_under_five_get_no_adult_education(inputs):
    _, margins, sample = inputs
    d = fit_maxent_design(margins, sample)
    assert np.all(d.table[:, :, 0, 2:] == 0.0)  # no primary-or-above among 0-4s
    assert np.all(d.table[:, :, 2:, 0] == 0.0)  # no NIU among adults


def test_stratum_missing_from_the_sample_is_left_unconstrained_and_reported(inputs):
    _, margins, sample = inputs
    keep = sample["age_band"] != 0
    thin = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in sample.items()}
    d = fit_maxent_design(margins, thin)
    assert d.unconstrained_strata == [0]
    # Max entropy over the admissible levels (NIU, less than primary): an even split.
    np.testing.assert_allclose(d.table[:, :, 0, 0], d.table[:, :, 0, 1], rtol=1e-9)
    assert np.all(d.table[:, :, 0, 2:] == 0.0)


def test_flow_education_is_associated_but_maxent_is_not(inputs):
    """The planted truth carries an education-origin association that pooled max entropy removes."""
    flow, margins, sample = inputs
    t = crosstab(flow, ["origin", "male", "age_band", "edattain"], {"origin": 3, "male": 2, "age_band": 8, "edattain": 5})
    strata = strata_of_age_bands()
    assert conditional_mutual_information(adult_axis_slice(t, strata), 3, (0, 2)) > 1e-3
    d = fit_maxent_design(margins, sample)
    assert conditional_mutual_information(adult_axis_slice(d.table, strata), 3, (0, 2)) < 1e-12


def test_empty_origin_cell_falls_back_to_pooled_and_is_reported(inputs):
    _, margins, sample = inputs
    drop = (sample["origin"] == 2) & (sample["age_band"] == 1)
    thin = {k: (v[~drop] if isinstance(v, np.ndarray) and v.shape[:1] == drop.shape else v) for k, v in sample.items()}
    d = fit_maxent_design(margins, thin, by=("origin",))
    assert {"origin": 2, "stratum": 1} in d.fallbacks
    pooled = fit_maxent_design(margins, thin)
    np.testing.assert_allclose(d.shares[2, 1], pooled.shares[1], atol=1e-12)


def test_margins_round_trip_through_json(inputs, tmp_path):
    _, margins, _ = inputs
    p = tmp_path / "margins.json"
    p.write_text(json.dumps(margins.as_dict()))
    back = load_margins(p)
    assert back.variables == margins.variables and back.controls == margins.controls
    for (a1, t1), (a2, t2) in zip(back.margins, margins.margins):
        assert a1 == a2
        np.testing.assert_array_equal(t1, t2)


def test_education_may_not_be_a_by_variable(inputs):
    _, margins, sample = inputs
    with pytest.raises(ValueError):
        fit_maxent_design(margins, sample, by=("edattain",))
