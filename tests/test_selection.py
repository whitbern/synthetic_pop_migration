"""Verification of the Sprint 2 selection mechanism, margins and sample.

Analytic cases first (known answers), then the Darién calibration on a
simulated truth, then the manufactured inputs.
"""

import numpy as np
import pytest

from simulated import ORIGINS, simulated_truth
from synthpop import ipf, marginal
from synthpop.margins import PlantedAttributeError, crosstab, manufacture_margins
from synthpop.selection import (
    CalibrationError,
    DarienTargets,
    calibrate_selection,
    composition,
    draw_survey_sample,
    realize_flow,
    select_flow,
    sweep_selection_strength,
)

SHARES = {"A": 0.71, "B": 0.22, "C": 0.07}


@pytest.fixture(scope="module")
def truth():
    return simulated_truth()


@pytest.fixture(scope="module")
def targets():
    return DarienTargets(origin_shares=SHARES)


def logit(p):
    return np.log(p / (1 - p))


# --- Analytic cases ----------------------------------------------------------


def test_single_binary_feature_has_closed_form():
    """One binary feature: θ = logit(target) − logit(base share), exactly."""
    rng = np.random.default_rng(1)
    x = (rng.random(5000) < 0.5).astype(float)
    w = rng.uniform(1, 3, 5000)
    base = (w @ x) / w.sum()
    res = calibrate_selection(w, {"x": x}, {"x": 0.715})
    assert res.converged
    assert res.theta["x"] == pytest.approx(logit(0.715) - logit(base), abs=1e-9)
    assert res.achieved["x"] == pytest.approx(0.715, abs=1e-12)


def test_targets_equal_to_the_truth_give_uniform_selection(truth):
    """If the truth already has the target composition, nobody is favoured."""
    w = truth["weight"]
    feats = {"male": truth["male"].astype(float), "age": truth["age"]}
    tgts = {k: float((w @ v) / w.sum()) for k, v in feats.items()}
    res = calibrate_selection(w, feats, tgts)
    np.testing.assert_allclose(res.probability, 1.0, atol=1e-9)
    assert res.iterations == 0


def test_free_tilt_alone_is_applied_exactly(truth):
    """With no calibrated features, p_i = exp(γ g_i) / max."""
    g = truth["ses"] - 0.5
    res = calibrate_selection(truth["weight"], {}, {}, {"ses": (g, 1.5)})
    np.testing.assert_allclose(res.probability, np.exp(1.5 * g) / np.exp(1.5 * g).max())


def test_calibration_is_invariant_to_scaling_the_weights(truth, targets):
    a = select_flow(truth, ORIGINS, targets, ses_strength=0.5)
    scaled = dict(truth, weight=truth["weight"] * 37.0)
    b = select_flow(scaled, ORIGINS, targets, ses_strength=0.5)
    np.testing.assert_allclose(a.probability, b.probability, rtol=1e-8)


# --- Darién calibration ------------------------------------------------------


@pytest.mark.parametrize("strength", [-2.0, 0.0, 2.0])
def test_flow_reproduces_the_darien_composition(truth, targets, strength):
    res = select_flow(truth, ORIGINS, targets, ses_strength=strength)
    comp = composition(truth, res.flow_weight, ORIGINS)
    assert comp["male"] == pytest.approx(0.715, abs=1e-8)
    assert comp["mean_age"] == pytest.approx(31.0, abs=1e-6)
    assert comp["age>=65"] == pytest.approx(0.005, abs=1e-8)
    for name, share in SHARES.items():
        assert comp[f"origin={name}"] == pytest.approx(share, abs=1e-8)
    assert 0 < res.probability.min() and res.probability.max() == pytest.approx(1.0)


def test_same_targets_different_joints(truth, targets):
    """The calibration is underdetermined: SES strength moves education, not the targets."""
    rows = sweep_selection_strength(truth, ORIGINS, targets, [-2.0, 0.0, 2.0])
    assert all(r["within_tolerance"] == 1.0 for r in rows)
    for k in ["male", "mean_age", "origin=A"]:
        assert np.ptp([r[k] for r in rows]) < 1e-6
    ses = [r["mean_ses"] for r in rows]
    university = [r["edattain=4"] for r in rows]
    assert ses[0] < ses[1] < ses[2]
    assert university[0] < university[1] < university[2]
    assert university[2] - university[0] > 0.05


def test_boundary_target_fails_loudly(truth):
    with pytest.raises(CalibrationError, match="outside the open range"):
        calibrate_selection(truth["weight"], {"male": truth["male"].astype(float)}, {"male": 1.0})


def test_jointly_unreachable_targets_fail_or_report_the_gap():
    """Each target is reachable alone but not together: no woman is old."""
    male = np.array([1, 1, 0, 0], dtype=float)
    old = np.array([1, 0, 0, 0], dtype=float)
    w = np.ones(4)
    feats, tgts = {"male": male, "old": old}, {"male": 0.2, "old": 0.5}
    with pytest.raises(CalibrationError) as err:
        calibrate_selection(w, feats, tgts, max_iter=50)
    res = err.value.result
    assert res is not None and not (res.converged and res.within_tolerance)
    soft = calibrate_selection(w, feats, tgts, max_iter=50, strict=False)
    assert not soft.within_tolerance
    assert max(abs(v) for v in soft.gaps.values()) > 0.01


def test_mismatched_origin_names_are_rejected(truth):
    with pytest.raises(ValueError, match="do not match"):
        select_flow(truth, ORIGINS, DarienTargets(origin_shares={"A": 0.5, "Z": 0.5}))


def test_origin_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        DarienTargets(origin_shares={"A": 0.7, "B": 0.2})


# --- Flow, margins, sample ---------------------------------------------------


@pytest.fixture(scope="module")
def flow(truth, targets):
    res = select_flow(truth, ORIGINS, targets, ses_strength=1.0)
    flow, idx = realize_flow(truth, res.flow_weight, 150_000, np.random.default_rng(3))
    return flow


def test_realized_flow_matches_targets_within_sampling_error(flow):
    comp = composition(flow, np.ones(len(flow["age"])), ORIGINS)
    assert len(flow["age"]) == 150_000
    se = np.sqrt(0.715 * 0.285 / 150_000)
    assert abs(comp["male"] - 0.715) < 5 * se
    assert abs(comp["origin=A"] - 0.71) < 5 * np.sqrt(0.71 * 0.29 / 150_000)
    assert abs(comp["mean_age"] - 31.0) < 0.3
    np.testing.assert_array_equal(flow["flow_id"], np.arange(150_000))


def test_margins_never_include_the_planted_attribute(flow):
    for controls in ([("origin", "edattain")], [("schl",)], [("origin",), ("male", "edattain")]):
        with pytest.raises(PlantedAttributeError):
            manufacture_margins(flow, len(ORIGINS), controls)


def test_margins_are_exact_consistent_and_fit_by_ipf(flow):
    m = manufacture_margins(flow, len(ORIGINS))
    assert m.variables == ("origin", "male", "age_band")
    assert "edattain" not in m.variables and "schl" not in m.variables
    full = crosstab(flow, m.variables, m.levels)
    for axes, target in m.margins:
        np.testing.assert_array_equal(marginal(full, axes), target)
    fit = ipf(np.ones(m.shape), m.margins)
    assert fit.converged


def test_margins_respect_the_order_variables_are_listed_in(flow):
    m = manufacture_margins(flow, len(ORIGINS), [("male", "origin")])
    axes, target = m.margins[0]
    assert target.shape == (2, len(ORIGINS))
    full = crosstab(flow, m.variables, m.levels)
    np.testing.assert_array_equal(target, full.T)
    ipf(np.ones(m.shape), m.margins)


def test_sample_is_without_replacement_and_reproducible(flow):
    tilt = {"ses": (flow["ses"] - 0.5, 1.0)}
    s1, i1 = draw_survey_sample(flow, 1369, tilt, np.random.default_rng(9))
    s2, i2 = draw_survey_sample(flow, 1369, tilt, np.random.default_rng(9))
    assert len(i1) == 1369 and len(np.unique(i1)) == 1369
    np.testing.assert_array_equal(i1, i2)
    assert "edattain" in s1  # the seed does observe the planted attribute


def test_sample_tilt_biases_in_the_stated_direction(flow):
    """Tilting toward SES raises the sample's SES and education relative to the flow."""
    rng = np.random.default_rng(11)
    flow_ses = flow["ses"].mean()
    flow_uni = (flow["edattain"] == 4).mean()
    up, down, srs = [], [], []
    for _ in range(20):
        s, _ = draw_survey_sample(flow, 1369, {"ses": (flow["ses"] - 0.5, 2.0)}, rng)
        up.append(s["ses"].mean())
        s, _ = draw_survey_sample(flow, 1369, {"ses": (flow["ses"] - 0.5, -2.0)}, rng)
        down.append(s["ses"].mean())
        s, _ = draw_survey_sample(flow, 1369, None, rng)
        srs.append(((s["edattain"] == 4).mean(), s["ses"].mean()))
    assert np.mean(down) < flow_ses - 0.05 < flow_ses + 0.05 < np.mean(up)
    srs = np.array(srs)
    assert abs(srs[:, 1].mean() - flow_ses) < 0.01
    assert abs(srs[:, 0].mean() - flow_uni) < 0.01


def test_weighted_sampling_matches_inclusion_odds_for_small_samples():
    """For n=1 the scheme draws record j with probability w_j / Σw exactly."""
    rng = np.random.default_rng(5)
    flow = {"x": np.array([0.0, 1.0, 2.0])}
    coef = np.log(2.0)  # weights 1, 2, 4
    draws = np.array([draw_survey_sample(flow, 1, {"x": (flow["x"], coef)}, rng)[1][0] for _ in range(20_000)])
    freq = np.bincount(draws, minlength=3) / len(draws)
    np.testing.assert_allclose(freq, [1 / 7, 2 / 7, 4 / 7], atol=0.015)
