"""Pooled and separate IPF, and the crossover experiment, on simulated inputs (no microdata).

What is checked, analytically where possible:

* both designs meet every control margin exactly where they fit;
* pooled: within every origin x sex x age band cell, the synthetic education
  distribution is the *whole sample's* for that sex x age band (IPF only
  rescales whole cells, since education is in no margin);
* separate: the same, but with the origin's *own* sample distribution;
* the origin-in-the-seed ("joint") run equals the separate runs, because
  every control is nested within origin;
* a thin group with an empty control cell fails loudly, its slice is left
  undefined (not patched), and the other groups are unaffected;
* scoring and the crossover locator on hand-made inputs.
"""

import numpy as np
import pytest

from simulated import ORIGINS, simulated_truth
from synthpop import marginal
from synthpop.crossover import crossover_point, run_crossover, summarize
from synthpop.design_ipf import coarsen_age_bands, fit_ipf_design
from synthpop.margins import crosstab, manufacture_margins
from synthpop.scoring import mutual_information, score_origin, srmse, tvd
from synthpop.selection import DarienTargets, draw_survey_sample, realize_flow, select_flow, survey_sample_keys
from synthpop.truth import subset


def _inputs(shares, sample_size, seed=11):
    rng = np.random.default_rng(seed)
    truth = simulated_truth()
    sel = select_flow(truth, ORIGINS, DarienTargets(origin_shares=dict(zip(ORIGINS, shares))), ses_strength=2.0)
    flow, _ = realize_flow(truth, sel.flow_weight, 30_000, rng)
    flow["weight"] = np.ones(len(flow["age"]))
    margins = manufacture_margins(flow, len(ORIGINS))
    tilt = {"ses": (flow["ses"] - 0.5, 1.0), "minor": ((flow["age"] < 18).astype(float), -1.0)}
    sample, _ = draw_survey_sample(flow, sample_size, tilt, rng)
    return flow, margins, sample, tilt


@pytest.fixture(scope="module")
def rich():
    """A large sample with merged child bands, so every group fits under both designs."""
    flow, margins, sample, tilt = _inputs((0.5, 0.3, 0.2), 6000)
    margins, (sample, flow), edges = coarsen_age_bands(margins, [sample, flow], [5])
    return flow, margins, sample, edges


@pytest.fixture(scope="module")
def thin():
    """The Darién-like case: a 2% group with ~25 sample members."""
    return _inputs((0.9, 0.08, 0.02), 1369)


def _edu_given_cell(table):
    with np.errstate(invalid="ignore", divide="ignore"):
        return table / table.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("variant", ["pooled", "separate"])
def test_controls_hold_exactly(rich, variant):
    _, margins, sample, edges = rich
    d = fit_ipf_design(margins, sample, variant, age_edges=edges)
    assert not d.failed
    ctrl = d.table.sum(axis=-1)
    for axes, target in margins.margins:
        np.testing.assert_allclose(marginal(ctrl, axes), target, rtol=1e-8, atol=1e-6)


def test_pooled_borrows_the_whole_sample_education_pattern(rich):
    _, margins, sample, edges = rich
    d = fit_ipf_design(margins, sample, "pooled", age_edges=edges)
    pooled = crosstab(sample, ["male", "age_band", "edattain"], dict(margins.levels, edattain=5))
    expected = _edu_given_cell(pooled)
    got = _edu_given_cell(d.table)
    for o in range(len(ORIGINS)):
        mask = d.table[o].sum(axis=-1) > 0
        np.testing.assert_allclose(got[o][mask], expected[mask], atol=1e-10)


def test_separate_uses_each_origins_own_slice(rich):
    _, margins, sample, edges = rich
    d = fit_ipf_design(margins, sample, "separate", age_edges=edges)
    own = crosstab(sample, ["origin", "male", "age_band", "edattain"], dict(margins.levels, edattain=5))
    mask = d.table.sum(axis=-1) > 0
    np.testing.assert_allclose(_edu_given_cell(d.table)[mask], _edu_given_cell(own)[mask], atol=1e-10)
    # ...so where origins differ in the sample, the designs differ.
    p = fit_ipf_design(margins, sample, "pooled", age_edges=edges)
    assert np.max(np.abs(d.table - p.table)) > 1.0


def test_origin_in_the_seed_is_the_separate_design(rich):
    _, margins, sample, edges = rich
    joint = fit_ipf_design(margins, sample, "joint", age_edges=edges)
    sep = fit_ipf_design(margins, sample, "separate", age_edges=edges)
    np.testing.assert_allclose(joint.table, sep.table, rtol=1e-8, atol=1e-6)


def test_thin_group_fails_loudly_and_is_not_patched(thin):
    flow, margins, sample, _ = thin
    sep = fit_ipf_design(margins, sample, "separate")
    c = ORIGINS.index("C")
    status = sep.groups[c]
    assert status["failure"] == "empty control cell"
    assert status["empty_control_cells"], "the empty cells must be listed"
    assert np.all(np.isnan(sep.table[c]))
    # The other groups' fits do not depend on C.
    for k, g in enumerate(sep.groups):
        if not g["failure"]:
            assert np.all(np.isfinite(sep.table[k]))
    truth = crosstab(flow, ("origin", "male", "age_band", "edattain"), dict(margins.levels, edattain=5))
    assert score_origin(sep.table, truth, ("origin", "male", "age_band", "edattain"), c)["srmse"] is None


def test_separate_cannot_fill_seed_zeros(rich):
    flow, margins, sample, edges = rich
    # Drop every university graduate of origin C from the sample: separate must give C none.
    c = ORIGINS.index("C")
    keep = ~((sample["origin"] == c) & (sample["edattain"] == 4))
    d = fit_ipf_design(margins, subset(sample, np.flatnonzero(keep)), "separate", age_edges=edges)
    assert d.table[c, ..., 4].sum() == 0
    truth = crosstab(flow, ("origin", "male", "age_band", "edattain"), dict(margins.levels, edattain=5))
    s = score_origin(d.table, truth, ("origin", "male", "age_band", "edattain"), c, band_edges=edges)
    assert s["false_zeros"] >= int((truth[c][:, 1:, 4] > 0).sum())
    assert s["univ_error"] < 0


def test_coarsened_margins_equal_margins_of_the_coarsened_flow(thin):
    flow, margins, _, _ = thin
    m2, (f2,), edges = coarsen_age_bands(margins, [flow], [5, 65])
    assert edges == (0, 18, 25, 35, 45, 55)
    direct_age = crosstab(f2, ["origin", "age_band"], {"origin": 3, "age_band": len(edges)})
    np.testing.assert_allclose(m2.margins[2][1], direct_age)
    with pytest.raises(ValueError):
        coarsen_age_bands(margins, [flow], [0])


def test_scores_on_hand_made_tables():
    assert srmse([1, 2, 3], [1, 2, 3]) == 0
    assert srmse([0, 0], [2, 2]) == pytest.approx(1.0)
    assert tvd([1, 0], [0, 1]) == 1.0
    assert tvd([2, 2], [1, 1]) == 0.0


def test_mutual_information_on_hand_made_tables():
    # Independent: product of margins.
    assert mutual_information(np.outer([1, 3], [2, 2, 4]), 0, 1) == pytest.approx(0.0, abs=1e-12)
    # Perfectly dependent 2x2 with uniform margins: log 2.
    assert mutual_information(np.eye(2), 0, 1) == pytest.approx(np.log(2))
    assert mutual_information(np.eye(2), 1, 0) == pytest.approx(np.log(2))
    # Other axes are summed out first.
    t = np.stack([np.eye(2), np.eye(2)], axis=1)  # axes (a, c, b)
    assert mutual_information(t, 0, 2) == pytest.approx(np.log(2))


def test_ipf_keeps_the_seeds_age_education_association(rich):
    """Pooled and separate IPF carry the sample's education-by-age pattern; maximum entropy would not."""
    flow, margins, sample, edges = rich
    variables = margins.variables + ("edattain",)
    truth = crosstab(flow, variables, dict(margins.levels, edattain=5))
    for variant in ("pooled", "separate"):
        d = fit_ipf_design(margins, sample, variant, age_edges=edges)
        for c in range(len(ORIGINS)):
            s = score_origin(d.table, truth, variables, c, band_edges=edges)
            assert s["mi_edattain_age_truth"] > 1e-4
            assert s["mi_edattain_age"] > 1e-4  # maximum entropy gives ~0 here


def test_draw_survey_sample_is_the_smallest_keys():
    flow, _, _, tilt = _inputs((0.9, 0.08, 0.02), 50)
    _, idx = draw_survey_sample(flow, 200, tilt, np.random.default_rng(3))
    keys = survey_sample_keys(flow, tilt, np.random.default_rng(3))
    np.testing.assert_array_equal(idx, np.sort(np.argsort(keys)[:200]))


def test_crossover_locator():
    assert crossover_point([10, 100], [1.0, -1.0]) == {"n": pytest.approx(np.sqrt(1000)), "status": "crossed"}
    assert crossover_point([10, 100], [1.0, 0.5])["status"] == "pooled better throughout"
    assert crossover_point([10, 100], [-1.0, -0.5])["status"] == "separate better throughout"
    assert crossover_point([10, 100], [None, None])["status"] == "no data"
    # A sign flip and back: the crossover is after the *last* size where pooled wins.
    assert crossover_point([10, 20, 40, 80], [1.0, -1.0, 1.0, -1.0])["n"] > 40


def test_crossover_experiment_runs_and_separate_stops_failing(thin):
    flow, margins, sample, tilt = thin
    margins, (sample, flow), edges = coarsen_age_bands(margins, [sample, flow], [5])
    c = ORIGINS.index("C")
    sizes = [2, 40, 400]
    rows = run_crossover(flow, margins, sample, c, sizes, 4, tilt, np.random.default_rng(5), edges)
    assert len(rows) == 4 * len(sizes) * 2
    s = summarize(rows)
    by_n = {e["n"]: e for e in s["by_size"]}
    assert by_n[2]["separate_failure_rate"] == 1.0  # two people cannot fill six age bands
    assert by_n[400]["separate_failure_rate"] < 1.0
    assert all(e["pooled_failure_rate"] == 0.0 for e in s["by_size"])
    m40, m400 = by_n[40]["separate"]["median"], by_n[400]["separate"]["median"]
    assert m400 is not None and (m40 is None or m400 < m40)


def test_heterogeneity_is_zero_for_a_group_like_the_pool():
    from synthpop.scoring import heterogeneity
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 10, (2, 8, 5))
    truth = np.stack([base * 3, base * 1])  # two origins, identical shape
    h = heterogeneity(truth, ("origin", "male", "age_band", "edattain"), 1)
    assert h["tvd_edattain_vs_pool"] == pytest.approx(0, abs=1e-12)
    assert h["pooled_limit_tvd"] == pytest.approx(0, abs=1e-12)


def test_pooled_limit_is_what_pooled_ipf_gives_with_the_truth_as_seed(rich):
    """With the whole flow as the seed, pooled IPF's error for a group is exactly the pooled-limit TVD."""
    from synthpop.scoring import adult_bands, heterogeneity
    flow, margins, _, edges = rich
    variables = ("origin", "male", "age_band", "edattain")
    truth = crosstab(flow, variables, dict(margins.levels, edattain=5))
    d = fit_ipf_design(margins, flow, "pooled", age_edges=edges)
    adults = adult_bands(edges)
    for k in range(len(ORIGINS)):
        h = heterogeneity(truth, variables, k, band_edges=edges)
        est, tru = d.table[k][:, adults], truth[k][:, adults]
        assert h["pooled_limit_tvd"] == pytest.approx(tvd(est, tru), abs=1e-8)
        assert h["pooled_limit_tvd"] > 0.01  # the simulated origins do differ
