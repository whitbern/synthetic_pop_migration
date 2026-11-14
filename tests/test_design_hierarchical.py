"""Hierarchical (partial-pooling) IPF on simulated inputs (no microdata).

Checks: kappa = 0 is the separate design and a large kappa approaches the
pooled one; the seed's education distribution is a convex combination of the
origin's own and the pooled one; controls hold exactly; a thin group that
separate IPF cannot fit is fitted by borrowing; empirical-Bayes kappa is large
when origins share one education pattern and small when they do not.
"""

import numpy as np
import pytest

from test_design_ipf import _edu_given_cell, _inputs, rich, thin  # noqa: F401  (fixtures)
from synthpop import marginal
from synthpop.design_hierarchical import (estimate_kappa, fit_hierarchical_design, hierarchical_seed, informative_cells,
                                          log_marginal_likelihood)
from synthpop.design_ipf import fit_ipf_design, ipf_seed


def test_kappa_zero_is_separate(rich):
    _, margins, sample, edges = rich
    h, krec = fit_hierarchical_design(margins, sample, 0.0, age_edges=edges)
    s = fit_ipf_design(margins, sample, "separate", age_edges=edges)
    assert krec["method"] == "fixed"
    np.testing.assert_allclose(h.table, s.table, atol=1e-8)


def test_large_kappa_approaches_pooled(rich):
    _, margins, sample, edges = rich
    h, _ = fit_hierarchical_design(margins, sample, 1e9, age_edges=edges)
    p = fit_ipf_design(margins, sample, "pooled", age_edges=edges)
    np.testing.assert_allclose(h.table, p.table, rtol=1e-4, atol=1e-3)


def test_seed_is_a_convex_combination(rich):
    _, margins, sample, _ = rich
    variables, _, raw = ipf_seed(sample, margins, "separate")
    o = variables.index("origin")
    kappa = 25.0
    seed = hierarchical_seed(raw, o, kappa)
    pooled = raw.sum(axis=o)
    own, pool = _edu_given_cell(raw), _edu_given_cell(pooled)
    got = _edu_given_cell(seed)
    for k in range(raw.shape[o]):
        n = raw[k].sum(axis=-1, keepdims=True)
        lam = n / (n + kappa)
        expected = np.where(n > 0, lam * np.nan_to_num(own[k]) + (1 - lam) * pool, pool)
        mask = seed[k].sum(axis=-1) > 0
        np.testing.assert_allclose(got[k][mask], expected[mask], atol=1e-12)


def test_controls_hold_and_thin_group_fits_by_borrowing(thin):
    flow, margins, sample, _ = thin
    sep = fit_ipf_design(margins, sample, "separate")
    assert sep.failed, "the thin fixture should defeat separate IPF"
    h, krec = fit_hierarchical_design(margins, sample)
    assert krec["method"] == "empirical Bayes" and krec["kappa"] > 0
    assert not h.failed
    for g in h.groups:
        if g["origin"] in sep.failed:
            assert g["empty_in_own_slice"] and not g["empty_control_cells"]
    ctrl = h.table.sum(axis=-1)
    for axes, target in margins.margins:
        np.testing.assert_allclose(marginal(ctrl, axes), target, rtol=1e-8, atol=1e-6)


def _synthetic_counts(rng, same, n_origins=4, cells=6, levels=4, n=400):
    base = rng.dirichlet(np.ones(levels), size=cells)
    out = np.zeros((n_origins, cells, levels))
    for k in range(n_origins):
        probs = base if same else rng.dirichlet(np.full(levels, 0.3), size=cells)
        for c in range(cells):
            out[k, c] = rng.multinomial(n, probs[c])
    return out


def test_eb_kappa_tracks_heterogeneity():
    rng = np.random.default_rng(3)
    same = estimate_kappa(_synthetic_counts(rng, True), 0)
    diff = estimate_kappa(_synthetic_counts(rng, False), 0)
    assert same["kappa"] > 100 * diff["kappa"]
    assert diff["at_bound"] is None and diff["kappa"] < 10


def test_eb_kappa_is_the_maximum():
    rng = np.random.default_rng(5)
    x = _synthetic_counts(rng, False)
    est = estimate_kappa(x, 0)
    for f in (0.8, 1.25):
        assert log_marginal_likelihood(x, 0, est["kappa"] * f) <= est["log_ml"] + 1e-9


def test_negative_kappa_rejected(rich):
    _, margins, sample, _ = rich
    with pytest.raises(ValueError):
        fit_hierarchical_design(margins, sample, -1.0)


def _dominated_counts(rng, cells=12, levels=4, sizes=(1100, 90, 50, 90)):
    """One origin is most of the pool; the thin ones follow their own pattern."""
    base = rng.dirichlet(np.full(levels, 2.0), size=cells)
    x = np.zeros((len(sizes), cells, levels))
    for k, n in enumerate(sizes):
        probs = base if k == 0 else rng.dirichlet(np.ones(levels), size=cells)
        for c in range(cells):
            x[k, c] = rng.multinomial(n // cells, probs[c])
    return x


def test_shared_kappa_is_swamped_but_per_origin_is_not():
    x = _dominated_counts(np.random.default_rng(1))
    assert estimate_kappa(x, 0)["at_bound"] == "upper"
    assert estimate_kappa(x, 0, origin=0)["kappa"] > 1e4
    for k in (1, 2, 3):
        assert estimate_kappa(x, 0, origin=k)["kappa"] < 100


def test_per_origin_kappas_in_the_seed():
    x = _dominated_counts(np.random.default_rng(2))
    kappas = [1e3, 5.0, 0.0, 50.0]
    seed = hierarchical_seed(x, 0, kappas)
    for k, kap in enumerate(kappas):
        np.testing.assert_allclose(seed[k], hierarchical_seed(x, 0, kap)[k])


def test_eb_origin_fits_and_falls_back_when_uninformative(thin):
    _, margins, sample, _ = thin
    h, krec = fit_hierarchical_design(margins, sample, "eb_origin")
    assert not h.failed
    assert len(krec["per_origin"]) == len(krec["kappas"]) == margins.levels["origin"]
    for p, kap in zip(krec["per_origin"], krec["kappas"]):
        assert p["kappa"] == kap
        assert p["fallback"] == (p["informative_cells"] < krec["min_informative_cells"])
        if p["fallback"]:
            assert kap == krec["kappa"]
    ctrl = h.table.sum(axis=-1)
    for axes, target in margins.margins:
        np.testing.assert_allclose(marginal(ctrl, axes), target, rtol=1e-8, atol=1e-6)


def test_uninformative_origin_has_flat_likelihood():
    x = np.zeros((2, 6, 3))
    x[0] = 20
    x[1, :3, 0] = 1  # three cells with one person each, the rest empty
    assert informative_cells(x, 0, 1) == 0
    vals = [log_marginal_likelihood(x, 0, k, origin=1) for k in (0.1, 10.0, 1e4)]
    np.testing.assert_allclose(vals, vals[0])


def test_unknown_kappa_string_rejected(rich):
    _, margins, sample, _ = rich
    with pytest.raises(ValueError):
        fit_hierarchical_design(margins, sample, "bayes")


def test_origin_below_cell_threshold_uses_shared_kappa(thin):
    _, margins, sample, _ = thin
    _, loose = fit_hierarchical_design(margins, sample, "eb_origin", min_informative_cells=1)
    cells = [p["informative_cells"] for p in loose["per_origin"]]
    threshold = max(cells)  # only the best-covered origin(s) keep their own kappa
    _, strict = fit_hierarchical_design(margins, sample, "eb_origin", min_informative_cells=threshold)
    for c, p, kap in zip(cells, strict["per_origin"], strict["kappas"]):
        assert p["fallback"] == (c < threshold)
        if p["fallback"]:
            assert kap == strict["kappa"]


# --- shrink_to="origin": borrow across cells within the origin ---------------


def test_cell_strata_follow_age_bands(rich):
    _, margins, sample, edges = rich
    from synthpop.design_hierarchical import cell_strata
    variables, levels, _ = ipf_seed(sample, margins, "separate")
    strata = cell_strata(variables, levels, edges)
    controls = [v for v in variables if v != "origin"][:-1]
    band = np.indices([levels[v] for v in controls])[controls.index("age_band")].ravel()
    lower = np.asarray(edges)[band]
    np.testing.assert_array_equal(strata, (lower >= 5).astype(int) + (lower >= 18).astype(int))


def test_large_kappa_gives_each_origin_its_own_adult_mix(rich):
    _, margins, sample, edges = rich
    h, _ = fit_hierarchical_design(margins, sample, 1e9, age_edges=edges, shrink_to="origin")
    variables = h.variables
    a = variables.index("age_band") - 1  # axis within an origin slice
    adult = [i for i, lo in enumerate(edges) if lo >= 18]
    for k in range(margins.levels["origin"]):
        rows = sample["origin"] == k
        own = np.bincount(sample["edattain"][rows & (sample["age"] >= 18)].astype(int), minlength=h.table.shape[-1])
        own = own / own.sum()
        cond = _edu_given_cell(np.take(h.table[k], adult, axis=a))
        mask = np.take(h.table[k], adult, axis=a).sum(axis=-1) > 0
        np.testing.assert_allclose(cond[mask], np.broadcast_to(own, cond.shape)[mask], atol=1e-6)


def test_kappa_zero_is_separate_for_either_target(rich):
    _, margins, sample, edges = rich
    a, _ = fit_hierarchical_design(margins, sample, 0.0, age_edges=edges, shrink_to="origin")
    b, _ = fit_hierarchical_design(margins, sample, 0.0, age_edges=edges, shrink_to="pool")
    np.testing.assert_allclose(a.table, b.table, atol=1e-8)


def test_own_target_eb_origin_fits_thin_groups(thin):
    _, margins, sample, _ = thin
    h, krec = fit_hierarchical_design(margins, sample, "eb_origin", shrink_to="origin")
    assert krec["shrink_to"] == "origin" and not h.failed
    ctrl = h.table.sum(axis=-1)
    for axes, target in margins.margins:
        np.testing.assert_allclose(marginal(ctrl, axes), target, rtol=1e-8, atol=1e-6)


def test_unknown_shrink_target_rejected(rich):
    _, margins, sample, _ = rich
    with pytest.raises(ValueError):
        fit_hierarchical_design(margins, sample, 1.0, shrink_to="neighbours")
