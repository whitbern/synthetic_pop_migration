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
from synthpop.design_hierarchical import (estimate_kappa, fit_hierarchical_design, hierarchical_seed,
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
