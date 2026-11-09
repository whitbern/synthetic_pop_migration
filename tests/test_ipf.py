"""Code verification for IPF: six analytic cases with known answers.

None of these tests use data. Each one checks a property that IPF must have
mathematically, so a failure means the implementation is wrong rather than the
inputs being unusual.

1. Odds-ratio preservation
2. Uniform seed gives the product of the margins
3. Structural zeros stay zero
4. Inconsistent margins fail loudly
5. Idempotence
6. Recovery of a planted three-way table
"""

import numpy as np
import pytest

from synthpop import InconsistentMarginsError, IPFConvergenceError, ipf, marginal

ATOL = 1e-9


@pytest.fixture
def rng():
    return np.random.default_rng(20261109)


def random_one_way_margins(rng, shape, total=1000.0):
    """One-way targets with a common total, so they are mutually consistent."""
    out = []
    for axis, n in enumerate(shape):
        p = rng.dirichlet(np.ones(n))
        out.append(((axis,), total * p))
    return out


def local_odds_ratios(t):
    """All adjacent 2x2 cross-product ratios of a 2-way table."""
    return (t[:-1, :-1] * t[1:, 1:]) / (t[:-1, 1:] * t[1:, :-1])


def three_way_interaction(t):
    """Ratio of conditional odds ratios across the third axis (adjacent cells)."""
    cor = (t[:-1, :-1, :] * t[1:, 1:, :]) / (t[:-1, 1:, :] * t[1:, :-1, :])
    return cor[:, :, 1:] / cor[:, :, :-1]


def assert_margins_hold(table, margins):
    """Margins must hold to the solver's default tolerance (1e-10 of the total)."""
    total = table.sum()
    for axes, target in margins:
        order = np.argsort(axes)
        np.testing.assert_allclose(
            marginal(table, axes), np.transpose(np.asarray(target), order), atol=1e-9 * total, rtol=0
        )


# 1. Odds-ratio preservation -------------------------------------------------


def test_two_way_odds_ratios_preserved(rng):
    seed = rng.uniform(0.5, 5.0, size=(4, 5))
    margins = random_one_way_margins(rng, seed.shape)
    fit = ipf(seed, margins).table

    assert_margins_hold(fit, margins)
    np.testing.assert_allclose(local_odds_ratios(fit), local_odds_ratios(seed), rtol=1e-9)


def test_three_way_conditional_odds_ratios_preserved(rng):
    """With only one-way targets, every conditional odds ratio of the seed survives."""
    seed = rng.uniform(0.5, 5.0, size=(3, 4, 2))
    margins = random_one_way_margins(rng, seed.shape)
    fit = ipf(seed, margins).table

    assert_margins_hold(fit, margins)
    for k in range(seed.shape[2]):
        np.testing.assert_allclose(
            local_odds_ratios(fit[:, :, k]), local_odds_ratios(seed[:, :, k]), rtol=1e-9
        )


# 2. Uniform seed -> product of margins --------------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 5), (2, 3, 2, 4)])
def test_uniform_seed_gives_product_of_margins(rng, shape):
    total = 1000.0
    margins = random_one_way_margins(rng, shape, total)
    result = ipf(np.ones(shape), margins)

    expected = total
    for _, target in margins:
        expected = np.multiply.outer(expected, target / total)
    np.testing.assert_allclose(result.table, expected, atol=ATOL, rtol=0)
    # Independence is reached exactly after one cycle.
    assert result.iterations == 1


# 3. Structural zeros stay zero ----------------------------------------------


def test_structural_zeros_stay_zero(rng):
    seed = rng.uniform(0.5, 5.0, size=(4, 4, 3))
    zeros = np.zeros(seed.shape, dtype=bool)
    zeros[0, 1, :] = True  # a whole fibre
    zeros[2, 3, 1] = True  # a single cell
    zeros[np.arange(4), np.arange(4), 0] = True  # a diagonal slice
    seed[zeros] = 0.0

    # Targets taken from a table with the same zero pattern, so they are feasible.
    truth = rng.uniform(0.5, 5.0, size=seed.shape)
    truth[zeros] = 0.0
    margins = [((0, 1), marginal(truth, (0, 1))), ((2,), marginal(truth, (2,)))]
    fit = ipf(seed, margins).table

    assert np.all(fit[zeros] == 0.0)  # exactly zero, not merely small
    assert np.all(fit[~zeros] > 0.0)
    assert_margins_hold(fit, margins)


def test_positive_target_on_structural_zero_is_rejected():
    seed = np.array([[1.0, 2.0], [0.0, 0.0]])  # row 1 is impossible
    margins = [((0,), [5.0, 5.0]), ((1,), [4.0, 6.0])]
    with pytest.raises(InconsistentMarginsError, match="structural zero"):
        ipf(seed, margins)


# 4. Inconsistent margins fail loudly ----------------------------------------


def test_different_totals_fail():
    margins = [((0,), [10.0, 20.0]), ((1,), [15.0, 16.0])]
    with pytest.raises(InconsistentMarginsError, match="totals differ"):
        ipf(np.ones((2, 2)), margins)


def test_overlapping_margins_that_disagree_fail(rng):
    a = rng.uniform(1, 5, size=(3, 4, 2))
    b = a.copy()
    b[0, :, :] *= 1.5
    b *= a.sum() / b.sum()  # same total, different axis-0 distribution
    margins = [((0, 1), marginal(a, (0, 1))), ((0, 2), marginal(b, (0, 2)))]
    with pytest.raises(InconsistentMarginsError, match=r"shared axes \(0,\)"):
        ipf(np.ones(a.shape), margins)


def test_infeasible_zero_pattern_fails_instead_of_returning_garbage():
    """Passes every pairwise check, but a diagonal seed forces row i == column i."""
    seed = np.eye(2)
    margins = [((0,), [1.0, 2.0]), ((1,), [2.0, 1.0])]
    with pytest.raises(IPFConvergenceError) as err:
        ipf(seed, margins, max_iter=200)
    assert not err.value.result.converged

    # strict=False hands back the table but flags it.
    result = ipf(seed, margins, max_iter=200, strict=False)
    assert not result.converged
    assert result.max_deviation > 1e-3


@pytest.mark.parametrize(
    "seed, margins, match",
    [
        (np.ones((2, 2)), [((0,), [1.0, 2.0, 3.0])], "shape"),
        (np.ones((2, 2)), [((0,), [-1.0, 2.0])], "negative"),
        (np.ones((2, 2)), [((0,), [np.nan, 2.0])], "non-finite"),
        (np.array([[1.0, -1.0], [1.0, 1.0]]), [((0,), [1.0, 1.0])], "negative"),
        (np.ones((2, 2)), [((2,), [1.0, 1.0])], "out of range"),
        (np.ones((2, 2)), [], "at least one"),
    ],
)
def test_malformed_inputs_fail(seed, margins, match):
    with pytest.raises(ValueError, match=match):
        ipf(seed, margins)


# 5. Idempotence --------------------------------------------------------------


def test_refitting_a_fitted_table_changes_nothing(rng):
    seed = rng.uniform(0.5, 5.0, size=(3, 4, 5))
    truth = rng.uniform(0.5, 5.0, size=seed.shape)
    margins = [((0, 1), marginal(truth, (0, 1))), ((1, 2), marginal(truth, (1, 2)))]

    first = ipf(seed, margins)
    second = ipf(first.table, margins)

    assert second.iterations == 0
    np.testing.assert_array_equal(second.table, first.table)


def test_seed_that_already_fits_is_returned_unchanged(rng):
    seed = rng.uniform(0.5, 5.0, size=(3, 4))
    margins = [((0,), seed.sum(axis=1)), ((1,), seed.sum(axis=0))]
    result = ipf(seed, margins)
    assert result.iterations == 0
    np.testing.assert_array_equal(result.table, seed)


# 6. Recovery of a planted three-way table -----------------------------------


def planted_problem(rng, shape=(3, 4, 5)):
    """Seed carries the three-way interaction; truth differs only in two-way terms.

    truth = seed * a[i,j] * b[i,k] * c[j,k]. Because the two-way margins are
    sufficient for the two-way terms, and IPF preserves the seed's three-way
    interaction, fitting the seed to truth's two-way margins must return truth
    exactly (the IPF solution is unique).
    """
    I, J, K = shape
    seed = rng.uniform(0.2, 5.0, size=shape)
    a = rng.uniform(0.3, 3.0, size=(I, J))[:, :, None]
    b = rng.uniform(0.3, 3.0, size=(I, K))[:, None, :]
    c = rng.uniform(0.3, 3.0, size=(J, K))[None, :, :]
    truth = seed * a * b * c
    truth *= 10_000.0 / truth.sum()
    margins = [((0, 1), marginal(truth, (0, 1))), ((0, 2), marginal(truth, (0, 2))), ((1, 2), marginal(truth, (1, 2)))]
    return seed, truth, margins


def test_planted_three_way_table_is_recovered(rng):
    seed, truth, margins = planted_problem(rng)
    result = ipf(seed, margins, tol=1e-13, max_iter=5000)

    assert result.converged
    np.testing.assert_allclose(result.table, truth, rtol=1e-7)
    np.testing.assert_allclose(
        three_way_interaction(result.table), three_way_interaction(truth), rtol=1e-7
    )


def test_planted_recovery_with_margin_axes_out_of_order(rng):
    seed, truth, margins = planted_problem(rng)
    flipped = [(tuple(reversed(ax)), np.transpose(t)) for ax, t in margins]
    result = ipf(seed, flipped, tol=1e-13, max_iter=5000)
    np.testing.assert_allclose(result.table, truth, rtol=1e-7)


def test_uniform_seed_cannot_recover_planted_three_way_interaction(rng):
    """Negative control: without the interaction in the seed, two-way margins alone
    give the no-three-way-interaction table, which is not the truth. This shows the
    recovery test above depends on the seed, not on IPF inventing structure."""
    _, truth, margins = planted_problem(rng)
    fit = ipf(np.ones(truth.shape), margins, tol=1e-12, max_iter=5000).table

    assert_margins_hold(fit, margins)
    np.testing.assert_allclose(three_way_interaction(fit), 1.0, rtol=1e-6)
    assert np.max(np.abs(fit - truth) / truth) > 0.05
