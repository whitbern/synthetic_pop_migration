"""Code verification for the maximum-entropy synthesizer: analytic cases.

As with ``test_ipf.py``, none of these use data; each checks a property the
maximum-entropy solution must have mathematically.

1. With one-way margins only, the answer is the product of the margins
2. With overlapping margins, the answer equals IPF from a uniform seed
3. Every constraint holds, including coarsened ("grouped") margins
4. It is the entropy maximum: no feasible perturbation raises entropy
5. It has the log-linear form, so uncontrolled interactions are exactly absent
6. An unconstrained attribute is (conditionally) independent of the controls
7. A planted table of the max-entropy form is recovered; one with an
   uncontrolled association is not (negative control)
8. Structural zeros stay zero; inconsistent constraints fail loudly
9. Idempotence and invariance to constraint order and duplication
"""

import numpy as np
import pytest

from synthpop import InconsistentMarginsError, ipf, marginal
from synthpop.design_maxent import conditional_mutual_information
from synthpop.maxent import (
    MaxEntConvergenceError,
    draw_population,
    entropy,
    grouped_margin,
    margin,
    maxent,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20261110)


def group_totals(table, c):
    cov = c.labels >= 0
    return np.bincount(c.labels[cov], table[cov], minlength=c.n_groups)


def assert_constraints_hold(table, constraints, rtol=1e-9):
    total = table.sum()
    for c in constraints:
        np.testing.assert_allclose(group_totals(table, c), c.target, atol=rtol * total, rtol=0)


def constraint_matrix(shape, constraints):
    """Rows: indicator of each group of each constraint, over the flattened cells."""
    rows = []
    for c in constraints:
        lab = c.labels.ravel()
        for g in range(c.n_groups):
            rows.append((lab == g).astype(float))
    return np.array(rows)


def random_margins(rng, truth, axes_list):
    return [margin(truth.shape, ax, marginal(truth, ax)) for ax in axes_list]


# 1. One-way margins -> product of the margins --------------------------------


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 5), (2, 3, 2, 4)])
def test_one_way_margins_give_product_of_margins(rng, shape):
    total = 1000.0
    targets = [total * rng.dirichlet(np.ones(n)) for n in shape]
    res = maxent(shape, [margin(shape, (a,), t) for a, t in enumerate(targets)])

    expected = total
    for t in targets:
        expected = np.multiply.outer(expected, t / total)
    np.testing.assert_allclose(res.table, expected, rtol=1e-12)
    assert res.iterations == 1


# 2. Equals IPF from a uniform seed --------------------------------------------


def test_overlapping_margins_match_ipf_from_uniform_seed(rng):
    truth = rng.uniform(0.5, 5.0, size=(3, 4, 5))
    axes = [(0, 1), (0, 2), (1, 2)]
    res = maxent(truth.shape, random_margins(rng, truth, axes), tol=1e-13)
    ref = ipf(np.ones(truth.shape), [(ax, marginal(truth, ax)) for ax in axes], tol=1e-13, max_iter=5000)
    np.testing.assert_allclose(res.table, ref.table, rtol=1e-9)


# 3. Constraints hold, including grouped margins --------------------------------


def test_grouped_margin_over_coarsened_axis_holds(rng):
    truth = rng.uniform(0.5, 5.0, size=(4, 6, 3))
    groups = np.array([0, 0, 1, 1, 1, 2])  # six bands -> three strata
    coarse = np.stack([truth[:, groups == s, :].sum(axis=1) for s in range(3)], axis=1)  # (4, 3, 3)
    cons = [margin(truth.shape, (0, 1), marginal(truth, (0, 1))),
            grouped_margin(truth.shape, (1, 2), coarse.sum(axis=0), groups={1: groups})]
    res = maxent(truth.shape, cons, tol=1e-12)
    assert_constraints_hold(res.table, cons)


def test_grouped_margin_can_leave_levels_out(rng):
    """Level -1 in a grouping: those cells are outside the constraint."""
    shape = (3, 4)
    groups = np.array([-1, 0, 0, 1])
    c = grouped_margin(shape, (1,), [30.0, 10.0], groups={1: groups})
    assert np.all(c.labels[:, 0] == -1)
    res = maxent(shape, [c, margin(shape, (0,), [20.0, 20.0, 20.0])], tol=1e-12)
    assert_constraints_hold(res.table, [c])
    assert res.table[:, 0].sum() == pytest.approx(20.0, abs=1e-8)


# 4. Entropy maximum -------------------------------------------------------------


def test_no_feasible_perturbation_increases_entropy(rng):
    truth = rng.uniform(0.5, 5.0, size=(3, 4, 2))
    cons = random_margins(rng, truth, [(0, 1), (2,)])
    p = maxent(truth.shape, cons, tol=1e-13).table
    A = constraint_matrix(truth.shape, cons)
    # Directions that keep every constraint: the null space of A.
    _, s, vt = np.linalg.svd(A)
    null = vt[(s > 1e-10).sum():]
    assert null.shape[0] > 0
    h0 = entropy(p)
    for _ in range(50):
        d = (rng.normal(size=null.shape[0]) @ null).reshape(p.shape)
        eps = 0.05 * p.min() / np.abs(d).max()
        q = p + eps * d
        np.testing.assert_allclose(A @ q.ravel(), A @ p.ravel(), atol=1e-8)
        assert entropy(q) < h0
    # And the gradient of entropy is orthogonal to every feasible direction (KKT).
    grad = -(np.log(p.ravel() / p.sum()) + 1)
    np.testing.assert_allclose(null @ grad, 0.0, atol=1e-8)


# 5. Log-linear form ---------------------------------------------------------------


def test_solution_is_log_linear_in_the_constraint_groups(rng):
    truth = rng.uniform(0.5, 5.0, size=(3, 4, 5))
    groups = np.array([0, 0, 1, 1])
    coarse = np.stack([truth[:, groups == s, :].sum(axis=1) for s in range(2)], axis=1)
    cons = [margin(truth.shape, (0, 1), marginal(truth, (0, 1))),
            grouped_margin(truth.shape, (1, 2), coarse.sum(axis=0), groups={1: groups})]
    p = maxent(truth.shape, cons, tol=1e-13).table
    A = constraint_matrix(truth.shape, cons)
    lam, *_ = np.linalg.lstsq(A.T, np.log(p.ravel()), rcond=None)
    np.testing.assert_allclose(A.T @ lam, np.log(p.ravel()), atol=1e-9)


def test_uncontrolled_interactions_are_exactly_absent(rng):
    """With two-way margins (0,1) and (1,2) only, axes 0 and 2 are independent given 1."""
    truth = rng.uniform(0.5, 5.0, size=(3, 4, 5))
    p = maxent(truth.shape, random_margins(rng, truth, [(0, 1), (1, 2)]), tol=1e-13).table
    for j in range(4):
        sl = p[:, j, :]
        odds = (sl[:-1, :-1] * sl[1:, 1:]) / (sl[:-1, 1:] * sl[1:, :-1])
        np.testing.assert_allclose(odds, 1.0, rtol=1e-9)
    assert conditional_mutual_information(p, 0, (2,), (1,)) == pytest.approx(0.0, abs=1e-12)


# 6. Independence of an unconstrained attribute ------------------------------------


def test_attribute_constrained_only_by_its_own_margin_is_independent(rng):
    """Controls over (origin, sex, age); education only by its one-way margin."""
    truth = rng.uniform(0.5, 5.0, size=(3, 2, 4, 5))
    cons = random_margins(rng, truth, [(0, 1), (0, 2), (3,)])
    p = maxent(truth.shape, cons, tol=1e-12).table
    assert conditional_mutual_information(p, 3, (0, 1, 2)) == pytest.approx(0.0, abs=1e-12)
    # Whereas the truth it came from is associated.
    assert conditional_mutual_information(truth, 3, (0, 1, 2)) > 1e-3


def test_attribute_constrained_within_origin_is_independent_given_origin(rng):
    truth = rng.uniform(0.5, 5.0, size=(3, 2, 4, 5))
    cons = random_margins(rng, truth, [(0, 1), (0, 2), (0, 3)])
    p = maxent(truth.shape, cons, tol=1e-12).table
    assert conditional_mutual_information(p, 3, (1, 2), (0,)) == pytest.approx(0.0, abs=1e-12)
    # It does keep the education-origin association the constraint carries.
    np.testing.assert_allclose(marginal(p, (0, 3)), marginal(truth, (0, 3)), rtol=1e-9)
    assert conditional_mutual_information(p, 3, (0,)) > 1e-3


# 7. Planted recovery ---------------------------------------------------------------


def test_planted_max_entropy_table_is_recovered(rng):
    """truth = a(o,s) b(o,g) c(g_strata, e): exactly the form max entropy can represent."""
    O, S, G, E = 3, 2, 6, 5
    strata = np.array([0, 1, 2, 2, 2, 2])
    a = rng.uniform(0.3, 3, (O, S, 1, 1))
    b = rng.uniform(0.3, 3, (O, 1, G, 1))
    c = rng.uniform(0.3, 3, (3, E))[strata][None, None, :, :]
    truth = a * b * c
    truth *= 10_000 / truth.sum()
    coarse = np.stack([truth[:, :, strata == s, :].sum(axis=(0, 1, 2)) for s in range(3)])
    cons = [margin(truth.shape, (0, 1), marginal(truth, (0, 1))),
            margin(truth.shape, (0, 2), marginal(truth, (0, 2))),
            grouped_margin(truth.shape, (2, 3), coarse, groups={2: strata})]
    res = maxent(truth.shape, cons, tol=1e-13)
    np.testing.assert_allclose(res.table, truth, rtol=1e-8)


def test_planted_uncontrolled_association_is_not_recovered(rng):
    """Negative control: truth links education to sex; the same constraints cannot see it."""
    O, S, G, E = 3, 2, 6, 5
    strata = np.array([0, 1, 2, 2, 2, 2])
    truth = rng.uniform(0.3, 3, (O, S, G, 1)) * rng.uniform(0.3, 3, (3, E))[strata][None, None]
    truth = truth * np.array([1.0, 3.0])[None, :, None, None] ** (np.arange(E) / E)  # sex x education
    coarse = np.stack([truth[:, :, strata == s, :].sum(axis=(0, 1, 2)) for s in range(3)])
    cons = [margin(truth.shape, (0, 1, 2), marginal(truth, (0, 1, 2))),
            grouped_margin(truth.shape, (2, 3), coarse, groups={2: strata})]
    p = maxent(truth.shape, cons, tol=1e-12).table
    adults = strata == 2
    assert conditional_mutual_information(p[:, :, adults], 3, (1,)) == pytest.approx(0.0, abs=1e-12)
    assert conditional_mutual_information(truth[:, :, adults], 3, (1,)) > 1e-3


# 8. Structural zeros and inconsistency ---------------------------------------------


def test_structural_zeros_stay_zero(rng):
    shape = (4, 3)
    support = np.ones(shape, dtype=bool)
    support[0, 2] = support[3, 0] = False
    truth = rng.uniform(0.5, 5, shape) * support
    cons = random_margins(rng, truth, [(0,), (1,)])
    p = maxent(shape, cons, support=support, tol=1e-12).table
    assert np.all(p[~support] == 0.0)
    assert np.all(p[support] > 0.0)
    assert_constraints_hold(p, cons)


def test_positive_target_on_structural_zero_is_rejected():
    support = np.array([[True, True], [False, False]])
    cons = [margin((2, 2), (0,), [5.0, 5.0])]
    with pytest.raises(InconsistentMarginsError, match="structural zero"):
        maxent((2, 2), cons, support=support)


def test_different_totals_fail():
    cons = [margin((2, 2), (0,), [10.0, 20.0]), margin((2, 2), (1,), [15.0, 16.0])]
    with pytest.raises(InconsistentMarginsError, match="totals differ"):
        maxent((2, 2), cons)


def test_grouped_margin_that_disagrees_with_a_finer_one_fails(rng):
    truth = rng.uniform(0.5, 5.0, size=(4, 3))
    groups = np.array([0, 0, 1, 1])
    wrong = np.array([truth[:2].sum() + 5.0, truth[2:].sum() - 5.0])  # same total, wrong split
    cons = [margin(truth.shape, (0,), truth.sum(axis=1)),
            grouped_margin(truth.shape, (0,), wrong, groups={0: groups})]
    with pytest.raises(InconsistentMarginsError, match="different totals"):
        maxent(truth.shape, cons)


def test_zero_pattern_that_forces_unequal_totals_is_caught_before_fitting():
    """A diagonal support makes row i equal column i, so these margins cannot both hold."""
    support = np.eye(2, dtype=bool)
    cons = [margin((2, 2), (0,), [1.0, 2.0]), margin((2, 2), (1,), [2.0, 1.0])]
    with pytest.raises(InconsistentMarginsError, match="different totals"):
        maxent((2, 2), cons, support=support)


def test_infeasible_constraints_fail_instead_of_returning_garbage():
    """Pairwise-consistent two-way margins with no joint: X=Y, Y=Z, but X!=Z."""
    same = np.array([[0.5, 0.0], [0.0, 0.5]])
    cons = [margin((2, 2, 2), (0, 1), same), margin((2, 2, 2), (1, 2), same),
            margin((2, 2, 2), (0, 2), same[::-1])]
    with pytest.raises(MaxEntConvergenceError) as err:
        maxent((2, 2, 2), cons, max_iter=200)
    assert not err.value.result.converged

    res = maxent((2, 2, 2), cons, max_iter=200, strict=False)
    assert not res.converged and res.max_deviation > 1e-3


def test_partial_constraints_need_an_explicit_total():
    c = grouped_margin((3,), (0,), [4.0], groups={0: [0, 0, -1]})
    with pytest.raises(ValueError, match="total"):
        maxent((3,), [c])
    p = maxent((3,), [c], total=10.0).table
    np.testing.assert_allclose(p, [2.0, 2.0, 6.0])


# 9. Idempotence and invariance --------------------------------------------------------


def test_order_and_duplication_of_constraints_do_not_matter(rng):
    truth = rng.uniform(0.5, 5.0, size=(3, 4, 5))
    cons = random_margins(rng, truth, [(0, 1), (1, 2), (0, 2)])
    a = maxent(truth.shape, cons, tol=1e-13).table
    b = maxent(truth.shape, cons[::-1] + cons[:1], tol=1e-13).table
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_constraints_already_met_by_uniform_need_no_iteration():
    cons = [margin((2, 3), (0,), [3.0, 3.0]), margin((2, 3), (1,), [2.0, 2.0, 2.0])]
    res = maxent((2, 3), cons)
    assert res.iterations == 0
    np.testing.assert_array_equal(res.table, np.ones((2, 3)))


def test_distance_to_the_solution_shrinks_every_cycle(rng):
    """Cyclic I-projection: KL(p* || p_n) is nonincreasing in n (Csiszar 1975)."""
    truth = rng.uniform(0.5, 5.0, size=(3, 4, 5))
    cons = random_margins(rng, truth, [(0, 1), (1, 2), (0, 2)])
    res = maxent(truth.shape, cons, tol=1e-13)
    star = res.table / res.table.sum()
    kl = []
    for n in range(8):
        p = maxent(truth.shape, cons, max_iter=n, strict=False).table
        p = p / p.sum()
        kl.append(float((star * np.log(star / p)).sum()))
    assert np.all(np.diff(kl) <= 1e-14)
    assert res.entropy_history[0] == pytest.approx(np.log(truth.size))
    assert res.entropy == pytest.approx(entropy(res.table))


# Drawing individuals -------------------------------------------------------------------


def test_draw_population_matches_table_in_expectation(rng):
    table = rng.uniform(0.5, 5.0, size=(3, 4))
    table *= 200_000 / table.sum()
    people = draw_population(table, rng)
    assert people.shape == (200_000, 2)
    counts = np.zeros(table.shape)
    np.add.at(counts, tuple(people.T), 1)
    # Multinomial: each cell within 5 standard deviations.
    sd = np.sqrt(table * (1 - table / table.sum()))
    assert np.all(np.abs(counts - table) < 5 * sd)


def test_draw_population_is_reproducible():
    table = np.arange(1.0, 7.0).reshape(2, 3)
    a = draw_population(table, np.random.default_rng(3), size=50)
    b = draw_population(table, np.random.default_rng(3), size=50)
    np.testing.assert_array_equal(a, b)
