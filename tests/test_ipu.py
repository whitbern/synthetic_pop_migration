"""Code verification for IPU: analytic cases with known answers.

1. Hand-computed two-household case (delta before and after, exact weights)
2. Household-only controls fit in one iteration
3. After each iteration the last constraint holds exactly
4. Zero weights stay zero; weights stay nonnegative
5. Infeasible controls stall and fail loudly (strict), or return converged=False
6. A constraint no household contributes to is rejected before fitting
7. The entropy reference solver fits planted feasible instances exactly
8. When IPU fits, it meets the same constraints the reference solver does
"""

import numpy as np
import pytest

from synthpop.ipu import (
    InfeasibleConstraintError,
    IPUConvergenceError,
    entropy_weights,
    goodness_of_fit,
    ipu,
)
from synthpop.ipu_convergence import planted_instance, run_instance


# 1. Hand-computed case ------------------------------------------------------


def test_hand_computed_two_household_case():
    # Columns: households of type 1, persons of type 1. A has 1 person, B has 2.
    freq = np.array([[1.0, 1.0], [1.0, 2.0]])
    targets = np.array([10.0, 15.0])
    # Before fitting: fitted (2, 3) vs (10, 15) -> |2-10|/10 = |3-15|/15 = 0.8.
    assert goodness_of_fit(freq, targets, np.ones(2)) == pytest.approx(0.8)
    res = ipu(freq, targets)
    # Constraint 1 scales both to 5; constraint 2 then sees 5 + 10 = 15 and stops.
    np.testing.assert_allclose(res.weights, [5.0, 5.0])
    assert res.iterations == 1
    assert res.history == pytest.approx([0.8, 0.0])
    assert res.converged and not res.stalled


# 2. Household-only controls -------------------------------------------------


def test_household_only_controls_fit_in_one_iteration():
    freq = np.eye(3)[[0, 0, 1, 2, 2, 2]]
    targets = np.array([10.0, 4.0, 30.0])
    res = ipu(freq, targets)
    assert res.iterations == 1
    np.testing.assert_allclose(freq.T @ res.weights, targets)
    np.testing.assert_allclose(res.weights, [5, 5, 4, 10, 10, 10])


# 3. Last constraint exact ---------------------------------------------------


def test_last_constraint_holds_after_each_iteration():
    rng = np.random.default_rng(20261117)
    freq, targets, _ = planted_instance(rng)
    for n in (1, 2, 5):
        res = ipu(freq, targets, max_iter=n, tol=0.0, stall_tol=None, strict=False)
        assert res.iterations == n
        assert freq[:, -1] @ res.weights == pytest.approx(targets[-1], rel=1e-12)
        assert len(res.history) == n + 1


# 4. Zeros and nonnegativity -------------------------------------------------


def test_zero_weights_stay_zero_and_weights_nonnegative():
    rng = np.random.default_rng(7)
    freq, targets, _ = planted_instance(rng)
    init = np.ones(freq.shape[0])
    init[[0, 3]] = 0.0
    res = ipu(freq, targets, init=init, strict=False)
    assert res.weights[0] == 0.0 and res.weights[3] == 0.0
    assert np.all(res.weights >= 0)


# 5. Infeasible controls fail loudly -----------------------------------------


def test_infeasible_controls_stall_and_raise():
    # Every household has exactly two persons, so persons must equal 2 x households.
    freq = np.array([[1.0, 2.0], [1.0, 2.0]])
    targets = np.array([10.0, 30.0])
    with pytest.raises(IPUConvergenceError) as exc:
        ipu(freq, targets)
    assert exc.value.result.stalled
    assert not exc.value.result.converged


def test_infeasible_controls_non_strict_returns_unconverged():
    freq = np.array([[1.0, 2.0], [1.0, 2.0]])
    res = ipu(freq, np.array([10.0, 30.0]), strict=False)
    assert not res.converged
    assert res.max_rel_residual > 0.1
    _, ok, _ = entropy_weights(freq, np.array([10.0, 30.0]))
    assert not ok


# 6. Empty constraint --------------------------------------------------------


def test_constraint_with_no_contributing_household_is_rejected():
    freq = np.array([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(InfeasibleConstraintError):
        ipu(freq, np.array([10.0, 5.0]))


# 7-8. Reference solver ------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_entropy_reference_fits_planted_instances(seed):
    rng = np.random.default_rng(seed)
    freq, targets, _ = planted_instance(rng, n_households=60)
    w, ok, resid = entropy_weights(freq, targets)
    assert ok, resid
    np.testing.assert_allclose(freq.T @ w, targets, rtol=1e-9)
    assert np.all(w > 0)


def test_ipu_fit_when_it_converges_matches_reference_constraints():
    freq = np.array([[1.0, 1.0], [1.0, 2.0]])
    targets = np.array([10.0, 15.0])
    w_ref, ok, _ = entropy_weights(freq, targets)
    res = ipu(freq, targets)
    assert ok and res.converged
    # Two households, two independent constraints: the solution is unique.
    np.testing.assert_allclose(res.weights, w_ref, rtol=1e-9)


# 9. The convergence experiment runs and certifies feasibility ---------------


def test_convergence_experiment_row_on_planted_instance():
    rng = np.random.default_rng(3)
    freq, targets, _ = planted_instance(rng, n_households=50)
    row, history = run_instance(freq, targets)
    assert row["reference_feasible"]
    assert len(history) == row["ipu_iterations"] + 1
    assert row["best_delta"] <= row["ipu_final_delta"]
