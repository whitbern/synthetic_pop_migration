"""Iterative proportional updating (IPU) for joint household and person controls.

IPU (Ye, Konduri, Pendyala, Sana & Waddell 2009) reweights a sample of
households so that weighted counts match control totals at two levels at once:
household types (one per household) and person types (counted over the
household's members). The input is a frequency matrix ``D`` (households x
constraints), where ``D[i, j]`` is the number of units of constraint type ``j``
in household ``i``, and a vector of control totals ``c``.

One iteration visits every constraint ``j`` in order and rescales the weights
of the households that contribute to it,

    w_i <- w_i * c_j / sum_k D[k, j] w_k      for every i with D[i, j] > 0,

so constraint ``j`` holds exactly right after its own update. Fit is measured
with Ye et al.'s goodness-of-fit statistic

    delta = mean_j | sum_i D[i, j] w_i - c_j | / c_j,

recorded after every iteration in ``IPUResult.history``. That trace is the
instrumented loop needed to check the convergence analysis of Ye, Tian, Lv,
Li & Wang (2022), who prove that IPU cannot in general converge to a weight
vector satisfying both levels of constraints.

Ye et al. (2009) stop when ``delta`` stops improving. This implementation
records that stall separately from fit: a run that stalls before every
constraint holds is reported as not converged (``strict=True`` raises),
following the same fail-loudly rule as ``synthpop.ipf``.

``entropy_weights`` is a reference solver for the same constraints (the
minimum-KL reweighting, solved by Newton's method on the dual). It returns an
exact fit whenever one with positive weights exists, so it certifies that a
test instance is feasible. That makes an IPU failure on that instance a property
of IPU, not of the instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "IPUConvergenceError",
    "IPUResult",
    "InfeasibleConstraintError",
    "entropy_weights",
    "goodness_of_fit",
    "ipu",
]


class InfeasibleConstraintError(ValueError):
    """A control total is positive but no household contributes to it."""


class IPUConvergenceError(RuntimeError):
    """IPU stopped (stall or ``max_iter``) before every constraint held to tolerance."""

    def __init__(self, message: str, result: "IPUResult"):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class IPUResult:
    """Weights plus convergence diagnostics.

    ``iterations`` counts full passes over the constraints. ``delta`` is the
    goodness-of-fit statistic at the end; ``max_rel_residual`` is the largest
    ``|fitted - target| / target`` over constraints. ``history`` holds
    ``delta`` before the first iteration and after each one. ``stalled`` is
    true when the run stopped because ``delta`` stopped improving.
    """

    weights: np.ndarray
    iterations: int
    converged: bool
    stalled: bool
    delta: float
    max_rel_residual: float
    history: list[float] = field(default_factory=list)


def _validate(freq: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq = np.asarray(freq, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if freq.ndim != 2:
        raise ValueError("frequency matrix must be 2-D (households x constraints)")
    if targets.shape != (freq.shape[1],):
        raise ValueError(f"expected {freq.shape[1]} targets, got shape {targets.shape}")
    if not (np.all(np.isfinite(freq)) and np.all(np.isfinite(targets))):
        raise ValueError("inputs contain non-finite values")
    if np.any(freq < 0) or np.any(targets < 0):
        raise ValueError("inputs contain negative values")
    if np.any(targets == 0):
        raise ValueError("zero control totals are not supported (goodness of fit divides by c_j)")
    empty = np.flatnonzero(freq.sum(axis=0) == 0)
    if empty.size:
        raise InfeasibleConstraintError(
            f"constraint {int(empty[0])} has target {targets[empty[0]]:g} but no household contributes to it"
        )
    return freq, targets


def goodness_of_fit(freq: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    """Ye et al. (2009) delta: mean relative absolute deviation over constraints."""
    fitted = np.asarray(freq, dtype=float).T @ np.asarray(weights, dtype=float)
    return float(np.mean(np.abs(fitted - targets) / targets))


def _max_rel_residual(freq: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
    return float(np.max(np.abs(freq.T @ weights - targets) / targets))


def ipu(
    freq: np.ndarray,
    targets: np.ndarray,
    *,
    init: np.ndarray | None = None,
    tol: float = 1e-10,
    stall_tol: float | None = 1e-14,
    max_iter: int = 10_000,
    strict: bool = True,
) -> IPUResult:
    """Reweight households by iterative proportional updating.

    Parameters
    ----------
    freq
        Households x constraints frequency matrix. Household-type columns are
        0/1; person-type columns count members of that type.
    targets
        Control total for each column, all positive.
    init
        Starting weights (default all ones). Zero weights stay zero.
    tol
        Convergence threshold on the largest relative residual over constraints.
    stall_tol
        Stop when one iteration improves ``delta`` by less than this (Ye et al.
        2009's stopping rule). ``None`` disables it, so only ``tol`` and
        ``max_iter`` stop the loop.
    max_iter
        Maximum number of full passes over the constraints.
    strict
        If true (default), raise ``IPUConvergenceError`` when the loop stops
        without meeting ``tol``. If false, return with ``converged=False``.
    """
    freq, targets = _validate(freq, targets)
    weights = np.ones(freq.shape[0]) if init is None else np.asarray(init, dtype=float).copy()
    if weights.shape != (freq.shape[0],) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("init must be a finite nonnegative vector with one weight per household")

    delta = goodness_of_fit(freq, targets, weights)
    history = [delta]
    resid = _max_rel_residual(freq, targets, weights)
    iterations = 0
    stalled = False
    while resid > tol and iterations < max_iter:
        for j in range(freq.shape[1]):
            col = freq[:, j]
            current = float(col @ weights)
            if current > 0:
                weights[col > 0] *= targets[j] / current
        iterations += 1
        new_delta = goodness_of_fit(freq, targets, weights)
        history.append(new_delta)
        resid = _max_rel_residual(freq, targets, weights)
        improved = delta - new_delta
        delta = new_delta
        if resid > tol and stall_tol is not None and improved < stall_tol:
            stalled = True
            break

    converged = resid <= tol
    result = IPUResult(weights, iterations, converged, stalled, delta, resid, history)
    if not converged and strict:
        why = "stalled" if stalled else f"reached max_iter={max_iter}"
        raise IPUConvergenceError(
            f"IPU {why} after {iterations} iterations with max relative residual {resid:g} > tol {tol:g} "
            f"(delta {delta:g})",
            result,
        )
    return result


def entropy_weights(
    freq: np.ndarray,
    targets: np.ndarray,
    *,
    init: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, bool, float]:
    """Minimum-KL weights meeting every constraint exactly, if any exist.

    Solves min sum_i w_i log(w_i / w0_i) - w_i + w0_i subject to D^T w = c by
    damped Newton on the dual, with w_i = w0_i exp((D lambda)_i). Returns
    ``(weights, converged, max_rel_residual)``. Convergence certifies that a
    positive exact solution exists; failure suggests (but does not prove) that
    none does.
    """
    freq, targets = _validate(freq, targets)
    w0 = np.ones(freq.shape[0]) if init is None else np.asarray(init, dtype=float)
    lam = np.zeros(freq.shape[1])

    def dual(l: np.ndarray) -> tuple[float, np.ndarray]:
        w = w0 * np.exp(np.clip(freq @ l, -700, 700))
        return float(w.sum() - targets @ l), w

    g, w = dual(lam)
    resid = _max_rel_residual(freq, targets, w)
    for _ in range(max_iter):
        if resid <= tol:
            break
        grad = freq.T @ w - targets
        hess = freq.T @ (w[:, None] * freq)
        step = np.linalg.lstsq(hess, -grad, rcond=None)[0]
        t = 1.0
        while t > 1e-12:
            g_new, w_new = dual(lam + t * step)
            if g_new <= g + 1e-4 * t * float(grad @ step):
                break
            t *= 0.5
        lam, g, w = lam + t * step, g_new, w_new
        resid = _max_rel_residual(freq, targets, w)
    return w, resid <= tol, resid
