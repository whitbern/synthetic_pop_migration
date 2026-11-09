"""Iterative proportional fitting (IPF) on dense N-way contingency tables.

IPF (Deming & Stephan 1940) rescales a nonnegative seed table so that its
marginals match a set of target marginals, cycling over the targets until all
of them hold to tolerance. Each target is a sub-table over a subset of the
seed's axes, so one-way, two-way and higher-order margins are all supported.

Properties the implementation relies on, and which the verification tests in
``tests/test_ipf.py`` check analytically:

* The fitted table keeps every interaction of the seed that the targets do not
  constrain (e.g. the cross-product ratios of a 2-way seed under 1-way
  targets). Equivalently, the fit minimises KL divergence to the seed subject
  to the margin constraints.
* Cells that are zero in the seed stay zero (structural zeros).
* Targets that cannot all hold at once are rejected before fitting rather than
  producing a silently wrong table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "IPFResult",
    "InconsistentMarginsError",
    "IPFConvergenceError",
    "ipf",
    "marginal",
]


class InconsistentMarginsError(ValueError):
    """The target margins cannot all be satisfied by any table with the seed's zeros."""


class IPFConvergenceError(RuntimeError):
    """IPF reached ``max_iter`` without meeting the tolerance."""

    def __init__(self, message: str, result: "IPFResult"):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class IPFResult:
    """Fitted table plus convergence diagnostics.

    ``iterations`` counts full cycles over all margins; it is 0 when the seed
    already satisfied every target. ``max_deviation`` is the largest absolute
    difference between a fitted and a target margin cell, divided by the
    target total. ``history`` holds that deviation after each cycle.
    """

    table: np.ndarray
    iterations: int
    converged: bool
    max_deviation: float
    history: list[float] = field(default_factory=list)


Margin = tuple[Sequence[int], np.ndarray]


def marginal(table: np.ndarray, axes: Iterable[int]) -> np.ndarray:
    """Sum ``table`` over every axis not in ``axes``; result axes are in sorted order."""
    keep = set(_normalise_axes(axes, table.ndim))
    drop = tuple(a for a in range(table.ndim) if a not in keep)
    return table.sum(axis=drop)


def _normalise_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
    out = []
    for a in axes:
        a = int(a)
        if a < 0:
            a += ndim
        if not 0 <= a < ndim:
            raise ValueError(f"axis {a} out of range for a {ndim}-way table")
        out.append(a)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {tuple(axes)}")
    if not out:
        raise ValueError("a margin must cover at least one axis")
    return tuple(out)


def _prepare_margins(seed: np.ndarray, margins: Sequence[Margin]) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """Validate targets and return them with axes sorted and targets transposed to match."""
    if len(margins) == 0:
        raise ValueError("at least one target margin is required")
    prepared = []
    for k, (axes, target) in enumerate(margins):
        axes = _normalise_axes(axes, seed.ndim)
        target = np.asarray(target, dtype=float)
        if target.shape != tuple(seed.shape[a] for a in axes):
            raise ValueError(
                f"margin {k} over axes {axes} has shape {target.shape}, "
                f"expected {tuple(seed.shape[a] for a in axes)}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError(f"margin {k} contains non-finite values")
        if np.any(target < 0):
            raise ValueError(f"margin {k} contains negative values")
        order = np.argsort(axes)
        prepared.append((tuple(axes[i] for i in order), np.transpose(target, order)))
    return prepared


def check_consistency(
    seed: np.ndarray,
    margins: Sequence[tuple[tuple[int, ...], np.ndarray]],
    rtol: float = 1e-8,
) -> None:
    """Raise ``InconsistentMarginsError`` if the targets cannot all hold.

    Three necessary conditions are checked:

    1. every margin has the same grand total;
    2. any two margins that share axes agree on their common sub-margin;
    3. no target cell is positive where the seed is zero over that whole cell
       (structural zeros cannot be filled by rescaling).

    These are not sufficient for every zero pattern, so ``ipf`` also raises
    ``IPFConvergenceError`` if fitting stalls.
    """
    totals = np.array([t.sum() for _, t in margins])
    scale = max(float(totals.max()), 1.0)
    if np.any(np.abs(totals - totals[0]) > rtol * scale):
        raise InconsistentMarginsError(f"margin totals differ: {totals.tolist()}")

    for (i, (ax_a, t_a)), (j, (ax_b, t_b)) in combinations(enumerate(margins), 2):
        common = tuple(sorted(set(ax_a) & set(ax_b)))
        if not common:
            continue
        sub_a = t_a.sum(axis=tuple(p for p, a in enumerate(ax_a) if a not in common))
        sub_b = t_b.sum(axis=tuple(p for p, a in enumerate(ax_b) if a not in common))
        gap = float(np.max(np.abs(sub_a - sub_b)))
        if gap > rtol * scale:
            raise InconsistentMarginsError(
                f"margins {i} {ax_a} and {j} {ax_b} disagree on shared axes {common} "
                f"(max abs difference {gap:g})"
            )

    for k, (axes, target) in enumerate(margins):
        support = marginal(seed, axes)
        bad = (support == 0) & (target > 0)
        if np.any(bad):
            cell = tuple(int(c) for c in np.argwhere(bad)[0])
            raise InconsistentMarginsError(
                f"margin {k} {axes} asks for a positive count in cell {cell}, "
                "but the seed is zero across that cell (structural zero)"
            )


def ipf(
    seed: np.ndarray,
    margins: Sequence[Margin],
    *,
    tol: float = 1e-10,
    max_iter: int = 1000,
    strict: bool = True,
    consistency_rtol: float = 1e-8,
) -> IPFResult:
    """Fit ``seed`` to ``margins`` by iterative proportional fitting.

    Parameters
    ----------
    seed
        Nonnegative N-way array. Zeros are treated as structural and preserved.
    margins
        Sequence of ``(axes, target)`` pairs. ``target`` has one dimension per
        entry of ``axes``, in the order given.
    tol
        Convergence threshold on ``max |fitted margin - target| / total``.
    max_iter
        Maximum number of full cycles over the margins.
    strict
        If true (default), raise ``IPFConvergenceError`` when ``max_iter`` is
        reached. If false, return the last table with ``converged=False``.
    consistency_rtol
        Relative tolerance for the pre-fit consistency checks.
    """
    seed = np.asarray(seed, dtype=float)
    if seed.ndim == 0:
        raise ValueError("seed must have at least one axis")
    if not np.all(np.isfinite(seed)):
        raise ValueError("seed contains non-finite values")
    if np.any(seed < 0):
        raise ValueError("seed contains negative values")

    prepared = _prepare_margins(seed, margins)
    check_consistency(seed, prepared, rtol=consistency_rtol)

    total = float(prepared[0][1].sum())
    if total == 0:
        return IPFResult(np.zeros_like(seed), 0, True, 0.0, [0.0])

    # Targets broadcast against the table: summed axes become length 1.
    shaped = []
    for axes, target in prepared:
        shape = tuple(seed.shape[a] if a in axes else 1 for a in range(seed.ndim))
        drop = tuple(a for a in range(seed.ndim) if a not in axes)
        shaped.append((drop, target.reshape(shape)))

    def deviation(t: np.ndarray) -> float:
        return max(
            float(np.max(np.abs(t.sum(axis=drop, keepdims=True) - target))) for drop, target in shaped
        ) / total

    table = seed.copy()
    dev = deviation(table)
    history = [dev]
    iterations = 0
    while dev > tol and iterations < max_iter:
        for drop, target in shaped:
            current = table.sum(axis=drop, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(current > 0, target / current, 0.0)
            table *= factor
        iterations += 1
        dev = deviation(table)
        history.append(dev)

    converged = dev <= tol
    result = IPFResult(table, iterations, converged, dev, history)
    if not converged and strict:
        raise IPFConvergenceError(
            f"IPF did not converge in {max_iter} iterations (max deviation {dev:g} > tol {tol:g})",
            result,
        )
    return result
