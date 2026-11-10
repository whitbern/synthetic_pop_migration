"""Maximum-entropy population synthesis (after Wu et al. 2018).

Wu et al. (2018, *ACM TKDD* 12(4), doi:10.1145/3182383) generate a synthetic
population by taking, among all joint distributions that satisfy the known
aggregate constraints, the one with the largest entropy, and then drawing
individuals from it. Nothing is assumed beyond the constraints.

Here the joint is a dense table over categorical attributes. Each constraint
fixes the totals of a set of disjoint *groups of cells* (a margin is the
special case where the groups are the cells of a sub-table; a margin over a
coarsened axis, such as "adult vs child" from an age-band axis, is another).
Maximising entropy subject to linear equality constraints of this kind gives
a log-linear table::

    p(x) = exp( sum_k lambda_k[g_k(x)] )   on the support, 0 elsewhere,

with one parameter per group per constraint. It is found by cyclic
proportional scaling over the constraints from a uniform start (Darroch &
Ratcliff 1972; Csiszár 1975): the I-projection of the uniform distribution
on the constraint set, which is the maximum-entropy solution.

The consequence the study uses: any attribute that no constraint links to
another comes out **independent** of it. In the Sprint 2 setting education
is constrained only through its distribution within education-universe strata
(under 5, 5-17, 18+) taken from the biased sample, so among adults the
synthetic education is exactly independent of origin, sex and age band. That
is the conditional-independence baseline Sprint 4 measures the other designs
against. The analytic tests in ``tests/test_maxent.py`` check it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from synthpop.ipf import InconsistentMarginsError

__all__ = [
    "Constraint",
    "MaxEntConvergenceError",
    "MaxEntResult",
    "entropy",
    "grouped_margin",
    "margin",
    "maxent",
    "draw_population",
]


class MaxEntConvergenceError(RuntimeError):
    """Scaling reached ``max_iter`` without meeting the tolerance."""

    def __init__(self, message: str, result: "MaxEntResult"):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class Constraint:
    """Totals for disjoint groups of cells.

    ``labels`` has the table's shape: the group index of each cell, or -1 for
    cells the constraint does not cover. ``target[g]`` is the required total of
    group ``g``.
    """

    labels: np.ndarray
    target: np.ndarray
    name: str = ""

    @property
    def n_groups(self) -> int:
        return len(self.target)


def margin(shape: Sequence[int], axes: Sequence[int], target, name: str = "") -> Constraint:
    """A margin over ``axes`` (in the order given), as a :class:`Constraint`."""
    return grouped_margin(shape, axes, target, groups=None, name=name)


def grouped_margin(
    shape: Sequence[int],
    axes: Sequence[int],
    target,
    groups: Mapping[int, Sequence[int]] | None = None,
    name: str = "",
) -> Constraint:
    """A margin over ``axes`` where some axes are first coarsened.

    ``groups[a][level]`` gives the coarse category of each level of axis
    ``a`` (or -1 to leave that level out of the constraint). ``target`` has one
    dimension per entry of ``axes``, of the coarse size for grouped axes.
    """
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    axes = tuple(int(a) + ndim if int(a) < 0 else int(a) for a in axes)
    if not axes:
        raise ValueError("a margin must cover at least one axis")
    if len(set(axes)) != len(axes) or any(not 0 <= a < ndim for a in axes):
        raise ValueError(f"bad axes {axes} for a {ndim}-way table")
    groups = {int(a) + ndim if int(a) < 0 else int(a): np.asarray(g, dtype=np.int64) for a, g in (groups or {}).items()}
    for a, g in groups.items():
        if a not in axes:
            raise ValueError(f"grouping given for axis {a}, which is not in the margin")
        if g.shape != (shape[a],):
            raise ValueError(f"grouping for axis {a} has {g.size} entries, expected {shape[a]}")
    maps = [groups.get(a, np.arange(shape[a])) for a in axes]
    coarse = tuple(int(m.max()) + 1 for m in maps)
    target = np.asarray(target, dtype=float)
    if target.shape != coarse:
        raise ValueError(f"margin {name or axes} has shape {target.shape}, expected {coarse}")

    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    codes = [m[grids[a]] for a, m in zip(axes, maps)]
    covered = np.all([c >= 0 for c in codes], axis=0)
    labels = np.full(shape, -1, dtype=np.int64)
    labels[covered] = np.ravel_multi_index([c[covered] for c in codes], coarse)
    return Constraint(labels, target.ravel(), name or f"axes{axes}")


@dataclass(frozen=True)
class MaxEntResult:
    """Fitted table plus diagnostics.

    ``max_deviation`` is the largest |group total - target| over all
    constraints, divided by the table total; ``history`` holds it after each
    cycle and ``entropy_history`` the entropy (nats) of the normalised table.
    """

    table: np.ndarray
    iterations: int
    converged: bool
    max_deviation: float
    entropy: float
    history: list[float] = field(default_factory=list)
    entropy_history: list[float] = field(default_factory=list)


def entropy(table: np.ndarray) -> float:
    """Shannon entropy (nats) of ``table`` normalised to sum to one."""
    p = np.asarray(table, dtype=float).ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def _validate(shape, constraints, support):
    prepared = []
    for k, c in enumerate(constraints):
        labels = np.asarray(c.labels, dtype=np.int64)
        target = np.asarray(c.target, dtype=float)
        name = c.name or f"constraint {k}"
        if labels.shape != shape:
            raise ValueError(f"{name}: labels have shape {labels.shape}, expected {shape}")
        if target.ndim != 1 or labels.max(initial=-1) >= len(target) or labels.min(initial=0) < -1:
            raise ValueError(f"{name}: labels must be -1 or index a group of target")
        if not np.all(np.isfinite(target)):
            raise ValueError(f"{name}: target contains non-finite values")
        if np.any(target < 0):
            raise ValueError(f"{name}: target contains negative values")
        cov = labels >= 0
        # Positive target on a group with no admissible cell cannot be met.
        room = np.bincount(labels[cov & support], minlength=len(target))
        bad = np.flatnonzero((room == 0) & (target > 0))
        if bad.size:
            raise InconsistentMarginsError(
                f"{name}: group {int(bad[0])} has a positive target but no cell in the support (structural zero)"
            )
        prepared.append((name, labels, target, cov))
    return prepared


def _check_pairs(prepared, support, rtol, scale):
    """Two constraints must agree on every block of cells both fully determine.

    Link groups of A and B that share a support cell; each connected block whose
    A-groups and B-groups cover the same support cells has its total fixed by
    both, so the two totals must match. (Necessary, not sufficient.)
    """
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            na, la, ta, ca = prepared[i]
            nb, lb, tb, cb = prepared[j]
            parent = list(range(len(ta) + len(tb)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            both = ca & cb & support
            for ga, gb in set(zip(la[both].tolist(), lb[both].tolist())):
                ra, rb = find(ga), find(len(ta) + gb)
                if ra != rb:
                    parent[ra] = rb
            # A block is closed when no A-group cell lies outside B's coverage and vice versa.
            open_ = set()
            for g in np.unique(la[ca & ~cb & support]):
                open_.add(find(int(g)))
            for g in np.unique(lb[cb & ~ca & support]):
                open_.add(find(len(ta) + int(g)))
            sums: dict[int, list[float]] = {}
            for g in np.unique(la[both]):
                sums.setdefault(find(int(g)), [0.0, 0.0])
            for g in range(len(ta)):
                r = find(g)
                if r in sums:
                    sums[r][0] += ta[g]
            for g in range(len(tb)):
                r = find(len(ta) + g)
                if r in sums:
                    sums[r][1] += tb[g]
            for r, (sa, sb) in sums.items():
                if r not in open_ and abs(sa - sb) > rtol * scale:
                    raise InconsistentMarginsError(
                        f"{na} and {nb} fix the same cells to different totals ({sa:g} vs {sb:g})"
                    )


def maxent(
    shape: Sequence[int],
    constraints: Sequence[Constraint],
    *,
    support: np.ndarray | None = None,
    total: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 5000,
    strict: bool = True,
    consistency_rtol: float = 1e-8,
) -> MaxEntResult:
    """Maximum-entropy table over ``shape`` satisfying ``constraints``.

    Parameters
    ----------
    shape
        Table shape (one axis per attribute).
    constraints
        :class:`Constraint` objects, e.g. from :func:`margin` and
        :func:`grouped_margin`.
    support
        Boolean array of admissible cells (default: all). Cells outside it are
        structural zeros and stay exactly zero.
    total
        Table total. Inferred from any constraint that covers the whole
        support; required if none does.
    tol
        Convergence threshold on ``max |group total - target| / total``.
    max_iter
        Maximum number of full cycles over the constraints.
    strict
        Raise :class:`MaxEntConvergenceError` if ``max_iter`` is reached
        (default); otherwise return the table with ``converged=False``.
    """
    shape = tuple(int(n) for n in shape)
    if not shape:
        raise ValueError("table must have at least one axis")
    support = np.ones(shape, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    if support.shape != shape:
        raise ValueError(f"support has shape {support.shape}, expected {shape}")
    if not support.any():
        raise ValueError("support is empty")
    if len(constraints) == 0 and total is None:
        raise ValueError("at least one constraint or an explicit total is required")

    prepared = _validate(shape, constraints, support)
    full = [t.sum() for _, _, t, cov in prepared if np.all(cov[support])]
    if total is None:
        if not full:
            raise ValueError("no constraint covers the whole support; pass total=")
        total = float(full[0])
    total = float(total)
    scale = max(total, 1.0)
    for t in full:
        if abs(t - total) > consistency_rtol * scale:
            raise InconsistentMarginsError(f"constraint totals differ: {sorted(set(map(float, full)))} vs total {total:g}")
    _check_pairs(prepared, support, consistency_rtol, scale)
    # The total is itself a constraint (it matters when no other covers every cell).
    whole = np.where(support, 0, -1).astype(np.int64)
    prepared.append(("total", whole, np.array([total]), whole >= 0))

    if total == 0:
        z = np.zeros(shape)
        return MaxEntResult(z, 0, True, 0.0, 0.0, [0.0], [0.0])

    table = np.where(support, total / support.sum(), 0.0)

    def deviation(t):
        if not prepared:
            return 0.0
        return max(
            float(np.max(np.abs(np.bincount(lab[cov], t[cov], minlength=len(tg)) - tg), initial=0.0))
            for _, lab, tg, cov in prepared
        ) / total

    dev = deviation(table)
    history, ent_history = [dev], [entropy(table)]
    iterations = 0
    while dev > tol and iterations < max_iter:
        for _, lab, tg, cov in prepared:
            current = np.bincount(lab[cov], table[cov], minlength=len(tg))
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(current > 0, tg / current, 0.0)
            table[cov] *= factor[lab[cov]]
        iterations += 1
        dev = deviation(table)
        history.append(dev)
        ent_history.append(entropy(table))

    converged = dev <= tol
    result = MaxEntResult(table, iterations, converged, dev, ent_history[-1], history, ent_history)
    if not converged and strict:
        raise MaxEntConvergenceError(
            f"max-entropy scaling did not converge in {max_iter} cycles (max deviation {dev:g} > tol {tol:g})",
            result,
        )
    return result


def draw_population(table: np.ndarray, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
    """Draw individuals from the fitted table: an ``(n, ndim)`` array of category codes.

    Each person is an independent draw from the normalised table (Wu et al.
    sample individuals from the fitted distribution). ``size`` defaults to the
    table total, rounded.
    """
    table = np.asarray(table, dtype=float)
    n = int(round(table.sum())) if size is None else int(size)
    counts = rng.multinomial(n, table.ravel() / table.sum())
    cells = np.repeat(np.arange(table.size), counts)
    rng.shuffle(cells)
    return np.column_stack(np.unravel_index(cells, table.shape)) if n else np.zeros((0, table.ndim), dtype=np.int64)
