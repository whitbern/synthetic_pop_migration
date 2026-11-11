"""First-pass scores of a synthetic table against the planted truth (Sprint 3).

Sprint 4 builds the full metric battery. This module holds the few scores the
Sprint 3 comparison and the crossover experiment need, all computed *within
each origin group* (where pooled and separate estimation should differ) and
on adults, because "less than primary" among children is mostly people who
have not finished school yet.

Every design meets the control margins (origin x sex, origin x age band)
exactly when it converges, so differences between designs live in the
planted attribute: education, and its joint with sex and age band.

Scores, for origin group *o*, over adult cells *c* = (sex, age band >= 18,
EDATTAIN):

* ``srmse``: standardized RMSE, sqrt(mean_c (x_c - t_c)^2) / mean_c t_c, of
  the synthetic counts *x* against the truth counts *t* (Knudsen & Fotheringham
  1986; the usual form in population synthesis).
* ``tvd_edattain``: total variation distance between the synthetic and true
  adult education distributions, 1/2 sum_e |p_e - q_e|.
* ``univ_error``: synthetic minus true adult university share.
* ``false_zeros``: adult cells that are zero in the synthetic table but
  positive in the truth (Auld, Mohammadian & Wies 2009). IPF cannot fill
  them: a zero in the seed stays zero.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from synthpop.truth import AGE_BANDS

__all__ = ["ADULT_AGE", "adult_bands", "heterogeneity", "score_origin", "score_table", "srmse", "tvd"]

ADULT_AGE = 18


def adult_bands(band_edges: Sequence[int] = AGE_BANDS, adult_age: int = ADULT_AGE) -> np.ndarray:
    """Indices of the age bands that start at or above ``adult_age``."""
    return np.flatnonzero(np.asarray(band_edges) >= adult_age)


def srmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Standardized root mean squared error over the cells of ``truth``."""
    est, t = np.asarray(estimate, dtype=float).ravel(), np.asarray(truth, dtype=float).ravel()
    mean = t.mean()
    if mean == 0:
        return float("nan")
    return float(np.sqrt(np.mean((est - t) ** 2)) / mean)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two (unnormalized) distributions."""
    p, q = np.asarray(p, dtype=float).ravel(), np.asarray(q, dtype=float).ravel()
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def score_origin(estimate: np.ndarray, truth: np.ndarray, variables: Sequence[str], origin: int,
                 attribute: str = "edattain", top_level: int = 4,
                 band_edges: Sequence[int] = AGE_BANDS) -> dict:
    """Scores for one origin group. ``estimate`` and ``truth`` share ``variables`` as axes."""
    variables = list(variables)
    o, a, e = variables.index("origin"), variables.index("age_band"), variables.index(attribute)
    adults = adult_bands(band_edges)
    est = np.take(np.take(np.asarray(estimate, dtype=float), [origin], axis=o), adults, axis=a)
    tru = np.take(np.take(np.asarray(truth, dtype=float), [origin], axis=o), adults, axis=a)
    if not np.all(np.isfinite(est)):
        return {"srmse": None, "tvd_edattain": None, "univ_error": None, "false_zeros": None,
                "true_cells": int((tru > 0).sum())}
    others = tuple(x for x in range(est.ndim) if x != e)
    pe, qe = est.sum(axis=others), tru.sum(axis=others)
    return {
        "srmse": srmse(est, tru),
        "tvd_edattain": tvd(pe, qe),
        "univ_error": float(pe[top_level] / pe.sum() - qe[top_level] / qe.sum()) if pe.sum() and qe.sum() else None,
        "false_zeros": int(((est == 0) & (tru > 0)).sum()),
        "true_cells": int((tru > 0).sum()),
    }


def score_table(estimate: np.ndarray, truth: np.ndarray, variables: Sequence[str],
                origin_names: Sequence[str], band_edges: Sequence[int] = AGE_BANDS) -> dict[str, dict]:
    """:func:`score_origin` for every origin group, keyed by name."""
    return {name: score_origin(estimate, truth, variables, k, band_edges=band_edges)
            for k, name in enumerate(origin_names)}


def heterogeneity(truth: np.ndarray, variables: Sequence[str], origin: int, attribute: str = "edattain",
                  band_edges: Sequence[int] = AGE_BANDS) -> dict:
    """How far one origin group's adult education is from the pool's, in the truth.

    * ``tvd_edattain_vs_pool``: TVD between the group's adult education
      distribution and that of all adults (every origin, the group included).
    * ``pooled_limit_tvd``: TVD, over the group's adult sex x age band x
      education cells, between the truth and what pooled IPF gives the group
      when its seed is the *whole truth* (a perfect, infinite sample): the
      pool's table fitted to the group's own margins on every control that is
      not origin. This is the bias pooled estimation keeps however large the
      sample, so it is the natural x-axis for the crossover. Assumes the
      default controls: one-way margins of every non-origin control within
      origin.
    """
    from synthpop.ipf import ipf

    variables = list(variables)
    o, a, e = variables.index("origin"), variables.index("age_band"), variables.index(attribute)
    t = np.moveaxis(np.asarray(truth, dtype=float), [o, e], [0, -1])  # origin first, attribute last
    group, pool = t[origin], t.sum(axis=0)
    ctrl_axes = range(group.ndim - 1)
    margins = [((k,), group.sum(axis=tuple(x for x in range(group.ndim) if x != k))) for k in ctrl_axes]
    limit = ipf(pool, margins, tol=1e-12, max_iter=10_000).table
    a_new = [v for v in variables if v not in ("origin", attribute)].index("age_band")
    adults = adult_bands(band_edges)
    g_ad, l_ad, p_ad = (np.take(x, adults, axis=a_new) for x in (group, limit, pool))
    flat = tuple(range(g_ad.ndim - 1))
    return {
        "tvd_edattain_vs_pool": tvd(g_ad.sum(axis=flat), p_ad.sum(axis=flat)),
        "pooled_limit_tvd": tvd(l_ad, g_ad),
    }
