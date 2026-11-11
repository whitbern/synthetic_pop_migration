"""The selection mechanism: who from the ACS truth enters the synthetic flow.

Each truth record ``i`` (base weight ``w_i``) gets a selection propensity

    p_i  ∝  exp( γ·g_i + θ·f_i )

* ``f_i`` are the **calibrated** features: origin indicators (all but the
  first), male, age, and an elderly indicator. Their coefficients ``θ`` are
  solved for so that the flow, weighted by ``w_i p_i``, reproduces the target
  composition exactly: origin shares, male share, mean age, elderly share.
* ``g_i`` are the **free** features (by default the SES rank, centred). Their
  coefficients ``γ`` are set by hand. They are the *selection strength*.

The calibration is exponential tilting (a maximum entropy problem): among all
reweightings of the truth that hit the targets, it picks the one closest to
the truth in KL divergence, after the free tilt ``γ·g`` has been applied. The
dual objective ``log Σ w_i exp(η_i) − θ·t`` is convex, so Newton's method
finds the unique solution when the targets lie strictly inside what the truth
can reach; when they do not, calibration fails loudly instead of returning a
flow that silently misses.

Three targets do not pin down a selection function. Many values of ``γ``
match the same composition while giving different joint behaviour (e.g. a
different education mix). :func:`sweep_selection_strength` makes that
visible; the plan reports results across a range of ``γ``, not one value.

Probabilities are scaled so the largest is 1. Only relative propensities
matter for the flow's composition; the flow's size is set separately when it
is drawn (:func:`realize_flow`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from synthpop.truth import Records, n_records, subset

__all__ = [
    "CalibrationError",
    "DarienTargets",
    "SelectionResult",
    "calibrate_selection",
    "composition",
    "darien_features",
    "draw_survey_sample",
    "realize_flow",
    "select_flow",
    "survey_sample_keys",
    "sweep_selection_strength",
]


class CalibrationError(RuntimeError):
    """The selection function could not reproduce the targets."""

    def __init__(self, message: str, result: "SelectionResult | None" = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class DarienTargets:
    """Target composition of the flow.

    ``origin_shares`` maps origin-group names to shares summing to 1, in the
    same groups as the truth. ``tolerance`` is the largest absolute gap
    (share points for shares, years for mean age) that counts as a match.
    """

    origin_shares: Mapping[str, float]
    male_share: float = 0.715
    mean_age: float = 31.0
    elderly_share: float = 0.005
    elderly_age: int = 65
    tolerance: Mapping[str, float] = field(
        default_factory=lambda: {"share": 0.005, "mean_age": 0.25}
    )

    def __post_init__(self):
        total = sum(self.origin_shares.values())
        if abs(total - 1) > 1e-6:
            raise ValueError(f"origin shares sum to {total}, not 1")
        for name, v in [("male_share", self.male_share), ("elderly_share", self.elderly_share)]:
            if not 0 < v < 1:
                raise ValueError(f"{name}={v} must be strictly between 0 and 1")
        if any(not 0 < s < 1 for s in self.origin_shares.values()) and len(self.origin_shares) > 1:
            raise ValueError("every origin share must be strictly between 0 and 1")


@dataclass
class SelectionResult:
    """Calibrated selection function plus how close it got."""

    probability: np.ndarray  # p_i, max 1
    flow_weight: np.ndarray  # w_i * p_i
    theta: dict[str, float]  # calibrated coefficients, original feature scale
    gamma: dict[str, float]  # free coefficients as given
    targets: dict[str, float]
    achieved: dict[str, float]
    gaps: dict[str, float]
    iterations: int
    converged: bool
    within_tolerance: bool
    history: list[float] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"{'moment':<24}{'target':>10}{'achieved':>12}{'gap':>12}"]
        for k in self.targets:
            lines.append(f"{k:<24}{self.targets[k]:>10.4f}{self.achieved[k]:>12.4f}{self.gaps[k]:>12.2e}")
        lines.append(f"converged={self.converged} within_tolerance={self.within_tolerance} iterations={self.iterations}")
        return "\n".join(lines)


def darien_features(
    records: Records, origin_names: Sequence[str], targets: DarienTargets
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Calibrated features and their targets for the Darién composition.

    The first origin group is the reference and gets no indicator (its share
    follows from the others).
    """
    missing = set(origin_names) - set(targets.origin_shares)
    extra = set(targets.origin_shares) - set(origin_names)
    if missing or extra:
        raise ValueError(f"origin shares do not match the truth's groups (missing {missing}, extra {extra})")
    feats: dict[str, np.ndarray] = {}
    tgts: dict[str, float] = {}
    for k, name in enumerate(origin_names):
        if k == 0:
            continue
        feats[f"origin={name}"] = (records["origin"] == k).astype(float)
        tgts[f"origin={name}"] = float(targets.origin_shares[name])
    feats["male"] = records["male"].astype(float)
    tgts["male"] = targets.male_share
    feats["age"] = records["age"].astype(float)
    tgts["age"] = targets.mean_age
    feats[f"age>={targets.elderly_age}"] = (records["age"] >= targets.elderly_age).astype(float)
    tgts[f"age>={targets.elderly_age}"] = targets.elderly_share
    return feats, tgts


def calibrate_selection(
    weight: np.ndarray,
    features: Mapping[str, np.ndarray],
    targets: Mapping[str, float],
    free: Mapping[str, tuple[np.ndarray, float]] | None = None,
    *,
    tolerance: Mapping[str, float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    strict: bool = True,
) -> SelectionResult:
    """Solve for ``θ`` so that the ``w·exp(γ·g + θ·f)``-weighted means of ``f`` equal ``targets``.

    Parameters
    ----------
    weight
        Base weights ``w_i`` (PWGTP).
    features, targets
        Calibrated features by name and their target weighted means.
    free
        ``name -> (g, γ)``: free features and their fixed coefficients.
    tolerance
        Gap that counts as a match, by name; names not given use ``1e-6``.
        ``within_tolerance`` reports whether every gap is inside it.
    tol
        Newton stopping threshold on the largest standardised gap.
    strict
        If true, raise :class:`CalibrationError` when Newton does not converge
        or a gap exceeds its tolerance; otherwise return the result with
        ``converged`` / ``within_tolerance`` false so the gap can be reported.
    """
    w = np.asarray(weight, dtype=float)
    if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and not all zero")
    names = list(features)
    if set(names) != set(targets):
        raise ValueError("features and targets must have the same names")
    free = dict(free or {})

    offset = np.zeros_like(w)
    for g, coef in free.values():
        offset += float(coef) * np.asarray(g, dtype=float)

    F = np.column_stack([np.asarray(features[k], dtype=float) for k in names]) if names else np.zeros((len(w), 0))
    t = np.array([float(targets[k]) for k in names])

    _check_reachable(F, w, t, names)

    # Standardise so Newton steps are well scaled; θ is mapped back at the end.
    mu = (w @ F) / w.sum()
    sd = np.sqrt((w @ (F - mu) ** 2) / w.sum())
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd
    tz = (t - mu) / sd

    def moments(theta):
        eta = offset + Z @ theta
        eta -= eta.max()
        m = w * np.exp(eta)
        m /= m.sum()
        mean = m @ Z
        return m, mean

    def dual(theta):
        eta = offset + Z @ theta
        top = eta.max()
        return top + np.log(np.sum(w * np.exp(eta - top))) - theta @ tz

    theta = np.zeros(len(names))
    history = []
    iterations = 0
    converged = False
    for iterations in range(1, max_iter + 1):
        m, mean = moments(theta)
        grad = mean - tz
        gap = float(np.max(np.abs(grad))) if len(grad) else 0.0
        history.append(gap)
        if gap <= tol:
            converged = True
            iterations -= 1
            break
        Zc = Z - mean
        H = (Zc * m[:, None]).T @ Zc
        try:
            step = np.linalg.solve(H + 1e-12 * np.eye(len(names)), grad)
        except np.linalg.LinAlgError:
            break
        # Backtracking line search on the convex dual.
        f0 = dual(theta)
        s = 1.0
        while s > 1e-10 and dual(theta - s * step) > f0 - 1e-4 * s * (grad @ step):
            s /= 2
        if s <= 1e-10:
            break
        theta = theta - s * step
    else:
        m, mean = moments(theta)
        history.append(float(np.max(np.abs(mean - tz))) if len(names) else 0.0)

    m, _ = moments(theta)
    eta = offset + Z @ theta
    p = np.exp(eta - eta.max())
    achieved_raw = m @ F if names else np.zeros(0)
    achieved = {k: float(v) for k, v in zip(names, achieved_raw)}
    gaps = {k: achieved[k] - float(targets[k]) for k in names}
    tolerance = dict(tolerance or {})
    within = all(abs(gaps[k]) <= tolerance.get(k, 1e-6) for k in names)
    theta_orig = {k: float(v) for k, v in zip(names, theta / sd)}
    result = SelectionResult(
        probability=p,
        flow_weight=w * p,
        theta=theta_orig,
        gamma={k: float(c) for k, (_, c) in free.items()},
        targets={k: float(targets[k]) for k in names},
        achieved=achieved,
        gaps=gaps,
        iterations=iterations,
        converged=converged,
        within_tolerance=within,
        history=history,
    )
    if strict and not (converged and within):
        raise CalibrationError(
            "selection calibration did not reach the targets:\n" + result.report(), result
        )
    return result


def _check_reachable(F, w, t, names):
    """Fail early on targets no reweighting of the truth can reach."""
    support = w > 0
    for j, k in enumerate(names):
        col = F[support, j]
        lo, hi = col.min(), col.max()
        if not lo < t[j] < hi:
            raise CalibrationError(
                f"target {k}={t[j]:g} is outside the open range ({lo:g}, {hi:g}) the truth can reach; "
                "exponential tilting cannot hit a boundary value"
            )


def select_flow(
    records: Records,
    origin_names: Sequence[str],
    targets: DarienTargets,
    ses_strength: float = 0.0,
    extra_free: Mapping[str, tuple[np.ndarray, float]] | None = None,
    strict: bool = True,
) -> SelectionResult:
    """Calibrate the Darién selection function at a given SES selection strength.

    ``ses_strength`` is ``γ`` on the centred SES rank: the log-ratio of
    selection propensity between the top and bottom of the truth's income
    distribution, other features held fixed. Negative values favour poorer
    people.
    """
    feats, tgts = darien_features(records, origin_names, targets)
    free = {"ses": (records["ses"] - 0.5, ses_strength)}
    free.update(extra_free or {})
    tolerance = {k: targets.tolerance["share"] for k in tgts if k != "age"}
    tolerance["age"] = targets.tolerance["mean_age"]
    return calibrate_selection(records["weight"], feats, tgts, free, tolerance=tolerance, strict=strict)


def composition(records: Records, weight: np.ndarray, origin_names: Sequence[str], elderly_age: int = 65) -> dict[str, float]:
    """Weighted composition summary: the calibration targets plus SES and education."""
    w = np.asarray(weight, dtype=float)
    w = w / w.sum()
    out = {f"origin={name}": float(w[records["origin"] == k].sum()) for k, name in enumerate(origin_names)}
    out["male"] = float(w @ records["male"])
    out["mean_age"] = float(w @ records["age"])
    out[f"age>={elderly_age}"] = float(w[records["age"] >= elderly_age].sum())
    out["mean_ses"] = float(w @ records["ses"])
    for level in range(5):
        out[f"edattain={level}"] = float(w[records["edattain"] == level].sum())
    return out


def sweep_selection_strength(
    records: Records,
    origin_names: Sequence[str],
    targets: DarienTargets,
    strengths: Sequence[float],
) -> list[dict[str, float]]:
    """Calibrate at each SES strength and report the resulting composition.

    Every row matches the same targets; the columns that move (``mean_ses``,
    ``edattain=*``) show how underdetermined the calibration is.
    """
    rows = []
    for g in strengths:
        res = select_flow(records, origin_names, targets, ses_strength=g, strict=False)
        row = {"ses_strength": float(g), "converged": float(res.converged), "within_tolerance": float(res.within_tolerance)}
        row.update(composition(records, res.flow_weight, origin_names, targets.elderly_age))
        rows.append(row)
    return rows


def realize_flow(records: Records, flow_weight: np.ndarray, size: int, rng: np.random.Generator) -> tuple[Records, np.ndarray]:
    """Draw a finite flow of ``size`` people from the truth, with replacement.

    Each person is a copy of a truth record chosen with probability
    proportional to ``flow_weight``. Returns the flow records (with a
    ``truth_index`` column pointing back to the truth row and a ``flow_id``)
    and the truth indices. The flow is the planted truth every design is
    scored against.
    """
    q = np.asarray(flow_weight, dtype=float)
    q = q / q.sum()
    counts = rng.multinomial(int(size), q)
    idx = np.repeat(np.arange(len(q)), counts)
    flow = subset(records, idx)
    flow["truth_index"] = idx
    flow["flow_id"] = np.arange(len(idx))
    return flow, idx


def draw_survey_sample(
    flow: Records,
    n: int,
    tilt: Mapping[str, tuple[np.ndarray, float]] | None,
    rng: np.random.Generator,
) -> tuple[Records, np.ndarray]:
    """A biased sample of ``n`` people from the flow, without replacement.

    ``tilt`` maps a name to ``(feature, coefficient)``; person ``j`` is drawn
    with propensity ``exp(Σ coef·feature_j)``, using the Efraimidis–Spirakis
    weighted sampling scheme. This stands in for the BPM: a non-probability
    survey whose inclusion propensities the designs are never told. An empty
    tilt gives a simple random sample.
    """
    N = n_records(flow)
    if not 0 < n <= N:
        raise ValueError(f"sample size {n} must be in 1..{N}")
    keys = survey_sample_keys(flow, tilt, rng)
    idx = np.sort(np.argpartition(keys, n - 1)[:n])
    return subset(flow, idx), idx


def survey_sample_keys(
    flow: Records,
    tilt: Mapping[str, tuple[np.ndarray, float]] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Efraimidis–Spirakis keys for :func:`draw_survey_sample`: smallest ``n`` = the sample.

    Exposed so that nested samples can be taken: the ``n`` smallest keys of
    any subset of the flow are a tilted sample of size ``n`` from that subset,
    and the samples for ``n < m`` are contained in the one for ``m``.
    """
    N = n_records(flow)
    eta = np.zeros(N)
    for feat, coef in (tilt or {}).values():
        eta += float(coef) * np.asarray(feat, dtype=float)
    # key_j = log(U_j) / w_j; the n largest keys are a weighted sample without
    # replacement. Work in logs: log(-log U) - eta, smallest n.
    u = rng.random(N)
    return np.log(-np.log(u)) - eta
