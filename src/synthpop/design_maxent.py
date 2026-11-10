"""The maximum-entropy design on the Sprint 2 inputs (margins + biased sample).

Usage::

    python -m synthpop.design_maxent configs/sprint2.json

For each SES selection strength already built by ``python -m synthpop.sprint2
build``, this reads ``margins.json`` (origin, origin x sex, origin x age band;
no education) and ``sample.csv.gz`` (the biased ~1,369 records, which carry
education), fits the maximum-entropy table over origin x sex x age band x
EDATTAIN, draws a synthetic population from it, and writes everything to
``data/sprint2/ses_<g>/maxent_<variant>/``. The flow (the planted truth) is
read only to print a comparison; it never enters the fit.

Constraints:

* every control margin, exactly;
* the distribution of education within each *education stratum* (default:
  under 5, 5-17, 18+, coarsened from the age bands), with the stratum sizes
  from the control margins and the within-stratum shares from the sample.
  ``pooled`` takes those shares over all origins, ``separate`` within each
  origin (falling back to the pooled shares where an origin has nobody in a
  stratum, which is reported).

Everything else is left at maximum entropy, so among adults the synthetic
education is independent of origin, sex and age band (``pooled``) or of sex
and age band given origin (``separate``).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from synthpop.margins import Margins, crosstab
from synthpop.maxent import MaxEntResult, draw_population, grouped_margin, margin, maxent
from synthpop.runlog import log_run
from synthpop.truth import AGE_BANDS, EDATTAIN_LABELS, Records, load_frozen_truth

__all__ = [
    "DEFAULT_STRATA_EDGES",
    "DEFAULT_VARIANTS",
    "MaxEntDesign",
    "conditional_mutual_information",
    "edattain_support",
    "fit_maxent_design",
    "load_margins",
    "strata_of_age_bands",
]

# Education universes: under 5 (NIU or pre-primary), school age, adults.
DEFAULT_STRATA_EDGES: tuple[int, ...] = (0, 5, 18)
DEFAULT_VARIANTS: dict[str, tuple[str, ...]] = {"pooled": (), "separate": ("origin",)}


def load_margins(path: str | Path) -> Margins:
    """Read a ``margins.json`` written by the Sprint 2 build."""
    with open(path) as f:
        d = json.load(f)
    variables = tuple(d["variables"])
    margins, controls = [], []
    for m in d["margins"]:
        controls.append(tuple(m["variables"]))
        margins.append((tuple(variables.index(v) for v in m["variables"]), np.asarray(m["target"], dtype=float)))
    return Margins(variables, {k: int(v) for k, v in d["levels"].items()}, margins, tuple(controls))


def edattain_support(n_age_bands: int = len(AGE_BANDS), n_levels: int = len(EDATTAIN_LABELS)) -> np.ndarray:
    """Admissible (age band, EDATTAIN) cells, from the coding alone.

    EDATTAIN 0 (NIU) means SCHL is missing, which the ACS does only under age
    3, so it is impossible outside the 0-4 band; and a 0-4 year old has at
    most kindergarten or grade 1, which is EDATTAIN 1. Nothing else is ruled
    out (a 17-year-old with a diploma is rare but possible).
    """
    if AGE_BANDS[1] != 5 or n_age_bands != len(AGE_BANDS):
        raise ValueError("edattain_support assumes the default age bands (0-4 first)")
    ok = np.ones((n_age_bands, n_levels), dtype=bool)
    ok[1:, 0] = False
    ok[0, 2:] = False
    return ok


def strata_of_age_bands(strata_edges: Sequence[int] = DEFAULT_STRATA_EDGES, band_edges: Sequence[int] = AGE_BANDS) -> np.ndarray:
    """Education stratum of each age band. Every stratum edge must be an age-band edge."""
    missing = set(strata_edges) - set(band_edges)
    if missing or list(strata_edges) != sorted(strata_edges) or strata_edges[0] != band_edges[0]:
        raise ValueError(f"strata edges {list(strata_edges)} must be increasing age-band edges starting at {band_edges[0]}")
    return np.searchsorted(np.asarray(strata_edges), np.asarray(band_edges), side="right") - 1


@dataclass(frozen=True)
class MaxEntDesign:
    """A fitted maximum-entropy design and what went into it."""

    variables: tuple[str, ...]
    levels: dict[str, int]
    table: np.ndarray
    fit: MaxEntResult
    by: tuple[str, ...]
    strata: np.ndarray
    shares: np.ndarray  # (by..., stratum, attribute) within-cell shares used as constraints
    fallbacks: list[dict] = field(default_factory=list)
    unconstrained_strata: list[int] = field(default_factory=list)


def fit_maxent_design(
    margins: Margins,
    sample: Records,
    attribute: str = "edattain",
    n_levels: int = len(EDATTAIN_LABELS),
    strata_edges: Sequence[int] = DEFAULT_STRATA_EDGES,
    by: Sequence[str] = (),
    *,
    support_rule=None,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> MaxEntDesign:
    """Maximum-entropy joint of the controls and the survey-only ``attribute``.

    The control margins are met exactly; ``attribute`` is constrained only to
    the sample's (weighted) distribution within each education stratum, and
    within each cell of ``by`` if given. ``support_rule(n_age_bands,
    n_levels)`` returns a boolean (age band x level) array of admissible
    cells; the default, :func:`edattain_support`, encodes the logical zeros of
    the EDATTAIN coding. Pass ``support_rule=False`` for no structural zeros.
    """
    if support_rule is None and attribute == "edattain":
        support_rule = edattain_support
    elif support_rule is False:
        support_rule = None
    by = tuple(by)
    if attribute in margins.variables:
        raise ValueError(f"{attribute} is already a control variable")
    if "age_band" not in margins.variables:
        raise ValueError("the control margins must include age_band to define education strata")
    bad = [v for v in by if v not in margins.variables or v == "age_band"]
    if bad:
        raise ValueError(f"'by' variables {bad} must be control variables other than age_band")

    variables = margins.variables + (attribute,)
    levels = dict(margins.levels, **{attribute: int(n_levels)})
    shape = tuple(levels[v] for v in variables)
    ctrl_shape = shape[:-1]
    strata = strata_of_age_bands(strata_edges)
    if len(strata) != levels["age_band"]:
        raise ValueError(f"{len(strata)} age bands in AGE_BANDS but {levels['age_band']} in the margins")
    n_strata = int(strata.max()) + 1

    # Stratum sizes per `by` cell, from the controls alone (exact when a
    # control covers by x age_band, the max-entropy estimate otherwise).
    base = maxent(ctrl_shape, [margin(ctrl_shape, ax, t) for ax, t in margins.margins], tol=tol, max_iter=max_iter)
    by_axes = tuple(margins.variables.index(v) for v in by)
    age_axis = margins.variables.index("age_band")
    keep = tuple(sorted(by_axes + (age_axis,)))
    sizes = base.table.sum(axis=tuple(a for a in range(len(ctrl_shape)) if a not in keep))
    sizes = np.moveaxis(sizes, [keep.index(a) for a in by_axes + (age_axis,)], range(len(keep)))
    grouped = np.zeros(sizes.shape[:-1] + (n_strata,))
    for b, s in enumerate(strata):
        grouped[..., s] += sizes[..., b]

    # Within-stratum shares of the attribute in the sample.
    w = np.asarray(sample.get("weight", np.ones(len(sample[attribute]))), dtype=float)
    counts_banded = crosstab(sample, list(by) + ["age_band", attribute], levels, weight=w)
    counts = np.zeros(counts_banded.shape[:-2] + (n_strata, n_levels))
    for b, s in enumerate(strata):
        counts[..., s, :] += counts_banded[..., b, :]
    pooled = counts.reshape((-1, n_strata, n_levels)).sum(axis=0)
    pooled_n = pooled.sum(axis=-1, keepdims=True)
    # A stratum nobody in the sample falls in is left unconstrained: maximum
    # entropy over the levels its support allows.
    unconstrained = [int(x) for x in np.flatnonzero(pooled_n[:, 0] == 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_shares = np.where(pooled_n > 0, pooled / pooled_n, 0.0)
    cell_n = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(cell_n > 0, counts / cell_n, pooled_shares)
    fallbacks = []
    for idx in np.argwhere(cell_n[..., 0] == 0):
        if by and int(idx[-1]) not in unconstrained:
            cell = {v: int(i) for v, i in zip(by, idx[:-1])}
            cell["stratum"] = int(idx[-1])
            fallbacks.append(cell)

    support = np.ones(shape, dtype=bool)
    if support_rule is not None:
        support = np.broadcast_to(support_rule(levels["age_band"], n_levels).reshape(
            [levels["age_band"] if v == "age_band" else (n_levels if v == attribute else 1) for v in variables]), shape).copy()

    target = grouped[..., None] * shares
    # Drop unconstrained strata from the education constraint.
    kept = [x for x in range(n_strata) if x not in unconstrained]
    remap = np.full(n_strata, -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept))
    strata_c = remap[strata]
    target = target[..., kept, :]
    constraints = [margin(shape, ax, t, name="control " + "x".join(c))
                   for (ax, t), c in zip(margins.margins, margins.controls)]
    attr_axis = len(variables) - 1
    if kept:
        constraints.append(grouped_margin(shape, by_axes + (age_axis, attr_axis), target, groups={age_axis: strata_c},
                                          name=f"{attribute} within strata" + (f" by {'x'.join(by)}" if by else "")))
    fit = maxent(shape, constraints, support=support, tol=tol, max_iter=max_iter)
    return MaxEntDesign(variables, levels, fit.table, fit, by, strata, shares, fallbacks, unconstrained)


def conditional_mutual_information(table: np.ndarray, a: int, b: Sequence[int], given: Sequence[int] = ()) -> float:
    """I(A; B | C) in nats for a table of counts, where B and C may be several axes."""
    t = np.asarray(table, dtype=float)
    t = t / t.sum()
    b, given = tuple(b), tuple(given)
    keep = tuple(sorted((a,) + b + given))
    t = t.sum(axis=tuple(x for x in range(t.ndim) if x not in keep), keepdims=True)

    def m(axes):
        return t.sum(axis=tuple(x for x in range(t.ndim) if x not in axes), keepdims=True)

    p_abc, p_ac, p_bc, p_c = t, m((a,) + given), m(b + given), m(given) if given else t.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p_abc > 0, p_abc * np.log(p_abc * p_c / (p_ac * p_bc)), 0.0)
    return float(max(terms.sum(), 0.0))


# --- Pipeline ----------------------------------------------------------------


def _write_population(path: Path, people: np.ndarray, variables: Sequence[str]) -> None:
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["synth_id", *variables])
        for i, row in enumerate(people.tolist()):
            w.writerow([i, *row])


def _adult_summary(table: np.ndarray, variables: Sequence[str], strata: np.ndarray, origin_names: Sequence[str]) -> dict:
    """Education among adults (top stratum): university share by origin, and its association with each control."""
    age = variables.index("age_band")
    adult = np.flatnonzero(strata == strata.max())
    t = np.take(table, adult, axis=age)
    o, e = variables.index("origin"), len(variables) - 1
    by_origin = t.sum(axis=tuple(x for x in range(t.ndim) if x not in (o, e)))
    uni = by_origin[:, 4] / by_origin.sum(axis=1)
    out = {"university_share_adults": float(t.sum(axis=tuple(range(t.ndim - 1)))[4] / t.sum())}
    out.update({f"university_share_adults|{n}": float(u) for n, u in zip(origin_names, uni)})
    for v in variables[:-1]:
        out[f"MI(edattain;{v})_adults"] = conditional_mutual_information(t, e, (variables.index(v),))
    others = tuple(variables.index(v) for v in variables[:-1] if v != "origin")
    out["MI(edattain;sex,age|origin)_adults"] = conditional_mutual_information(t, e, others, (o,))
    return out


def run(cfg: Mapping) -> list[dict]:
    """Fit every maxent variant at every SES strength in the config."""
    mcfg = dict(cfg.get("maxent", {}))
    strata_edges = tuple(mcfg.get("strata_edges", DEFAULT_STRATA_EDGES))
    variants = {k: tuple(v) for k, v in mcfg.get("variants", DEFAULT_VARIANTS).items()}
    draw = bool(mcfg.get("population", True))
    root = Path(cfg.get("output_dir", "data/sprint2"))
    seed = int(cfg.get("seed", 20261109))
    origin_names = [o["name"] for o in cfg["origins"]]

    summaries = []
    for strength in cfg["flow"]["ses_strengths"]:
        d = root / f"ses_{strength:+.2f}"
        if not (d / "margins.json").is_file():
            raise FileNotFoundError(f"{d / 'margins.json'} not found; run `python -m synthpop.sprint2 build` first")
        margins = load_margins(d / "margins.json")
        sample, _ = load_frozen_truth(d / "sample.csv.gz")
        flow, _ = load_frozen_truth(d / "flow.csv.gz")
        print(f"\n=== SES selection strength {strength:+g} ===")
        flow_table = crosstab(flow, list(margins.variables) + ["edattain"],
                              dict(margins.levels, edattain=len(EDATTAIN_LABELS)))
        strata = strata_of_age_bands(strata_edges)
        truth_summary = _adult_summary(flow_table, list(margins.variables) + ["edattain"], strata, origin_names)
        sample_table = crosstab(sample, list(margins.variables) + ["edattain"],
                                dict(margins.levels, edattain=len(EDATTAIN_LABELS)), weight=sample["weight"])
        sample_summary = _adult_summary(sample_table, list(margins.variables) + ["edattain"], strata, origin_names)

        rows = {"flow (truth, scoring only)": truth_summary, "sample": sample_summary}
        for k, (name, by) in enumerate(variants.items()):
            design = fit_maxent_design(margins, sample, strata_edges=strata_edges, by=by)
            out = d / f"maxent_{name}"
            out.mkdir(parents=True, exist_ok=True)
            outputs = [out / "table.json", out / "report.json"]
            with open(out / "table.json", "w") as f:
                json.dump({"variables": list(design.variables), "levels": design.levels,
                           "table": design.table.tolist()}, f)
            summary = _adult_summary(design.table, design.variables, design.strata, origin_names)
            rows[f"maxent {name}"] = summary
            if draw:
                rng = np.random.default_rng([seed, int(round(strength * 1000)) & 0xFFFFFFFF, 1000 + k])
                people = draw_population(design.table, rng)
                _write_population(out / "population.csv.gz", people, design.variables)
                outputs.append(out / "population.csv.gz")
            report = {
                "variant": name, "by": list(by), "ses_strength": strength, "strata_edges": list(strata_edges),
                "iterations": design.fit.iterations, "converged": design.fit.converged,
                "max_deviation": design.fit.max_deviation, "entropy": design.fit.entropy,
                "fallbacks_to_pooled": design.fallbacks, "unconstrained_strata": design.unconstrained_strata,
                "adults": summary,
            }
            with open(out / "report.json", "w") as f:
                json.dump(report, f, indent=2)
            log_run(
                cfg.get("run_log", "runs/run_log.jsonl"),
                f"Maximum-entropy design ({name}) at SES strength {strength:+g}",
                {"variant": name, "by": list(by), "strata_edges": list(strata_edges), "ses_strength": strength},
                seed=seed,
                inputs=[d / "margins.json", d / "sample.csv.gz"],
                outputs=outputs,
                results={k2: report[k2] for k2 in ("iterations", "converged", "max_deviation", "fallbacks_to_pooled",
                                                  "unconstrained_strata")},
            )
            if design.unconstrained_strata:
                print(f"NOTE: {name}: nobody in the sample in strata {design.unconstrained_strata}; "
                      "education there is left at maximum entropy over its admissible levels.")
            if design.fallbacks:
                print(f"NOTE: {name}: no sample records in {design.fallbacks}; used pooled shares there.")
            summaries.append(report)

        cols = ["university_share_adults"] + [f"university_share_adults|{n}" for n in origin_names] + \
               ["MI(edattain;origin)_adults", "MI(edattain;age_band)_adults"]
        heads = ["univ all"] + [f"univ {n[:9]}" for n in origin_names] + ["MI origin", "MI age"]
        print(f"{'':28}" + "".join(f"{h:>14}" for h in heads))
        for label, s in rows.items():
            print(f"{label:28}" + "".join(f"{s[c]:>14.4f}" for c in cols))
    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    from synthpop.sprint2 import load_config

    parser = argparse.ArgumentParser(prog="python -m synthpop.design_maxent", description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
