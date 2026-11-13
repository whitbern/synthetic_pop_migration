"""Pooled and separate IPF on the Sprint 2 inputs (Sprint 3).

Usage::

    python -m synthpop.design_ipf configs/sprint2.json

Both designs fit a table over origin x sex x age band x EDATTAIN by
iterative proportional fitting: the biased sample is the seed, the control
margins (origin, origin x sex, origin x age band; no education) are the
targets. Education is never a target, so whatever the synthetic population
says about education comes from the seed's cross-classification. They differ
only in which seed each origin group gets:

* ``pooled``: one IPF run over all origins at once, with origin carried as an
  attribute of the table. Every origin's slice of the seed is the *whole*
  sample's sex x age band x education table, so the education pattern is
  borrowed across origins and the margins make it origin-specific only
  through the origin's own sex and age mix.
* ``separate``: one IPF run per origin group, each with that group's own
  margins and its own slice of the sample as the seed. Nothing is shared.

A third seed, the full sample table *with* origin (``joint``), is kept for
verification only: because every control margin is nested within origin, IPF
on that seed factorizes into the per-origin runs, so ``joint`` and
``separate`` give the same table. That is why "origin as an attribute" alone
does not make a design pooled; pooling has to happen in the seed.

Failures are recorded, not patched. A group whose sample slice has nobody in
a sex or age-band cell its margins ask for cannot be fitted (IPF can only
rescale, so a seed zero stays zero); that group's table is left undefined
(NaN) and the empty cells are listed in the report. Its adult cells that are
zero in the seed but present in the truth (false zeros) are counted when it
does fit.

For each SES strength this writes ``data/sprint2/ses_<g>/ipf_<variant>/``
holding ``table.json``, ``population.csv.gz`` (synthetic people from the
groups that fitted) and ``report.json``, logs the run, and prints per-origin
scores against the flow (the planted truth), which is read only for scoring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from synthpop.design_maxent import _write_population, load_margins
from synthpop.ipf import InconsistentMarginsError, IPFConvergenceError, ipf, marginal
from synthpop.margins import Margins, crosstab
from synthpop.maxent import draw_population
from synthpop.runlog import log_run
from synthpop.scoring import score_table
from synthpop.truth import AGE_BANDS, EDATTAIN_LABELS, Records, load_frozen_truth

__all__ = ["VARIANTS", "IPFDesign", "coarsen_age_bands", "fit_ipf_design", "ipf_seed"]

VARIANTS: tuple[str, ...] = ("pooled", "separate")
_ALL_VARIANTS = VARIANTS + ("joint",)


@dataclass(frozen=True)
class IPFDesign:
    """A fitted IPF design. ``table`` is NaN over the origin slices that failed."""

    variant: str
    variables: tuple[str, ...]
    levels: dict[str, int]
    table: np.ndarray
    seed: np.ndarray
    groups: list[dict] = field(default_factory=list)  # one status entry per origin
    age_edges: tuple[int, ...] = AGE_BANDS

    @property
    def failed(self) -> list[int]:
        return [g["origin"] for g in self.groups if g["failure"]]


def _labels(variable: str, n: int, age_edges: Sequence[int] = AGE_BANDS) -> list[str]:
    if variable == "male":
        return ["female", "male"]
    if variable == "age_band":
        edges = list(age_edges)
        return [f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
    return [str(i) for i in range(n)]


def ipf_seed(sample: Records, margins: Margins, variant: str, attribute: str = "edattain",
             n_levels: int = len(EDATTAIN_LABELS)) -> tuple[tuple[str, ...], dict[str, int], np.ndarray]:
    """The seed table for ``variant``: the sample's weighted counts over controls x ``attribute``.

    ``pooled`` collapses origin and gives every origin the same slice;
    ``separate`` and ``joint`` keep each origin's own slice.
    """
    if variant not in _ALL_VARIANTS:
        raise ValueError(f"unknown IPF variant {variant!r}; expected one of {_ALL_VARIANTS}")
    if attribute in margins.variables:
        raise ValueError(f"{attribute} is already a control variable")
    if "origin" not in margins.variables:
        raise ValueError("the control margins must include origin")
    variables = margins.variables + (attribute,)
    levels = dict(margins.levels, **{attribute: int(n_levels)})
    w = np.asarray(sample.get("weight", np.ones(len(sample[attribute]))), dtype=float)
    seed = crosstab(sample, variables, levels, weight=w)
    if variant == "pooled":
        o = variables.index("origin")
        seed = np.broadcast_to(seed.sum(axis=o, keepdims=True), seed.shape).copy()
    return variables, levels, seed


def _origin_margins(margins: Margins, k: int) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """Origin ``k``'s slice of every control, as margins on the table without the origin axis."""
    o = margins.variables.index("origin")
    out = []
    for (axes, target), control in zip(margins.margins, margins.controls):
        if o not in axes:
            raise ValueError(f"the separate design needs every control to include origin; {control} does not")
        pos = list(axes).index(o)
        rest = tuple(a - (a > o) for a in axes if a != o)
        if rest:
            out.append((rest, np.take(np.asarray(target), k, axis=pos)))
    if not out:
        raise ValueError("no control besides origin itself; nothing for the separate design to fit")
    return out


def _empty_control_cells(seed: np.ndarray, sub_margins, variables: Sequence[str], levels: Mapping[str, int],
                         age_edges: Sequence[int] = AGE_BANDS) -> list[str]:
    """Control cells the margins ask for but the seed has nobody in."""
    cells = []
    for axes, target in sub_margins:
        support = marginal(seed, axes)  # axes in sorted order
        t = np.transpose(np.asarray(target), np.argsort(axes))
        sorted_axes = tuple(sorted(axes))
        for idx in np.argwhere((support == 0) & (t > 0)):
            cells.append(", ".join(f"{variables[a]}={_labels(variables[a], levels[variables[a]], age_edges)[i]}"
                                   for a, i in zip(sorted_axes, idx)))
    return cells


def _fit(seed: np.ndarray, sub_margins, tol: float, max_iter: int) -> tuple[np.ndarray | None, dict]:
    try:
        res = ipf(seed, sub_margins, tol=tol, max_iter=max_iter)
    except InconsistentMarginsError as err:
        return None, {"failure": "empty control cell", "detail": str(err)}
    except IPFConvergenceError as err:
        return None, {"failure": "nonconvergence", "detail": str(err),
                      "iterations": err.result.iterations, "max_deviation": err.result.max_deviation}
    return res.table, {"failure": None, "iterations": res.iterations, "max_deviation": res.max_deviation}


def fit_ipf_design(margins: Margins, sample: Records, variant: str, attribute: str = "edattain",
                   n_levels: int = len(EDATTAIN_LABELS), *, age_edges: Sequence[int] = AGE_BANDS,
                   tol: float = 1e-10, max_iter: int = 1000) -> IPFDesign:
    """Fit the ``pooled``, ``separate`` (or, for verification, ``joint``) IPF design.

    ``age_edges`` only labels the age bands in the failure report; pass the
    edges returned by :func:`coarsen_age_bands` if the bands were merged.
    """
    variables, levels, seed = ipf_seed(sample, margins, variant, attribute, n_levels)
    o = variables.index("origin")
    n_origins = levels["origin"]
    n_seed = crosstab(sample, ["origin"], levels)
    table = np.full(seed.shape, np.nan)
    groups = []
    sub_vars = tuple(v for v in variables if v != "origin")

    if variant == "separate":
        for k in range(n_origins):
            sub_seed = np.take(seed, k, axis=o)
            sub_margins = _origin_margins(margins, k)
            fitted, status = _fit(sub_seed, sub_margins, tol, max_iter)
            if fitted is not None:
                idx = [slice(None)] * seed.ndim
                idx[o] = k
                table[tuple(idx)] = fitted
            status["empty_control_cells"] = _empty_control_cells(sub_seed, sub_margins, sub_vars, levels, age_edges)
            groups.append({"origin": k, "n_seed": int(n_seed[k]), **status})
    else:
        fitted, status = _fit(seed, margins.margins, tol, max_iter)
        if fitted is not None:
            table = fitted
        for k in range(n_origins):
            sub_seed = np.take(seed, k, axis=o)
            empty = _empty_control_cells(sub_seed, _origin_margins(margins, k), sub_vars, levels, age_edges)
            groups.append({"origin": k, "n_seed": int(n_seed[k]), **status, "empty_control_cells": empty})
    return IPFDesign(variant, variables, levels, table, seed, groups, tuple(age_edges))


def coarsen_age_bands(margins: Margins, records: Sequence[Records], drop_edges: Sequence[int] = (),
                      age_edges: Sequence[int] = AGE_BANDS) -> tuple[Margins, list[Records], tuple[int, ...]]:
    """Merge age bands by removing lower edges, in the margins and in every record set alike.

    ``drop_edges=[5]`` merges 0-4 into 5-17, giving 0-17. This is a *global,
    reported* aggregation: it applies to every origin and both designs, so it
    never merges origins and never adapts to which group is thin (unlike the
    adaptive aggregation of Auld, Mohammadian & Wies 2009). With no edges to
    drop, everything is returned unchanged.
    """
    drop = {int(e) for e in drop_edges}
    edges = tuple(int(e) for e in age_edges)
    if not drop:
        return margins, list(records), edges
    if drop - set(edges[1:]):
        raise ValueError(f"can only drop interior age-band edges {list(edges[1:])}, not {sorted(drop - set(edges[1:]))}")
    new_edges = tuple(e for e in edges if e not in drop)
    remap = np.searchsorted(np.asarray(new_edges), np.asarray(edges), side="right") - 1
    a = margins.variables.index("age_band")
    n_new = len(new_edges)
    new_margins = []
    for axes, target in margins.margins:
        target = np.asarray(target, dtype=float)
        if a in axes:
            pos = list(axes).index(a)
            moved = np.moveaxis(target, pos, 0)
            summed = np.zeros((n_new,) + moved.shape[1:])
            np.add.at(summed, remap, moved)
            target = np.moveaxis(summed, 0, pos)
        new_margins.append((axes, target))
    levels = dict(margins.levels, age_band=n_new)
    out = []
    for r in records:
        r = dict(r)
        r["age_band"] = remap[np.asarray(r["age_band"]).astype(np.int64)]
        out.append(r)
    return Margins(margins.variables, levels, new_margins, margins.controls), out, new_edges


# --- Pipeline ----------------------------------------------------------------


def _fmt(x, spec=".4f"):
    return f"{x:{spec}}" if isinstance(x, (int, float)) and x is not None else "FAILED"


def run(cfg: Mapping) -> list[dict]:
    """Fit both IPF designs at every SES strength in the config and score them."""
    icfg = dict(cfg.get("ipf", {}))
    variants = tuple(icfg.get("variants", VARIANTS))
    draw = bool(icfg.get("population", True))
    root = Path(cfg.get("output_dir", "data/sprint2"))
    seed = int(cfg.get("seed", 20261109))
    origin_names = [o["name"] for o in cfg["origins"]]
    drop_edges = [int(e) for e in icfg.get("drop_age_edges", [])]

    reports = []
    for strength in cfg["flow"]["ses_strengths"]:
        d = root / f"ses_{strength:+.2f}"
        if not (d / "margins.json").is_file():
            raise FileNotFoundError(f"{d / 'margins.json'} not found; run `python -m synthpop.sprint2 build` first")
        margins = load_margins(d / "margins.json")
        sample, _ = load_frozen_truth(d / "sample.csv.gz")
        flow, _ = load_frozen_truth(d / "flow.csv.gz")
        print(f"\n=== SES selection strength {strength:+g} ===")
        margins, (sample, flow), edges = coarsen_age_bands(margins, [sample, flow], drop_edges)
        if drop_edges:
            print(f"Age bands merged for IPF (reported parameter drop_age_edges={drop_edges}): edges {list(edges)}")
        variables = margins.variables + ("edattain",)
        levels = dict(margins.levels, edattain=len(EDATTAIN_LABELS))
        truth_table = crosstab(flow, variables, levels)

        scored: dict[str, dict] = {}
        for k, name in enumerate(variants):
            design = fit_ipf_design(margins, sample, name, age_edges=edges)
            out = d / f"ipf_{name}"
            out.mkdir(parents=True, exist_ok=True)
            outputs = [out / "table.json", out / "report.json"]
            table_list = design.table.tolist()
            for g in design.failed:
                table_list[g] = None  # origin is axis 0: a failed group has no table
            with open(out / "table.json", "w") as f:
                json.dump({"variables": list(design.variables), "levels": design.levels, "table": table_list,
                           "failed_origins": [origin_names[g] for g in design.failed]}, f)
            if draw and len(design.failed) < len(origin_names):
                rng = np.random.default_rng([seed, int(round(strength * 1000)) & 0xFFFFFFFF, 2000 + k])
                people = draw_population(np.nan_to_num(design.table, nan=0.0), rng)
                _write_population(out / "population.csv.gz", people, design.variables)
                outputs.append(out / "population.csv.gz")
            scores = score_table(design.table, truth_table, variables, origin_names, edges)
            scored[f"ipf {name}"] = scores
            groups = [dict(g, origin=origin_names[g["origin"]]) for g in design.groups]
            report = {"variant": name, "ses_strength": strength, "drop_age_edges": drop_edges,
                      "age_edges": list(edges), "groups": groups, "scores": scores,
                      "failed_origins": [origin_names[g] for g in design.failed]}
            with open(out / "report.json", "w") as f:
                json.dump(report, f, indent=2)
            log_run(
                cfg.get("run_log", "runs/run_log.jsonl"),
                f"IPF design ({name}) at SES strength {strength:+g}",
                {"variant": name, "ses_strength": strength, "drop_age_edges": drop_edges, "tol": 1e-10,
                 "max_iter": 1000},
                seed=seed,
                inputs=[d / "margins.json", d / "sample.csv.gz", d / "flow.csv.gz"],
                outputs=outputs,
                results={"failed_origins": report["failed_origins"],
                         "srmse": {n: s["srmse"] for n, s in scores.items()}},
            )
            if name == "separate":
                for g in groups:
                    if g["failure"]:
                        cells = "; ".join(g["empty_control_cells"]) or "none listed"
                        print(f"FAILED: ipf separate, {g['origin']} (n_seed={g['n_seed']}): {g['failure']}. "
                              f"Empty control cells: {cells}")
            elif groups and groups[0]["failure"]:
                cells = sorted({c for g in groups for c in g["empty_control_cells"]})
                print(f"FAILED: ipf {name} (one run over all origins): {groups[0]['failure']}. "
                      f"Empty control cells in the pooled seed: {'; '.join(cells) or 'none listed'}")
            reports.append(report)

        # Max-entropy tables from `python -m synthpop.design_maxent`, if present, for comparison.
        for mdir in sorted(d.glob("maxent_*")):
            if (mdir / "table.json").is_file():
                with open(mdir / "table.json") as f:
                    t = json.load(f)
                if tuple(t["variables"]) == variables and not drop_edges:
                    scored[mdir.name.replace("_", " ")] = score_table(np.asarray(t["table"]), truth_table,
                                                                    variables, origin_names)

        print("Adult SRMSE (sex x age band x EDATTAIN) within origin, against the flow:")
        print(f"{'':20}" + "".join(f"{n[:11]:>13}" for n in origin_names))
        for label, scores in scored.items():
            print(f"{label:20}" + "".join(f"{_fmt(scores[n]['srmse']):>13}" for n in origin_names))
        print("Adult university share error (synthetic - truth):")
        for label, scores in scored.items():
            print(f"{label:20}" + "".join(f"{_fmt(scores[n]['univ_error'], '+.4f'):>13}" for n in origin_names))
        first = next(iter(scored.values()))
        for key, what in (("mi_edattain_age", "age band"), ("mi_edattain_sex", "sex")):
            print(f"Adult MI(education; {what}) within origin, nats (maximum entropy predicts 0):")
            print(f"{'flow (truth)':20}" + "".join(f"{_fmt(first[n][key + '_truth']):>13}" for n in origin_names))
            for label, scores in scored.items():
                print(f"{label:20}" + "".join(f"{_fmt(scores[n][key]):>13}" for n in origin_names))
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    from synthpop.sprint2 import load_config

    parser = argparse.ArgumentParser(prog="python -m synthpop.design_ipf", description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
