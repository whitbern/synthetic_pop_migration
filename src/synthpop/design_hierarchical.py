"""Hierarchical (partial-pooling) IPF design.

Usage::

    python -m synthpop.design_hierarchical configs/sprint2.json

Pooled IPF borrows the whole sample's education pattern for every origin;
separate IPF uses only the origin's own slice and fails when that slice has an
empty control cell. The hierarchical design sits between them. Within each
origin ``o`` and control cell ``c`` (sex x age band) the seed's education
distribution is shrunk toward the pooled one,

    p_o(e | c) = (x_oce + kappa * q_ce) / (x_oc + kappa),

where ``x`` are the origin's sample counts and ``q_ce`` is the pooled sample's
education distribution in cell ``c``. The cell itself gets mass
``x_oc + kappa * w_c`` (``w_c``: the pooled share of cell ``c``), so a cell
the origin's slice misses but the pool has is no longer zero and IPF can fit
it. The seed ``(x_oc + kappa w_c) p_o(e | c)`` is then fitted to the origin's
own margins by IPF, exactly as in the separate design. Education is still in
no margin, so IPF keeps each cell's shrunken education distribution.

``kappa`` is a prior sample size: ``kappa = 0`` is the separate design and
``kappa -> infinity`` is the pooled one. By default it is estimated by
empirical Bayes, maximizing the Dirichlet-multinomial marginal likelihood of
the origins' cell counts given the pooled distributions (one ``kappa`` for
all origins). It can be fixed in the config (``hierarchical.kappa``), and
``hierarchical.kappa_sweep`` adds a list of fixed values for a sensitivity
check against the planted truth.

For each SES strength this writes ``data/sprint2/ses_<g>/hierarchical/``
(``table.json``, ``report.json``, optionally ``population.csv.gz``), logs the
run, and prints SRMSE, university-share error and MI next to pooled and
separate IPF (fitted here for comparison) and any max-entropy tables on disk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from synthpop.design_ipf import (IPFDesign, _empty_control_cells, _fit, _fmt, _origin_margins, coarsen_age_bands,
                                 fit_ipf_design, ipf_seed)
from synthpop.design_maxent import _write_population, load_margins
from synthpop.margins import Margins, crosstab
from synthpop.maxent import draw_population
from synthpop.runlog import log_run
from synthpop.scoring import score_table
from synthpop.truth import AGE_BANDS, EDATTAIN_LABELS, Records, load_frozen_truth

__all__ = ["KAPPA_BOUNDS", "estimate_kappa", "fit_hierarchical_design", "hierarchical_seed", "log_marginal_likelihood"]

KAPPA_BOUNDS: tuple[float, float] = (1e-2, 1e6)
_lgamma = np.vectorize(math.lgamma, otypes=[float])


def _origin_first(raw: np.ndarray, o: int) -> np.ndarray:
    """Counts as (origin, cell, attribute): origin first, attribute (last axis) last, cells flattened."""
    t = np.moveaxis(np.asarray(raw, dtype=float), o, 0)
    return t.reshape(t.shape[0], -1, t.shape[-1])


def _pooled(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pooled conditional q (cell x attribute) and pooled cell shares w, from origin-first counts."""
    pooled = x.sum(axis=0)
    cell = pooled.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(cell > 0, pooled / cell, 0.0)
    w = cell[:, 0] / cell.sum()
    return q, w


def hierarchical_seed(raw: np.ndarray, o: int, kappa: float) -> np.ndarray:
    """Shrink every origin's slice of ``raw`` toward the pooled slice by ``kappa`` pseudo-counts.

    ``raw`` is the separate seed (counts over controls x attribute, attribute
    last, origin on axis ``o``). ``kappa = 0`` returns ``raw`` unchanged.
    """
    if not kappa >= 0:
        raise ValueError(f"kappa must be nonnegative, got {kappa}")
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite; use the pooled IPF design for kappa = infinity")
    x = _origin_first(raw, o)
    q, w = _pooled(x)
    n = x.sum(axis=-1, keepdims=True)  # x_oc
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (x + kappa * q[None]) / (n + kappa)  # shrunken p_o(e | c); 0/0 only if kappa = 0 and x_oc = 0
    seed = np.nan_to_num((n + kappa * w[None, :, None]) * p, nan=0.0)
    shape = np.moveaxis(np.asarray(raw), o, 0).shape
    return np.moveaxis(seed.reshape(shape), 0, o)


def log_marginal_likelihood(raw: np.ndarray, o: int, kappa: float) -> float:
    """Dirichlet-multinomial log marginal likelihood of each origin's cell counts, prior ``Dir(kappa q_c)``.

    Constants that do not involve ``kappa`` are dropped.
    """
    x = _origin_first(raw, o)
    q, _ = _pooled(x)
    n = x.sum(axis=-1)  # (origin, cell)
    ll = float(np.sum(np.where(n > 0, math.lgamma(kappa) - _lgamma(kappa + n), 0.0)))
    a = kappa * np.broadcast_to(q, x.shape)
    pos = a > 0
    ll += float(np.sum(_lgamma(x[pos] + a[pos]) - _lgamma(a[pos])))
    return ll


def estimate_kappa(raw: np.ndarray, o: int, bounds: tuple[float, float] = KAPPA_BOUNDS) -> dict:
    """Empirical-Bayes ``kappa``: maximize :func:`log_marginal_likelihood` over ``log kappa``.

    A log grid locates the maximum, golden-section search refines it. Returns
    ``{"kappa", "log_ml", "at_bound"}``, where ``at_bound`` is ``"lower"``
    (the origins look unrelated: close to separate), ``"upper"`` (they look
    identical: close to pooled) or ``None``.
    """
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    grid = np.linspace(lo, hi, 81)
    f = lambda t: log_marginal_likelihood(raw, o, math.exp(t))  # noqa: E731
    vals = np.array([f(t) for t in grid])
    i = int(np.argmax(vals))
    if i in (0, len(grid) - 1):
        return {"kappa": math.exp(grid[i]), "log_ml": float(vals[i]), "at_bound": "lower" if i == 0 else "upper"}
    a, b = grid[i - 1], grid[i + 1]
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    t = (a + b) / 2
    return {"kappa": math.exp(t), "log_ml": f(t), "at_bound": None}


def fit_hierarchical_design(margins: Margins, sample: Records, kappa: float | str = "eb",
                            attribute: str = "edattain", n_levels: int = len(EDATTAIN_LABELS), *,
                            age_edges: Sequence[int] = AGE_BANDS, tol: float = 1e-10,
                            max_iter: int = 1000) -> tuple[IPFDesign, dict]:
    """Fit the hierarchical design. ``kappa="eb"`` estimates it; a number fixes it.

    Returns the design (``variant="hierarchical"``) and the kappa record.
    """
    variables, levels, raw = ipf_seed(sample, margins, "separate", attribute, n_levels)
    o = variables.index("origin")
    if kappa == "eb":
        krec = dict(estimate_kappa(raw, o), method="empirical Bayes")
    else:
        kappa = float(kappa)
        if not kappa >= 0:
            raise ValueError(f"kappa must be nonnegative, got {kappa}")
        krec = {"kappa": kappa, "log_ml": log_marginal_likelihood(raw, o, kappa) if kappa > 0 else None,
                "at_bound": None, "method": "fixed"}
    seed = hierarchical_seed(raw, o, krec["kappa"])
    n_seed = crosstab(sample, ["origin"], levels)
    table = np.full(seed.shape, np.nan)
    sub_vars = tuple(v for v in variables if v != "origin")
    groups = []
    for k in range(levels["origin"]):
        sub_seed = np.take(seed, k, axis=o)
        sub_margins = _origin_margins(margins, k)
        fitted, status = _fit(sub_seed, sub_margins, tol, max_iter)
        if fitted is not None:
            idx = [slice(None)] * seed.ndim
            idx[o] = k
            table[tuple(idx)] = fitted
        status["empty_control_cells"] = _empty_control_cells(sub_seed, sub_margins, sub_vars, levels, age_edges)
        status["empty_in_own_slice"] = _empty_control_cells(np.take(raw, k, axis=o), sub_margins, sub_vars,
                                                            levels, age_edges)
        groups.append({"origin": k, "n_seed": int(n_seed[k]), **status})
    return IPFDesign("hierarchical", variables, levels, table, seed, groups, tuple(age_edges)), krec


# --- Pipeline ----------------------------------------------------------------


def _print_scores(scored: Mapping[str, dict], origin_names: Sequence[str]) -> None:
    print("Adult SRMSE (sex x age band x EDATTAIN) within origin, against the flow:")
    print(f"{'':22}" + "".join(f"{n[:11]:>13}" for n in origin_names))
    for label, scores in scored.items():
        print(f"{label:22}" + "".join(f"{_fmt(scores[n]['srmse']):>13}" for n in origin_names))
    print("Adult university share error (synthetic - truth):")
    for label, scores in scored.items():
        print(f"{label:22}" + "".join(f"{_fmt(scores[n]['univ_error'], '+.4f'):>13}" for n in origin_names))
    first = next(iter(scored.values()))
    for key, what in (("mi_edattain_age", "age band"), ("mi_edattain_sex", "sex")):
        print(f"Adult MI(education; {what}) within origin, nats:")
        print(f"{'flow (truth)':22}" + "".join(f"{_fmt(first[n][key + '_truth']):>13}" for n in origin_names))
        for label, scores in scored.items():
            print(f"{label:22}" + "".join(f"{_fmt(scores[n][key]):>13}" for n in origin_names))


def run(cfg: Mapping) -> list[dict]:
    """Fit the hierarchical design at every SES strength in the config and score it."""
    hcfg = dict(cfg.get("hierarchical", {}))
    kappa = hcfg.get("kappa", "eb")
    sweep = [float(k) for k in hcfg.get("kappa_sweep", [])]
    draw = bool(hcfg.get("population", True))
    drop_edges = [int(e) for e in hcfg.get("drop_age_edges", cfg.get("ipf", {}).get("drop_age_edges", []))]
    root = Path(cfg.get("output_dir", "data/sprint2"))
    seed = int(cfg.get("seed", 20261109))
    origin_names = [o["name"] for o in cfg["origins"]]

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
        variables = margins.variables + ("edattain",)
        levels = dict(margins.levels, edattain=len(EDATTAIN_LABELS))
        truth_table = crosstab(flow, variables, levels)

        design, krec = fit_hierarchical_design(margins, sample, kappa, age_edges=edges)
        bound = {"lower": " (at lower bound: close to separate)",
                 "upper": " (at upper bound: close to pooled)"}.get(krec["at_bound"], "")
        print(f"kappa ({krec['method']}) = {krec['kappa']:.4g}{bound}")
        scores = score_table(design.table, truth_table, variables, origin_names, edges)
        groups = [dict(g, origin=origin_names[g["origin"]]) for g in design.groups]
        for g in groups:
            if g["failure"]:
                print(f"FAILED: hierarchical, {g['origin']} (n_seed={g['n_seed']}): {g['failure']}. "
                      f"Empty control cells: {'; '.join(g['empty_control_cells']) or 'none listed'}")
            elif g["empty_in_own_slice"]:
                print(f"Filled by borrowing: {g['origin']} (n_seed={g['n_seed']}) had no one in "
                      f"{'; '.join(g['empty_in_own_slice'])}")

        sweep_scores = {}
        for k in sweep:
            dk, _ = fit_hierarchical_design(margins, sample, k, age_edges=edges)
            sweep_scores[k] = score_table(dk.table, truth_table, variables, origin_names, edges)

        out = d / "hierarchical"
        out.mkdir(parents=True, exist_ok=True)
        outputs = [out / "table.json", out / "report.json"]
        table_list = design.table.tolist()
        for g in design.failed:
            table_list[g] = None  # origin is axis 0
        with open(out / "table.json", "w") as f:
            json.dump({"variables": list(design.variables), "levels": design.levels, "table": table_list,
                       "kappa": krec["kappa"], "failed_origins": [origin_names[g] for g in design.failed]}, f)
        if draw and len(design.failed) < len(origin_names):
            rng = np.random.default_rng([seed, int(round(strength * 1000)) & 0xFFFFFFFF, 3000])
            people = draw_population(np.nan_to_num(design.table, nan=0.0), rng)
            _write_population(out / "population.csv.gz", people, design.variables)
            outputs.append(out / "population.csv.gz")
        report = {"variant": "hierarchical", "ses_strength": strength, "kappa": krec, "drop_age_edges": drop_edges,
                  "age_edges": list(edges), "groups": groups, "scores": scores,
                  "failed_origins": [origin_names[g] for g in design.failed],
                  "kappa_sweep": {str(k): {n: s["srmse"] for n, s in sc.items()} for k, sc in sweep_scores.items()}}
        with open(out / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        log_run(
            cfg.get("run_log", "runs/run_log.jsonl"),
            f"Hierarchical IPF design at SES strength {strength:+g}",
            {"variant": "hierarchical", "ses_strength": strength, "kappa": kappa, "kappa_sweep": sweep,
             "drop_age_edges": drop_edges, "tol": 1e-10, "max_iter": 1000},
            seed=seed,
            inputs=[d / "margins.json", d / "sample.csv.gz", d / "flow.csv.gz"],
            outputs=outputs,
            results={"kappa": krec["kappa"], "failed_origins": report["failed_origins"],
                     "srmse": {n: s["srmse"] for n, s in scores.items()}},
        )
        reports.append(report)

        scored: dict[str, dict] = {}
        for name in ("pooled", "separate"):
            ipf_d = fit_ipf_design(margins, sample, name, age_edges=edges)
            scored[f"ipf {name}"] = score_table(ipf_d.table, truth_table, variables, origin_names, edges)
        if not drop_edges:
            for mdir in sorted(d.glob("maxent_*")):
                if (mdir / "table.json").is_file():
                    with open(mdir / "table.json") as f:
                        t = json.load(f)
                    if tuple(t["variables"]) == variables:
                        scored[mdir.name.replace("_", " ")] = score_table(np.asarray(t["table"]), truth_table,
                                                                        variables, origin_names)
        scored[f"hierarchical k={krec['kappa']:.3g}"] = scores
        _print_scores(scored, origin_names)
        if sweep_scores:
            print("Kappa sweep, adult SRMSE (0 = separate; large = pooled):")
            for k, sc in sweep_scores.items():
                print(f"{'k=' + format(k, 'g'):22}" + "".join(f"{_fmt(sc[n]['srmse']):>13}" for n in origin_names))
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    from synthpop.sprint2 import load_config

    parser = argparse.ArgumentParser(prog="python -m synthpop.design_hierarchical",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
