"""The thin-group crossover experiment (Sprint 3).

Usage::

    python -m synthpop.crossover configs/sprint2.json

Question: at what sample size does separate estimation of a group stop losing
to pooled estimation? Pooled IPF borrows the education pattern across origins
(low variance, biased by however much the group differs from the others);
separate IPF uses only the group's own sample members (unbiased in the limit,
noisy and full of false zeros when thin, and unable to fit at all when a
control cell is empty). Somewhere in between the lines cross.

Procedure, at each SES strength:

1. Keep every other origin's members of the Sprint 2 sample as they are.
2. For the thin group (default: the smallest origin in the flow), draw
   Efraimidis–Spirakis keys over all its members *in the flow*, with the same
   tilt the Sprint 2 sample used (SES +1, minor -1). The ``n`` smallest keys
   are a tilted sample of size ``n``, and the samples are nested: each size
   is a subsample of the next larger one, so the curves share random numbers.
   This oversamples the group beyond its Sprint 2 size (about 9 Cameroonians
   in 1,369) so that there is a range to subsample down through; see the
   Sprint 3 write-up for why that is a departure from the plan.
3. At each size, replace the thin group's sample members with that draw, fit
   pooled and separate IPF, and score the thin group's adult sex x age band x
   EDATTAIN table against the flow.
4. Repeat for ``replicates`` independent key draws.

Outputs (``data/sprint3/`` by default): ``crossover_runs.csv`` (one row per
replicate x size x design), ``crossover_summary.json`` (medians, quartiles,
separate's failure rate, and the estimated crossover per SES strength) and
``crossover.png``. All are aggregates over synthetic tables; no microdata.

The crossover is located on paired differences: for each replicate in which
separate fitted, separate's SRMSE minus pooled's on the *same* sample. It is
the size at which the median paired difference changes sign from positive
(pooled better) to negative (separate better), interpolated on log n. Failure
rates are reported beside it, not folded into it: a separate run that cannot
fit has no error to compare.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from synthpop.design_ipf import coarsen_age_bands, fit_ipf_design
from synthpop.design_maxent import load_margins
from synthpop.margins import Margins, crosstab
from synthpop.runlog import log_run
from synthpop.scoring import heterogeneity, score_origin
from synthpop.selection import survey_sample_keys
from synthpop.truth import AGE_BANDS, EDATTAIN_LABELS, Records, load_frozen_truth, subset

__all__ = ["DEFAULT_SIZES", "crossover_point", "run_crossover", "summarize"]

DEFAULT_SIZES: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320, 640)
_COLUMNS = ("origin", "male", "age_band", "edattain", "weight")


def _stack(a: Records, b: Records) -> Records:
    return {c: np.concatenate([np.asarray(a[c]), np.asarray(b[c])]) for c in _COLUMNS}


def run_crossover(
    flow: Records,
    margins: Margins,
    sample: Records,
    thin: int,
    sizes: Sequence[int],
    replicates: int,
    tilt: Mapping[str, tuple[np.ndarray, float]] | None,
    rng: np.random.Generator,
    age_edges: Sequence[int] = AGE_BANDS,
) -> list[dict]:
    """One row per replicate x size x design, scoring origin ``thin``.

    ``tilt`` is over the whole flow (as for :func:`draw_survey_sample`); it is
    restricted to the thin group's members here.
    """
    members = np.flatnonzero(np.asarray(flow["origin"]) == thin)
    sizes = sorted(int(n) for n in sizes)
    if not members.size:
        raise ValueError(f"origin {thin} has nobody in the flow")
    if sizes[0] < 1 or sizes[-1] > members.size:
        raise ValueError(f"sizes must lie in 1..{members.size} (the thin group's size in the flow)")
    variables = margins.variables + ("edattain",)
    truth = crosstab(flow, variables, dict(margins.levels, edattain=len(EDATTAIN_LABELS)))
    others = subset(sample, np.flatnonzero(np.asarray(sample["origin"]) != thin))
    group = subset(flow, members)
    group.setdefault("weight", np.ones(members.size))
    group_tilt = {k: (np.asarray(f)[members], c) for k, (f, c) in (tilt or {}).items()}

    rows = []
    for r in range(replicates):
        order = np.argsort(survey_sample_keys(group, group_tilt, rng))
        for n in sizes:
            draw = subset(group, np.sort(order[:n]))
            draw["weight"] = np.ones(n)
            combined = _stack(others, draw)
            for variant in ("pooled", "separate"):
                design = fit_ipf_design(margins, combined, variant, age_edges=age_edges)
                status = design.groups[thin]
                s = score_origin(design.table, truth, variables, thin, band_edges=age_edges)
                rows.append({"replicate": r, "n": n, "design": variant, "failure": status["failure"] or "",
                             "empty_control_cells": "; ".join(status["empty_control_cells"]), **s})
    return rows


def crossover_point(sizes: Sequence[float], diff: Sequence[float]) -> dict:
    """Where the median paired difference (separate - pooled) turns from >0 to <=0.

    Returns ``{"n": value or None, "status": ...}``: ``crossed`` (interpolated on
    log n), ``separate better throughout``, ``pooled better throughout``, or
    ``no data``. Sizes with no difference (separate never fitted) are skipped.
    """
    pts = [(float(n), float(d)) for n, d in zip(sizes, diff) if d is not None and np.isfinite(d)]
    if not pts:
        return {"n": None, "status": "no data"}
    if all(d <= 0 for _, d in pts):
        return {"n": None, "status": "separate better throughout"}
    # Last size at which pooled is still better, then the next size where separate is.
    last_pos = max(i for i, (_, d) in enumerate(pts) if d > 0)
    if last_pos == len(pts) - 1:
        return {"n": None, "status": "pooled better throughout"}
    (n0, d0), (n1, d1) = pts[last_pos], pts[last_pos + 1]
    x0, x1 = np.log(n0), np.log(n1)
    x = x0 + (x1 - x0) * d0 / (d0 - d1)
    return {"n": float(np.exp(x)), "status": "crossed"}


def _q(values, q):
    v = [x for x in values if x is not None and x != "" and np.isfinite(float(x))]
    return float(np.quantile(np.asarray(v, dtype=float), q)) if v else None


def summarize(rows: Sequence[Mapping], metric: str = "srmse") -> dict:
    """Per size: quartiles of ``metric`` by design, separate's failure rate, paired difference."""
    sizes = sorted({int(r["n"]) for r in rows})
    out = {"metric": metric, "sizes": sizes, "by_size": []}
    for n in sizes:
        at = [r for r in rows if int(r["n"]) == n]
        pooled = {r["replicate"]: r for r in at if r["design"] == "pooled"}
        sep = {r["replicate"]: r for r in at if r["design"] == "separate"}
        ok = [k for k, r in sep.items() if not r["failure"] and not pooled[k]["failure"]]
        diffs = [float(sep[k][metric]) - float(pooled[k][metric]) for k in ok]
        entry = {"n": n, "replicates": len(sep),
                 "separate_failure_rate": 1 - len([r for r in sep.values() if not r["failure"]]) / max(len(sep), 1),
                 "pooled_failure_rate": 1 - len([r for r in pooled.values() if not r["failure"]]) / max(len(pooled), 1),
                 "paired_median_diff": _q(diffs, 0.5),
                 "separate_wins_share": float(np.mean([d < 0 for d in diffs])) if diffs else None}
        for name, d in (("pooled", pooled), ("separate", sep)):
            vals = [r[metric] for r in d.values() if not r["failure"]]
            entry[name] = {"q25": _q(vals, 0.25), "median": _q(vals, 0.5), "q75": _q(vals, 0.75)}
        out["by_size"].append(entry)
    out["crossover"] = crossover_point(sizes, [e["paired_median_diff"] for e in out["by_size"]])
    return out


# --- Plot --------------------------------------------------------------------

_POOLED, _SEPARATE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


METRIC_LABELS = {"srmse": "adult SRMSE (sex x age x education)", "tvd_edattain": "adult education TVD"}


def plot_crossover(summaries: Mapping[str, dict], path: str | Path, thin_name: str,
                   sprint2_n: Mapping[str, int] | None = None, metric: str = "srmse") -> None:
    """Small multiples, one column per SES strength: error (top) and separate's failure rate (bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(summaries)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys) + 0.6, 5.4), sharex=True,
                             sharey="row", gridspec_kw={"height_ratios": [3, 1.3]}, squeeze=False)
    for j, key in enumerate(keys):
        s = summaries[key]
        n = np.array(s["sizes"], dtype=float)
        top, bot = axes[0, j], axes[1, j]
        for name, color in (("pooled", _POOLED), ("separate", _SEPARATE)):
            med = np.array([np.nan if e[name]["median"] is None else e[name]["median"] for e in s["by_size"]])
            lo = np.array([np.nan if e[name]["q25"] is None else e[name]["q25"] for e in s["by_size"]])
            hi = np.array([np.nan if e[name]["q75"] is None else e[name]["q75"] for e in s["by_size"]])
            top.fill_between(n, lo, hi, color=color, alpha=0.15, linewidth=0)
            top.plot(n, med, color=color, linewidth=2, marker="o", markersize=4, label=name)
        c = s["crossover"]
        if c["n"]:
            top.axvline(c["n"], color=_MUTED, linewidth=1, linestyle="--")
            top.annotate(f"crossover ≈ {c['n']:.0f}", (c["n"], 0.97), xycoords=("data", "axes fraction"),
                         ha="right", va="top", fontsize=8, color=_MUTED, xytext=(-3, 0), textcoords="offset points")
        else:
            top.text(0.03, 0.97, c["status"], transform=top.transAxes, ha="left", va="top", fontsize=8, color=_MUTED)
        if sprint2_n and key in sprint2_n:
            for ax in (top, bot):
                ax.axvline(sprint2_n[key], color=_MUTED, linewidth=1, linestyle=":")
        fail = [e["separate_failure_rate"] for e in s["by_size"]]
        bot.bar(n, fail, width=n * 0.35, color=_SEPARATE, edgecolor="white", linewidth=1)
        bot.set_ylim(0, 1)
        top.set_title(key, fontsize=10, color=_INK)
        for ax in (top, bot):
            ax.set_xscale("log")
            ax.grid(True, color=_GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=_MUTED, labelsize=8)
        bot.set_xlabel(f"{thin_name} sample size n", fontsize=9, color=_INK)
    axes[0, 0].set_ylabel(f"{thin_name} {METRIC_LABELS.get(metric, metric)}\n(median, IQR band)", fontsize=8,
                          color=_INK)
    axes[1, 0].set_ylabel("separate\nfails to fit", fontsize=9, color=_INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle(f"Pooled vs separate IPF as the {thin_name} sample shrinks "
                 "(dotted: Sprint 2 sample size; dashed: crossover)", fontsize=10, color=_INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Pipeline ----------------------------------------------------------------


def plot_heterogeneity(points: Sequence[Mapping], path: str | Path, metrics: Sequence[str]) -> None:
    """Crossover n against pooled-limit TVD, one panel per metric, one colour per origin, labelled by γ."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots, fixed order
    origins = list(dict.fromkeys(p["origin"] for p in points))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.0), squeeze=False, sharex=True)
    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        for k, origin in enumerate(origins):
            pts = [p for p in points if p["origin"] == origin and p["metric"] == metric and p["crossover_n"]]
            if pts:
                x = [p["pooled_limit_tvd"] for p in pts]
                y = [p["crossover_n"] for p in pts]
                ax.scatter(x, y, s=40, color=colors[k % len(colors)], edgecolor="white", linewidth=1.5,
                           label=origin, zorder=3)
                for p in pts:
                    ax.annotate(f"{p['ses_strength']:+g}", (p["pooled_limit_tvd"], p["crossover_n"]), fontsize=7,
                                color=_MUTED, xytext=(4, 3), textcoords="offset points")
        n_missing = sum(1 for p in points if p["metric"] == metric and not p["crossover_n"])
        if n_missing:
            ax.text(0.98, 0.02, f"{n_missing} without a crossover (see CSV)", transform=ax.transAxes,
                    fontsize=7, color=_MUTED, ha="right")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"Crossover on {METRIC_LABELS.get(metric, metric)}", fontsize=9, color=_INK)
        ax.set_xlabel("pooled-limit TVD (how far the group is from the pool)", fontsize=8, color=_INK)
        ax.grid(True, color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=_MUTED, labelsize=8)
    axes[0, 0].set_ylabel("crossover sample size n", fontsize=9, color=_INK)
    handles, labels = {}, []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), frameon=False, fontsize=8, loc="upper right",
               ncol=len(handles) or 1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _origins_from_config(ccfg: Mapping) -> list[str] | None:
    if "origins" in ccfg:
        return list(ccfg["origins"])
    if ccfg.get("origin"):
        return [ccfg["origin"]]
    return None


def run(cfg: Mapping) -> dict:
    from synthpop.sprint2 import feature

    ccfg = dict(cfg.get("crossover", {}))
    origin_names = [o["name"] for o in cfg["origins"]]
    root = Path(cfg.get("output_dir", "data/sprint2"))
    out = Path(ccfg.get("output_dir", "data/sprint3"))
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 20261109))
    replicates = int(ccfg.get("replicates", 50))
    metrics = list(ccfg.get("metrics", ["srmse"]))
    tilt_cfg = cfg.get("sample", {}).get("tilt", {})
    drop_edges = [int(e) for e in cfg.get("ipf", {}).get("drop_age_edges", [])]
    thin_names = _origins_from_config(ccfg)

    all_rows, inputs, points = [], [], []
    summaries: dict[str, dict[str, dict]] = {}  # "<origin>|<metric>" -> {γ key -> summary}
    sprint2_n: dict[str, dict[str, int]] = {}
    for strength in cfg["flow"]["ses_strengths"]:
        d = root / f"ses_{strength:+.2f}"
        margins = load_margins(d / "margins.json")
        sample, _ = load_frozen_truth(d / "sample.csv.gz")
        flow, _ = load_frozen_truth(d / "flow.csv.gz")
        inputs += [d / "margins.json", d / "sample.csv.gz", d / "flow.csv.gz"]
        counts = np.bincount(flow["origin"], minlength=len(origin_names))
        if thin_names is None:
            thin_names = [origin_names[int(np.argmin(counts))]]
        tilt = {name: (feature(flow, name, origin_names), coef) for name, coef in tilt_cfg.items()}
        margins, (sample, flow), edges = coarsen_age_bands(margins, [sample, flow], drop_edges)
        variables = margins.variables + ("edattain",)
        truth = crosstab(flow, variables, dict(margins.levels, edattain=len(EDATTAIN_LABELS)))
        key = f"γ = {strength:+g}"
        for j, thin_name in enumerate(thin_names):
            thin = origin_names.index(thin_name)
            sizes = [n for n in ccfg.get("sizes", DEFAULT_SIZES) if n <= counts[thin]]
            het = heterogeneity(truth, variables, thin, band_edges=edges)
            # Same stream as the single-origin runs for the first origin, so earlier results reproduce.
            rng = np.random.default_rng([seed, int(round(strength * 1000)) & 0xFFFFFFFF, 3000 + 1000 * j])
            n_s2 = int((sample["origin"] == thin).sum())
            print(f"\n=== SES selection strength {strength:+g}: {thin_name}, {counts[thin]} in the flow, "
                  f"{n_s2} in the Sprint 2 sample; pooled-limit TVD {het['pooled_limit_tvd']:.3f}, "
                  f"education TVD vs pool {het['tvd_edattain_vs_pool']:.3f} ===")
            rows = run_crossover(flow, margins, sample, thin, sizes, replicates, tilt, rng, edges)
            for r in rows:
                r["ses_strength"] = strength
                r["origin"] = thin_name
            all_rows += rows
            sprint2_n.setdefault(thin_name, {})[key] = n_s2
            for metric in metrics:
                summ = summarize(rows, metric)
                summ["heterogeneity"] = het
                summaries.setdefault(f"{thin_name}|{metric}", {})[key] = summ
                c = summ["crossover"]
                points.append({"origin": thin_name, "metric": metric, "ses_strength": strength,
                               "crossover_n": c["n"], "status": c["status"], **het})
                print(f"[{metric}]")
                print(f"{'n':>6}{'pooled':>10}{'separate':>10}{'sep-pool':>10}{'sep wins':>10}{'sep fails':>11}")
                for e in summ["by_size"]:
                    f = lambda x, s=".4f": "-" if x is None else f"{x:{s}}"  # noqa: E731
                    print(f"{e['n']:>6}{f(e['pooled']['median']):>10}{f(e['separate']['median']):>10}"
                          f"{f(e['paired_median_diff'], '+.4f'):>10}{f(e['separate_wins_share'], '.2f'):>10}"
                          f"{e['separate_failure_rate']:>11.2f}")
                print(f"Crossover [{metric}]: " + (f"n ≈ {c['n']:.0f}" if c["n"] else c["status"]))

    fields = ["origin", "ses_strength", "replicate", "n", "design", "failure", "srmse", "tvd_edattain",
              "univ_error", "false_zeros", "true_cells", "empty_control_cells"]
    with open(out / "crossover_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})
    pfields = ["origin", "metric", "ses_strength", "crossover_n", "status", "pooled_limit_tvd", "tvd_edattain_vs_pool"]
    with open(out / "crossover_vs_heterogeneity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        for p in points:
            w.writerow({k: p.get(k) for k in pfields})
    result = {"origins": thin_names, "metrics": metrics, "drop_age_edges": drop_edges, "replicates": replicates,
              "sprint2_sample_n": sprint2_n, "summaries": summaries, "crossover_vs_heterogeneity": points}
    with open(out / "crossover_summary.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    outputs = [out / "crossover_runs.csv", out / "crossover_vs_heterogeneity.csv", out / "crossover_summary.json"]

    print("\nCrossover against heterogeneity (pooled-limit TVD):")
    print(f"{'origin':>12}{'metric':>14}{'γ':>6}{'limit TVD':>11}{'edu TVD':>9}{'crossover':>28}")
    for p in sorted(points, key=lambda p: (p["metric"], p["pooled_limit_tvd"])):
        cn = f"{p['crossover_n']:.0f}" if p["crossover_n"] else p["status"]
        print(f"{p['origin']:>12}{p['metric']:>14}{p['ses_strength']:>+6g}{p['pooled_limit_tvd']:>11.3f}"
              f"{p['tvd_edattain_vs_pool']:>9.3f}{cn:>28}")
    try:
        for name, by_gamma in summaries.items():
            origin, metric = name.split("|")
            png = out / f"crossover_{origin}_{metric}.png"
            plot_crossover(by_gamma, png, origin, sprint2_n[origin], metric)
            outputs.append(png)
        plot_heterogeneity(points, out / "crossover_vs_heterogeneity.png", metrics)
        outputs.append(out / "crossover_vs_heterogeneity.png")
        print(f"\nPlots in {out}")
    except ImportError:
        print("\nmatplotlib is not installed; skipped the plots.")
    log_run(
        cfg.get("run_log", "runs/run_log.jsonl"),
        f"Pooled vs separate IPF crossover on {', '.join(thin_names)} ({', '.join(metrics)})",
        {"origins": thin_names, "metrics": metrics, "drop_age_edges": drop_edges,
         "sizes": list(ccfg.get("sizes", DEFAULT_SIZES)), "replicates": replicates, "tilt": tilt_cfg,
         "ses_strengths": list(cfg["flow"]["ses_strengths"])},
        seed=seed,
        inputs=inputs,
        outputs=outputs,
        results={"crossover_vs_heterogeneity": points},
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    from synthpop.sprint2 import load_config

    parser = argparse.ArgumentParser(prog="python -m synthpop.crossover", description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
