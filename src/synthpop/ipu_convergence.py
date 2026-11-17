"""Convergence verification for IPU on planted two-level instances.

Usage::

    python -m synthpop.ipu_convergence [--reps 200] [--out data/ipu_convergence] [--plot docs/figures/ipu_delta_traces.png]

Each instance is a random household sample (two household types, three
person types, 1-4 members) whose control totals are computed from known
positive weights, so an exact positive solution exists by construction. The
entropy reference solver confirms that numerically. IPU is then run three ways:

* to tolerance, with no stall rule and a large iteration cap, to ask whether it
  reaches a solution that exists;
* with Ye et al. (2009)'s stopping rule (stop when delta stops improving), to
  see where that rule stops and what fit it reports;
* with its full delta trace kept, to locate the best iterate and test whether
  delta decreases monotonically.

No microdata is used. Outputs are a per-instance CSV and a JSON summary; the
run is appended to ``runs/run_log.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from synthpop.ipu import entropy_weights, ipu
from synthpop.runlog import log_run

__all__ = ["planted_instance", "run_instance", "main"]

SIZES = (8, 12, 20, 50, 200)
TOL = 1e-8
MAX_ITER = 20_000


def planted_instance(rng, n_households=40, n_hh_types=2, n_person_types=3, max_size=4):
    """Random two-level instance with targets computed from known positive weights.

    Returns ``(freq, targets, true_weights)``. Columns with no contributing
    household are dropped.
    """
    hh_type = rng.integers(0, n_hh_types, n_households)
    hh_cols = np.eye(n_hh_types)[hh_type]
    sizes = rng.integers(1, max_size + 1, n_households)
    person_cols = np.zeros((n_households, n_person_types))
    for i, s in enumerate(sizes):
        person_cols[i] = np.bincount(rng.integers(0, n_person_types, s), minlength=n_person_types)
    freq = np.hstack([hh_cols, person_cols])
    freq = freq[:, freq.sum(axis=0) > 0]
    w_true = rng.uniform(5, 50, n_households)
    return freq, freq.T @ w_true, w_true


def run_instance(freq, targets) -> dict:
    """All three IPU runs plus the reference solver on one instance."""
    _, ref_ok, ref_resid = entropy_weights(freq, targets, tol=TOL)
    full = ipu(freq, targets, tol=TOL, stall_tol=None, max_iter=MAX_ITER, strict=False)
    rule = ipu(freq, targets, tol=TOL, stall_tol=1e-14, max_iter=MAX_ITER, strict=False)
    h = np.asarray(full.history)
    best = int(np.argmin(h))
    return {
        "reference_feasible": bool(ref_ok),
        "reference_residual": float(ref_resid),
        "ipu_converged": bool(full.converged),
        "ipu_iterations": int(full.iterations),
        "ipu_final_delta": float(h[-1]),
        "ipu_final_max_rel_residual": float(full.max_rel_residual),
        "delta_0": float(h[0]),
        "delta_1": float(h[1]) if len(h) > 1 else float(h[0]),
        "best_iteration": best,
        "best_delta": float(h[best]),
        "delta_monotone": bool(np.all(np.diff(h) <= 1e-15)),
        "rule_converged": bool(rule.converged),
        "rule_iterations": int(rule.iterations),
        "rule_delta": float(rule.delta),
        "min_weight_final": float(full.weights.min()),
    }, h


def summarise(rows: list[dict]) -> dict:
    out = {}
    for n in sorted({r["n_households"] for r in rows}):
        sub = [r for r in rows if r["n_households"] == n]
        feas = [r for r in sub if r["reference_feasible"]]
        failed = [r for r in feas if not r["ipu_converged"]]
        out[str(n)] = {
            "instances": len(sub),
            "reference_feasible": len(feas),
            "ipu_converged_share_of_feasible": len(feas) and (len(feas) - len(failed)) / len(feas),
            "ipu_failed_on_feasible": len(failed),
            "median_iterations_when_converged": float(
                np.median([r["ipu_iterations"] for r in feas if r["ipu_converged"]] or [np.nan])
            ),
            "non_monotone_delta_share": float(np.mean([not r["delta_monotone"] for r in sub])),
            "failed_median_final_delta": float(np.median([r["ipu_final_delta"] for r in failed] or [np.nan])),
            "failed_max_final_rel_residual": float(
                max([r["ipu_final_max_rel_residual"] for r in failed] or [np.nan])
            ),
            "failed_best_iteration_median": float(np.median([r["best_iteration"] for r in failed] or [np.nan])),
            "failed_best_delta_median": float(np.median([r["best_delta"] for r in failed] or [np.nan])),
            "stall_rule_stopped_unconverged_share": float(np.mean([not r["rule_converged"] for r in feas])),
            "stall_rule_median_iterations_when_unconverged": float(
                np.median([r["rule_iterations"] for r in feas if not r["rule_converged"]] or [np.nan])
            ),
            "median_delta_1_over_delta_0": float(np.median([r["delta_1"] / r["delta_0"] for r in sub])),
        }
    return out


def plot_traces(traces: list[tuple[str, np.ndarray]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, h in traces:
        it = np.arange(len(h))
        ax.plot(it[1:], np.maximum(h[1:], 1e-12), lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IPU iteration")
    ax.set_ylabel("goodness of fit δ (log)")
    ax.axhline(TOL, color="0.6", lw=0.8, ls="--")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("IPU δ traces on feasible 8-household instances", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--out", default="data/ipu_convergence")
    p.add_argument("--plot", default=None, help="optional path for a delta-trace figure")
    p.add_argument("--log", default="runs/run_log.jsonl")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = np.random.default_rng(args.seed)
    rows, examples = [], []
    for n in SIZES:
        for rep in range(args.reps):
            rng = np.random.default_rng(root.integers(2**63))
            freq, targets, _ = planted_instance(rng, n_households=n)
            res, h = run_instance(freq, targets)
            rows.append({"n_households": n, "rep": rep, **res})
            if n == SIZES[0] and len(examples) < 4 and res["reference_feasible"]:
                if (not res["ipu_converged"]) or len(examples) == 0:
                    tag = "converged" if res["ipu_converged"] else "fails"
                    examples.append((f"rep {rep} ({tag})", h))

    with open(out / "instances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    outputs = [out / "instances.csv", out / "summary.json"]
    if args.plot:
        plot_traces(examples, Path(args.plot))
        outputs.append(Path(args.plot))
    log_run(
        args.log,
        "IPU convergence verification on planted two-level instances (Ye et al. 2022 check)",
        {"sizes": list(SIZES), "reps": args.reps, "tol": TOL, "max_iter": MAX_ITER},
        args.seed,
        outputs=outputs,
        results={"summary": summary},
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
