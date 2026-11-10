"""Sprint 2 pipeline: truth → calibrated flow → margins + biased sample.

Usage::

    python -m synthpop.sprint2 freeze configs/sprint2.json
    python -m synthpop.sprint2 build  configs/sprint2.json

``freeze`` reads the raw PUMS CSVs, keeps the origin groups and writes the
frozen truth (gzipped CSV + manifest with its SHA-256). ``build`` reads the
frozen truth and, for each SES selection strength in the config, calibrates
the selection function, draws the flow, and writes the manufactured margins
(no education) and the biased sample. Both append to the run log.

See ``configs/sprint2.example.json`` for every key.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from synthpop.margins import DEFAULT_CONTROLS, manufacture_margins
from synthpop.runlog import log_run
from synthpop.selection import (
    DarienTargets,
    composition,
    draw_survey_sample,
    realize_flow,
    select_flow,
)
from synthpop.truth import OriginGroup, Records, freeze_truth, load_frozen_truth, load_pums_persons

__all__ = ["build", "expand_paths", "feature", "freeze", "load_config", "targets_from_config"]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    cfg["_path"] = str(path)
    return cfg


def origins_from_config(cfg: Mapping) -> list[OriginGroup]:
    return [OriginGroup.from_mapping(g["name"], g["pobp"]) for g in cfg["origins"]]


def targets_from_config(cfg: Mapping) -> DarienTargets:
    t = cfg["targets"]
    shares = {k: v for k, v in t["origin_shares"].items() if not k.startswith("_")}
    unset = [k for k, v in shares.items() if v is None]
    if unset:
        raise ValueError(f"targets.origin_shares not set for {unset}; fill them in from the SENAFRONT series")
    # Shares may be entered as raw counts or as shares of the whole flow
    # (including nationalities outside the study's groups); renormalise to the
    # groups studied.
    total = float(sum(shares.values()))
    if total <= 0:
        raise ValueError("targets.origin_shares must be positive")
    shares = {k: float(v) / total for k, v in shares.items()}
    return DarienTargets(
        origin_shares=shares,
        male_share=t.get("male_share", 0.715),
        mean_age=t.get("mean_age", 31.0),
        elderly_share=t.get("elderly_share", 0.005),
        elderly_age=t.get("elderly_age", 65),
        tolerance=t.get("tolerance", {"share": 0.005, "mean_age": 0.25}),
    )


def feature(records: Records, name: str, origin_names: Sequence[str]) -> np.ndarray:
    """A named feature for the survey-sample tilt.

    ``ses`` (centred rank), ``male``, ``age_decades`` (age/10, centred at 3),
    ``minor`` (under 18), ``edattain`` (0–4, centred at 2), or
    ``origin=<name>``.
    """
    if name == "ses":
        return records["ses"] - 0.5
    if name == "male":
        return records["male"].astype(float)
    if name == "age_decades":
        return records["age"] / 10 - 3
    if name == "minor":
        return (records["age"] < 18).astype(float)
    if name == "edattain":
        return records["edattain"] - 2.0
    if name.startswith("origin="):
        k = list(origin_names).index(name.split("=", 1)[1])
        return (records["origin"] == k).astype(float)
    raise ValueError(f"unknown tilt feature {name!r}")


def expand_paths(entries: Sequence[str], what: str) -> list[str]:
    """Expand glob patterns (e.g. ``data/raw/psam_pus*.csv``) and check every file exists.

    Also warns when the folder holds files that look like further parts of
    the same split file (``psam_pusa.csv`` listed, ``psam_pusc.csv`` present
    but not listed), since leaving a part out silently drops whole states.
    """
    out: list[str] = []
    for entry in entries:
        if any(ch in entry for ch in "*?["):
            matches = sorted(glob.glob(entry))
            if not matches:
                raise FileNotFoundError(f"{what}: pattern {entry!r} matches no files")
            out.extend(matches)
        else:
            if not Path(entry).is_file():
                raise FileNotFoundError(f"{what}: {entry!r} does not exist")
            out.append(entry)
    out = list(dict.fromkeys(out))
    listed = {Path(p).resolve() for p in out}
    for p in out:
        stem = Path(p).stem
        if len(stem) > 1:
            for sibling in Path(p).parent.glob(stem[:-1] + "?" + Path(p).suffix):
                if sibling.resolve() not in listed:
                    print(f"WARNING: {what}: {sibling} looks like another part of the same file but is not listed")
    return out


def freeze(cfg: Mapping) -> dict:
    origins = origins_from_config(cfg)
    src = dict(cfg["truth"])
    src["person_csvs"] = expand_paths(src["person_csvs"], "person_csvs")
    src["household_csvs"] = expand_paths(src.get("household_csvs", ()), "household_csvs")
    print("Reading person files: " + ", ".join(src["person_csvs"]))
    if src["household_csvs"]:
        print("Reading household files: " + ", ".join(src["household_csvs"]))
    records = load_pums_persons(src["person_csvs"], origins, src["household_csvs"])
    manifest = freeze_truth(
        records,
        src["frozen_path"],
        origins,
        meta={"source": src.get("source", "ACS 2020-2024 5-year PUMS"), "person_csvs": list(src["person_csvs"]),
              "household_csvs": list(src["household_csvs"])},
    )
    log_run(
        cfg.get("run_log", "runs/run_log.jsonl"),
        "Sprint 2: freeze ACS truth",
        {"origins": cfg["origins"], "household_csvs": bool(src.get("household_csvs"))},
        seed=None,
        inputs=list(src["person_csvs"]) + list(src.get("household_csvs", ())),
        outputs=[src["frozen_path"]],
        results={"sha256": manifest["sha256"], "records_per_origin": manifest["records_per_origin"]},
    )
    print(json.dumps({k: manifest[k] for k in ("sha256", "n_records", "records_per_origin")}, indent=2))
    return manifest


def build(cfg: Mapping) -> list[dict]:
    truth_path = cfg["truth"]["frozen_path"]
    truth, manifest = load_frozen_truth(truth_path)
    origin_names = [o["name"] for o in sorted(manifest["origins"], key=lambda o: o["index"])]
    targets = targets_from_config(cfg)
    flow_cfg = cfg["flow"]
    sample_cfg = cfg["sample"]
    controls = [tuple(c) for c in cfg.get("controls", DEFAULT_CONTROLS)]
    out_root = Path(cfg.get("output_dir", "data/sprint2"))
    seed = int(cfg.get("seed", 20261109))

    summaries = []
    for strength in flow_cfg["ses_strengths"]:
        rng = np.random.default_rng([seed, int(round(strength * 1000)) & 0xFFFFFFFF])
        sel = select_flow(truth, origin_names, targets, ses_strength=strength, strict=False)
        print(f"\n=== SES selection strength {strength:+g} ===\n{sel.report()}")
        if not sel.within_tolerance:
            print("WARNING: calibration misses the targets; reported as a gap, not forced.")

        flow, _ = realize_flow(truth, sel.flow_weight, flow_cfg["size"], rng)
        flow.pop("repwt", None)
        flow["weight"] = np.ones(len(flow["age"]))
        margins = manufacture_margins(flow, len(origin_names), controls)

        tilt = {name: (feature(flow, name, origin_names), coef) for name, coef in sample_cfg.get("tilt", {}).items()}
        sample, _ = draw_survey_sample(flow, sample_cfg["size"], tilt, rng)

        out = out_root / f"ses_{strength:+.2f}"
        out.mkdir(parents=True, exist_ok=True)
        flow_out = freeze_truth(flow, out / "flow.csv.gz", origins_from_manifest(manifest),
                                meta={"truth_sha256": manifest["sha256"], "ses_strength": strength})
        sample_out = freeze_truth(sample, out / "sample.csv.gz", origins_from_manifest(manifest),
                                  meta={"flow_sha256": flow_out["sha256"], "tilt": sample_cfg.get("tilt", {})})
        with open(out / "margins.json", "w") as f:
            json.dump(margins.as_dict(), f, indent=2)
        flow_comp = composition(flow, flow["weight"], origin_names, targets.elderly_age)
        sample_comp = composition(sample, sample["weight"], origin_names, targets.elderly_age)
        realized_gaps = _realized_gaps(flow_comp, targets, origin_names)
        print("Realized flow vs targets: " + ", ".join(f"{k} {v:+.4f}" for k, v in realized_gaps.items()))
        report = {
            "ses_strength": strength,
            "theta": sel.theta,
            "targets": sel.targets,
            "achieved_expected": sel.achieved,
            "gaps": sel.gaps,
            "converged": sel.converged,
            "within_tolerance": sel.within_tolerance,
            "flow_composition": flow_comp,
            "realized_gaps": realized_gaps,
            "sample_composition": sample_comp,
        }
        with open(out / "selection.json", "w") as f:
            json.dump(report, f, indent=2)
        log_run(
            cfg.get("run_log", "runs/run_log.jsonl"),
            f"Sprint 2: flow, margins and sample at SES strength {strength:+g}",
            {"targets": cfg["targets"], "flow": flow_cfg, "sample": sample_cfg, "controls": controls,
             "ses_strength": strength},
            seed=seed,
            inputs=[truth_path],
            outputs=[str(out / n) for n in ("flow.csv.gz", "sample.csv.gz", "margins.json", "selection.json")],
            results={"within_tolerance": sel.within_tolerance, "gaps": sel.gaps,
                     "flow_sha256": flow_out["sha256"], "sample_sha256": sample_out["sha256"]},
        )
        summaries.append(report)

    print("\nComposition of the flow by SES strength (same targets, different joints):")
    keys = ["mean_ses"] + [f"edattain={k}" for k in range(5)]
    print(f"{'strength':>9}" + "".join(f"{k:>14}" for k in keys))
    for r in summaries:
        print(f"{r['ses_strength']:>+9.2f}" + "".join(f"{r['flow_composition'][k]:>14.4f}" for k in keys))
    return summaries


def _realized_gaps(comp: Mapping[str, float], targets: DarienTargets, origin_names: Sequence[str]) -> dict[str, float]:
    """Gap between the drawn flow's composition and the targets (sampling error only)."""
    gaps = {f"origin={n}": comp[f"origin={n}"] - targets.origin_shares[n] for n in origin_names}
    gaps["male"] = comp["male"] - targets.male_share
    gaps["mean_age"] = comp["mean_age"] - targets.mean_age
    gaps[f"age>={targets.elderly_age}"] = comp[f"age>={targets.elderly_age}"] - targets.elderly_share
    return gaps


def origins_from_manifest(manifest: Mapping) -> list[OriginGroup]:
    return [OriginGroup(o["name"], tuple(o["pobp"])) for o in sorted(manifest["origins"], key=lambda o: o["index"])]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m synthpop.sprint2", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["freeze", "build"])
    parser.add_argument("config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    {"freeze": freeze, "build": build}[args.command](cfg)


if __name__ == "__main__":
    main()
