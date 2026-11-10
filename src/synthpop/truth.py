"""Planted ground truth: load, recode and freeze ACS PUMS person records.

The truth is the ACS 2020–2024 5-year PUMS restricted to the chosen origin
groups (place of birth, ``POBP``). Every later step (selection, margins, the
biased sample, every synthesis design) reads the frozen file written by
:func:`freeze_truth`, never the raw extract, so each run can be traced back to
one file by its SHA-256 hash.

Records are held as a plain ``dict[str, np.ndarray]`` (one array per column,
equal length). That keeps the core numpy-only; pandas is needed only to read
the raw PUMS CSVs.

Census PUMS column names are used throughout (``AGEP``, ``SEX``, ``POBP``,
``SCHL``, ``PWGTP``, ``PWGTP1``–``PWGTP80``, ``PINCP``, ``HINCP``,
``ADJINC``). Check the codes you pass for ``POBP`` against the PUMS data
dictionary for the vintage you downloaded.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "AGE_BANDS",
    "EDATTAIN_LABELS",
    "OriginGroup",
    "Records",
    "age_band",
    "collapse_schl_to_edattain",
    "file_sha256",
    "freeze_truth",
    "load_frozen_truth",
    "load_pums_persons",
    "n_records",
    "records_from_frame",
    "subset",
    "weighted_rank",
]

Records = dict[str, np.ndarray]

N_REPLICATES = 80
REPLICATE_COLUMNS = [f"PWGTP{r}" for r in range(1, N_REPLICATES + 1)]

# Lower edges of the age bands used for margins. The last band is open.
AGE_BANDS: tuple[int, ...] = (0, 5, 18, 25, 35, 45, 55, 65)

# IPUMS International EDATTAIN, general version. 0 = not in universe.
EDATTAIN_LABELS = {
    0: "NIU (under 3 / not reported)",
    1: "Less than primary completed",
    2: "Primary completed",
    3: "Secondary completed",
    4: "University completed",
}


@dataclass(frozen=True)
class OriginGroup:
    """An origin group: a label and the ``POBP`` codes that make it up."""

    name: str
    pobp: tuple[int, ...]

    @classmethod
    def from_mapping(cls, name: str, codes: Iterable[int]) -> "OriginGroup":
        codes = tuple(int(c) for c in codes)
        if not codes:
            raise ValueError(f"origin group {name!r} has no POBP codes")
        return cls(name, codes)


def n_records(records: Records) -> int:
    lengths = {len(v) for v in records.values()}
    if len(lengths) != 1:
        raise ValueError(f"columns have different lengths: {sorted(lengths)}")
    return lengths.pop()


def subset(records: Records, index: np.ndarray) -> Records:
    """Rows of ``records`` selected by a boolean mask or an integer index."""
    return {k: v[index] for k, v in records.items()}


# --- Recodes -----------------------------------------------------------------


def collapse_schl_to_edattain(schl: np.ndarray) -> np.ndarray:
    """Collapse ACS ``SCHL`` (1–24) to IPUMS International ``EDATTAIN`` (0–4).

    The plan requires this before the robustness replication, so the ACS truth
    and the pooled IPUMS International truth measure the same outcome.
    Primary completion is taken as six years (UN/ISCED convention used by
    EDATTAIN):

    ========  ======================================  ========
    SCHL      Meaning                                 EDATTAIN
    ========  ======================================  ========
    missing   under 3 years old                       0 (NIU)
    1–8       none, preschool, K, grades 1–5          1
    9–15      grade 6 through grade 12, no diploma    2
    16–20     diploma/GED, some college, associate    3
    21–24     bachelor's and above                    4
    ========  ======================================  ========

    Putting the associate's degree (20) in "secondary completed" is a choice:
    EDATTAIN's "university completed" means a first university degree.
    """
    schl = np.asarray(schl, dtype=float)
    out = np.zeros(schl.shape, dtype=np.int64)
    valid = np.isfinite(schl)
    bad = valid & ((schl < 1) | (schl > 24) | (schl != np.round(schl)))
    if np.any(bad):
        raise ValueError(f"SCHL values outside 1–24: {np.unique(schl[bad])[:10].tolist()}")
    out[valid & (schl <= 8)] = 1
    out[valid & (schl >= 9) & (schl <= 15)] = 2
    out[valid & (schl >= 16) & (schl <= 20)] = 3
    out[valid & (schl >= 21)] = 4
    return out


def age_band(age: np.ndarray, edges: Sequence[int] = AGE_BANDS) -> np.ndarray:
    """Index of the age band each age falls in (0 = youngest)."""
    age = np.asarray(age, dtype=float)
    if np.any(age < edges[0]):
        raise ValueError("age below the first band edge")
    return np.searchsorted(np.asarray(edges, dtype=float), age, side="right") - 1


def weighted_rank(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mid-rank of each value in [0, 1]; ties share a rank.

    Used to turn income into a scale-free SES proxy, so a selection
    coefficient on SES means the same thing whatever the income scale.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    total = ws.sum()
    uniq, start = np.unique(xs, return_index=True)
    cum = np.concatenate([[0.0], np.cumsum(ws)])
    end = np.append(start[1:], len(xs))
    mid = (cum[start] + cum[end]) / 2 / total
    ranks = np.empty_like(x)
    ranks[order] = np.repeat(mid, end - start)
    return ranks


# --- Loading -----------------------------------------------------------------

PERSON_COLUMNS = ["SERIALNO", "SPORDER", "AGEP", "SEX", "POBP", "SCHL", "PINCP", "ADJINC", "PWGTP"]


def records_from_frame(df, origins: Sequence[OriginGroup], ses_income: str = "HINCP") -> Records:
    """Recode a PUMS person frame (already filtered to the origin groups).

    Output columns:

    ``serialno`` (str), ``sporder``, ``origin`` (index into ``origins``),
    ``age``, ``male`` (1/0), ``age_band``, ``schl`` (raw, NaN under 3),
    ``edattain``, ``income`` (ADJINC-adjusted dollars), ``ses`` (weighted
    rank of ``income`` within the truth, 0–1), ``weight`` (PWGTP) and
    ``repwt`` (n × 80 replicate weights).
    """
    code_to_origin = {}
    for k, group in enumerate(origins):
        for c in group.pobp:
            if c in code_to_origin:
                raise ValueError(f"POBP code {c} is in more than one origin group")
            code_to_origin[c] = k
    pobp = df["POBP"].to_numpy().astype(int)
    origin = np.array([code_to_origin.get(c, -1) for c in pobp], dtype=np.int64)
    if np.any(origin < 0):
        raise ValueError("frame contains POBP codes outside the origin groups")

    sex = df["SEX"].to_numpy().astype(int)
    if not set(np.unique(sex)) <= {1, 2}:
        raise ValueError("SEX must be coded 1 (male) / 2 (female)")
    age = df["AGEP"].to_numpy().astype(float)
    schl = df["SCHL"].to_numpy().astype(float)
    adjinc = df["ADJINC"].to_numpy().astype(float) / 1e6
    if ses_income not in df.columns:
        raise ValueError(f"income column {ses_income!r} not in frame")
    income = df[ses_income].to_numpy().astype(float) * adjinc
    # Children and people in group quarters have no income reported; treat as 0
    # so they rank at the bottom rather than dropping out.
    income = np.where(np.isfinite(income), income, 0.0)
    weight = df["PWGTP"].to_numpy().astype(float)

    rep_cols = [c for c in REPLICATE_COLUMNS if c in df.columns]
    if rep_cols and len(rep_cols) != N_REPLICATES:
        raise ValueError(f"expected {N_REPLICATES} replicate weights, found {len(rep_cols)}")
    repwt = df[rep_cols].to_numpy().astype(float) if rep_cols else np.zeros((len(df), 0))

    return {
        "serialno": df["SERIALNO"].astype(str).to_numpy(),
        "sporder": df["SPORDER"].to_numpy().astype(np.int64),
        "origin": origin,
        "age": age,
        "male": (sex == 1).astype(np.int64),
        "age_band": age_band(age),
        "schl": schl,
        "edattain": collapse_schl_to_edattain(schl),
        "income": income,
        "ses": weighted_rank(income, weight),
        "weight": weight,
        "repwt": repwt,
    }


def load_pums_persons(
    person_paths: Sequence[str | Path],
    origins: Sequence[OriginGroup],
    household_paths: Sequence[str | Path] = (),
    chunksize: int = 500_000,
) -> Records:
    """Read raw PUMS person CSVs, keep the origin groups, recode.

    If household CSVs are given, ``HINCP`` (household income) is joined on
    ``SERIALNO`` and used as the SES proxy; otherwise personal income
    ``PINCP`` is used, which ranks every child at zero.
    """
    import pandas as pd  # only needed for raw extracts

    keep = {c for g in origins for c in g.pobp}
    usecols = PERSON_COLUMNS + REPLICATE_COLUMNS
    parts = []
    for path in person_paths:
        for chunk in pd.read_csv(
            path, usecols=lambda c: c in usecols, chunksize=chunksize, dtype={"SERIALNO": str}
        ):
            parts.append(chunk[chunk["POBP"].isin(keep)])
    if not parts:
        raise ValueError("no person files read")
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        raise ValueError("no records match the origin groups' POBP codes")

    ses_income = "PINCP"
    if household_paths:
        serials = set(df["SERIALNO"])
        hh = []
        for path in household_paths:
            for chunk in pd.read_csv(
                path, usecols=["SERIALNO", "HINCP"], chunksize=chunksize, dtype={"SERIALNO": str}
            ):
                hh.append(chunk[chunk["SERIALNO"].isin(serials)])
        hh = pd.concat(hh, ignore_index=True).drop_duplicates("SERIALNO")
        df = df.merge(hh, on="SERIALNO", how="left", validate="many_to_one")
        ses_income = "HINCP"
    return records_from_frame(df, origins, ses_income=ses_income)


# --- Freezing ----------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_SCALAR_COLUMNS = ["serialno", "sporder", "origin", "age", "male", "age_band", "schl", "edattain", "income", "ses", "weight"]
# Written when present: links from a flow back to the truth, and flow ids.
_OPTIONAL_COLUMNS = ["truth_index", "flow_id"]


def freeze_truth(
    records: Records,
    path: str | Path,
    origins: Sequence[OriginGroup],
    meta: Mapping[str, object] | None = None,
) -> dict:
    """Write the truth to a gzipped CSV plus a JSON manifest; return the manifest.

    The CSV is written deterministically (fixed column order, fixed float
    format, gzip mtime 0, no stored file name), so the same records always
    give the same hash.
    The manifest sits beside it as ``<path>.manifest.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = n_records(records)
    repwt = records.get("repwt", np.zeros((n, 0)))
    scalar = _SCALAR_COLUMNS + [c for c in _OPTIONAL_COLUMNS if c in records]
    header = scalar + [f"repwt{r + 1}" for r in range(repwt.shape[1])]
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        text = (line.encode() for line in _csv_lines(records, repwt, scalar, header))
        for line in text:
            gz.write(line)
    manifest = {
        "file": path.name,
        "sha256": file_sha256(path),
        "n_records": n,
        "weighted_total": float(records["weight"].sum()),
        "origins": [{"index": k, "name": g.name, "pobp": list(g.pobp)} for k, g in enumerate(origins)],
        "records_per_origin": {g.name: int((records["origin"] == k).sum()) for k, g in enumerate(origins)},
        "columns": header,
        "meta": dict(meta or {}),
    }
    with open(str(path) + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _fmt(v) -> str:
    if isinstance(v, (str, np.str_)):
        return str(v)
    v = float(v)
    if np.isnan(v):
        return ""
    if v == int(v):
        return str(int(v))
    return repr(v)


def _csv_lines(records: Records, repwt: np.ndarray, scalar: list[str], header: list[str]):
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    yield buf.getvalue()
    for i in range(n_records(records)):
        buf.seek(0)
        buf.truncate()
        row = [_fmt(records[c][i]) for c in scalar] + [_fmt(x) for x in repwt[i]]
        writer.writerow(row)
        yield buf.getvalue()


def load_frozen_truth(path: str | Path, verify: bool = True) -> tuple[Records, dict]:
    """Read a frozen truth and its manifest; check the hash unless ``verify=False``."""
    path = Path(path)
    with open(str(path) + ".manifest.json") as f:
        manifest = json.load(f)
    if verify:
        digest = file_sha256(path)
        if digest != manifest["sha256"]:
            raise ValueError(f"{path} hash {digest} does not match manifest {manifest['sha256']}")
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    cols = list(zip(*rows)) if rows else [()] * len(header)
    data = dict(zip(header, cols))

    def num(name, dtype=float):
        return np.array([float(x) if x != "" else np.nan for x in data[name]], dtype=float).astype(dtype)

    records: Records = {"serialno": np.array(data["serialno"], dtype=str)}
    for c in ["sporder", "origin", "male", "age_band", "edattain"] + [c for c in _OPTIONAL_COLUMNS if c in data]:
        records[c] = num(c).astype(np.int64)
    for c in ["age", "schl", "income", "ses", "weight"]:
        records[c] = num(c)
    rep = [c for c in header if c.startswith("repwt")]
    records["repwt"] = np.column_stack([num(c) for c in rep]) if rep else np.zeros((len(rows), 0))
    return records, manifest
