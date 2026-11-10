"""Truth loading, recodes and freezing."""

import json

import numpy as np
import pytest

from simulated import ORIGINS, simulated_truth
from synthpop.truth import (
    OriginGroup,
    age_band,
    collapse_schl_to_edattain,
    freeze_truth,
    load_frozen_truth,
    weighted_rank,
)

GROUPS = [OriginGroup(n, (100 + k,)) for k, n in enumerate(ORIGINS)]


def test_schl_to_edattain_boundaries():
    schl = np.array([np.nan, 1, 8, 9, 15, 16, 17, 19, 20, 21, 24])
    np.testing.assert_array_equal(collapse_schl_to_edattain(schl), [0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4])


def test_schl_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        collapse_schl_to_edattain(np.array([0.0]))
    with pytest.raises(ValueError):
        collapse_schl_to_edattain(np.array([25.0]))


def test_age_bands():
    np.testing.assert_array_equal(age_band(np.array([0, 4, 5, 17, 18, 64, 65, 99])), [0, 0, 1, 1, 2, 6, 7, 7])


def test_weighted_rank_ties_and_weights():
    r = weighted_rank(np.array([3.0, 1.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0, 2.0]))
    # sorted: 1,1 (w=2) | 2 (w=2) | 3 (w=1); total 5
    np.testing.assert_allclose(r, [4.5 / 5, 1 / 5, 1 / 5, 3 / 5])


def test_freeze_is_deterministic_and_round_trips(tmp_path):
    rec = simulated_truth(n=500)
    m1 = freeze_truth(rec, tmp_path / "a.csv.gz", GROUPS, meta={"note": "test"})
    m2 = freeze_truth(rec, tmp_path / "b.csv.gz", GROUPS, meta={"note": "test"})
    assert m1["sha256"] == m2["sha256"]
    back, manifest = load_frozen_truth(tmp_path / "a.csv.gz")
    assert manifest["n_records"] == 500
    for k in rec:
        if rec[k].dtype.kind == "f":
            np.testing.assert_array_equal(back[k], rec[k])  # repr() floats round-trip exactly
        else:
            np.testing.assert_array_equal(back[k], rec[k])


def test_tampered_truth_fails_the_hash_check(tmp_path):
    rec = simulated_truth(n=50)
    path = tmp_path / "t.csv.gz"
    freeze_truth(rec, path, GROUPS)
    mpath = tmp_path / "t.csv.gz.manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["sha256"] = "0" * 64
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        load_frozen_truth(path)


def test_load_pums_persons_from_csv(tmp_path):
    pd = pytest.importorskip("pandas")
    from synthpop.truth import load_pums_persons

    rows = []
    for i, (pobp, age, sex, schl) in enumerate(
        [(100, 30, 1, 21), (101, 2, 2, None), (999, 40, 1, 16), (102, 70, 2, 9), (100, 25, 2, 16)]
    ):
        row = {"SERIALNO": f"2022HU{i // 2:04d}", "SPORDER": i % 2 + 1, "AGEP": age, "SEX": sex, "POBP": pobp,
               "SCHL": schl, "PINCP": 1000 * (i + 1), "ADJINC": 1_042_311, "PWGTP": 10 + i}
        row.update({f"PWGTP{r}": 10 + i for r in range(1, 81)})
        rows.append(row)
    persons = tmp_path / "psam_p.csv"
    pd.DataFrame(rows).to_csv(persons, index=False)
    hh = tmp_path / "psam_h.csv"
    pd.DataFrame({"SERIALNO": ["2022HU0000", "2022HU0001", "2022HU0002"], "HINCP": [50_000, 20_000, 80_000]}).to_csv(hh, index=False)

    rec = load_pums_persons([persons], GROUPS)
    assert len(rec["age"]) == 4  # POBP 999 dropped
    np.testing.assert_array_equal(rec["origin"], [0, 1, 2, 0])
    np.testing.assert_array_equal(rec["male"], [1, 0, 0, 0])
    np.testing.assert_array_equal(rec["edattain"], [4, 0, 2, 3])
    assert rec["repwt"].shape == (4, 80)
    np.testing.assert_allclose(rec["income"][0], 1000 * 1.042311)

    rec_hh = load_pums_persons([persons], GROUPS, household_paths=[hh])
    np.testing.assert_allclose(rec_hh["income"], np.array([50_000, 50_000, 20_000, 80_000]) * 1.042311)


def test_duplicate_pobp_across_groups_is_rejected():
    pd = pytest.importorskip("pandas")
    from synthpop.truth import records_from_frame

    df = pd.DataFrame({"SERIALNO": ["a"], "SPORDER": [1], "AGEP": [30], "SEX": [1], "POBP": [100],
                       "SCHL": [16], "PINCP": [0], "ADJINC": [1_000_000], "PWGTP": [1]})
    with pytest.raises(ValueError, match="more than one"):
        records_from_frame(df, [OriginGroup("x", (100,)), OriginGroup("y", (100,))], ses_income="PINCP")
