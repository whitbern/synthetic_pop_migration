"""A simulated stand-in for the ACS truth, so Sprint 2 tests need no microdata."""

import numpy as np

from synthpop.truth import AGE_BANDS, age_band, collapse_schl_to_edattain, weighted_rank

ORIGINS = ("A", "B", "C")


def simulated_truth(n=20_000, seed=7, repwt=4):
    """Three origin groups of decreasing size with different age and education profiles.

    Education rises with income, so selecting on SES changes the education mix:
    the joint the calibration targets do not pin down.
    """
    rng = np.random.default_rng(seed)
    origin = rng.choice(3, size=n, p=[0.6, 0.3, 0.1])
    mean_age = np.array([38.0, 33.0, 28.0])[origin]
    age = np.clip(rng.gamma(4.0, mean_age / 4.0), 0, 95).round()
    male = (rng.random(n) < np.array([0.48, 0.5, 0.55])[origin]).astype(np.int64)
    income = rng.lognormal(10 + 0.3 * origin, 0.8)
    ses = weighted_rank(income, np.ones(n))
    # SCHL: higher with SES; missing (NaN) under age 3; capped by age for children.
    latent = 9 + 12 * ses + rng.normal(0, 3, n) - 1.5 * (origin == 2)
    schl = np.clip(np.round(latent), 1, 24)
    schl = np.where(age < 18, np.clip(np.round(age - 2), 1, 15), schl)
    schl = np.where(age < 3, np.nan, schl)
    weight = rng.uniform(5, 50, n).round()
    records = {
        "serialno": np.array([f"2022HU{i:07d}" for i in range(n)]),
        "sporder": np.ones(n, dtype=np.int64),
        "origin": origin.astype(np.int64),
        "age": age,
        "male": male,
        "age_band": age_band(age),
        "schl": schl,
        "edattain": collapse_schl_to_edattain(schl),
        "income": income,
        "ses": weighted_rank(income, weight),
        "weight": weight,
        "repwt": weight[:, None] * rng.uniform(0.8, 1.2, (n, repwt)),
    }
    assert records["age_band"].max() < len(AGE_BANDS)
    return records
