"""Manufactured inputs: control margins and contingency tables from records.

The margins stand in for SENAFRONT's counts. They are computed exactly from
the flow, over categorical variables only, and **never** over the planted
survey-only attribute: the designs may see education only through the biased
sample. :func:`manufacture_margins` enforces that.

The output is in the ``(axes, target)`` format :func:`synthpop.ipf` takes, on
a table whose axes are ``Margins.variables`` in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from synthpop.truth import AGE_BANDS, Records

__all__ = [
    "DEFAULT_CONTROLS",
    "PLANTED_ATTRIBUTES",
    "Margins",
    "PlantedAttributeError",
    "crosstab",
    "manufacture_margins",
]

# The survey-only attribute and every recode of it.
PLANTED_ATTRIBUTES: frozenset[str] = frozenset({"schl", "edattain"})

# One-way origin, then sex and age band within origin. Shared axes agree by
# construction, since every margin is summed from the same flow.
DEFAULT_CONTROLS: tuple[tuple[str, ...], ...] = (
    ("origin",),
    ("origin", "male"),
    ("origin", "age_band"),
)


class PlantedAttributeError(ValueError):
    """A control margin was requested over the withheld survey-only attribute."""


@dataclass(frozen=True)
class Margins:
    """Control margins over ``variables``, ready for :func:`synthpop.ipf`."""

    variables: tuple[str, ...]
    levels: dict[str, int]
    margins: list[tuple[tuple[int, ...], np.ndarray]]
    controls: tuple[tuple[str, ...], ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.levels[v] for v in self.variables)

    def as_dict(self) -> dict:
        return {
            "variables": list(self.variables),
            "levels": dict(self.levels),
            "margins": [
                {"variables": [self.variables[a] for a in axes], "target": target.tolist()}
                for axes, target in self.margins
            ],
        }


def default_levels(n_origins: int) -> dict[str, int]:
    return {"origin": n_origins, "male": 2, "age_band": len(AGE_BANDS), "edattain": 5}


def crosstab(
    records: Records,
    variables: Sequence[str],
    levels: Mapping[str, int],
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted (default: unit-weight) counts over ``variables``."""
    shape = tuple(int(levels[v]) for v in variables)
    codes = []
    for v, n in zip(variables, shape):
        c = np.asarray(records[v]).astype(np.int64)
        if np.any((c < 0) | (c >= n)):
            raise ValueError(f"{v} has codes outside 0..{n - 1}")
        codes.append(c)
    flat = np.ravel_multi_index(codes, shape) if codes else np.zeros(0, dtype=np.int64)
    w = np.ones(len(flat)) if weight is None else np.asarray(weight, dtype=float)
    return np.bincount(flat, weights=w, minlength=int(np.prod(shape))).reshape(shape)


def manufacture_margins(
    flow: Records,
    n_origins: int,
    controls: Sequence[Sequence[str]] = DEFAULT_CONTROLS,
    planted: frozenset[str] = PLANTED_ATTRIBUTES,
) -> Margins:
    """Exact margins of the flow over each control, never over ``planted``."""
    controls = tuple(tuple(c) for c in controls)
    used = {v for c in controls for v in c}
    leaked = used & set(planted)
    if leaked:
        raise PlantedAttributeError(
            f"control margins may not include the survey-only attribute(s) {sorted(leaked)}"
        )
    variables = tuple(v for v in ("origin", "male", "age_band") if v in used) + tuple(
        sorted(used - {"origin", "male", "age_band"})
    )
    levels = {v: default_levels(n_origins)[v] for v in variables}
    full = crosstab(flow, variables, levels)
    margins = []
    for c in controls:
        axes = tuple(variables.index(v) for v in c)
        drop = tuple(a for a in range(len(variables)) if a not in axes)
        target = full.sum(axis=drop)
        # full.sum leaves kept axes in table order; reorder to match ``c``.
        order = np.argsort(np.argsort(axes))
        margins.append((axes, np.transpose(target, order) if target.ndim > 1 else target))
    return Margins(variables, levels, margins, controls)
