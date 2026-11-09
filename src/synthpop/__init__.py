"""Population synthesis designs and their verification."""

from synthpop.ipf import (
    InconsistentMarginsError,
    IPFConvergenceError,
    IPFResult,
    check_consistency,
    ipf,
    marginal,
)
from synthpop.margins import Margins, PlantedAttributeError, crosstab, manufacture_margins
from synthpop.selection import (
    CalibrationError,
    DarienTargets,
    SelectionResult,
    calibrate_selection,
    draw_survey_sample,
    realize_flow,
    select_flow,
    sweep_selection_strength,
)

__all__ = [
    "InconsistentMarginsError",
    "IPFConvergenceError",
    "IPFResult",
    "check_consistency",
    "ipf",
    "marginal",
    "CalibrationError",
    "DarienTargets",
    "Margins",
    "PlantedAttributeError",
    "SelectionResult",
    "calibrate_selection",
    "crosstab",
    "draw_survey_sample",
    "manufacture_margins",
    "realize_flow",
    "select_flow",
    "sweep_selection_strength",
]
