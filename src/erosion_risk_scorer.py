"""
Soil erosion risk scoring for reforestation site assessment.

This module computes an integrated erosion-risk index for candidate
reforestation sites by combining slope steepness, rainfall erosivity,
soil erodibility (derived from soil type), and current land cover. The
formulation is inspired by the Revised Universal Soil Loss Equation
(RUSLE) factor framework (R, K, LS, C) but is rescaled to a bounded
0-100 risk score suitable for first-order site prioritisation rather
than quantitative soil-loss prediction.

A higher score indicates **higher erosion risk** and therefore a
greater need for protective interventions (terracing, contour planting,
cover crops, fast-establishing pioneer species).

Key design choices
------------------
- Each component contributes a sub-score in [0, 100] and is combined
  via a weighted sum with weights that **must sum to 1.0**.
- All public functions return new objects; inputs are never mutated.
- Categorical lookups are case-insensitive and reject unknown values
  with a clear ``ValueError``.

References
----------
- Wischmeier, W.H. & Smith, D.D. (1978). *Predicting Rainfall Erosion
  Losses* (USDA Agriculture Handbook 537).
- Renard, K.G. et al. (1997). *Predicting Soil Erosion by Water:
  RUSLE* (USDA Agriculture Handbook 703).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default weights for the four risk components (must sum to 1.0).
DEFAULT_EROSION_WEIGHTS: Dict[str, float] = {
    "slope": 0.40,
    "rainfall": 0.25,
    "soil": 0.20,
    "land_cover": 0.15,
}

#: Tolerance for the weight-sum check.
_WEIGHT_SUM_TOLERANCE: float = 1e-6

#: Soil erodibility (K-factor analogue) on a 0-100 risk scale.
#: Higher = more easily eroded.
_SOIL_ERODIBILITY: Dict[str, float] = {
    "sandy": 85.0,
    "silt": 90.0,
    "loam": 50.0,
    "clay": 35.0,
    "laterite": 65.0,
    "peat": 30.0,
    "alluvial": 55.0,
    "volcanic": 45.0,
}

#: Land-cover protection (C-factor analogue) on a 0-100 risk scale.
#: Higher = less protective cover, thus higher erosion risk.
_LAND_COVER_RISK: Dict[str, float] = {
    "bare_land": 95.0,
    "grassland": 55.0,
    "shrubland": 40.0,
    "agricultural": 70.0,
    "plantation": 35.0,
    "degraded_forest": 30.0,
    "mixed_vegetation": 25.0,
}

#: Required columns for DataFrame-level erosion scoring.
_REQUIRED_COLUMNS = (
    "slope_pct",
    "annual_rainfall_mm",
    "soil_type",
    "current_land_use",
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErosionRiskBreakdown:
    """Per-site decomposition of an erosion-risk score.

    Attributes:
        slope_score: Slope contribution in [0, 100].
        rainfall_score: Rainfall-erosivity contribution in [0, 100].
        soil_score: Soil-erodibility contribution in [0, 100].
        land_cover_score: Land-cover protection contribution in [0, 100].
        composite_score: Weighted composite risk in [0, 100].
        risk_class: Categorical band: 'low', 'moderate', 'high', 'severe'.
    """

    slope_score: float
    rainfall_score: float
    soil_score: float
    land_cover_score: float
    composite_score: float
    risk_class: str


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def score_slope_erosion(slope_pct: float) -> float:
    """Score slope contribution to erosion risk (0-100, higher = riskier).

    Mirrors RUSLE's LS factor in spirit: risk grows non-linearly with
    slope and saturates above ~40 %.

    Args:
        slope_pct: Terrain slope as a percentage. Must be >= 0.

    Returns:
        Erosion-risk sub-score in [0, 100].

    Raises:
        ValueError: If ``slope_pct`` is negative.
    """
    if slope_pct < 0:
        raise ValueError(f"slope_pct must be non-negative, got {slope_pct}.")
    if slope_pct >= 40:
        return 100.0
    # Quadratic ramp: gentle at low slopes, steep above ~15 %.
    return float(np.clip((slope_pct / 40.0) ** 1.4 * 100.0, 0.0, 100.0))


def score_rainfall_erosivity(annual_rainfall_mm: float) -> float:
    """Score rainfall erosivity contribution to erosion risk (0-100).

    A simplified R-factor analogue: very low rainfall produces little
    erosion; rainfall above ~3500 mm/yr is highly erosive in the tropics.

    Args:
        annual_rainfall_mm: Mean annual precipitation in mm. Must be >= 0.

    Returns:
        Erosion-risk sub-score in [0, 100].

    Raises:
        ValueError: If ``annual_rainfall_mm`` is negative.
    """
    if annual_rainfall_mm < 0:
        raise ValueError(
            f"annual_rainfall_mm must be non-negative, got {annual_rainfall_mm}."
        )
    if annual_rainfall_mm <= 500:
        return float(annual_rainfall_mm / 500.0 * 20.0)
    if annual_rainfall_mm <= 3500:
        return float(20.0 + (annual_rainfall_mm - 500.0) / 3000.0 * 75.0)
    return float(min(100.0, 95.0 + (annual_rainfall_mm - 3500.0) / 1000.0 * 5.0))


def score_soil_erodibility(soil_type: str) -> float:
    """Score soil-type erodibility contribution to erosion risk (0-100).

    Args:
        soil_type: One of the recognised soil-type strings
            (case-insensitive). See ``_SOIL_ERODIBILITY`` keys.

    Returns:
        Erosion-risk sub-score in [0, 100].

    Raises:
        ValueError: If ``soil_type`` is unrecognised.
    """
    key = str(soil_type).strip().lower()
    if key not in _SOIL_ERODIBILITY:
        raise ValueError(
            f"soil_type '{soil_type}' is not recognised. "
            f"Valid values: {sorted(_SOIL_ERODIBILITY)}."
        )
    return _SOIL_ERODIBILITY[key]


def score_land_cover_protection(current_land_use: str) -> float:
    """Score land-cover contribution to erosion risk (0-100).

    Args:
        current_land_use: One of the recognised land-use strings
            (case-insensitive). See ``_LAND_COVER_RISK`` keys.

    Returns:
        Erosion-risk sub-score in [0, 100].

    Raises:
        ValueError: If ``current_land_use`` is unrecognised.
    """
    key = str(current_land_use).strip().lower()
    if key not in _LAND_COVER_RISK:
        raise ValueError(
            f"current_land_use '{current_land_use}' is not recognised. "
            f"Valid values: {sorted(_LAND_COVER_RISK)}."
        )
    return _LAND_COVER_RISK[key]


# ---------------------------------------------------------------------------
# Composite scorer & helpers
# ---------------------------------------------------------------------------

def _classify_risk(score: float) -> str:
    """Map a numeric risk score to a categorical band."""
    if score < 30.0:
        return "low"
    if score < 55.0:
        return "moderate"
    if score < 80.0:
        return "high"
    return "severe"


def _validate_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Return a copy of ``weights`` after validating keys, signs, and sum."""
    expected = set(DEFAULT_EROSION_WEIGHTS)
    provided = set(weights)
    if expected != provided:
        raise ValueError(
            f"Erosion weight keys mismatch. Expected {sorted(expected)}, "
            f"got {sorted(provided)}."
        )
    for name, val in weights.items():
        if val < 0:
            raise ValueError(
                f"Weight '{name}' must be non-negative, got {val}."
            )
    total = float(sum(weights.values()))
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"Erosion weights must sum to 1.0 (got {total:.6f})."
        )
    return dict(weights)


def compute_erosion_risk(
    slope_pct: float,
    annual_rainfall_mm: float,
    soil_type: str,
    current_land_use: str,
    weights: Mapping[str, float] | None = None,
) -> ErosionRiskBreakdown:
    """Compute an integrated erosion-risk score for a single site.

    Args:
        slope_pct: Terrain slope percentage (>= 0).
        annual_rainfall_mm: Mean annual precipitation in mm (>= 0).
        soil_type: Soil classification string. Must be a recognised value.
        current_land_use: Current land-use classification.  Must be a
            recognised value.
        weights: Optional custom weight mapping. Keys must match
            ``DEFAULT_EROSION_WEIGHTS``. Values must be non-negative and
            sum to 1.0. Defaults to ``DEFAULT_EROSION_WEIGHTS``.

    Returns:
        An :class:`ErosionRiskBreakdown` with sub-scores, the composite
        risk score in [0, 100], and a categorical risk class.

    Raises:
        ValueError: If any numeric input is negative, any categorical
            input is unrecognised, or the weights are invalid.
    """
    resolved = _validate_weights(weights) if weights is not None else dict(DEFAULT_EROSION_WEIGHTS)

    slope_s = score_slope_erosion(slope_pct)
    rain_s = score_rainfall_erosivity(annual_rainfall_mm)
    soil_s = score_soil_erodibility(soil_type)
    cover_s = score_land_cover_protection(current_land_use)

    composite = (
        slope_s * resolved["slope"]
        + rain_s * resolved["rainfall"]
        + soil_s * resolved["soil"]
        + cover_s * resolved["land_cover"]
    )
    composite = round(float(np.clip(composite, 0.0, 100.0)), 2)

    return ErosionRiskBreakdown(
        slope_score=round(slope_s, 2),
        rainfall_score=round(rain_s, 2),
        soil_score=round(soil_s, 2),
        land_cover_score=round(cover_s, 2),
        composite_score=composite,
        risk_class=_classify_risk(composite),
    )


# ---------------------------------------------------------------------------
# DataFrame-level scorer
# ---------------------------------------------------------------------------

def score_erosion_dataframe(
    df: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute erosion-risk scores for every row in a site DataFrame.

    The input must contain the columns: ``slope_pct``,
    ``annual_rainfall_mm``, ``soil_type``, ``current_land_use``.

    Args:
        df: Site DataFrame (typically the validated, preprocessed output
            of :class:`SiteAssessor`).
        weights: Optional custom weight mapping (see
            :func:`compute_erosion_risk`).

    Returns:
        A *new* DataFrame (the input is never mutated) with these added
        columns: ``erosion_slope_score``, ``erosion_rainfall_score``,
        ``erosion_soil_score``, ``erosion_land_cover_score``,
        ``erosion_risk_score``, ``erosion_risk_class``.

    Raises:
        KeyError: If any required column is absent.
        ValueError: If any row contains an invalid value or weights are
            invalid.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s) for erosion scoring: {missing}. "
            f"Required: {list(_REQUIRED_COLUMNS)}."
        )

    if df.empty:
        # Return a copy with the expected new columns but no rows.
        result = df.copy()
        for col in (
            "erosion_slope_score",
            "erosion_rainfall_score",
            "erosion_soil_score",
            "erosion_land_cover_score",
            "erosion_risk_score",
        ):
            result[col] = pd.Series(dtype=float)
        result["erosion_risk_class"] = pd.Series(dtype=object)
        return result

    breakdowns = [
        compute_erosion_risk(
            slope_pct=float(row["slope_pct"]),
            annual_rainfall_mm=float(row["annual_rainfall_mm"]),
            soil_type=row["soil_type"],
            current_land_use=row["current_land_use"],
            weights=weights,
        )
        for _, row in df.iterrows()
    ]

    return df.assign(
        erosion_slope_score=[b.slope_score for b in breakdowns],
        erosion_rainfall_score=[b.rainfall_score for b in breakdowns],
        erosion_soil_score=[b.soil_score for b in breakdowns],
        erosion_land_cover_score=[b.land_cover_score for b in breakdowns],
        erosion_risk_score=[b.composite_score for b in breakdowns],
        erosion_risk_class=[b.risk_class for b in breakdowns],
    )
