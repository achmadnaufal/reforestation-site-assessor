"""
Suitability scoring engine for reforestation site assessment.

Each scoring function returns a value in the range 0–100, where 100
represents the most suitable condition for tropical/Indonesian reforestation.
The composite score is a weighted average of individual sub-scores.

Scoring philosophy
------------------
- High rainfall is positive up to a threshold, then neutral.
- Steeper slopes reduce suitability; very steep terrain is unsuitable.
- Deeper soils are always better.
- Proximity to existing forest edges raises success probability.
- Road proximity aids logistics but very close proximity may indicate
  pressure; a moderate optimum is used.
- Severe degradation implies higher restoration potential (more urgent
  need) and therefore a *higher* priority score for intervention.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "rainfall": 0.20,
    "slope": 0.20,
    "soil_depth": 0.15,
    "forest_proximity": 0.20,
    "road_proximity": 0.10,
    "degradation": 0.15,
}

# Ensure weights are immutable at module level
DEFAULT_WEIGHTS = dict(DEFAULT_WEIGHTS)  # copy for safety


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def score_rainfall(annual_rainfall_mm: float) -> float:
    """Score annual rainfall suitability for tropical reforestation (0–100).

    Scoring logic:
    - < 500 mm  : very low (arid), score scales from 0 to 40.
    - 500–1200  : moderate, score scales from 40 to 70.
    - 1200–3000 : optimal tropical range, score 70–100.
    - > 3000 mm : waterlogging risk begins, score gently decreases toward 60.

    Args:
        annual_rainfall_mm: Mean annual precipitation in millimetres.
            Must be >= 0.

    Returns:
        Suitability score in range [0, 100].

    Raises:
        ValueError: If annual_rainfall_mm is negative.
    """
    if annual_rainfall_mm < 0:
        raise ValueError(
            f"annual_rainfall_mm must be non-negative, got {annual_rainfall_mm}."
        )

    if annual_rainfall_mm < 500:
        return float(np.clip(annual_rainfall_mm / 500 * 40, 0, 40))
    if annual_rainfall_mm < 1200:
        return float(40 + (annual_rainfall_mm - 500) / 700 * 30)
    if annual_rainfall_mm <= 3000:
        return float(70 + (annual_rainfall_mm - 1200) / 1800 * 30)
    # Excess rainfall — gentle penalty
    excess = annual_rainfall_mm - 3000
    return float(max(60.0, 100.0 - excess / 100))


def score_slope(slope_pct: float) -> float:
    """Score terrain slope suitability for reforestation (0–100).

    Gentler slopes retain soil and moisture and are easier to plant.

    Scoring logic:
    - 0–10 %  : ideal flat to gentle, 90–100.
    - 10–30 % : moderate, scores decline linearly from 90 to 50.
    - 30–60 % : steep, scores decline from 50 to 10.
    - > 60 %  : impractical, score = 0.

    Args:
        slope_pct: Terrain slope as a percentage (rise / run × 100).
            Must be in range [0, 100].

    Returns:
        Suitability score in range [0, 100].

    Raises:
        ValueError: If slope_pct is negative.
    """
    if slope_pct < 0.0:
        raise ValueError(
            f"slope_pct must be non-negative, got {slope_pct}."
        )
    # Values above 100 are physically extreme — treat as maximally steep (score 0)
    if slope_pct > 100.0:
        return 0.0

    if slope_pct <= 10:
        return float(90 + (10 - slope_pct) / 10 * 10)
    if slope_pct <= 30:
        return float(90 - (slope_pct - 10) / 20 * 40)
    if slope_pct <= 60:
        return float(50 - (slope_pct - 30) / 30 * 40)
    return 0.0


def score_soil_depth(soil_depth_cm: float) -> float:
    """Score soil depth suitability for tree establishment (0–100).

    Deeper soils support larger root systems and better water retention.

    Scoring logic:
    - < 20 cm  : very shallow, score scales 0–30.
    - 20–50 cm : shallow-moderate, score 30–70.
    - 50–100 cm: good depth, score 70–95.
    - > 100 cm : excellent, score 95–100.

    Args:
        soil_depth_cm: Effective rooting depth in centimetres. Must be >= 0.

    Returns:
        Suitability score in range [0, 100].

    Raises:
        ValueError: If soil_depth_cm is negative.
    """
    if soil_depth_cm < 0:
        raise ValueError(
            f"soil_depth_cm must be non-negative, got {soil_depth_cm}."
        )

    if soil_depth_cm < 20:
        return float(soil_depth_cm / 20 * 30)
    if soil_depth_cm < 50:
        return float(30 + (soil_depth_cm - 20) / 30 * 40)
    if soil_depth_cm <= 100:
        return float(70 + (soil_depth_cm - 50) / 50 * 25)
    return float(min(100.0, 95 + (soil_depth_cm - 100) / 100 * 5))


def score_forest_proximity(distance_to_forest_km: float) -> float:
    """Score proximity to existing forest for seed dispersal and microclimate (0–100).

    Proximity to forest remnants accelerates natural regeneration, reduces
    planting costs, and improves survival rates.

    Scoring logic:
    - 0–1 km   : directly adjacent, score 90–100.
    - 1–5 km   : close, score 60–90.
    - 5–15 km  : moderate distance, score 30–60.
    - > 15 km  : isolated, score diminishes toward 0.

    Args:
        distance_to_forest_km: Straight-line distance to the nearest
            forest boundary in kilometres. Must be >= 0.

    Returns:
        Suitability score in range [0, 100].

    Raises:
        ValueError: If distance_to_forest_km is negative.
    """
    if distance_to_forest_km < 0:
        raise ValueError(
            f"distance_to_forest_km must be non-negative, got {distance_to_forest_km}."
        )

    if distance_to_forest_km <= 1:
        return float(90 + (1 - distance_to_forest_km) * 10)
    if distance_to_forest_km <= 5:
        return float(90 - (distance_to_forest_km - 1) / 4 * 30)
    if distance_to_forest_km <= 15:
        return float(60 - (distance_to_forest_km - 5) / 10 * 30)
    return float(max(0.0, 30 - (distance_to_forest_km - 15) * 2))


def score_road_proximity(distance_to_road_km: float) -> float:
    """Score road accessibility for logistics and maintenance (0–100).

    Moderate proximity optimises planting crew access while reducing
    poaching/clearing pressure.

    Scoring logic:
    - < 0.5 km : very close — logistically easy but higher anthropogenic
                 pressure, score 70–80.
    - 0.5–5 km : optimal range, score 80–100.
    - 5–20 km  : increasing difficulty, score 100 → 40.
    - > 20 km  : remote, score diminishes toward 10.

    Args:
        distance_to_road_km: Distance to nearest accessible road in km.
            Must be >= 0.

    Returns:
        Suitability score in range [0, 100].

    Raises:
        ValueError: If distance_to_road_km is negative.
    """
    if distance_to_road_km < 0:
        raise ValueError(
            f"distance_to_road_km must be non-negative, got {distance_to_road_km}."
        )

    if distance_to_road_km < 0.5:
        return float(70 + distance_to_road_km / 0.5 * 10)
    if distance_to_road_km <= 5:
        return float(80 + (distance_to_road_km - 0.5) / 4.5 * 20)
    if distance_to_road_km <= 20:
        return float(100 - (distance_to_road_km - 5) / 15 * 60)
    return float(max(10.0, 40 - (distance_to_road_km - 20) * 1.5))


_DEGRADATION_BASE_SCORES: Dict[str, float] = {
    "low": 40.0,
    "medium": 60.0,
    "high": 80.0,
    "severe": 100.0,
}


def score_degradation(degradation_level: str) -> float:
    """Score degradation urgency for reforestation prioritisation (0–100).

    More severely degraded land receives a higher priority score because
    intervention is more urgent and ecological benefit is greater.

    Args:
        degradation_level: One of 'low', 'medium', 'high', or 'severe'
            (case-insensitive).

    Returns:
        Priority score in range [0, 100].

    Raises:
        ValueError: If degradation_level is not a recognised value.
    """
    normalised = degradation_level.strip().lower()
    if normalised not in _DEGRADATION_BASE_SCORES:
        raise ValueError(
            f"degradation_level '{degradation_level}' is not recognised. "
            f"Valid values: {list(_DEGRADATION_BASE_SCORES)}."
        )
    return _DEGRADATION_BASE_SCORES[normalised]


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_composite_score(
    rainfall_score: float,
    slope_score: float,
    soil_depth_score: float,
    forest_proximity_score: float,
    road_proximity_score: float,
    degradation_score: float,
    weights: Dict[str, float] | None = None,
) -> float:
    """Compute the weighted composite suitability score for a single site.

    Args:
        rainfall_score: Sub-score from :func:`score_rainfall` (0–100).
        slope_score: Sub-score from :func:`score_slope` (0–100).
        soil_depth_score: Sub-score from :func:`score_soil_depth` (0–100).
        forest_proximity_score: Sub-score from :func:`score_forest_proximity`
            (0–100).
        road_proximity_score: Sub-score from :func:`score_road_proximity`
            (0–100).
        degradation_score: Sub-score from :func:`score_degradation` (0–100).
        weights: Optional custom weight mapping. Keys must match
            ``DEFAULT_WEIGHTS``. Values must be non-negative and should sum
            to 1.0. Defaults to ``DEFAULT_WEIGHTS`` when ``None``.

    Returns:
        Composite score rounded to two decimal places in range [0, 100].

    Raises:
        ValueError: If any sub-score is outside [0, 100] or if weight keys
            do not match the expected set.
    """
    resolved_weights = weights if weights is not None else DEFAULT_WEIGHTS

    expected_keys = set(DEFAULT_WEIGHTS)
    provided_keys = set(resolved_weights)
    if expected_keys != provided_keys:
        raise ValueError(
            f"Weight keys mismatch. Expected {sorted(expected_keys)}, "
            f"got {sorted(provided_keys)}."
        )

    sub_scores: Dict[str, float] = {
        "rainfall": rainfall_score,
        "slope": slope_score,
        "soil_depth": soil_depth_score,
        "forest_proximity": forest_proximity_score,
        "road_proximity": road_proximity_score,
        "degradation": degradation_score,
    }

    for name, val in sub_scores.items():
        if not (0.0 <= val <= 100.0):
            raise ValueError(
                f"Sub-score '{name}' must be in [0, 100], got {val}."
            )

    composite = sum(
        sub_scores[k] * resolved_weights[k] for k in sub_scores
    )
    return round(float(np.clip(composite, 0.0, 100.0)), 2)


# ---------------------------------------------------------------------------
# DataFrame-level scoring
# ---------------------------------------------------------------------------

def score_dataframe(
    df: pd.DataFrame,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply all scoring functions to a validated site DataFrame.

    The input DataFrame must contain these columns (after preprocessing):
    ``elevation_m``, ``slope_pct``, ``annual_rainfall_mm``, ``soil_type``,
    ``distance_to_road_km``, ``distance_to_forest_km``, ``degradation_level``.

    A synthetic ``soil_depth_cm`` column is derived from soil_type when a
    ``soil_depth_cm`` column is not already present (see
    :func:`_derive_soil_depth`).

    Args:
        df: Preprocessed site DataFrame.
        weights: Optional custom weight mapping forwarded to
            :func:`compute_composite_score`.

    Returns:
        A *new* DataFrame (original is never modified) with additional
        columns: ``score_rainfall``, ``score_slope``, ``score_soil_depth``,
        ``score_forest_proximity``, ``score_road_proximity``,
        ``score_degradation``, and ``composite_score``.

    Raises:
        KeyError: If a required column is absent from the DataFrame.
    """
    result = df.copy()

    if "soil_depth_cm" not in result.columns:
        result = result.assign(soil_depth_cm=result["soil_type"].map(_derive_soil_depth))

    result = result.assign(
        score_rainfall=result["annual_rainfall_mm"].apply(score_rainfall),
        score_slope=result["slope_pct"].apply(score_slope),
        score_soil_depth=result["soil_depth_cm"].apply(score_soil_depth),
        score_forest_proximity=result["distance_to_forest_km"].apply(score_forest_proximity),
        score_road_proximity=result["distance_to_road_km"].apply(score_road_proximity),
        score_degradation=result["degradation_level"].apply(score_degradation),
    )

    result = result.assign(
        composite_score=result.apply(
            lambda row: compute_composite_score(
                rainfall_score=row["score_rainfall"],
                slope_score=row["score_slope"],
                soil_depth_score=row["score_soil_depth"],
                forest_proximity_score=row["score_forest_proximity"],
                road_proximity_score=row["score_road_proximity"],
                degradation_score=row["score_degradation"],
                weights=weights,
            ),
            axis=1,
        )
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SOIL_TYPE_DEPTH_MAP: Dict[str, float] = {
    "volcanic": 120.0,
    "alluvial": 110.0,
    "loam": 90.0,
    "peat": 80.0,
    "clay": 70.0,
    "silt": 65.0,
    "laterite": 40.0,
    "sandy": 35.0,
}


def _derive_soil_depth(soil_type: str) -> float:
    """Map a soil type string to an estimated rooting depth in centimetres.

    Uses typical tropical soil profiles. Returns the median value (70 cm)
    for unrecognised soil types to avoid hard failures on novel inputs.

    Args:
        soil_type: Soil classification string (case-insensitive).

    Returns:
        Estimated soil depth in centimetres.
    """
    return _SOIL_TYPE_DEPTH_MAP.get(str(soil_type).strip().lower(), 70.0)


def prioritise_sites(
    scored_df: pd.DataFrame,
    top_n: int = 10,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Return the highest-priority sites from a scored DataFrame.

    Args:
        scored_df: DataFrame produced by :func:`score_dataframe`, containing
            a ``composite_score`` column.
        top_n: Maximum number of sites to return. Must be > 0.
        min_score: Minimum composite score threshold (inclusive).
            Sites below this score are excluded before ranking.

    Returns:
        A new DataFrame sorted by ``composite_score`` descending, limited
        to ``top_n`` rows and filtered by ``min_score``.

    Raises:
        ValueError: If top_n <= 0 or min_score is outside [0, 100].
        KeyError: If ``composite_score`` column is absent.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}.")
    if not (0.0 <= min_score <= 100.0):
        raise ValueError(f"min_score must be in [0, 100], got {min_score}.")
    if "composite_score" not in scored_df.columns:
        raise KeyError("'composite_score' column is missing. Run score_dataframe() first.")

    filtered = scored_df[scored_df["composite_score"] >= min_score].copy()
    return filtered.sort_values("composite_score", ascending=False).head(top_n).reset_index(
        drop=True
    )
