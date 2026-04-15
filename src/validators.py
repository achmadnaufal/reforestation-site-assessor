"""
Input validation utilities for reforestation site assessment data.

Provides schema-based validation with clear error messages for all
system boundary inputs including coordinates, environmental metrics,
and scoring fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATITUDE_RANGE: Tuple[float, float] = (-90.0, 90.0)
LONGITUDE_RANGE: Tuple[float, float] = (-180.0, 180.0)
ELEVATION_RANGE_M: Tuple[float, float] = (-500.0, 9000.0)
SLOPE_RANGE_PCT: Tuple[float, float] = (0.0, 100.0)
RAINFALL_RANGE_MM: Tuple[float, float] = (0.0, 15000.0)
SCORE_RANGE: Tuple[float, float] = (0.0, 100.0)
DISTANCE_RANGE_KM: Tuple[float, float] = (0.0, 10000.0)

REQUIRED_COLUMNS: List[str] = [
    "site_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_pct",
    "annual_rainfall_mm",
    "soil_type",
    "current_land_use",
    "distance_to_road_km",
    "distance_to_forest_km",
    "degradation_level",
]

VALID_SOIL_TYPES: List[str] = [
    "clay", "sandy", "loam", "silt", "peat", "laterite", "volcanic", "alluvial"
]

VALID_LAND_USE_TYPES: List[str] = [
    "bare_land", "grassland", "shrubland", "degraded_forest",
    "agricultural", "plantation", "mixed_vegetation",
]

VALID_DEGRADATION_LEVELS: List[str] = ["low", "medium", "high", "severe"]


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Immutable result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed without errors.
        errors: List of human-readable error messages, empty when valid.
        warnings: List of non-blocking advisory messages.
    """

    is_valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

    @classmethod
    def ok(cls, warnings: Optional[Tuple[str, ...]] = None) -> "ValidationResult":
        """Create a passing validation result.

        Args:
            warnings: Optional advisory messages that do not block processing.

        Returns:
            A ValidationResult with is_valid=True.
        """
        return cls(is_valid=True, errors=(), warnings=warnings or ())

    @classmethod
    def fail(
        cls,
        errors: Tuple[str, ...],
        warnings: Optional[Tuple[str, ...]] = None,
    ) -> "ValidationResult":
        """Create a failing validation result.

        Args:
            errors: Tuple of error messages describing what failed.
            warnings: Optional advisory messages.

        Returns:
            A ValidationResult with is_valid=False.
        """
        return cls(is_valid=False, errors=errors, warnings=warnings or ())


# ---------------------------------------------------------------------------
# Field-level validators
# ---------------------------------------------------------------------------

def validate_coordinate(
    latitude: float, longitude: float, site_id: str = "unknown"
) -> ValidationResult:
    """Validate that latitude and longitude are within valid geographic bounds.

    Args:
        latitude: Decimal degrees latitude. Valid range: -90 to 90.
        longitude: Decimal degrees longitude. Valid range: -180 to 180.
        site_id: Identifier used in error messages for traceability.

    Returns:
        ValidationResult indicating success or describing out-of-range values.
    """
    errors: List[str] = []

    if not (LATITUDE_RANGE[0] <= latitude <= LATITUDE_RANGE[1]):
        errors.append(
            f"Site '{site_id}': latitude {latitude} is outside valid range "
            f"{LATITUDE_RANGE[0]} to {LATITUDE_RANGE[1]}."
        )

    if not (LONGITUDE_RANGE[0] <= longitude <= LONGITUDE_RANGE[1]):
        errors.append(
            f"Site '{site_id}': longitude {longitude} is outside valid range "
            f"{LONGITUDE_RANGE[0]} to {LONGITUDE_RANGE[1]}."
        )

    if errors:
        return ValidationResult.fail(tuple(errors))
    return ValidationResult.ok()


def validate_elevation(elevation_m: float, site_id: str = "unknown") -> ValidationResult:
    """Validate that elevation is within a physically plausible range.

    Args:
        elevation_m: Elevation above sea level in metres.
        site_id: Identifier used in error messages.

    Returns:
        ValidationResult. Values below -500 m or above 9000 m are rejected.
    """
    lo, hi = ELEVATION_RANGE_M
    if not (lo <= elevation_m <= hi):
        return ValidationResult.fail(
            (
                f"Site '{site_id}': elevation_m {elevation_m} is outside valid range "
                f"{lo} to {hi} m.",
            )
        )
    return ValidationResult.ok()


def validate_slope(slope_pct: float, site_id: str = "unknown") -> ValidationResult:
    """Validate that slope percentage is within 0–100.

    Args:
        slope_pct: Terrain slope expressed as a percentage (rise/run * 100).
        site_id: Identifier used in error messages.

    Returns:
        ValidationResult. Negative values or values above 100 are rejected.
    """
    lo, hi = SLOPE_RANGE_PCT
    if not (lo <= slope_pct <= hi):
        return ValidationResult.fail(
            (
                f"Site '{site_id}': slope_pct {slope_pct} is outside valid range "
                f"{lo} to {hi}%.",
            )
        )
    return ValidationResult.ok()


def validate_rainfall(
    annual_rainfall_mm: float, site_id: str = "unknown"
) -> ValidationResult:
    """Validate that annual rainfall is within a physically plausible range.

    Args:
        annual_rainfall_mm: Annual rainfall total in millimetres.
        site_id: Identifier used in error messages.

    Returns:
        ValidationResult. Negative values or values above 15 000 mm are rejected.
    """
    lo, hi = RAINFALL_RANGE_MM
    if not (lo <= annual_rainfall_mm <= hi):
        return ValidationResult.fail(
            (
                f"Site '{site_id}': annual_rainfall_mm {annual_rainfall_mm} is outside "
                f"valid range {lo} to {hi} mm.",
            )
        )
    return ValidationResult.ok()


def validate_score(
    score: float, field_name: str = "score", site_id: str = "unknown"
) -> ValidationResult:
    """Validate that a suitability score lies within 0–100.

    Args:
        score: Numeric score to validate.
        field_name: Column name used in error messages.
        site_id: Identifier used in error messages.

    Returns:
        ValidationResult. Values outside 0–100 are rejected.
    """
    lo, hi = SCORE_RANGE
    if not (lo <= score <= hi):
        return ValidationResult.fail(
            (
                f"Site '{site_id}': {field_name} {score} is outside valid range "
                f"{lo} to {hi}.",
            )
        )
    return ValidationResult.ok()


def validate_distance(
    distance_km: float, field_name: str = "distance_km", site_id: str = "unknown"
) -> ValidationResult:
    """Validate that a distance value is non-negative and within a sane upper bound.

    Args:
        distance_km: Distance in kilometres.
        field_name: Column name used in error messages.
        site_id: Identifier used in error messages.

    Returns:
        ValidationResult. Negative values or values above 10 000 km are rejected.
    """
    lo, hi = DISTANCE_RANGE_KM
    if not (lo <= distance_km <= hi):
        return ValidationResult.fail(
            (
                f"Site '{site_id}': {field_name} {distance_km} is outside valid range "
                f"{lo} to {hi} km.",
            )
        )
    return ValidationResult.ok()


def validate_categorical(
    value: str,
    valid_values: List[str],
    field_name: str,
    site_id: str = "unknown",
) -> ValidationResult:
    """Validate that a categorical field contains a recognised value.

    Args:
        value: The string value to check.
        valid_values: List of accepted values (case-insensitive comparison).
        field_name: Column name used in error messages.
        site_id: Identifier used in error messages.

    Returns:
        ValidationResult. Unrecognised values are rejected with a hint listing
        all valid options.
    """
    normalised = value.strip().lower()
    accepted = [v.lower() for v in valid_values]
    if normalised not in accepted:
        return ValidationResult.fail(
            (
                f"Site '{site_id}': {field_name} '{value}' is not recognised. "
                f"Valid values: {valid_values}.",
            )
        )
    return ValidationResult.ok()


# ---------------------------------------------------------------------------
# DataFrame-level validator
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Validate an entire site-assessment DataFrame against domain rules.

    Checks are performed in this order:
    1. Required columns are present.
    2. No completely empty rows remain.
    3. Per-row field validation (coordinates, ranges, categorical values).

    Rows with missing values in numeric fields produce warnings rather than
    hard errors, so that partial data is still usable.

    Args:
        df: DataFrame containing site records. Column names must match those
            defined in REQUIRED_COLUMNS (case-insensitive after normalisation
            by :func:`~src.main.SiteAssessor.preprocess`).

    Returns:
        ValidationResult aggregating all errors and warnings found.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}.")

    if df.empty:
        errors.append("DataFrame contains no rows.")

    if errors:
        return ValidationResult.fail(tuple(errors), tuple(warnings))

    # 2. Per-row validation
    for idx, row in df.iterrows():
        site_id = str(row.get("site_id", f"row_{idx}"))

        # Coordinates
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.notna(lat) and pd.notna(lon):
            result = validate_coordinate(float(lat), float(lon), site_id)
            errors.extend(result.errors)
        else:
            warnings.append(f"Site '{site_id}': latitude/longitude is missing.")

        # Elevation
        elev = row.get("elevation_m")
        if pd.notna(elev):
            result = validate_elevation(float(elev), site_id)
            errors.extend(result.errors)
        else:
            warnings.append(f"Site '{site_id}': elevation_m is missing.")

        # Slope
        slope = row.get("slope_pct")
        if pd.notna(slope):
            result = validate_slope(float(slope), site_id)
            errors.extend(result.errors)

        # Rainfall
        rain = row.get("annual_rainfall_mm")
        if pd.notna(rain):
            result = validate_rainfall(float(rain), site_id)
            errors.extend(result.errors)

        # Distances
        for dist_field in ("distance_to_road_km", "distance_to_forest_km"):
            dist_val = row.get(dist_field)
            if pd.notna(dist_val):
                result = validate_distance(float(dist_val), dist_field, site_id)
                errors.extend(result.errors)

        # Categorical fields
        soil = row.get("soil_type")
        if pd.notna(soil):
            result = validate_categorical(str(soil), VALID_SOIL_TYPES, "soil_type", site_id)
            errors.extend(result.errors)

        land_use = row.get("current_land_use")
        if pd.notna(land_use):
            result = validate_categorical(
                str(land_use), VALID_LAND_USE_TYPES, "current_land_use", site_id
            )
            errors.extend(result.errors)

        degradation = row.get("degradation_level")
        if pd.notna(degradation):
            result = validate_categorical(
                str(degradation),
                VALID_DEGRADATION_LEVELS,
                "degradation_level",
                site_id,
            )
            errors.extend(result.errors)

    if errors:
        return ValidationResult.fail(tuple(errors), tuple(warnings))
    return ValidationResult.ok(tuple(warnings))
