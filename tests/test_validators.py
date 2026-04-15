"""
Unit tests for src/validators.py.

Tests cover:
- Individual field validators (coordinates, elevation, slope, rainfall,
  score, distance, categorical)
- ValidationResult immutability
- DataFrame-level validation (required columns, per-row checks)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.validators import (
    REQUIRED_COLUMNS,
    VALID_DEGRADATION_LEVELS,
    VALID_LAND_USE_TYPES,
    VALID_SOIL_TYPES,
    ValidationResult,
    validate_categorical,
    validate_coordinate,
    validate_dataframe,
    validate_distance,
    validate_elevation,
    validate_rainfall,
    validate_score,
    validate_slope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_row() -> dict:
    return {
        "site_id": "TEST-0001",
        "latitude": -2.5,
        "longitude": 117.0,
        "elevation_m": 200.0,
        "slope_pct": 15.0,
        "annual_rainfall_mm": 2000.0,
        "soil_type": "loam",
        "current_land_use": "degraded_forest",
        "distance_to_road_km": 3.0,
        "distance_to_forest_km": 4.0,
        "degradation_level": "high",
    }


def _valid_df(n: int = 3) -> pd.DataFrame:
    row = _valid_row()
    rows = [{**row, "site_id": f"TEST-{i:04d}"} for i in range(n)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_ok_factory_sets_is_valid_true(self):
        result = ValidationResult.ok()
        assert result.is_valid is True

    def test_ok_factory_has_empty_errors(self):
        result = ValidationResult.ok()
        assert result.errors == ()

    def test_fail_factory_sets_is_valid_false(self):
        result = ValidationResult.fail(("some error",))
        assert result.is_valid is False

    def test_fail_factory_stores_errors(self):
        errors = ("error 1", "error 2")
        result = ValidationResult.fail(errors)
        assert result.errors == errors

    def test_ok_with_warnings(self):
        result = ValidationResult.ok(warnings=("advisory",))
        assert result.is_valid is True
        assert "advisory" in result.warnings

    def test_result_is_immutable(self):
        result = ValidationResult.ok()
        with pytest.raises((AttributeError, TypeError)):
            result.is_valid = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_coordinate
# ---------------------------------------------------------------------------

class TestValidateCoordinate:
    def test_valid_indonesian_coordinates(self):
        result = validate_coordinate(-2.5, 117.0)
        assert result.is_valid

    def test_extreme_valid_coordinates(self):
        assert validate_coordinate(-90.0, -180.0).is_valid
        assert validate_coordinate(90.0, 180.0).is_valid

    def test_latitude_too_high(self):
        result = validate_coordinate(91.0, 0.0)
        assert not result.is_valid
        assert any("latitude" in e for e in result.errors)

    def test_latitude_too_low(self):
        result = validate_coordinate(-91.0, 0.0)
        assert not result.is_valid

    def test_longitude_too_high(self):
        result = validate_coordinate(0.0, 181.0)
        assert not result.is_valid
        assert any("longitude" in e for e in result.errors)

    def test_longitude_too_low(self):
        result = validate_coordinate(0.0, -181.0)
        assert not result.is_valid

    def test_both_invalid_produces_two_errors(self):
        result = validate_coordinate(91.0, 181.0)
        assert len(result.errors) == 2

    def test_site_id_appears_in_error(self):
        result = validate_coordinate(91.0, 0.0, site_id="SITE-XYZ")
        assert any("SITE-XYZ" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_elevation
# ---------------------------------------------------------------------------

class TestValidateElevation:
    def test_valid_elevation(self):
        assert validate_elevation(500.0).is_valid

    def test_sea_level_valid(self):
        assert validate_elevation(0.0).is_valid

    def test_slightly_below_sea_level_valid(self):
        assert validate_elevation(-100.0).is_valid

    def test_too_low_elevation(self):
        result = validate_elevation(-600.0)
        assert not result.is_valid

    def test_too_high_elevation(self):
        result = validate_elevation(9500.0)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_slope
# ---------------------------------------------------------------------------

class TestValidateSlope:
    def test_valid_slope(self):
        assert validate_slope(30.0).is_valid

    def test_zero_slope_valid(self):
        assert validate_slope(0.0).is_valid

    def test_100pct_slope_valid(self):
        assert validate_slope(100.0).is_valid

    def test_negative_slope_invalid(self):
        result = validate_slope(-1.0)
        assert not result.is_valid

    def test_above_100_invalid(self):
        result = validate_slope(101.0)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_rainfall
# ---------------------------------------------------------------------------

class TestValidateRainfall:
    def test_valid_tropical_rainfall(self):
        assert validate_rainfall(2500.0).is_valid

    def test_zero_rainfall_valid(self):
        assert validate_rainfall(0.0).is_valid

    def test_negative_rainfall_invalid(self):
        result = validate_rainfall(-1.0)
        assert not result.is_valid

    def test_extreme_rainfall_invalid(self):
        result = validate_rainfall(16000.0)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_score
# ---------------------------------------------------------------------------

class TestValidateScore:
    def test_valid_score(self):
        assert validate_score(75.0).is_valid

    def test_zero_score_valid(self):
        assert validate_score(0.0).is_valid

    def test_100_score_valid(self):
        assert validate_score(100.0).is_valid

    def test_negative_score_invalid(self):
        result = validate_score(-1.0)
        assert not result.is_valid

    def test_above_100_invalid(self):
        result = validate_score(100.1)
        assert not result.is_valid

    def test_field_name_in_error(self):
        result = validate_score(-5.0, field_name="suitability_score")
        assert any("suitability_score" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_distance
# ---------------------------------------------------------------------------

class TestValidateDistance:
    def test_valid_distance(self):
        assert validate_distance(5.0).is_valid

    def test_zero_distance_valid(self):
        assert validate_distance(0.0).is_valid

    def test_negative_distance_invalid(self):
        result = validate_distance(-0.1)
        assert not result.is_valid

    def test_excessive_distance_invalid(self):
        result = validate_distance(15000.0)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# validate_categorical
# ---------------------------------------------------------------------------

class TestValidateCategorical:
    def test_valid_soil_type(self):
        for soil in VALID_SOIL_TYPES:
            assert validate_categorical(soil, VALID_SOIL_TYPES, "soil_type").is_valid

    def test_case_insensitive(self):
        result = validate_categorical("LOAM", VALID_SOIL_TYPES, "soil_type")
        assert result.is_valid

    def test_whitespace_stripped(self):
        result = validate_categorical("  clay  ", VALID_SOIL_TYPES, "soil_type")
        assert result.is_valid

    def test_unknown_soil_type_invalid(self):
        result = validate_categorical("gravel", VALID_SOIL_TYPES, "soil_type")
        assert not result.is_valid

    def test_error_message_contains_field_name(self):
        result = validate_categorical("unknown", VALID_SOIL_TYPES, "soil_type", "S001")
        assert any("soil_type" in e for e in result.errors)

    def test_valid_degradation_levels(self):
        for level in VALID_DEGRADATION_LEVELS:
            assert validate_categorical(
                level, VALID_DEGRADATION_LEVELS, "degradation_level"
            ).is_valid

    def test_valid_land_use_types(self):
        for land_use in VALID_LAND_USE_TYPES:
            assert validate_categorical(
                land_use, VALID_LAND_USE_TYPES, "current_land_use"
            ).is_valid


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_valid_dataframe_passes(self):
        result = validate_dataframe(_valid_df(3))
        assert result.is_valid

    def test_empty_dataframe_fails(self):
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result = validate_dataframe(empty)
        assert not result.is_valid

    def test_missing_required_column_fails(self):
        df = _valid_df(2).drop(columns=["latitude"])
        result = validate_dataframe(df)
        assert not result.is_valid
        assert any("latitude" in e for e in result.errors)

    def test_multiple_missing_columns_reported(self):
        df = _valid_df(2).drop(columns=["latitude", "longitude"])
        result = validate_dataframe(df)
        assert not result.is_valid

    def test_invalid_coordinate_row_fails(self):
        df = _valid_df(2)
        df.at[0, "latitude"] = 95.0  # Invalid
        result = validate_dataframe(df)
        assert not result.is_valid

    def test_invalid_slope_row_fails(self):
        df = _valid_df(2)
        df.at[0, "slope_pct"] = 200.0
        result = validate_dataframe(df)
        assert not result.is_valid

    def test_invalid_rainfall_row_fails(self):
        df = _valid_df(2)
        df.at[0, "annual_rainfall_mm"] = -500.0
        result = validate_dataframe(df)
        assert not result.is_valid

    def test_invalid_soil_type_fails(self):
        df = _valid_df(2)
        df.at[0, "soil_type"] = "moon_dust"
        result = validate_dataframe(df)
        assert not result.is_valid

    def test_invalid_degradation_level_fails(self):
        df = _valid_df(2)
        df.at[0, "degradation_level"] = "catastrophic"
        result = validate_dataframe(df)
        assert not result.is_valid

    def test_missing_coordinates_produce_warnings_not_errors(self):
        df = _valid_df(2)
        df.at[0, "latitude"] = None
        df.at[0, "longitude"] = None
        result = validate_dataframe(df)
        # Row with missing coords should trigger a warning
        assert len(result.warnings) > 0

    def test_warnings_do_not_block_valid_result(self):
        df = _valid_df(2)
        df.at[0, "elevation_m"] = None
        result = validate_dataframe(df)
        # Missing elevation produces a warning; other fields are fine
        assert result.is_valid
        assert len(result.warnings) > 0
