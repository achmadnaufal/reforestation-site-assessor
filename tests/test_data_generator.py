"""
Unit tests for src/data_generator.py.

Tests cover:
- generate_sample: shape, columns, value ranges, reproducibility
- Edge cases: n=1, include_anomalies flag, invalid n
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.data_generator import generate_sample
from src.validators import (
    VALID_DEGRADATION_LEVELS,
    VALID_LAND_USE_TYPES,
    VALID_SOIL_TYPES,
)


class TestGenerateSample:
    def test_returns_dataframe(self):
        df = generate_sample(10)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = generate_sample(20)
        assert len(df) == 20

    def test_required_columns_present(self):
        df = generate_sample(5)
        required = [
            "site_id", "latitude", "longitude", "elevation_m",
            "slope_pct", "annual_rainfall_mm", "soil_type",
            "current_land_use", "distance_to_road_km",
            "distance_to_forest_km", "degradation_level",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_site_ids_are_unique(self):
        df = generate_sample(50)
        assert df["site_id"].nunique() == 50

    def test_latitude_in_valid_range(self):
        df = generate_sample(100)
        assert (df["latitude"] >= -90).all()
        assert (df["latitude"] <= 90).all()

    def test_longitude_in_valid_range(self):
        df = generate_sample(100)
        assert (df["longitude"] >= -180).all()
        assert (df["longitude"] <= 180).all()

    def test_elevation_non_negative(self):
        df = generate_sample(100)
        assert (df["elevation_m"] >= 0).all()

    def test_slope_in_valid_range(self):
        df = generate_sample(100)
        assert (df["slope_pct"] >= 0).all()
        assert (df["slope_pct"] <= 100).all()

    def test_rainfall_positive(self):
        df = generate_sample(100)
        assert (df["annual_rainfall_mm"] > 0).all()

    def test_distances_non_negative(self):
        df = generate_sample(50)
        assert (df["distance_to_road_km"] >= 0).all()
        assert (df["distance_to_forest_km"] >= 0).all()

    def test_soil_types_are_valid(self):
        df = generate_sample(50)
        for val in df["soil_type"]:
            assert val.lower() in [s.lower() for s in VALID_SOIL_TYPES], (
                f"Unexpected soil_type: {val}"
            )

    def test_land_use_values_are_valid(self):
        df = generate_sample(50)
        for val in df["current_land_use"]:
            assert val.lower() in [s.lower() for s in VALID_LAND_USE_TYPES], (
                f"Unexpected current_land_use: {val}"
            )

    def test_degradation_levels_are_valid(self):
        df = generate_sample(50)
        for val in df["degradation_level"]:
            assert val.lower() in [s.lower() for s in VALID_DEGRADATION_LEVELS], (
                f"Unexpected degradation_level: {val}"
            )

    def test_reproducibility_with_same_seed(self):
        df1 = generate_sample(10, seed=123)
        df2 = generate_sample(10, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_sample(20, seed=1)
        df2 = generate_sample(20, seed=2)
        assert not df1["latitude"].equals(df2["latitude"])

    def test_n_equals_one(self):
        df = generate_sample(1)
        assert len(df) == 1

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            generate_sample(0)

    def test_include_anomalies_still_produces_correct_shape(self):
        df = generate_sample(50, include_anomalies=True)
        assert len(df) == 50

    def test_no_fully_null_rows(self):
        df = generate_sample(30)
        assert not df.isnull().all(axis=1).any()
