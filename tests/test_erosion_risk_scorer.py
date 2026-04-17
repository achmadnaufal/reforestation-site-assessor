"""
Unit tests for src/erosion_risk_scorer.py.

Coverage areas:
- Component scorers (slope, rainfall, soil, land cover): valid ranges,
  monotonicity, edge cases, invalid inputs.
- compute_erosion_risk: happy path, immutability, custom weights,
  weight-sum validation, unrecognised categorical values.
- score_erosion_dataframe: happy path, empty DataFrame, missing columns,
  immutability of input, expected output columns.
- _classify_risk: band boundaries.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.erosion_risk_scorer import (
    DEFAULT_EROSION_WEIGHTS,
    ErosionRiskBreakdown,
    _classify_risk,
    compute_erosion_risk,
    score_erosion_dataframe,
    score_land_cover_protection,
    score_rainfall_erosivity,
    score_slope_erosion,
    score_soil_erodibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_df(n: int = 3) -> pd.DataFrame:
    """Return a minimal DataFrame with the columns required for erosion scoring."""
    return pd.DataFrame(
        {
            "site_id": [f"SITE-{i:03d}" for i in range(1, n + 1)],
            "slope_pct": [5.0, 18.0, 42.0][:n],
            "annual_rainfall_mm": [800.0, 2400.0, 3800.0][:n],
            "soil_type": ["loam", "clay", "sandy"][:n],
            "current_land_use": ["plantation", "shrubland", "bare_land"][:n],
        }
    )


# ---------------------------------------------------------------------------
# DEFAULT_EROSION_WEIGHTS
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    def test_keys_are_complete(self) -> None:
        assert set(DEFAULT_EROSION_WEIGHTS) == {
            "slope",
            "rainfall",
            "soil",
            "land_cover",
        }

    def test_weights_sum_to_one(self) -> None:
        assert sum(DEFAULT_EROSION_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

class TestSlopeErosionScore:
    def test_zero_slope_returns_zero(self) -> None:
        assert score_slope_erosion(0.0) == 0.0

    def test_score_grows_with_slope(self) -> None:
        values = [score_slope_erosion(s) for s in [1, 5, 10, 20, 30, 39]]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_saturates_at_high_slope(self) -> None:
        assert score_slope_erosion(40.0) == 100.0
        assert score_slope_erosion(75.0) == 100.0

    def test_negative_slope_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            score_slope_erosion(-1.0)


class TestRainfallErosivityScore:
    def test_zero_rainfall_returns_zero(self) -> None:
        assert score_rainfall_erosivity(0.0) == 0.0

    def test_low_rainfall_low_score(self) -> None:
        assert score_rainfall_erosivity(250.0) == pytest.approx(10.0)

    def test_high_tropical_rainfall_high_score(self) -> None:
        score = score_rainfall_erosivity(3500.0)
        assert score == pytest.approx(95.0)

    def test_extreme_rainfall_capped(self) -> None:
        assert score_rainfall_erosivity(10_000.0) == 100.0

    def test_negative_rainfall_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            score_rainfall_erosivity(-50.0)


class TestSoilErodibilityScore:
    @pytest.mark.parametrize("soil", ["sandy", "silt", "loam", "clay", "peat"])
    def test_known_soils_return_expected_range(self, soil: str) -> None:
        assert 0.0 <= score_soil_erodibility(soil) <= 100.0

    def test_case_insensitive(self) -> None:
        assert score_soil_erodibility("Loam") == score_soil_erodibility("loam")

    def test_unknown_soil_raises(self) -> None:
        with pytest.raises(ValueError, match="not recognised"):
            score_soil_erodibility("regolith")


class TestLandCoverProtectionScore:
    def test_bare_land_is_riskiest(self) -> None:
        bare = score_land_cover_protection("bare_land")
        veg = score_land_cover_protection("mixed_vegetation")
        assert bare > veg

    def test_unknown_land_use_raises(self) -> None:
        with pytest.raises(ValueError, match="not recognised"):
            score_land_cover_protection("urban")


# ---------------------------------------------------------------------------
# _classify_risk
# ---------------------------------------------------------------------------

class TestClassifyRisk:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, "low"),
            (29.99, "low"),
            (30.0, "moderate"),
            (54.99, "moderate"),
            (55.0, "high"),
            (79.99, "high"),
            (80.0, "severe"),
            (100.0, "severe"),
        ],
    )
    def test_band_boundaries(self, score: float, expected: str) -> None:
        assert _classify_risk(score) == expected


# ---------------------------------------------------------------------------
# compute_erosion_risk
# ---------------------------------------------------------------------------

class TestComputeErosionRisk:
    def test_happy_path_returns_breakdown(self) -> None:
        result = compute_erosion_risk(
            slope_pct=20.0,
            annual_rainfall_mm=2500.0,
            soil_type="loam",
            current_land_use="shrubland",
        )
        assert isinstance(result, ErosionRiskBreakdown)
        assert 0.0 <= result.composite_score <= 100.0
        assert result.risk_class in {"low", "moderate", "high", "severe"}

    def test_low_risk_site(self) -> None:
        # Flat, dry, protective cover, resistant soil.
        result = compute_erosion_risk(
            slope_pct=1.0,
            annual_rainfall_mm=600.0,
            soil_type="clay",
            current_land_use="mixed_vegetation",
        )
        assert result.risk_class in {"low", "moderate"}
        assert result.composite_score < 55.0

    def test_severe_risk_site(self) -> None:
        # Steep, very wet, sandy, bare land.
        result = compute_erosion_risk(
            slope_pct=45.0,
            annual_rainfall_mm=4000.0,
            soil_type="sandy",
            current_land_use="bare_land",
        )
        assert result.risk_class == "severe"
        assert result.composite_score >= 80.0

    def test_invalid_weights_keys_raise(self) -> None:
        bad = {"slope": 0.5, "rainfall": 0.5}  # missing soil & land_cover
        with pytest.raises(ValueError, match="keys mismatch"):
            compute_erosion_risk(10.0, 2000.0, "loam", "plantation", weights=bad)

    def test_weights_not_summing_to_one_raise(self) -> None:
        bad = {"slope": 0.5, "rainfall": 0.3, "soil": 0.3, "land_cover": 0.3}
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_erosion_risk(10.0, 2000.0, "loam", "plantation", weights=bad)

    def test_negative_weight_raises(self) -> None:
        bad = {"slope": -0.1, "rainfall": 0.4, "soil": 0.4, "land_cover": 0.3}
        with pytest.raises(ValueError, match="non-negative"):
            compute_erosion_risk(10.0, 2000.0, "loam", "plantation", weights=bad)

    def test_custom_weights_change_score(self) -> None:
        base = compute_erosion_risk(30.0, 2500.0, "sandy", "bare_land")
        # All weight on the (low) rainfall component should reduce composite.
        slope_heavy = compute_erosion_risk(
            30.0,
            2500.0,
            "sandy",
            "bare_land",
            weights={"slope": 1.0, "rainfall": 0.0, "soil": 0.0, "land_cover": 0.0},
        )
        assert slope_heavy.composite_score != base.composite_score

    def test_unknown_soil_propagates_value_error(self) -> None:
        with pytest.raises(ValueError, match="not recognised"):
            compute_erosion_risk(10.0, 2000.0, "moondust", "plantation")


# ---------------------------------------------------------------------------
# score_erosion_dataframe
# ---------------------------------------------------------------------------

class TestScoreErosionDataframe:
    def test_happy_path_adds_expected_columns(self) -> None:
        df = _valid_df(3)
        result = score_erosion_dataframe(df)
        for col in (
            "erosion_slope_score",
            "erosion_rainfall_score",
            "erosion_soil_score",
            "erosion_land_cover_score",
            "erosion_risk_score",
            "erosion_risk_class",
        ):
            assert col in result.columns
        assert len(result) == 3

    def test_input_dataframe_not_mutated(self) -> None:
        df = _valid_df(2)
        cols_before = list(df.columns)
        _ = score_erosion_dataframe(df)
        assert list(df.columns) == cols_before

    def test_empty_dataframe_returns_empty_with_columns(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "site_id",
                "slope_pct",
                "annual_rainfall_mm",
                "soil_type",
                "current_land_use",
            ]
        )
        result = score_erosion_dataframe(empty)
        assert result.empty
        assert "erosion_risk_score" in result.columns
        assert "erosion_risk_class" in result.columns

    def test_missing_required_column_raises(self) -> None:
        df = _valid_df(2).drop(columns=["soil_type"])
        with pytest.raises(KeyError, match="soil_type"):
            score_erosion_dataframe(df)

    def test_risk_classes_are_valid(self) -> None:
        df = _valid_df(3)
        result = score_erosion_dataframe(df)
        valid = {"low", "moderate", "high", "severe"}
        assert set(result["erosion_risk_class"]).issubset(valid)

    def test_custom_weights_applied(self) -> None:
        df = _valid_df(2)
        default_result = score_erosion_dataframe(df)
        custom = {"slope": 1.0, "rainfall": 0.0, "soil": 0.0, "land_cover": 0.0}
        custom_result = score_erosion_dataframe(df, weights=custom)
        assert not (
            default_result["erosion_risk_score"].equals(
                custom_result["erosion_risk_score"]
            )
        )
