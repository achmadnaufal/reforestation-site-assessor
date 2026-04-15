"""
Unit tests for src/scorer.py.

Tests cover:
- Individual scoring functions (normal inputs, edge cases, invalid inputs)
- Composite score computation with default and custom weights
- DataFrame-level scoring
- Site prioritisation
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.scorer import (
    DEFAULT_WEIGHTS,
    compute_composite_score,
    prioritise_sites,
    score_dataframe,
    score_degradation,
    score_forest_proximity,
    score_rainfall,
    score_road_proximity,
    score_slope,
    score_soil_depth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_site_row() -> dict:
    """Return a single valid site record as a dict."""
    return {
        "site_id": "TEST-0001",
        "latitude": -2.5,
        "longitude": 117.0,
        "elevation_m": 200.0,
        "slope_pct": 15.0,
        "annual_rainfall_mm": 2000.0,
        "soil_type": "loam",
        "soil_depth_cm": 90.0,
        "current_land_use": "degraded_forest",
        "distance_to_road_km": 3.0,
        "distance_to_forest_km": 4.0,
        "degradation_level": "high",
    }


def _make_df(n: int = 3) -> pd.DataFrame:
    """Create a small valid DataFrame with n identical rows."""
    row = _minimal_site_row()
    rows = [{**row, "site_id": f"TEST-{i:04d}"} for i in range(n)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# score_rainfall
# ---------------------------------------------------------------------------

class TestScoreRainfall:
    def test_optimal_rainfall_returns_high_score(self):
        # 2000 mm falls in the 1200–3000 optimal range; score ≈ 83.3
        score = score_rainfall(2000.0)
        assert score >= 80.0

    def test_very_low_rainfall_returns_low_score(self):
        score = score_rainfall(100.0)
        assert score < 20.0

    def test_zero_rainfall_returns_zero(self):
        assert score_rainfall(0.0) == 0.0

    def test_moderate_rainfall_in_expected_range(self):
        score = score_rainfall(800.0)
        assert 40.0 <= score <= 75.0

    def test_excessive_rainfall_penalised(self):
        low_excess = score_rainfall(3500.0)
        high_excess = score_rainfall(5000.0)
        assert low_excess > high_excess

    def test_boundary_at_500mm(self):
        below = score_rainfall(499.0)
        above = score_rainfall(501.0)
        assert above > below

    def test_boundary_at_1200mm(self):
        below = score_rainfall(1199.0)
        above = score_rainfall(1201.0)
        assert above > below

    def test_negative_rainfall_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            score_rainfall(-1.0)

    def test_return_type_is_float(self):
        assert isinstance(score_rainfall(1500.0), float)


# ---------------------------------------------------------------------------
# score_slope
# ---------------------------------------------------------------------------

class TestScoreSlope:
    def test_flat_slope_returns_near_100(self):
        assert score_slope(0.0) == 100.0

    def test_gentle_slope_returns_high_score(self):
        assert score_slope(5.0) >= 90.0

    def test_steep_slope_returns_low_score(self):
        assert score_slope(50.0) < 25.0

    def test_extreme_slope_returns_zero(self):
        assert score_slope(70.0) == 0.0
        assert score_slope(100.0) == 0.0

    def test_boundary_at_10pct(self):
        # 10% boundary: both sides should differ
        at_10 = score_slope(10.0)
        above_10 = score_slope(11.0)
        assert at_10 > above_10

    def test_negative_slope_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            score_slope(-1.0)

    def test_slope_above_100_returns_zero(self):
        # Values above 100% are physically extreme — treated as maximally steep
        assert score_slope(101.0) == 0.0
        assert score_slope(999.0) == 0.0

    def test_return_type_is_float(self):
        assert isinstance(score_slope(20.0), float)


# ---------------------------------------------------------------------------
# score_soil_depth
# ---------------------------------------------------------------------------

class TestScoreSoilDepth:
    def test_very_shallow_soil_returns_low_score(self):
        assert score_soil_depth(5.0) < 15.0

    def test_zero_depth_returns_zero(self):
        assert score_soil_depth(0.0) == 0.0

    def test_deep_soil_returns_high_score(self):
        assert score_soil_depth(120.0) >= 90.0

    def test_moderate_depth_in_expected_range(self):
        score = score_soil_depth(60.0)
        assert 70.0 <= score <= 100.0

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            score_soil_depth(-10.0)

    def test_score_is_monotonically_non_decreasing(self):
        depths = [0, 10, 20, 40, 60, 80, 100, 150]
        scores = [score_soil_depth(d) for d in depths]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Score decreased between depth {depths[i]} and {depths[i+1]}"
            )


# ---------------------------------------------------------------------------
# score_forest_proximity
# ---------------------------------------------------------------------------

class TestScoreForestProximity:
    def test_adjacent_forest_returns_near_100(self):
        assert score_forest_proximity(0.0) == 100.0

    def test_close_forest_high_score(self):
        assert score_forest_proximity(0.5) >= 90.0

    def test_distant_forest_low_score(self):
        assert score_forest_proximity(30.0) < 15.0

    def test_score_decreases_with_distance(self):
        distances = [0, 1, 3, 5, 10, 15, 20, 30]
        scores = [score_forest_proximity(d) for d in distances]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score increased from distance {distances[i]} to {distances[i+1]}"
            )

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            score_forest_proximity(-1.0)


# ---------------------------------------------------------------------------
# score_road_proximity
# ---------------------------------------------------------------------------

class TestScoreRoadProximity:
    def test_optimal_range_returns_high_score(self):
        assert score_road_proximity(2.0) >= 85.0

    def test_very_close_road_slightly_lower(self):
        very_close = score_road_proximity(0.1)
        optimal = score_road_proximity(2.0)
        assert optimal >= very_close

    def test_very_distant_road_low_score(self):
        assert score_road_proximity(40.0) < 25.0

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            score_road_proximity(-0.5)

    def test_return_type_is_float(self):
        assert isinstance(score_road_proximity(5.0), float)


# ---------------------------------------------------------------------------
# score_degradation
# ---------------------------------------------------------------------------

class TestScoreDegradation:
    def test_severe_returns_100(self):
        assert score_degradation("severe") == 100.0

    def test_high_returns_80(self):
        assert score_degradation("high") == 80.0

    def test_medium_returns_60(self):
        assert score_degradation("medium") == 60.0

    def test_low_returns_40(self):
        assert score_degradation("low") == 40.0

    def test_case_insensitive(self):
        assert score_degradation("SEVERE") == score_degradation("severe")
        assert score_degradation("High") == score_degradation("high")

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            score_degradation("unknown")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            score_degradation("")


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def _all_max_scores(self) -> dict:
        return dict(
            rainfall_score=100.0,
            slope_score=100.0,
            soil_depth_score=100.0,
            forest_proximity_score=100.0,
            road_proximity_score=100.0,
            degradation_score=100.0,
        )

    def test_all_max_scores_returns_100(self):
        score = compute_composite_score(**self._all_max_scores())
        assert score == 100.0

    def test_all_zero_scores_returns_0(self):
        score = compute_composite_score(0, 0, 0, 0, 0, 0)
        assert score == 0.0

    def test_default_weights_sum_to_1(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights_applied(self):
        equal_weights = {k: 1 / 6 for k in DEFAULT_WEIGHTS}
        score_default = compute_composite_score(100, 0, 0, 0, 0, 0)
        score_equal = compute_composite_score(100, 0, 0, 0, 0, 0, weights=equal_weights)
        # With equal weights, rainfall (100) contributes 100/6 ≈ 16.67
        assert abs(score_equal - 100 / 6) < 0.1

    def test_out_of_range_sub_score_raises(self):
        with pytest.raises(ValueError, match="Sub-score"):
            compute_composite_score(150, 50, 50, 50, 50, 50)

    def test_wrong_weight_keys_raises(self):
        bad_weights = {"rainfall": 1.0, "wrong_key": 0.0}
        with pytest.raises(ValueError, match="Weight keys mismatch"):
            compute_composite_score(50, 50, 50, 50, 50, 50, weights=bad_weights)

    def test_result_is_rounded_to_two_decimals(self):
        score = compute_composite_score(33.3, 66.7, 50.0, 75.0, 80.0, 60.0)
        assert score == round(score, 2)


# ---------------------------------------------------------------------------
# score_dataframe
# ---------------------------------------------------------------------------

class TestScoreDataframe:
    def test_adds_all_score_columns(self):
        df = _make_df(5)
        scored = score_dataframe(df)
        expected_cols = [
            "score_rainfall", "score_slope", "score_soil_depth",
            "score_forest_proximity", "score_road_proximity",
            "score_degradation", "composite_score",
        ]
        for col in expected_cols:
            assert col in scored.columns, f"Missing column: {col}"

    def test_does_not_mutate_input(self):
        df = _make_df(3)
        original_cols = list(df.columns)
        score_dataframe(df)
        assert list(df.columns) == original_cols

    def test_composite_score_in_valid_range(self):
        df = _make_df(10)
        scored = score_dataframe(df)
        assert (scored["composite_score"] >= 0).all()
        assert (scored["composite_score"] <= 100).all()

    def test_derives_soil_depth_from_soil_type_when_missing(self):
        df = _make_df(3).drop(columns=["soil_depth_cm"])
        scored = score_dataframe(df)
        assert "score_soil_depth" in scored.columns

    def test_preserves_row_count(self):
        df = _make_df(7)
        scored = score_dataframe(df)
        assert len(scored) == 7

    def test_custom_weights_change_scores(self):
        df = _make_df(3)
        default_scored = score_dataframe(df)

        # Maximise weight on rainfall (should shift composite scores)
        custom_weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
        custom_weights["rainfall"] = 1.0
        custom_scored = score_dataframe(df, weights=custom_weights)

        # With all weight on rainfall, composite should equal rainfall score
        # (allowing for rounding to 2 decimal places)
        assert (
            (custom_scored["composite_score"] - custom_scored["score_rainfall"]).abs() < 0.01
        ).all()
        # And default-weighted scores should differ
        assert not (
            default_scored["composite_score"] == custom_scored["composite_score"]
        ).all()


# ---------------------------------------------------------------------------
# prioritise_sites
# ---------------------------------------------------------------------------

class TestPrioritiseSites:
    def test_returns_correct_number_of_sites(self):
        df = score_dataframe(_make_df(20))
        top5 = prioritise_sites(df, top_n=5)
        assert len(top5) <= 5

    def test_sorted_descending_by_composite_score(self):
        df = score_dataframe(_make_df(15))
        ranked = prioritise_sites(df, top_n=15)
        scores = ranked["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filter_applied(self):
        df = score_dataframe(_make_df(10))
        max_score = df["composite_score"].max()
        ranked = prioritise_sites(df, top_n=10, min_score=max_score + 1)
        assert len(ranked) == 0

    def test_zero_top_n_raises(self):
        df = score_dataframe(_make_df(5))
        with pytest.raises(ValueError, match="top_n must be > 0"):
            prioritise_sites(df, top_n=0)

    def test_invalid_min_score_raises(self):
        df = score_dataframe(_make_df(5))
        with pytest.raises(ValueError, match="min_score"):
            prioritise_sites(df, top_n=5, min_score=150.0)

    def test_missing_composite_score_column_raises(self):
        df = _make_df(5)  # No composite_score column
        with pytest.raises(KeyError, match="composite_score"):
            prioritise_sites(df)

    def test_returns_new_dataframe(self):
        df = score_dataframe(_make_df(5))
        ranked = prioritise_sites(df, top_n=3)
        assert ranked is not df
