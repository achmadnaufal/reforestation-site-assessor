"""
Unit tests for src/carbon_sequestration_estimator.py.

Coverage areas:
- _chapman_richards_agb: growth curve values and validation
- estimate_site_sequestration: happy path, edge cases, invalid inputs
- estimate_dataframe_sequestration: happy path, single row, missing cols, empty df
- summarise_portfolio: empty, single, multi-site
- Determinism and immutability checks
- Parametrized tests across climate zones and rotation lengths
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.carbon_sequestration_estimator import (
    VALID_CLIMATE_ZONES,
    SequestrationEstimate,
    _chapman_richards_agb,
    estimate_dataframe_sequestration,
    estimate_site_sequestration,
    summarise_portfolio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_df(n: int = 2) -> pd.DataFrame:
    """Return a minimal valid DataFrame with *n* site rows."""
    return pd.DataFrame(
        {
            "site_id": [f"SITE-{i:03d}" for i in range(1, n + 1)],
            "area_ha": [50.0 + i * 10 for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# _chapman_richards_agb
# ---------------------------------------------------------------------------

class TestChapmanRichardsAgb:
    def test_year_one_less_than_peak(self) -> None:
        agb = _chapman_richards_agb(1, 300.0, 0.18)
        assert 0 < agb < 300.0

    def test_monotonically_increasing(self) -> None:
        values = [_chapman_richards_agb(yr, 300.0, 0.18) for yr in range(1, 51)]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_asymptote_approaches_peak(self) -> None:
        """AGB at year 200 should be within 1% of peak."""
        agb = _chapman_richards_agb(200, 300.0, 0.18)
        assert agb > 300.0 * 0.99

    def test_invalid_year_raises(self) -> None:
        with pytest.raises(ValueError, match="year must be >= 1"):
            _chapman_richards_agb(0, 300.0, 0.18)

    def test_invalid_peak_agb_raises(self) -> None:
        with pytest.raises(ValueError, match="peak_agb must be > 0"):
            _chapman_richards_agb(10, 0.0, 0.18)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be > 0"):
            _chapman_richards_agb(10, 300.0, 0.0)


# ---------------------------------------------------------------------------
# estimate_site_sequestration — happy path
# ---------------------------------------------------------------------------

class TestEstimateSiteSequestration:
    def test_returns_sequestration_estimate(self) -> None:
        result = estimate_site_sequestration("SITE-001", 100.0)
        assert isinstance(result, SequestrationEstimate)

    def test_site_id_preserved(self) -> None:
        result = estimate_site_sequestration("SITE-KAL-007", 75.0)
        assert result.site_id == "SITE-KAL-007"

    def test_area_ha_preserved(self) -> None:
        result = estimate_site_sequestration("X", 123.456)
        assert math.isclose(result.area_ha, 123.456)

    def test_total_co2e_positive(self) -> None:
        result = estimate_site_sequestration("X", 50.0)
        assert result.total_co2e_Mg > 0

    def test_annual_co2e_series_length(self) -> None:
        result = estimate_site_sequestration("X", 50.0, rotation_years=20)
        assert len(result.annual_co2e_series) == 20

    def test_annual_co2e_sums_to_total(self) -> None:
        result = estimate_site_sequestration("X", 50.0, rotation_years=30)
        assert math.isclose(sum(result.annual_co2e_series), result.total_co2e_Mg, rel_tol=1e-9)

    def test_co2e_per_ha_per_year_consistency(self) -> None:
        result = estimate_site_sequestration("X", 80.0, rotation_years=25)
        expected = result.total_co2e_Mg / (80.0 * 25)
        assert math.isclose(result.co2e_per_ha_per_year, expected, rel_tol=1e-9)

    def test_biomass_factor_scales_output(self) -> None:
        base = estimate_site_sequestration("X", 50.0, biomass_factor=1.0)
        double = estimate_site_sequestration("X", 50.0, biomass_factor=2.0)
        assert math.isclose(double.total_co2e_Mg, base.total_co2e_Mg * 2.0, rel_tol=1e-9)

    def test_result_is_immutable(self) -> None:
        result = estimate_site_sequestration("X", 50.0)
        with pytest.raises((TypeError, AttributeError)):
            result.total_co2e_Mg = 999.0  # type: ignore[misc]

    def test_deterministic(self) -> None:
        """Same inputs must always produce identical outputs."""
        r1 = estimate_site_sequestration("X", 50.0, "tropical_moist", 30, 1.0)
        r2 = estimate_site_sequestration("X", 50.0, "tropical_moist", 30, 1.0)
        assert r1 == r2

    def test_larger_area_proportional_total(self) -> None:
        small = estimate_site_sequestration("X", 10.0)
        large = estimate_site_sequestration("X", 100.0)
        assert math.isclose(large.total_co2e_Mg, small.total_co2e_Mg * 10.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# estimate_site_sequestration — invalid inputs
# ---------------------------------------------------------------------------

class TestEstimateSiteInvalidInputs:
    def test_empty_site_id_raises(self) -> None:
        with pytest.raises(ValueError, match="site_id"):
            estimate_site_sequestration("", 50.0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError, match="area_ha"):
            estimate_site_sequestration("X", 0.0)

    def test_negative_area_raises(self) -> None:
        with pytest.raises(ValueError, match="area_ha"):
            estimate_site_sequestration("X", -10.0)

    def test_invalid_climate_zone_raises(self) -> None:
        with pytest.raises(ValueError, match="climate_zone"):
            estimate_site_sequestration("X", 50.0, climate_zone="arctic")

    def test_rotation_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rotation_years"):
            estimate_site_sequestration("X", 50.0, rotation_years=0)

    def test_rotation_over_100_raises(self) -> None:
        with pytest.raises(ValueError, match="rotation_years"):
            estimate_site_sequestration("X", 50.0, rotation_years=101)

    def test_zero_biomass_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="biomass_factor"):
            estimate_site_sequestration("X", 50.0, biomass_factor=0.0)


# ---------------------------------------------------------------------------
# Parametrized: all climate zones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zone", sorted(VALID_CLIMATE_ZONES))
def test_all_climate_zones_produce_positive_co2e(zone: str) -> None:
    result = estimate_site_sequestration("TEST", 50.0, climate_zone=zone, rotation_years=30)
    assert result.total_co2e_Mg > 0
    assert result.co2e_per_ha_per_year > 0


# ---------------------------------------------------------------------------
# Parametrized: rotation period edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("years", [1, 10, 50, 100])
def test_rotation_years_parametrized(years: int) -> None:
    result = estimate_site_sequestration("TEST", 50.0, rotation_years=years)
    assert len(result.annual_co2e_series) == years
    assert result.total_co2e_Mg > 0


# ---------------------------------------------------------------------------
# estimate_dataframe_sequestration
# ---------------------------------------------------------------------------

class TestEstimateDataframeSequestration:
    def test_happy_path_columns(self) -> None:
        df = _minimal_df(3)
        result = estimate_dataframe_sequestration(df)
        for col in ("total_co2e_Mg", "co2e_per_ha_per_year", "peak_agb_ha"):
            assert col in result.columns

    def test_row_count_preserved(self) -> None:
        df = _minimal_df(5)
        result = estimate_dataframe_sequestration(df)
        assert len(result) == 5

    def test_single_row(self) -> None:
        df = _minimal_df(1)
        result = estimate_dataframe_sequestration(df)
        assert len(result) == 1
        assert result["total_co2e_Mg"].iloc[0] > 0

    def test_input_df_not_mutated(self) -> None:
        df = _minimal_df(3)
        original_cols = list(df.columns)
        estimate_dataframe_sequestration(df)
        assert list(df.columns) == original_cols

    def test_per_row_climate_zone(self) -> None:
        df = pd.DataFrame(
            {
                "site_id": ["A", "B"],
                "area_ha": [50.0, 50.0],
                "zone": ["tropical_moist", "tropical_dry"],
            }
        )
        result = estimate_dataframe_sequestration(df, climate_zone_col="zone")
        # Moist zone should yield higher CO2e than dry zone for same area
        moist_co2e = result.loc[result["site_id"] == "A", "total_co2e_Mg"].iloc[0]
        dry_co2e = result.loc[result["site_id"] == "B", "total_co2e_Mg"].iloc[0]
        assert moist_co2e > dry_co2e

    def test_empty_df_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            estimate_dataframe_sequestration(pd.DataFrame())

    def test_missing_site_id_col_raises(self) -> None:
        df = pd.DataFrame({"area_ha": [50.0]})
        with pytest.raises(ValueError, match="site_id"):
            estimate_dataframe_sequestration(df)

    def test_missing_area_col_raises(self) -> None:
        df = pd.DataFrame({"site_id": ["X"]})
        with pytest.raises(ValueError, match="area_ha"):
            estimate_dataframe_sequestration(df)

    def test_invalid_default_climate_zone_raises(self) -> None:
        df = _minimal_df(2)
        with pytest.raises(ValueError, match="default_climate_zone"):
            estimate_dataframe_sequestration(df, default_climate_zone="unknown")

    def test_non_dataframe_raises(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            estimate_dataframe_sequestration([{"site_id": "X", "area_ha": 50.0}])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# summarise_portfolio
# ---------------------------------------------------------------------------

class TestSummarisePortfolio:
    def test_empty_returns_zeros(self) -> None:
        summary = summarise_portfolio([])
        assert summary["total_sites"] == 0
        assert summary["total_co2e_Mg"] == 0.0

    def test_single_site(self) -> None:
        est = estimate_site_sequestration("A", 50.0)
        summary = summarise_portfolio([est])
        assert summary["total_sites"] == 1
        assert math.isclose(summary["total_co2e_Mg"], est.total_co2e_Mg)
        assert math.isclose(summary["max_co2e_Mg"], est.total_co2e_Mg)
        assert math.isclose(summary["min_co2e_Mg"], est.total_co2e_Mg)

    def test_multi_site_totals(self) -> None:
        est1 = estimate_site_sequestration("A", 50.0)
        est2 = estimate_site_sequestration("B", 100.0, "tropical_dry")
        summary = summarise_portfolio([est1, est2])
        assert summary["total_sites"] == 2
        assert math.isclose(
            summary["total_co2e_Mg"], est1.total_co2e_Mg + est2.total_co2e_Mg
        )
        assert math.isclose(summary["total_area_ha"], 150.0)

    def test_max_min_correctness(self) -> None:
        small = estimate_site_sequestration("S", 10.0)
        large = estimate_site_sequestration("L", 500.0)
        summary = summarise_portfolio([small, large])
        assert math.isclose(summary["max_co2e_Mg"], large.total_co2e_Mg)
        assert math.isclose(summary["min_co2e_Mg"], small.total_co2e_Mg)
