"""
Unit tests for src/main.py (SiteAssessor).

Tests cover:
- load_data: file not found, unsupported extension, valid CSV
- preprocess: column normalisation, empty-row dropping, string stripping
- validate: empty DataFrame guard, delegation to validate_dataframe
- analyze: structure of returned dict, numeric columns
- run_pipeline: full pipeline, strict mode, mutation safety
- prioritise: delegates to prioritise_sites with config values
- run: legacy convenience method
- to_dataframe: flattening nested dicts
- get_summary: structure of summary dict
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.main import SiteAssessor


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
        "soil_depth_cm": 90.0,
        "current_land_use": "degraded_forest",
        "distance_to_road_km": 3.0,
        "distance_to_forest_km": 4.0,
        "degradation_level": "high",
    }


def _valid_df(n: int = 3) -> pd.DataFrame:
    row = _valid_row()
    rows = [{**row, "site_id": f"TEST-{i:04d}"} for i in range(n)]
    return pd.DataFrame(rows)


def _write_temp_csv(df: pd.DataFrame) -> str:
    """Write df to a temporary CSV and return the file path."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        return f.name


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestSiteAssessorInit:
    def test_default_config_is_empty_dict(self):
        assessor = SiteAssessor()
        assert assessor.config == {}

    def test_provided_config_is_copied(self):
        original = {"top_n": 5}
        assessor = SiteAssessor(config=original)
        original["top_n"] = 99  # Mutate original
        assert assessor.config["top_n"] == 5  # Assessor unaffected


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_load_valid_csv(self):
        df = _valid_df(5)
        path = _write_temp_csv(df)
        assessor = SiteAssessor()
        loaded = assessor.load_data(path)
        assert len(loaded) == 5

    def test_file_not_found_raises(self):
        assessor = SiteAssessor()
        with pytest.raises(FileNotFoundError, match="not found"):
            assessor.load_data("/nonexistent/path/data.csv")

    def test_unsupported_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = f.name
        assessor = SiteAssessor()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            assessor.load_data(path)

    def test_returns_dataframe(self):
        df = _valid_df(2)
        path = _write_temp_csv(df)
        assessor = SiteAssessor()
        result = assessor.load_data(path)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_column_names_lowercased(self):
        df = pd.DataFrame({"Site_ID": [1], "RAINFALL_MM": [2000]})
        assessor = SiteAssessor()
        result = assessor.preprocess(df)
        assert "site_id" in result.columns
        assert "rainfall_mm" in result.columns

    def test_spaces_in_column_names_replaced(self):
        df = pd.DataFrame({"Site ID": [1], "Slope Pct": [10]})
        assessor = SiteAssessor()
        result = assessor.preprocess(df)
        assert "site_id" in result.columns
        assert "slope_pct" in result.columns

    def test_fully_empty_rows_dropped(self):
        df = pd.DataFrame({"a": [1, None, 2], "b": [10, None, 20]})
        assessor = SiteAssessor()
        result = assessor.preprocess(df)
        assert len(result) == 2

    def test_string_whitespace_stripped(self):
        df = pd.DataFrame({"soil_type": ["  loam  ", " clay"]})
        assessor = SiteAssessor()
        result = assessor.preprocess(df)
        assert result["soil_type"].tolist() == ["loam", "clay"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"Site ID": [1]})
        original_cols = list(df.columns)
        assessor = SiteAssessor()
        assessor.preprocess(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_empty_dataframe_raises_value_error(self):
        assessor = SiteAssessor()
        with pytest.raises(ValueError, match="empty"):
            assessor.validate(pd.DataFrame())

    def test_valid_dataframe_returns_ok_result(self):
        assessor = SiteAssessor()
        result = assessor.validate(_valid_df(3))
        assert result.is_valid

    def test_invalid_df_returns_failing_result(self):
        df = _valid_df(2)
        df.at[0, "slope_pct"] = 999.0
        assessor = SiteAssessor()
        result = assessor.validate(df)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_total_records(self):
        assessor = SiteAssessor()
        result = assessor.analyze(_valid_df(5))
        assert result["total_records"] == 5

    def test_returns_columns_list(self):
        assessor = SiteAssessor()
        result = assessor.analyze(_valid_df(2))
        assert isinstance(result["columns"], list)

    def test_returns_missing_pct(self):
        assessor = SiteAssessor()
        result = assessor.analyze(_valid_df(2))
        assert "missing_pct" in result

    def test_includes_summary_stats_for_numeric(self):
        assessor = SiteAssessor()
        result = assessor.analyze(_valid_df(4))
        assert "summary_stats" in result

    def test_no_summary_stats_for_non_numeric(self):
        df = pd.DataFrame({"site_id": ["A", "B"], "soil_type": ["loam", "clay"]})
        assessor = SiteAssessor()
        result = assessor.analyze(df)
        assert "summary_stats" not in result


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_scored_dataframe(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(5))
        assert "composite_score" in scored.columns

    def test_composite_scores_in_valid_range(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(5))
        assert (scored["composite_score"] >= 0).all()
        assert (scored["composite_score"] <= 100).all()

    def test_empty_dataframe_raises_after_preprocess(self):
        assessor = SiteAssessor()
        with pytest.raises(ValueError, match="No data remains"):
            assessor.run_pipeline(pd.DataFrame())

    def test_strict_mode_raises_on_validation_errors(self):
        df = _valid_df(3)
        df.at[0, "slope_pct"] = 999.0
        assessor = SiteAssessor()
        with pytest.raises(ValueError, match="Validation failed"):
            assessor.run_pipeline(df, strict=True)

    def test_non_strict_mode_continues_despite_errors(self):
        df = _valid_df(3)
        df.at[0, "slope_pct"] = 999.0  # Invalid but non-strict
        assessor = SiteAssessor()
        # Should not raise
        scored = assessor.run_pipeline(df, strict=False)
        assert scored is not None

    def test_does_not_mutate_input(self):
        df = _valid_df(3)
        original_cols = list(df.columns)
        assessor = SiteAssessor()
        assessor.run_pipeline(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# prioritise
# ---------------------------------------------------------------------------

class TestPrioritise:
    def test_respects_top_n_from_config(self):
        assessor = SiteAssessor(config={"top_n": 2})
        scored = assessor.run_pipeline(_valid_df(10))
        top = assessor.prioritise(scored)
        assert len(top) <= 2

    def test_default_top_n_is_10(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(20))
        top = assessor.prioritise(scored)
        assert len(top) <= 10

    def test_results_sorted_descending(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(15))
        top = assessor.prioritise(scored)
        scores = top["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# run (legacy)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_returns_analysis_dict(self):
        df = _valid_df(5)
        path = _write_temp_csv(df)
        assessor = SiteAssessor()
        result = assessor.run(path)
        assert "total_records" in result
        assert result["total_records"] == 5


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    def test_flat_key_becomes_single_row(self):
        assessor = SiteAssessor()
        result = assessor.to_dataframe({"total_records": 5})
        assert len(result) == 1
        assert result.iloc[0]["metric"] == "total_records"

    def test_nested_dict_expanded_with_dot_notation(self):
        assessor = SiteAssessor()
        result = assessor.to_dataframe({"means": {"rainfall": 2000.0}})
        assert result.iloc[0]["metric"] == "means.rainfall"

    def test_returns_dataframe(self):
        assessor = SiteAssessor()
        result = assessor.to_dataframe({"a": 1})
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_summary_contains_expected_keys(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(10))
        summary = assessor.get_summary(scored)
        assert "total_sites" in summary
        assert "mean_composite_score" in summary
        assert "score_distribution" in summary
        assert "top_site" in summary
        assert "score_stats" in summary

    def test_total_sites_matches_dataframe(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(7))
        summary = assessor.get_summary(scored)
        assert summary["total_sites"] == 7

    def test_distribution_bands_sum_to_total(self):
        assessor = SiteAssessor()
        scored = assessor.run_pipeline(_valid_df(10))
        summary = assessor.get_summary(scored)
        band_total = sum(summary["score_distribution"].values())
        assert band_total == 10
