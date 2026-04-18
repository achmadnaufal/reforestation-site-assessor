"""
Unit tests for src/visualization.py.

Coverage:
- ASCII histogram: happy path, NaN handling, empty input, degenerate
  (all-equal) values, explicit value_range, invalid inputs.
- Score distribution table: band counts and percentages, missing column,
  empty DataFrame.
- Top-N formatter: default column inference, explicit columns, empty input,
  validation errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    ascii_histogram,
    format_top_sites,
    score_distribution_table,
)


# ---------------------------------------------------------------------------
# ascii_histogram
# ---------------------------------------------------------------------------

class TestAsciiHistogram:
    def test_renders_bins_and_counts(self) -> None:
        out = ascii_histogram([1, 2, 2, 3, 3, 3], bins=3, width=10)
        lines = out.splitlines()
        assert len(lines) == 3
        # Last bin has the most entries (3).
        assert "(3)" in lines[-1]

    def test_ignores_nan_values(self) -> None:
        out_with_nan = ascii_histogram([1.0, 2.0, np.nan, 3.0], bins=3, width=10)
        out_without = ascii_histogram([1.0, 2.0, 3.0], bins=3, width=10)
        assert out_with_nan == out_without

    def test_empty_input_returns_empty_string(self) -> None:
        assert ascii_histogram([], bins=5, width=20) == ""
        assert ascii_histogram([np.nan, np.nan], bins=5, width=20) == ""

    def test_all_equal_values_single_line(self) -> None:
        out = ascii_histogram([5.0, 5.0, 5.0], bins=3, width=10)
        assert out.count("\n") == 0
        assert "(3)" in out

    def test_custom_value_range(self) -> None:
        out = ascii_histogram([1, 2, 3], bins=2, width=10, value_range=(0.0, 4.0))
        lines = out.splitlines()
        assert len(lines) == 2
        # All three values should fall into range.
        total = sum(int(line.split("(")[-1].rstrip(")")) for line in lines)
        assert total == 3

    def test_invalid_bins(self) -> None:
        with pytest.raises(ValueError, match="bins"):
            ascii_histogram([1, 2, 3], bins=0)

    def test_invalid_width(self) -> None:
        with pytest.raises(ValueError, match="width"):
            ascii_histogram([1, 2, 3], bins=3, width=0)

    def test_invalid_value_range(self) -> None:
        with pytest.raises(ValueError, match="value_range"):
            ascii_histogram([1, 2, 3], bins=3, value_range=(5.0, 5.0))


# ---------------------------------------------------------------------------
# score_distribution_table
# ---------------------------------------------------------------------------

class TestScoreDistributionTable:
    def test_counts_per_band(self) -> None:
        df = pd.DataFrame(
            {"composite_score": [10.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]}
        )
        out = score_distribution_table(df)
        assert "8 site(s)" in out
        # Count substrings present for each band.
        assert "low" in out and "moderate" in out
        assert "good" in out and "excellent" in out

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"other": [50.0]})
        with pytest.raises(KeyError):
            score_distribution_table(df)

    def test_raises_on_empty_dataframe(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            score_distribution_table(pd.DataFrame({"composite_score": []}))

    def test_custom_bands(self) -> None:
        df = pd.DataFrame({"composite_score": [25.0, 75.0]})
        out = score_distribution_table(
            df,
            bands=(("bucket_low", 0.0, 50.0), ("bucket_high", 50.0, 100.01)),
        )
        assert "bucket_low" in out
        assert "bucket_high" in out


# ---------------------------------------------------------------------------
# format_top_sites
# ---------------------------------------------------------------------------

class TestFormatTopSites:
    def test_default_columns_inferred(self) -> None:
        df = pd.DataFrame(
            {
                "site_id": ["A", "B", "C"],
                "composite_score": [80.0, 90.0, 70.0],
                "extra": [1, 2, 3],
            }
        ).sort_values("composite_score", ascending=False)
        out = format_top_sites(df, top_n=2)
        assert "site_id" in out
        assert "composite_score" in out
        # 'extra' should not appear because the default columns are inferred.
        assert "extra" not in out

    def test_explicit_columns(self) -> None:
        df = pd.DataFrame({"site_id": ["A"], "slope": [5.0], "rain": [2000]})
        out = format_top_sites(df, top_n=1, columns=["site_id", "rain"])
        assert "rain" in out
        assert "slope" not in out

    def test_missing_explicit_column_raises(self) -> None:
        df = pd.DataFrame({"site_id": ["A"]})
        with pytest.raises(KeyError):
            format_top_sites(df, top_n=1, columns=["missing"])

    def test_invalid_top_n(self) -> None:
        df = pd.DataFrame({"site_id": ["A"]})
        with pytest.raises(ValueError, match="top_n"):
            format_top_sites(df, top_n=0)

    def test_empty_dataframe_returns_placeholder(self) -> None:
        out = format_top_sites(pd.DataFrame(columns=["site_id"]))
        assert out == "(no sites)"
