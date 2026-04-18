"""
Unit tests for src/mcda_topsis.py.

Coverage:
- Happy-path ranking with mixed benefit/cost criteria.
- Immutability of the input DataFrame.
- Equal-weights default.
- Custom weights (re-normalisation to sum to 1.0).
- Threshold filtering with one-sided bounds.
- Edge cases: empty inputs, missing columns, NaN handling, zero-variance
  columns, invalid criterion types, bad weights.
- Detailed result dataclass shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mcda_topsis import (
    TopsisResult,
    VALID_CRITERIA_TYPES,
    filter_by_thresholds,
    rank_sites_topsis,
    rank_sites_topsis_detailed,
)


def _three_site_frame() -> pd.DataFrame:
    """A minimal frame where site A should dominate on a benefit/cost mix."""
    return pd.DataFrame(
        {
            "site_id": ["A", "B", "C"],
            "rainfall": [3000.0, 2000.0, 1000.0],  # benefit
            "slope": [5.0, 20.0, 35.0],             # cost
            "road_km": [2.0, 6.0, 15.0],            # cost
        }
    )


# ---------------------------------------------------------------------------
# rank_sites_topsis
# ---------------------------------------------------------------------------

class TestRankSitesTopsis:
    def test_happy_path_best_site_wins(self) -> None:
        df = _three_site_frame()
        out = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost", "road_km": "cost"},
        )

        assert list(out["topsis_rank"]) == [1, 2, 3]
        assert out.iloc[0]["site_id"] == "A"
        assert out.iloc[-1]["site_id"] == "C"
        # Scores are closeness in [0, 1] and monotonically decreasing.
        scores = out["topsis_score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_input_dataframe_is_not_mutated(self) -> None:
        df = _three_site_frame()
        original_cols = list(df.columns)
        original_values = df.copy()

        _ = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
        )

        assert list(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_values)

    def test_equal_weights_when_weights_none(self) -> None:
        df = _three_site_frame()
        out = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
            weights=None,
        )
        assert "topsis_score" in out.columns
        # With equal weights on this data, A still wins.
        assert out.iloc[0]["site_id"] == "A"

    def test_custom_weights_are_renormalised(self) -> None:
        df = _three_site_frame()
        # Non-normalised weights — should still work, just re-normalised.
        out_norm = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
            weights={"rainfall": 0.6, "slope": 0.4},
        )
        out_scaled = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
            weights={"rainfall": 60.0, "slope": 40.0},
        )
        pd.testing.assert_series_equal(
            out_norm["topsis_score"], out_scaled["topsis_score"], check_names=False
        )

    def test_weights_must_match_criteria_keys(self) -> None:
        df = _three_site_frame()
        with pytest.raises(ValueError, match="must match criteria keys"):
            rank_sites_topsis(
                df,
                criteria={"rainfall": "benefit"},
                weights={"slope": 1.0},
            )

    def test_negative_weight_is_rejected(self) -> None:
        df = _three_site_frame()
        with pytest.raises(ValueError, match="non-negative"):
            rank_sites_topsis(
                df,
                criteria={"rainfall": "benefit", "slope": "cost"},
                weights={"rainfall": -0.5, "slope": 1.5},
            )

    def test_all_zero_weights_rejected(self) -> None:
        df = _three_site_frame()
        with pytest.raises(ValueError, match="positive sum"):
            rank_sites_topsis(
                df,
                criteria={"rainfall": "benefit", "slope": "cost"},
                weights={"rainfall": 0.0, "slope": 0.0},
            )

    def test_invalid_criterion_type(self) -> None:
        df = _three_site_frame()
        with pytest.raises(ValueError, match="benefit"):
            rank_sites_topsis(df, criteria={"rainfall": "maximise"})

    def test_empty_criteria(self) -> None:
        df = _three_site_frame()
        with pytest.raises(ValueError, match="must not be empty"):
            rank_sites_topsis(df, criteria={})

    def test_empty_dataframe(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            rank_sites_topsis(
                pd.DataFrame(columns=["rainfall", "slope"]),
                criteria={"rainfall": "benefit", "slope": "cost"},
            )

    def test_missing_column_raises_keyerror(self) -> None:
        df = _three_site_frame().drop(columns=["slope"])
        with pytest.raises(KeyError, match="slope"):
            rank_sites_topsis(
                df,
                criteria={"rainfall": "benefit", "slope": "cost"},
            )

    def test_nan_rows_dropped_by_default(self) -> None:
        df = _three_site_frame().copy()
        df.loc[1, "slope"] = np.nan
        out = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
        )
        assert "B" not in set(out["site_id"])
        assert len(out) == 2

    def test_nan_rows_raise_when_drop_na_false(self) -> None:
        df = _three_site_frame().copy()
        df.loc[1, "slope"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            rank_sites_topsis(
                df,
                criteria={"rainfall": "benefit", "slope": "cost"},
                drop_na=False,
            )

    def test_zero_variance_column_does_not_crash(self) -> None:
        # All rows share identical slope — column contributes nothing.
        df = pd.DataFrame(
            {
                "site_id": ["A", "B", "C"],
                "rainfall": [3000.0, 2000.0, 1000.0],
                "slope": [10.0, 10.0, 10.0],
            }
        )
        out = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
        )
        assert len(out) == 3
        # Ranking is determined by rainfall alone here.
        assert out.iloc[0]["site_id"] == "A"

    def test_single_row_returns_neutral_score(self) -> None:
        df = pd.DataFrame(
            {"site_id": ["only"], "rainfall": [2500.0], "slope": [10.0]}
        )
        out = rank_sites_topsis(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
        )
        assert len(out) == 1
        assert out.iloc[0]["topsis_score"] == 0.5
        assert out.iloc[0]["topsis_rank"] == 1

    def test_valid_criteria_types_constant(self) -> None:
        assert VALID_CRITERIA_TYPES == frozenset({"benefit", "cost"})


# ---------------------------------------------------------------------------
# rank_sites_topsis_detailed
# ---------------------------------------------------------------------------

class TestRankSitesTopsisDetailed:
    def test_returns_topsis_result_dataclass(self) -> None:
        df = _three_site_frame()
        result = rank_sites_topsis_detailed(
            df,
            criteria={"rainfall": "benefit", "slope": "cost"},
        )
        assert isinstance(result, TopsisResult)
        assert "topsis_score" in result.ranked_df.columns
        assert result.ideal_best.shape == (2,)
        assert result.ideal_worst.shape == (2,)
        # Weights are re-normalised to sum to 1.
        assert pytest.approx(sum(result.weights.values()), abs=1e-9) == 1.0
        assert set(result.criteria) == {"rainfall", "slope"}

    def test_topsis_result_is_frozen(self) -> None:
        df = _three_site_frame()
        result = rank_sites_topsis_detailed(
            df, criteria={"rainfall": "benefit", "slope": "cost"}
        )
        with pytest.raises((AttributeError, Exception)):
            result.weights = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# filter_by_thresholds
# ---------------------------------------------------------------------------

class TestFilterByThresholds:
    def test_drops_rows_outside_bounds(self) -> None:
        df = pd.DataFrame({"slope": [5.0, 25.0, 45.0], "rainfall": [1500, 2500, 3500]})
        out = filter_by_thresholds(
            df,
            thresholds={"slope": (None, 30.0), "rainfall": (2000, None)},
        )
        assert len(out) == 1
        assert out.iloc[0]["slope"] == 25.0

    def test_returns_copy_not_view(self) -> None:
        df = pd.DataFrame({"slope": [5.0, 25.0]})
        out = filter_by_thresholds(df, thresholds={"slope": (None, 30.0)})
        out.loc[0, "slope"] = 999.0
        assert df.loc[0, "slope"] == 5.0

    def test_empty_thresholds_returns_copy(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = filter_by_thresholds(df, thresholds={})
        pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))
        assert out is not df

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError):
            filter_by_thresholds(df, thresholds={"missing": (0, 1)})

    def test_malformed_threshold_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="tuple"):
            filter_by_thresholds(df, thresholds={"a": [0, 1]})  # type: ignore[dict-item]

    def test_min_greater_than_max_raises(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="min .* > max"):
            filter_by_thresholds(df, thresholds={"a": (10, 5)})
