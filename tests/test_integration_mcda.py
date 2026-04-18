"""
Integration tests that wire together the new MCDA + threshold filtering
with the existing scoring pipeline and bundled sample data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.main import SiteAssessor
from src.mcda_topsis import filter_by_thresholds, rank_sites_topsis
from src.visualization import format_top_sites, score_distribution_table


DEMO_CSV = Path(__file__).resolve().parents[1] / "demo" / "sample_data.csv"


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


def test_topsis_ranks_all_demo_sites(demo_df: pd.DataFrame) -> None:
    ranked = rank_sites_topsis(
        demo_df,
        criteria={
            "annual_rainfall_mm": "benefit",
            "slope_pct": "cost",
            "distance_to_forest_km": "cost",
            "distance_to_road_km": "cost",
        },
        weights={
            "annual_rainfall_mm": 0.35,
            "slope_pct": 0.25,
            "distance_to_forest_km": 0.25,
            "distance_to_road_km": 0.15,
        },
    )
    assert len(ranked) == len(demo_df)
    assert ranked["topsis_rank"].iloc[0] == 1
    assert set(ranked["site_id"]) == set(demo_df["site_id"])


def test_threshold_filter_narrows_candidates(demo_df: pd.DataFrame) -> None:
    filtered = filter_by_thresholds(
        demo_df,
        thresholds={
            "slope_pct": (None, 25.0),
            "annual_rainfall_mm": (2000.0, None),
        },
    )
    assert len(filtered) < len(demo_df)
    assert (filtered["slope_pct"] <= 25.0).all()
    assert (filtered["annual_rainfall_mm"] >= 2000.0).all()


def test_assessor_pipeline_plus_visualization(demo_df: pd.DataFrame) -> None:
    assessor = SiteAssessor()
    scored = assessor.run_pipeline(demo_df)
    top_txt = format_top_sites(
        scored.sort_values("composite_score", ascending=False),
        top_n=3,
    )
    dist_txt = score_distribution_table(scored)
    assert "site_id" in top_txt
    assert "Score distribution" in dist_txt
