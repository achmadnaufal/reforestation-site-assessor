"""
reforestation-site-assessor
===========================

Site suitability scoring and prioritisation for reforestation projects.

Public API::

    from src.main import SiteAssessor
    from src.scorer import score_dataframe, prioritise_sites
    from src.validators import validate_dataframe
"""

from src.main import SiteAssessor
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
from src.validators import (
    ValidationResult,
    validate_coordinate,
    validate_dataframe,
    validate_elevation,
    validate_rainfall,
    validate_score,
    validate_slope,
)
from src.erosion_risk_scorer import (
    DEFAULT_EROSION_WEIGHTS,
    ErosionRiskBreakdown,
    compute_erosion_risk,
    score_erosion_dataframe,
    score_land_cover_protection,
    score_rainfall_erosivity,
    score_slope_erosion,
    score_soil_erodibility,
)
from src.mcda_topsis import (
    TopsisResult,
    VALID_CRITERIA_TYPES,
    filter_by_thresholds,
    rank_sites_topsis,
    rank_sites_topsis_detailed,
)
from src.visualization import (
    ascii_histogram,
    format_top_sites,
    score_distribution_table,
)

__all__ = [
    "SiteAssessor",
    "DEFAULT_WEIGHTS",
    "compute_composite_score",
    "prioritise_sites",
    "score_dataframe",
    "score_degradation",
    "score_forest_proximity",
    "score_rainfall",
    "score_road_proximity",
    "score_slope",
    "score_soil_depth",
    "ValidationResult",
    "validate_coordinate",
    "validate_dataframe",
    "validate_elevation",
    "validate_rainfall",
    "validate_score",
    "validate_slope",
    "DEFAULT_EROSION_WEIGHTS",
    "ErosionRiskBreakdown",
    "compute_erosion_risk",
    "score_erosion_dataframe",
    "score_land_cover_protection",
    "score_rainfall_erosivity",
    "score_slope_erosion",
    "score_soil_erodibility",
    "TopsisResult",
    "VALID_CRITERIA_TYPES",
    "filter_by_thresholds",
    "rank_sites_topsis",
    "rank_sites_topsis_detailed",
    "ascii_histogram",
    "format_top_sites",
    "score_distribution_table",
]
