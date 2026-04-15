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
]
