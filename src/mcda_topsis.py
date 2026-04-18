"""
Multi-Criteria Decision Analysis (MCDA) via TOPSIS for reforestation sites.

TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
ranks alternatives based on their geometric distance to an *ideal* and an
*anti-ideal* solution constructed from the decision matrix. It is a common
MCDA method in restoration prioritisation because it handles benefit and
cost criteria in a single, transparent framework.

This module provides a light, dependency-free (pandas/numpy only)
implementation tailored to reforestation site ranking. All functions
return *new* objects and never mutate their inputs.

Typical use::

    from src.mcda_topsis import rank_sites_topsis

    ranked = rank_sites_topsis(
        df,
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
    print(ranked[["site_id", "topsis_score", "topsis_rank"]].head())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

VALID_CRITERIA_TYPES: frozenset = frozenset({"benefit", "cost"})


@dataclass(frozen=True)
class TopsisResult:
    """Immutable container for TOPSIS ranking output.

    Attributes:
        ranked_df: New DataFrame sorted by descending ``topsis_score`` with
            two added columns: ``topsis_score`` (float in [0, 1]) and
            ``topsis_rank`` (1-indexed integer).
        ideal_best: Per-criterion ideal (positive) reference values, in the
            order of the criteria keys.
        ideal_worst: Per-criterion anti-ideal (negative) reference values.
        criteria: The resolved criteria mapping (criterion -> 'benefit' or
            'cost') used for this ranking.
        weights: The normalised weights actually applied (sum to 1.0).
    """

    ranked_df: pd.DataFrame
    ideal_best: np.ndarray
    ideal_worst: np.ndarray
    criteria: Dict[str, str]
    weights: Dict[str, float]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_criteria(criteria: Mapping[str, str]) -> Dict[str, str]:
    """Validate a criteria mapping and return a normalised copy.

    Args:
        criteria: Mapping of column name -> ``'benefit'`` or ``'cost'``.

    Returns:
        New dict with lowercased criterion-type values.

    Raises:
        ValueError: If ``criteria`` is empty or contains an invalid type.
    """
    if not criteria:
        raise ValueError("criteria mapping must not be empty.")

    normalised: Dict[str, str] = {}
    for column, kind in criteria.items():
        if not isinstance(kind, str):
            raise ValueError(
                f"criterion type for '{column}' must be a string, got {type(kind).__name__}."
            )
        kind_lower = kind.strip().lower()
        if kind_lower not in VALID_CRITERIA_TYPES:
            raise ValueError(
                f"criterion type for '{column}' must be one of "
                f"{sorted(VALID_CRITERIA_TYPES)}, got '{kind}'."
            )
        normalised[column] = kind_lower
    return normalised


def _resolve_weights(
    criteria: Mapping[str, str],
    weights: Mapping[str, float] | None,
) -> Dict[str, float]:
    """Resolve and normalise weights so that they sum to 1.0.

    If ``weights`` is ``None``, equal weights are used. Otherwise keys must
    match the criteria keys exactly, values must be non-negative, and at
    least one must be positive.

    Args:
        criteria: Validated criteria mapping.
        weights: Optional user-supplied weights.

    Returns:
        New dict of weights summing to 1.0.

    Raises:
        ValueError: On key mismatch, negative weight, or all-zero weights.
    """
    if weights is None:
        n = len(criteria)
        equal = 1.0 / n
        return {column: equal for column in criteria}

    if set(weights) != set(criteria):
        raise ValueError(
            f"weights keys {sorted(weights)} must match criteria keys "
            f"{sorted(criteria)}."
        )

    values: Dict[str, float] = {}
    for column, weight in weights.items():
        if weight < 0:
            raise ValueError(
                f"weight for '{column}' must be non-negative, got {weight}."
            )
        values[column] = float(weight)

    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("weights must have a positive sum.")

    return {column: weight / total for column, weight in values.items()}


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise ``KeyError`` if any column is missing from the DataFrame."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required criterion column(s): {missing}."
        )


# ---------------------------------------------------------------------------
# Core TOPSIS implementation
# ---------------------------------------------------------------------------

def rank_sites_topsis(
    df: pd.DataFrame,
    criteria: Mapping[str, str],
    weights: Mapping[str, float] | None = None,
    *,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Rank reforestation candidate sites using TOPSIS MCDA.

    The algorithm:

    1. Build the decision matrix from the requested columns.
    2. Vector-normalise each criterion column (divide by its Euclidean norm).
    3. Multiply by the (re-normalised) weight vector.
    4. Compute the ideal (best) and anti-ideal (worst) per criterion,
       respecting whether the criterion is a *benefit* (more is better) or
       *cost* (less is better).
    5. Compute each alternative's Euclidean distance to the ideal
       (``d_plus``) and anti-ideal (``d_minus``).
    6. Compute the relative closeness: ``score = d_minus / (d_plus + d_minus)``.
    7. Rank in descending order of closeness (1 = best site).

    A criterion column that has zero variance (all rows equal) contributes
    nothing to discrimination and is handled without dividing by zero.

    Args:
        df: Site DataFrame containing at least the criterion columns.
        criteria: Mapping of column name -> ``'benefit'`` or ``'cost'``.
            Must be non-empty and each column must exist in ``df``.
        weights: Optional mapping of column name -> non-negative weight.
            Keys must match ``criteria``. Values are re-normalised to sum
            to 1.0. If ``None``, equal weights are used.
        drop_na: When ``True`` (default), rows with missing values in any
            criterion column are dropped before ranking. When ``False``
            and NaNs are present, a ``ValueError`` is raised.

    Returns:
        New DataFrame (original is never mutated) with all original columns
        plus ``topsis_score`` (float in [0, 1]; higher is better) and
        ``topsis_rank`` (1-indexed integer, 1 is the top-ranked site),
        sorted by ``topsis_score`` descending.

    Raises:
        ValueError: If inputs are invalid (empty df, empty criteria, invalid
            criterion type, bad weights, or NaNs with ``drop_na=False``).
        KeyError: If a criterion column is missing from ``df``.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "site_id": ["A", "B", "C"],
        ...     "rainfall": [2000, 3000, 1500],
        ...     "slope":    [10.0, 25.0, 5.0],
        ... })
        >>> out = rank_sites_topsis(
        ...     df,
        ...     criteria={"rainfall": "benefit", "slope": "cost"},
        ... )
        >>> list(out["site_id"])[0]  # best site
        'A'
    """
    if df.empty:
        raise ValueError("Cannot rank an empty DataFrame.")

    criteria_norm = _validate_criteria(criteria)
    weights_norm = _resolve_weights(criteria_norm, weights)

    columns = list(criteria_norm.keys())
    _ensure_columns(df, columns)

    working = df.copy()

    # Handle NaNs up front so downstream math is well-defined.
    na_mask = working[columns].isna().any(axis=1)
    if na_mask.any():
        if not drop_na:
            raise ValueError(
                f"{int(na_mask.sum())} row(s) contain NaN in criterion columns; "
                "pass drop_na=True to exclude them."
            )
        working = working.loc[~na_mask].copy()

    if working.empty:
        raise ValueError("No rows remain after dropping NaNs in criterion columns.")

    matrix = working[columns].to_numpy(dtype=float)

    # Step 1: vector normalisation (Euclidean norm per criterion).
    norms = np.linalg.norm(matrix, axis=0)
    # Avoid division by zero: a zero-norm column (all zeros) contributes
    # nothing; replace norm with 1 so the column stays zero.
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalised = matrix / safe_norms

    # Step 2: weighted matrix.
    weight_vec = np.array([weights_norm[c] for c in columns], dtype=float)
    weighted = normalised * weight_vec

    # Step 3: ideal best / worst per criterion.
    ideal_best = np.empty(len(columns), dtype=float)
    ideal_worst = np.empty(len(columns), dtype=float)
    for idx, column in enumerate(columns):
        column_values = weighted[:, idx]
        if criteria_norm[column] == "benefit":
            ideal_best[idx] = column_values.max()
            ideal_worst[idx] = column_values.min()
        else:  # cost
            ideal_best[idx] = column_values.min()
            ideal_worst[idx] = column_values.max()

    # Step 4: distances to ideal and anti-ideal.
    d_plus = np.linalg.norm(weighted - ideal_best, axis=1)
    d_minus = np.linalg.norm(weighted - ideal_worst, axis=1)

    denom = d_plus + d_minus
    # If a row is coincident with both ideals (degenerate one-row input or
    # all-equal matrix), assign a neutral 0.5 score.
    with np.errstate(invalid="ignore", divide="ignore"):
        closeness = np.where(denom == 0, 0.5, d_minus / np.where(denom == 0, 1.0, denom))

    working = working.assign(topsis_score=np.round(closeness, 6))
    ranked = working.sort_values(
        "topsis_score", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    ranked = ranked.assign(topsis_rank=np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def rank_sites_topsis_detailed(
    df: pd.DataFrame,
    criteria: Mapping[str, str],
    weights: Mapping[str, float] | None = None,
    *,
    drop_na: bool = True,
) -> TopsisResult:
    """Run TOPSIS and return a rich :class:`TopsisResult` record.

    Same arguments and semantics as :func:`rank_sites_topsis` but returns
    an immutable :class:`TopsisResult` dataclass exposing the ideal and
    anti-ideal reference points and the resolved criteria/weights. Useful
    for auditing or reporting.

    Args:
        df: See :func:`rank_sites_topsis`.
        criteria: See :func:`rank_sites_topsis`.
        weights: See :func:`rank_sites_topsis`.
        drop_na: See :func:`rank_sites_topsis`.

    Returns:
        :class:`TopsisResult` with the ranked DataFrame and diagnostic data.
    """
    criteria_norm = _validate_criteria(criteria)
    weights_norm = _resolve_weights(criteria_norm, weights)

    ranked_df = rank_sites_topsis(
        df,
        criteria=criteria_norm,
        weights=weights_norm,
        drop_na=drop_na,
    )

    # Recompute ideal/anti-ideal for diagnostics (cheap on typical sizes).
    columns = list(criteria_norm.keys())
    matrix = df[columns].dropna().to_numpy(dtype=float) if drop_na else df[columns].to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(norms == 0, 1.0, norms)
    weight_vec = np.array([weights_norm[c] for c in columns], dtype=float)
    weighted = (matrix / safe_norms) * weight_vec

    ideal_best = np.empty(len(columns), dtype=float)
    ideal_worst = np.empty(len(columns), dtype=float)
    for idx, column in enumerate(columns):
        vals = weighted[:, idx]
        if criteria_norm[column] == "benefit":
            ideal_best[idx] = vals.max()
            ideal_worst[idx] = vals.min()
        else:
            ideal_best[idx] = vals.min()
            ideal_worst[idx] = vals.max()

    return TopsisResult(
        ranked_df=ranked_df,
        ideal_best=ideal_best,
        ideal_worst=ideal_worst,
        criteria=dict(criteria_norm),
        weights=dict(weights_norm),
    )


# ---------------------------------------------------------------------------
# Threshold filtering helper
# ---------------------------------------------------------------------------

def filter_by_thresholds(
    df: pd.DataFrame,
    thresholds: Mapping[str, tuple],
) -> pd.DataFrame:
    """Filter a site DataFrame by per-column (min, max) thresholds.

    A threshold bound may be ``None`` to leave that side unbounded. The
    input DataFrame is never mutated.

    Args:
        df: Site DataFrame.
        thresholds: Mapping of column name -> ``(min, max)`` tuple. Either
            bound may be ``None`` for half-open ranges. Both bounds are
            inclusive when provided.

    Returns:
        New DataFrame containing only rows that satisfy every threshold.

    Raises:
        KeyError: If a threshold column is not in ``df``.
        ValueError: If a threshold is malformed or if ``min > max``.

    Example:
        >>> out = filter_by_thresholds(
        ...     df,
        ...     thresholds={
        ...         "slope_pct": (None, 30.0),
        ...         "annual_rainfall_mm": (1500.0, None),
        ...     },
        ... )
    """
    if not thresholds:
        return df.copy()

    _ensure_columns(df, list(thresholds.keys()))

    mask = pd.Series(True, index=df.index)
    for column, bounds in thresholds.items():
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise ValueError(
                f"threshold for '{column}' must be a (min, max) tuple, got {bounds!r}."
            )
        lower, upper = bounds
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"threshold min ({lower}) > max ({upper}) for column '{column}'."
            )
        if lower is not None:
            mask = mask & (df[column] >= lower)
        if upper is not None:
            mask = mask & (df[column] <= upper)

    return df.loc[mask].copy().reset_index(drop=True)
