"""
Generate synthetic site-assessment data for reforestation-site-assessor.

Run this script to create test data::

    python src/data_generator.py

The generated dataset contains realistic values for Indonesian/tropical
field sites, including geographic coordinates, terrain attributes,
environmental metrics, and categorical land classifications.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.validators import (
    VALID_DEGRADATION_LEVELS,
    VALID_LAND_USE_TYPES,
    VALID_SOIL_TYPES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indonesian / tropical bounding box (Sumatra, Kalimantan, Java, Sulawesi)
_LAT_MIN: float = -8.0
_LAT_MAX: float = 5.5
_LON_MIN: float = 95.0
_LON_MAX: float = 141.0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_sample(
    n: int = 300,
    seed: int = 42,
    include_anomalies: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic site-assessment dataset with realistic tropical values.

    All column distributions are calibrated to Indonesian/tropical conditions:
    - High rainfall (1 000–4 000 mm typical)
    - Coordinates within the Indonesian archipelago
    - Soil types common to Kalimantan/Sumatra peatlands and volcanic Java

    Args:
        n: Number of site records to generate. Must be >= 1.
        seed: Random seed for reproducibility.
        include_anomalies: When ``True``, inject ~5% rows with edge-case
            values (zero rainfall, very steep slopes, etc.) to support
            robustness testing.

    Returns:
        DataFrame with columns defined in
        :data:`~src.validators.REQUIRED_COLUMNS` plus ``soil_depth_cm``.

    Raises:
        ValueError: If ``n`` < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    site_ids: List[str] = [f"SITE-{i:04d}" for i in range(1, n + 1)]

    latitudes: np.ndarray = rng.uniform(_LAT_MIN, _LAT_MAX, n)
    longitudes: np.ndarray = rng.uniform(_LON_MIN, _LON_MAX, n)

    # Elevation: mostly lowland / foothills (0–1 500 m), occasional highlands
    elevations: np.ndarray = np.abs(rng.normal(300, 250, n)).clip(0, 2500).round(1)

    # Slope: right-skewed, most sites gently sloped
    slopes: np.ndarray = np.abs(rng.exponential(12, n)).clip(0, 80).round(1)

    # Rainfall: tropical range
    rainfalls: np.ndarray = rng.normal(2200, 600, n).clip(800, 5000).round(0)

    # Soil depth: varies by soil type; generated here as a proxy
    soil_depths: np.ndarray = rng.normal(75, 30, n).clip(10, 200).round(1)

    soil_types: List[str] = random.choices(VALID_SOIL_TYPES, k=n)
    land_uses: List[str] = random.choices(VALID_LAND_USE_TYPES, k=n)
    degradation_levels: List[str] = random.choices(VALID_DEGRADATION_LEVELS, k=n)

    dist_road: np.ndarray = np.abs(rng.exponential(5, n)).clip(0.1, 50).round(2)
    dist_forest: np.ndarray = np.abs(rng.exponential(8, n)).clip(0.1, 80).round(2)

    df = pd.DataFrame(
        {
            "site_id": site_ids,
            "latitude": latitudes.round(6),
            "longitude": longitudes.round(6),
            "elevation_m": elevations,
            "slope_pct": slopes,
            "annual_rainfall_mm": rainfalls,
            "soil_type": soil_types,
            "soil_depth_cm": soil_depths,
            "current_land_use": land_uses,
            "distance_to_road_km": dist_road,
            "distance_to_forest_km": dist_forest,
            "degradation_level": degradation_levels,
        }
    )

    if include_anomalies:
        df = _inject_anomalies(df, rng)

    return df


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject edge-case values into ~5% of rows for robustness testing.

    Returns a new DataFrame; the input is not modified.

    Args:
        df: Source DataFrame produced by :func:`generate_sample`.
        rng: Seeded random number generator.

    Returns:
        New DataFrame with anomalous values injected.
    """
    anomaly_count = max(1, len(df) // 20)
    indices = rng.choice(len(df), size=anomaly_count, replace=False)

    modified = df.copy()
    for idx in indices:
        anomaly_type = rng.integers(0, 4)
        if anomaly_type == 0:
            modified.at[idx, "annual_rainfall_mm"] = 0.0
        elif anomaly_type == 1:
            modified.at[idx, "slope_pct"] = 75.0
        elif anomaly_type == 2:
            modified.at[idx, "elevation_m"] = 0.0
        else:
            modified.at[idx, "soil_depth_cm"] = 5.0

    return modified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    df = generate_sample(300)
    out_path = output_dir / "sample.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} records -> {out_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
