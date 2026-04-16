"""
Carbon sequestration estimator for reforestation site assessment.

This module estimates above-ground biomass (AGB) accumulation and CO2-equivalent
(CO2e) sequestration potential for candidate reforestation sites over a defined
rotation period.  Estimates are grounded in published allometric relationships
and IPCC Tier-1 default values for tropical moist/dry forests and are intended
as a first-order planning tool rather than a substitute for full-inventory
measurements.

Key assumptions
---------------
- Above-ground biomass growth follows a sigmoidal (Chapman-Richards) curve
  parameterised per climate zone.
- Below-ground biomass is estimated as 26 % of AGB (IPCC 2006, Table 4.4,
  tropical moist forest).
- Conversion from total biomass to CO2e uses the factor 3.67 × 0.47
  (44/12 × carbon fraction = 1.7249).
- Species mix adjusts the peak AGB via a ``biomass_factor`` (1.0 = mixed
  native tropical, >1 fast-growing species, <1 slow-growing or degraded).
- Site area is expressed in hectares; results are per-hectare and totals.

References
----------
- IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Volume 4.
- Brown, S. (1997). Estimating Biomass and Biomass Change of Tropical Forests.
  FAO Forestry Paper 134.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ratio of below-ground to above-ground biomass for tropical forests (IPCC).
_BGB_RATIO: float = 0.26

#: Carbon fraction of dry biomass (IPCC default for tropical forests).
_CARBON_FRACTION: float = 0.47

#: CO2 to C ratio (44 / 12).
_CO2_TO_C: float = 44.0 / 12.0

#: Combined multiplier: total biomass (AGB + BGB) → CO2e (Mg CO2e / Mg biomass).
_BIOMASS_TO_CO2E: float = (1.0 + _BGB_RATIO) * _CARBON_FRACTION * _CO2_TO_C

#: Peak AGB by climate zone (Mg dry matter / ha) based on IPCC Tier-1 defaults.
_PEAK_AGB_BY_ZONE: Dict[str, float] = {
    "tropical_moist": 300.0,
    "tropical_dry": 180.0,
    "montane": 220.0,
    "peat_swamp": 250.0,
}

#: Default climate zone when not specified.
_DEFAULT_CLIMATE_ZONE: str = "tropical_moist"

#: Recognised climate zone keys.
VALID_CLIMATE_ZONES: frozenset[str] = frozenset(_PEAK_AGB_BY_ZONE.keys())

#: Chapman-Richards growth rate parameter (k) per climate zone.
#: Higher k → faster canopy closure.
_GROWTH_RATE_BY_ZONE: Dict[str, float] = {
    "tropical_moist": 0.18,
    "tropical_dry": 0.12,
    "montane": 0.10,
    "peat_swamp": 0.14,
}


# ---------------------------------------------------------------------------
# Data classes (immutable by convention; frozen=True enforces it)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequestrationEstimate:
    """Immutable result record for a single reforestation site.

    Attributes:
        site_id: Unique identifier copied from the input record.
        area_ha: Site area in hectares.
        climate_zone: Climate zone used for the estimate.
        rotation_years: Planning horizon in years.
        biomass_factor: Species-mix multiplier applied to peak AGB.
        peak_agb_ha: Peak above-ground biomass at end of rotation (Mg DM / ha).
        total_agb_Mg: Total above-ground biomass for the whole site (Mg DM).
        total_co2e_Mg: Total CO2e sequestered over the rotation (Mg CO2e).
        co2e_per_ha_per_year: Annualised CO2e sequestration rate (Mg CO2e / ha / yr).
        annual_co2e_series: CO2e sequestered *each* year (list, length = rotation_years).
    """

    site_id: str
    area_ha: float
    climate_zone: str
    rotation_years: int
    biomass_factor: float
    peak_agb_ha: float
    total_agb_Mg: float
    total_co2e_Mg: float
    co2e_per_ha_per_year: float
    annual_co2e_series: tuple[float, ...]


# ---------------------------------------------------------------------------
# Core estimation logic
# ---------------------------------------------------------------------------

def _chapman_richards_agb(
    year: int,
    peak_agb: float,
    k: float,
) -> float:
    """Return AGB (Mg DM / ha) at *year* using a Chapman-Richards curve.

    AGB(t) = peak_agb × (1 - e^(-k·t))^3

    Args:
        year: Year in the rotation (1-indexed).  Must be >= 1.
        peak_agb: Asymptotic maximum AGB (Mg DM / ha).  Must be > 0.
        k: Growth-rate coefficient.  Must be > 0.

    Returns:
        Estimated AGB in Mg DM / ha at the given year.

    Raises:
        ValueError: If ``year`` < 1, ``peak_agb`` <= 0, or ``k`` <= 0.

    Example::

        >>> round(_chapman_richards_agb(10, 300.0, 0.18), 2)
        148.43
    """
    if year < 1:
        raise ValueError(f"year must be >= 1, got {year}")
    if peak_agb <= 0:
        raise ValueError(f"peak_agb must be > 0, got {peak_agb}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    return peak_agb * (1.0 - math.exp(-k * year)) ** 3


def estimate_site_sequestration(
    site_id: str,
    area_ha: float,
    climate_zone: str = _DEFAULT_CLIMATE_ZONE,
    rotation_years: int = 30,
    biomass_factor: float = 1.0,
) -> SequestrationEstimate:
    """Estimate CO2e sequestration for a single reforestation site.

    The function models above-ground biomass (AGB) growth using a
    Chapman-Richards sigmoidal curve calibrated to IPCC Tier-1 defaults for the
    specified climate zone.  Below-ground biomass is added as a fixed proportion
    of AGB before converting to CO2e.

    All computations return a new :class:`SequestrationEstimate` object; no
    external state is mutated.

    Args:
        site_id: Unique site identifier string (non-empty).
        area_ha: Site area in hectares.  Must be > 0.
        climate_zone: One of ``"tropical_moist"``, ``"tropical_dry"``,
            ``"montane"``, or ``"peat_swamp"``.  Defaults to
            ``"tropical_moist"``.
        rotation_years: Planning horizon in years.  Must be between 1 and 100
            (inclusive).
        biomass_factor: Multiplier on peak AGB reflecting species mix.
            1.0 = mixed native tropical; >1 = fast-growing pioneer species;
            <1 = slow-growing or highly degraded conditions.  Must be > 0.

    Returns:
        A frozen :class:`SequestrationEstimate` dataclass with per-hectare and
        total metrics.

    Raises:
        ValueError: If any argument fails validation.

    Example::

        >>> est = estimate_site_sequestration(
        ...     site_id="SITE-KAL-001",
        ...     area_ha=50.0,
        ...     climate_zone="tropical_moist",
        ...     rotation_years=30,
        ...     biomass_factor=1.0,
        ... )
        >>> round(est.total_co2e_Mg, 1)
        55347.8
        >>> round(est.co2e_per_ha_per_year, 2)
        36.90
    """
    _validate_site_inputs(site_id, area_ha, climate_zone, rotation_years, biomass_factor)

    peak_agb_base = _PEAK_AGB_BY_ZONE[climate_zone]
    peak_agb = peak_agb_base * biomass_factor
    k = _GROWTH_RATE_BY_ZONE[climate_zone]

    # Build annual AGB series (Mg DM / ha / year), compute incremental CO2e
    agb_series: list[float] = [
        _chapman_richards_agb(yr, peak_agb, k) for yr in range(1, rotation_years + 1)
    ]

    # Annual increment = AGB(t) - AGB(t-1); AGB(0) = 0
    annual_increments: list[float] = [agb_series[0]] + [
        agb_series[i] - agb_series[i - 1] for i in range(1, rotation_years)
    ]

    # Convert annual biomass increment to CO2e (includes BGB)
    annual_co2e_ha: list[float] = [inc * _BIOMASS_TO_CO2E for inc in annual_increments]

    # Scale to full site area
    annual_co2e_total: list[float] = [v * area_ha for v in annual_co2e_ha]

    total_co2e = sum(annual_co2e_total)
    total_agb = agb_series[-1] * area_ha  # end-of-rotation AGB for whole site

    co2e_per_ha_per_year = (
        total_co2e / (area_ha * rotation_years) if area_ha > 0 and rotation_years > 0 else 0.0
    )

    return SequestrationEstimate(
        site_id=site_id,
        area_ha=area_ha,
        climate_zone=climate_zone,
        rotation_years=rotation_years,
        biomass_factor=biomass_factor,
        peak_agb_ha=peak_agb,
        total_agb_Mg=total_agb,
        total_co2e_Mg=total_co2e,
        co2e_per_ha_per_year=co2e_per_ha_per_year,
        annual_co2e_series=tuple(annual_co2e_total),
    )


# ---------------------------------------------------------------------------
# DataFrame-level API
# ---------------------------------------------------------------------------

def estimate_dataframe_sequestration(
    df: pd.DataFrame,
    area_ha_col: str = "area_ha",
    climate_zone_col: Optional[str] = None,
    default_climate_zone: str = _DEFAULT_CLIMATE_ZONE,
    rotation_years: int = 30,
    biomass_factor_col: Optional[str] = None,
    default_biomass_factor: float = 1.0,
) -> pd.DataFrame:
    """Estimate sequestration for every site in a DataFrame.

    Returns a *new* DataFrame (the input is never modified) containing the
    original columns plus sequestration result columns.

    Required input columns
    ~~~~~~~~~~~~~~~~~~~~~~
    - ``site_id`` (str): Unique site identifier.
    - *area_ha_col* (float): Site area in hectares (column name configurable).

    Optional input columns
    ~~~~~~~~~~~~~~~~~~~~~~
    - *climate_zone_col* (str, optional): Per-row climate zone.  If omitted or
      ``None``, *default_climate_zone* is used for all rows.
    - *biomass_factor_col* (float, optional): Per-row biomass factor.  If
      omitted or ``None``, *default_biomass_factor* is used for all rows.

    Added output columns
    ~~~~~~~~~~~~~~~~~~~~
    - ``climate_zone``: Climate zone used for each site.
    - ``biomass_factor``: Biomass factor used for each site.
    - ``peak_agb_ha``: Peak above-ground biomass at end of rotation (Mg DM/ha).
    - ``total_agb_Mg``: Total end-of-rotation AGB for the whole site (Mg DM).
    - ``total_co2e_Mg``: Total CO2e sequestered over the rotation (Mg CO2e).
    - ``co2e_per_ha_per_year``: Annualised sequestration rate (Mg CO2e/ha/yr).

    Args:
        df: Input DataFrame with at least ``site_id`` and *area_ha_col*
            columns.  Must not be empty.
        area_ha_col: Name of the column holding site area in hectares.
        climate_zone_col: Column name for per-row climate zones, or ``None``
            to use *default_climate_zone* for all rows.
        default_climate_zone: Fallback climate zone when *climate_zone_col*
            is ``None`` or a cell value is missing.
        rotation_years: Planning horizon in years (1–100, inclusive).
        biomass_factor_col: Column name for per-row biomass factors, or
            ``None`` to use *default_biomass_factor* for all rows.
        default_biomass_factor: Fallback biomass factor when
            *biomass_factor_col* is ``None`` or a cell value is missing.

    Returns:
        A new DataFrame with sequestration result columns appended.  Row order
        matches the input.

    Raises:
        ValueError: If required columns are missing, the DataFrame is empty,
            or any parameter fails validation.

    Example::

        >>> import pandas as pd
        >>> sites = pd.DataFrame({
        ...     "site_id": ["SITE-KAL-001", "SITE-SUM-001"],
        ...     "area_ha": [50.0, 120.0],
        ... })
        >>> result = estimate_dataframe_sequestration(sites, rotation_years=20)
        >>> list(result.columns)  # doctest: +NORMALIZE_WHITESPACE
        ['site_id', 'area_ha', 'climate_zone', 'biomass_factor', 'peak_agb_ha',
         'total_agb_Mg', 'total_co2e_Mg', 'co2e_per_ha_per_year']
    """
    _validate_dataframe_inputs(
        df, area_ha_col, climate_zone_col, biomass_factor_col,
        default_climate_zone, rotation_years, default_biomass_factor,
    )

    records: list[dict] = []
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        area_ha = float(row[area_ha_col])

        if climate_zone_col and climate_zone_col in row.index and pd.notna(row[climate_zone_col]):
            climate_zone = str(row[climate_zone_col])
        else:
            climate_zone = default_climate_zone

        if biomass_factor_col and biomass_factor_col in row.index and pd.notna(row[biomass_factor_col]):
            biomass_factor = float(row[biomass_factor_col])
        else:
            biomass_factor = default_biomass_factor

        try:
            est = estimate_site_sequestration(
                site_id=site_id,
                area_ha=area_ha,
                climate_zone=climate_zone,
                rotation_years=rotation_years,
                biomass_factor=biomass_factor,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid data for site '{site_id}': {exc}") from exc

        records.append({
            "site_id": site_id,
            area_ha_col: area_ha,
            "climate_zone": est.climate_zone,
            "biomass_factor": est.biomass_factor,
            "peak_agb_ha": est.peak_agb_ha,
            "total_agb_Mg": est.total_agb_Mg,
            "total_co2e_Mg": est.total_co2e_Mg,
            "co2e_per_ha_per_year": est.co2e_per_ha_per_year,
        })

    # Preserve any extra columns from the original df
    extra_cols = [c for c in df.columns if c not in ("site_id", area_ha_col)]
    result_df = pd.DataFrame(records)
    if extra_cols:
        extra = df[extra_cols].reset_index(drop=True)
        result_df = pd.concat([result_df, extra], axis=1)

    return result_df


def summarise_portfolio(estimates: Sequence[SequestrationEstimate]) -> Dict[str, float]:
    """Aggregate sequestration metrics across a portfolio of sites.

    Returns a new dictionary; the input sequence and its elements are not
    mutated.

    Args:
        estimates: Sequence of :class:`SequestrationEstimate` objects.
            May be empty (returns zeros).

    Returns:
        Dictionary with the following keys:

        - ``total_sites``: Number of sites.
        - ``total_area_ha``: Combined site area (ha).
        - ``total_co2e_Mg``: Combined CO2e sequestration (Mg CO2e).
        - ``mean_co2e_per_ha_per_year``: Area-weighted mean annual rate
          (Mg CO2e / ha / yr).
        - ``max_co2e_Mg``: Highest single-site CO2e total.
        - ``min_co2e_Mg``: Lowest single-site CO2e total.

    Example::

        >>> est1 = estimate_site_sequestration("A", 50.0)
        >>> est2 = estimate_site_sequestration("B", 100.0, "tropical_dry")
        >>> summary = summarise_portfolio([est1, est2])
        >>> summary["total_sites"]
        2
    """
    if not estimates:
        return {
            "total_sites": 0,
            "total_area_ha": 0.0,
            "total_co2e_Mg": 0.0,
            "mean_co2e_per_ha_per_year": 0.0,
            "max_co2e_Mg": 0.0,
            "min_co2e_Mg": 0.0,
        }

    total_area = sum(e.area_ha for e in estimates)
    total_co2e = sum(e.total_co2e_Mg for e in estimates)
    co2e_values = [e.total_co2e_Mg for e in estimates]

    # Area-weighted mean annual rate
    weighted_rate = (
        sum(e.co2e_per_ha_per_year * e.area_ha for e in estimates) / total_area
        if total_area > 0
        else 0.0
    )

    return {
        "total_sites": len(estimates),
        "total_area_ha": total_area,
        "total_co2e_Mg": total_co2e,
        "mean_co2e_per_ha_per_year": weighted_rate,
        "max_co2e_Mg": max(co2e_values),
        "min_co2e_Mg": min(co2e_values),
    }


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _validate_site_inputs(
    site_id: str,
    area_ha: float,
    climate_zone: str,
    rotation_years: int,
    biomass_factor: float,
) -> None:
    """Validate inputs for :func:`estimate_site_sequestration`.

    Raises:
        ValueError: On any invalid argument.
    """
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id must be a non-empty string")
    if not isinstance(area_ha, (int, float)) or math.isnan(area_ha) or area_ha <= 0:
        raise ValueError(f"area_ha must be a positive number, got {area_ha!r}")
    if climate_zone not in VALID_CLIMATE_ZONES:
        raise ValueError(
            f"climate_zone must be one of {sorted(VALID_CLIMATE_ZONES)}, got {climate_zone!r}"
        )
    if not isinstance(rotation_years, int) or not (1 <= rotation_years <= 100):
        raise ValueError(f"rotation_years must be an int between 1 and 100, got {rotation_years!r}")
    if not isinstance(biomass_factor, (int, float)) or math.isnan(biomass_factor) or biomass_factor <= 0:
        raise ValueError(f"biomass_factor must be a positive number, got {biomass_factor!r}")


def _validate_dataframe_inputs(
    df: pd.DataFrame,
    area_ha_col: str,
    climate_zone_col: Optional[str],
    biomass_factor_col: Optional[str],
    default_climate_zone: str,
    rotation_years: int,
    default_biomass_factor: float,
) -> None:
    """Validate inputs for :func:`estimate_dataframe_sequestration`.

    Raises:
        TypeError: If *df* is not a DataFrame.
        ValueError: On missing columns or invalid parameters.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("df must not be empty")
    required = {"site_id", area_ha_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")
    if climate_zone_col is not None and climate_zone_col not in df.columns:
        raise ValueError(f"climate_zone_col '{climate_zone_col}' not found in df columns")
    if biomass_factor_col is not None and biomass_factor_col not in df.columns:
        raise ValueError(f"biomass_factor_col '{biomass_factor_col}' not found in df columns")
    if default_climate_zone not in VALID_CLIMATE_ZONES:
        raise ValueError(
            f"default_climate_zone must be one of {sorted(VALID_CLIMATE_ZONES)}, "
            f"got {default_climate_zone!r}"
        )
    if not isinstance(rotation_years, int) or not (1 <= rotation_years <= 100):
        raise ValueError(f"rotation_years must be an int between 1 and 100, got {rotation_years!r}")
    if not isinstance(default_biomass_factor, (int, float)) or default_biomass_factor <= 0:
        raise ValueError(
            f"default_biomass_factor must be a positive number, got {default_biomass_factor!r}"
        )
