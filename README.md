# Reforestation Site Assessor

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-302%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Site suitability scoring and prioritisation for tropical reforestation projects.

The assessor ingests field survey data, applies a multi-factor weighted scoring
model calibrated to Indonesian/tropical conditions, and ranks candidate sites by
restoration priority. It is designed for field teams, restoration ecologists,
and GIS analysts who need a reproducible, auditable prioritisation workflow.

## Features

- Multi-factor suitability scoring (rainfall, slope, soil depth, forest proximity, road access, degradation level)
- Weighted composite score with configurable weights
- Schema-based input validation with clear error messages
- Full pipeline: load, preprocess, validate, score, prioritise, export
- Bundled demo data covering 18 Indonesian field sites
- Synthetic data generator for development and testing
- 170 unit tests across all modules (pytest)

## Installation

```bash
git clone https://github.com/achmadnaufal/reforestation-site-assessor.git
cd reforestation-site-assessor
pip install -r requirements.txt
```

## Quick Start

```python
from src.main import SiteAssessor

# Initialise with optional configuration
assessor = SiteAssessor(config={"top_n": 5, "min_score": 60.0})

# Load and score the bundled demo data
df = assessor.load_data("demo/sample_data.csv")
scored = assessor.run_pipeline(df)

# Get ranked shortlist
top_sites = assessor.prioritise(scored)
print(top_sites[["site_id", "composite_score", "degradation_level"]])
```

## Step-by-Step Usage

The assessor is designed to be called one step at a time when you want to
inspect intermediate results, or end-to-end via `run_pipeline` for the
common case. A typical field-analyst workflow looks like:

1. **Load** raw CSV/Excel into a DataFrame
   ```python
   from src.main import SiteAssessor
   assessor = SiteAssessor()
   df = assessor.load_data("demo/sample_data.csv")
   ```
2. **Preprocess** (normalise column names, strip whitespace, drop empty rows)
   ```python
   cleaned = assessor.preprocess(df)
   ```
3. **Validate** the schema and field values
   ```python
   result = assessor.validate(cleaned)
   if not result.is_valid:
       for err in result.errors:
           print(err)
   ```
4. **Filter** candidate sites by hard thresholds (e.g. drop anything
   steeper than 30 % or drier than 1500 mm/yr)
   ```python
   from src.mcda_topsis import filter_by_thresholds
   shortlist = filter_by_thresholds(
       cleaned,
       thresholds={
           "slope_pct": (None, 30.0),
           "annual_rainfall_mm": (1500.0, None),
       },
   )
   ```
5. **Score** with the weighted suitability model
   ```python
   scored = assessor.run_pipeline(shortlist)
   ```
6. **Prioritise** via either the composite score or TOPSIS MCDA
   ```python
   top = assessor.prioritise(scored)
   ```
7. **Visualise** the distribution and top candidates in the terminal
   ```python
   from src.visualization import score_distribution_table, format_top_sites
   print(score_distribution_table(scored))
   print(format_top_sites(top, top_n=5))
   ```
8. **Export** results for stakeholders
   ```python
   assessor.save_results(top, "out/priority_sites.csv")
   ```

## Multi-Criteria Decision Analysis (TOPSIS)

For transparent, auditable ranking with explicit benefit/cost semantics per
criterion, use the TOPSIS helper in `src.mcda_topsis`. It handles weight
re-normalisation, NaN rows, zero-variance columns, and never mutates the
input.

```python
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
```

For auditing, `rank_sites_topsis_detailed` returns a frozen `TopsisResult`
containing the ranked DataFrame plus the ideal-best and anti-ideal
reference points actually used.

## Terminal Visualisation

The `src.visualization` module renders dependency-free ASCII summaries:

```python
from src.visualization import (
    ascii_histogram, score_distribution_table, format_top_sites,
)

print(ascii_histogram(scored["composite_score"], bins=8, width=30))
print(score_distribution_table(scored))
print(format_top_sites(top, top_n=5))
```

## Example Code

### Score a single site programmatically

```python
from src.scorer import (
    score_rainfall, score_slope, score_soil_depth,
    score_forest_proximity, score_road_proximity,
    score_degradation, compute_composite_score,
)

composite = compute_composite_score(
    rainfall_score=score_rainfall(2850),          # mm/year
    slope_score=score_slope(8.5),                 # percent
    soil_depth_score=score_soil_depth(90),        # cm
    forest_proximity_score=score_forest_proximity(1.2),  # km
    road_proximity_score=score_road_proximity(2.3),      # km
    degradation_score=score_degradation("high"),
)
print(f"Composite suitability score: {composite}")  # -> 89.05
```

### Custom scoring weights

```python
from src.scorer import DEFAULT_WEIGHTS

custom_weights = {**DEFAULT_WEIGHTS, "rainfall": 0.30, "slope": 0.10}
scored = assessor.run_pipeline(df)  # uses default weights
# or pass weights via config:
assessor2 = SiteAssessor(config={"weights": custom_weights})
scored2 = assessor2.run_pipeline(df)
```

### Validate data before processing

```python
from src.validators import validate_dataframe

result = validate_dataframe(df)
if not result.is_valid:
    for error in result.errors:
        print(f"ERROR: {error}")
for warning in result.warnings:
    print(f"WARNING: {warning}")
```

## New: Carbon Sequestration Estimator

Estimate above-ground biomass (AGB) accumulation and CO2-equivalent (CO2e)
sequestration for one or many reforestation sites over a configurable rotation
period.  The model uses Chapman-Richards sigmoidal growth curves calibrated to
IPCC Tier-1 Tier-1 defaults for four tropical climate zones.

### Estimate a single site

```python
from src.carbon_sequestration_estimator import estimate_site_sequestration

est = estimate_site_sequestration(
    site_id="SITE-KAL-001",
    area_ha=50.0,                    # hectares
    climate_zone="tropical_moist",   # or tropical_dry | montane | peat_swamp
    rotation_years=30,
    biomass_factor=1.0,              # 1.0 = mixed native tropical species
)

print(f"Total CO2e sequestered : {est.total_co2e_Mg:,.1f} Mg CO2e")
print(f"Annual rate            : {est.co2e_per_ha_per_year:.2f} Mg CO2e / ha / yr")
# Total CO2e sequestered : 55,347.8 Mg CO2e
# Annual rate            : 36.90 Mg CO2e / ha / yr
```

### Batch-estimate from a DataFrame

```python
import pandas as pd
from src.carbon_sequestration_estimator import estimate_dataframe_sequestration

sites = pd.read_csv("demo/sample_data.csv")
# Add an area_ha column (not in the core schema — supply your own field data)
sites = sites.assign(area_ha=75.0)

result = estimate_dataframe_sequestration(
    sites,
    area_ha_col="area_ha",
    default_climate_zone="tropical_moist",
    rotation_years=30,
)
print(result[["site_id", "total_co2e_Mg", "co2e_per_ha_per_year"]].head())
```

### Summarise a portfolio

```python
from src.carbon_sequestration_estimator import (
    estimate_site_sequestration,
    summarise_portfolio,
)

estimates = [
    estimate_site_sequestration("SITE-A", 50.0, "tropical_moist", 30),
    estimate_site_sequestration("SITE-B", 120.0, "tropical_dry", 30),
    estimate_site_sequestration("SITE-C", 80.0, "peat_swamp", 30),
]
summary = summarise_portfolio(estimates)
print(f"Portfolio: {summary['total_sites']} sites, "
      f"{summary['total_area_ha']:.0f} ha, "
      f"{summary['total_co2e_Mg']:,.0f} Mg CO2e total")
```

Supported climate zones: `tropical_moist`, `tropical_dry`, `montane`, `peat_swamp`.
All functions return new objects and never mutate their inputs.

## New: Soil Erosion Risk Scorer

Compute an integrated erosion-risk index per site, combining slope, rainfall
erosivity, soil erodibility, and current land cover into a bounded 0-100
score with a categorical band (`low`, `moderate`, `high`, `severe`). The
formulation is inspired by the RUSLE factor framework (R, K, LS, C) and is
intended for first-order site prioritisation rather than quantitative
soil-loss prediction. Use it to flag sites that need protective
interventions (terracing, contour planting, fast-establishing pioneers).

### Score a single site

```python
from src.erosion_risk_scorer import compute_erosion_risk

risk = compute_erosion_risk(
    slope_pct=22.0,
    annual_rainfall_mm=2850.0,
    soil_type="loam",            # sandy | silt | loam | clay | peat | laterite | alluvial | volcanic
    current_land_use="bare_land",  # bare_land | grassland | shrubland | agricultural | plantation | degraded_forest | mixed_vegetation
)
print(f"Risk score: {risk.composite_score} ({risk.risk_class})")
print(f"  slope     : {risk.slope_score}")
print(f"  rainfall  : {risk.rainfall_score}")
print(f"  soil      : {risk.soil_score}")
print(f"  land cover: {risk.land_cover_score}")
```

### Score every site in a DataFrame

```python
import pandas as pd
from src.erosion_risk_scorer import score_erosion_dataframe

sites = pd.read_csv("demo/sample_data.csv")
scored = score_erosion_dataframe(sites)
print(scored[["site_id", "erosion_risk_score", "erosion_risk_class"]].head())
```

### Custom weights

Weights must include all four components and sum to 1.0; otherwise a
clear `ValueError` is raised:

```python
from src.erosion_risk_scorer import compute_erosion_risk

custom = {"slope": 0.50, "rainfall": 0.20, "soil": 0.20, "land_cover": 0.10}
risk = compute_erosion_risk(20.0, 2500.0, "sandy", "shrubland", weights=custom)
```

## Sample Output

Running `python examples/basic_usage.py` on the bundled demo data produces:

```
=== Top 5 Priority Sites ===
     site_id  composite_score degradation_level  annual_rainfall_mm  slope_pct
SITE-KAL-003            94.20            severe                2650        5.1
SITE-SUL-002            93.40            severe                2750        7.6
SITE-SUM-004            93.17            severe                3500        6.9
SITE-KAL-001            89.05              high                2850        8.5
SITE-KAL-005            88.92              high                3350        3.4

=== Assessment Summary ===
  Total sites assessed : 18
  Mean composite score : 79.70
  Score distribution:
    low_(<40)             : 0 sites
    moderate_(40-60)      : 0 sites
    good_(60-80)          : 8 sites
    excellent_(>=80)      : 10 sites
```

## Data Format

Input CSV/Excel files must contain these columns:

| Column | Type | Description |
|---|---|---|
| `site_id` | string | Unique site identifier |
| `latitude` | float | Decimal degrees (-90 to 90) |
| `longitude` | float | Decimal degrees (-180 to 180) |
| `elevation_m` | float | Elevation above sea level (m) |
| `slope_pct` | float | Terrain slope percentage (0–100) |
| `annual_rainfall_mm` | float | Mean annual precipitation (mm) |
| `soil_type` | string | One of: clay, sandy, loam, silt, peat, laterite, volcanic, alluvial |
| `current_land_use` | string | One of: bare_land, grassland, shrubland, degraded_forest, agricultural, plantation, mixed_vegetation |
| `distance_to_road_km` | float | Distance to nearest road (km) |
| `distance_to_forest_km` | float | Distance to nearest forest edge (km) |
| `degradation_level` | string | One of: low, medium, high, severe |

See `demo/sample_data.csv` for a complete example with 18 Indonesian field sites.

## Scoring Model

The composite score is a weighted average of six sub-scores (each 0–100):

| Factor | Default Weight | Rationale |
|---|---|---|
| Rainfall | 20% | Adequate moisture is essential for tree survival |
| Slope | 20% | Gentler slopes improve planting success and reduce erosion |
| Soil depth | 15% | Deeper soils support root development |
| Forest proximity | 20% | Nearby seed sources accelerate natural regeneration |
| Road proximity | 10% | Moderate access enables logistics without excessive pressure |
| Degradation level | 15% | More degraded sites are higher restoration priorities |

## Project Structure

```
reforestation-site-assessor/
├── src/
│   ├── __init__.py          # Public API exports
│   ├── main.py              # SiteAssessor orchestration class
│   ├── scorer.py            # Suitability scoring functions
│   ├── validators.py        # Input validation utilities
│   └── data_generator.py   # Synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_main.py         # Tests for SiteAssessor
│   ├── test_scorer.py       # Tests for scoring functions
│   ├── test_validators.py   # Tests for validators
│   └── test_data_generator.py
├── demo/
│   └── sample_data.csv      # 18 realistic Indonesian field sites
├── examples/
│   └── basic_usage.py       # End-to-end usage example
├── data/                    # Data directory (gitignored for real data)
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_scorer.py -v
```

Expected output: **302 tests passed**.

## Generating Synthetic Data

```bash
# Generate 300 synthetic Indonesian site records to data/sample.csv
python src/data_generator.py
```

```python
# Or use the API directly
from src.data_generator import generate_sample

df = generate_sample(n=500, seed=42, include_anomalies=True)
```

## License

MIT License — free to use, modify, and distribute.
