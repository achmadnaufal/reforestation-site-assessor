# Reforestation Site Assessor

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-170%20passed-brightgreen)
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

Expected output: **170 tests passed**.

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
