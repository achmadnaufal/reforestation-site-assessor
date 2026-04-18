# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] - 2026-04-19

### Added

- `src/mcda_topsis.py`: New multi-criteria decision-analysis module implementing
  TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
  Handles mixed benefit/cost criteria, weight re-normalisation to sum 1.0,
  NaN handling (`drop_na` flag), zero-variance columns, and single-row inputs.
  Public API:
  - `rank_sites_topsis` -- returns a new DataFrame with `topsis_score` and
    `topsis_rank` columns, sorted best-first.
  - `rank_sites_topsis_detailed` -- returns a frozen `TopsisResult` dataclass
    exposing the ideal-best and anti-ideal reference vectors for auditing.
  - `filter_by_thresholds` -- filters a site DataFrame by per-column
    `(min, max)` bounds (either side optional).
- `src/visualization.py`: Dependency-free text-based visualisation helpers
  (`ascii_histogram`, `score_distribution_table`, `format_top_sites`) for
  terminal and CI-friendly output.
- `tests/test_mcda_topsis.py`, `tests/test_visualization.py`,
  `tests/test_integration_mcda.py`: 44 new pytest cases covering happy paths,
  immutability, edge cases (empty input, missing columns, NaNs, zero variance,
  invalid weights), and end-to-end integration with the bundled sample data.

### Changed

- `src/__init__.py`: Re-exports the new MCDA and visualisation public API.
- `README.md`: Added "Step-by-Step Usage", "Multi-Criteria Decision Analysis
  (TOPSIS)", and "Terminal Visualisation" sections. Updated test-count badge
  and expected pytest output to 302.

---

## [Unreleased] - 2026-04-18

### Added

- `src/erosion_risk_scorer.py`: New module computing an integrated soil
  erosion-risk score (0-100) per reforestation site by combining slope,
  rainfall erosivity, soil erodibility, and current land-cover protection.
  Inspired by the RUSLE factor framework (R, K, LS, C) but rescaled to a
  bounded site-prioritisation index. Public API:
  - `score_slope_erosion`, `score_rainfall_erosivity`,
    `score_soil_erodibility`, `score_land_cover_protection` -- component
    scorers, each returning a value in [0, 100].
  - `compute_erosion_risk` -- single-site composite scorer returning an
    immutable `ErosionRiskBreakdown` dataclass with sub-scores, the
    composite score, and a categorical band (`low`, `moderate`, `high`,
    `severe`). Validates that custom weights cover all four components,
    are non-negative, and sum to 1.0.
  - `score_erosion_dataframe` -- batch scorer returning a new DataFrame
    with six added columns (`erosion_slope_score`, ...,
    `erosion_risk_class`); the input is never mutated and missing
    columns raise a clear `KeyError`.
  - `DEFAULT_EROSION_WEIGHTS` -- default weight mapping (slope 0.40,
    rainfall 0.25, soil 0.20, land_cover 0.15).
- `tests/test_erosion_risk_scorer.py`: 42 pytest tests covering component
  scorers, monotonicity, saturation, case-insensitive lookups, risk-band
  boundaries, custom-weight validation (missing keys, negative weight,
  weights not summing to 1.0), DataFrame happy path, empty DataFrame,
  missing-column errors, and input immutability.

### Changed

- `src/__init__.py`: Re-exports the new erosion-risk public API.
- `README.md`: Added "Soil Erosion Risk Scorer" section with single-site,
  DataFrame, and custom-weight examples.

---

## [Unreleased] - 2026-04-17

### Added

- `src/carbon_sequestration_estimator.py`: New module estimating above-ground
  biomass (AGB) accumulation and CO2-equivalent (CO2e) sequestration for
  reforestation sites.  Key functions:
  - `estimate_site_sequestration` — single-site estimate using a
    Chapman-Richards sigmoidal growth curve parameterised per climate zone
    (tropical_moist, tropical_dry, montane, peat_swamp) with IPCC Tier-1
    default peak-AGB values and below-ground biomass ratio.
  - `estimate_dataframe_sequestration` — batch estimation over a pandas
    DataFrame with optional per-row climate-zone and biomass-factor columns;
    returns a new DataFrame (input is never mutated).
  - `summarise_portfolio` — aggregate CO2e totals, area-weighted annual rate,
    and min/max across a list of `SequestrationEstimate` dataclass records.
  - `SequestrationEstimate` — frozen dataclass holding per-site metrics
    including the full annual CO2e time series.
- `tests/test_carbon_sequestration_estimator.py`: 46 pytest tests covering
  growth-curve behaviour, happy-path estimation, all climate zones
  (parametrized), rotation-period edge cases (1, 10, 50, 100 years),
  invalid-input validation, immutability, determinism, single-row DataFrames,
  missing-column errors, and portfolio aggregation.

---

## [0.2.0] - 2026-04-16

### Added

- `src/scorer.py`: Six domain-specific scoring functions (`score_rainfall`,
  `score_slope`, `score_soil_depth`, `score_forest_proximity`,
  `score_road_proximity`, `score_degradation`) and a weighted composite scorer
  (`compute_composite_score`). DataFrame-level scoring via `score_dataframe`.
  Site prioritisation via `prioritise_sites`.

- `src/validators.py`: Schema-based input validation with `ValidationResult`
  (immutable dataclass), field-level validators for coordinates, elevation,
  slope, rainfall, scores, distances, and categorical fields, plus full
  DataFrame validation via `validate_dataframe`.

- `src/main.py`: Extended `SiteAssessor` with `run_pipeline` (full scored
  output), `prioritise`, `get_summary`, `save_results`, and `strict` mode for
  validation. Added comprehensive docstrings and type hints throughout.

- `tests/`: Full pytest test suite (170 tests) across four files:
  `test_scorer.py`, `test_validators.py`, `test_main.py`,
  `test_data_generator.py`. Covers normal inputs, edge cases, error paths,
  immutability, and mutation safety.

- `demo/sample_data.csv`: 18 realistic field sites spanning Kalimantan,
  Sumatra, Java, Sulawesi, and Papua with tropical-calibrated attribute values.

- `CHANGELOG.md`: This file.

### Changed

- `src/main.py`: `preprocess` now returns a new DataFrame (immutable pattern)
  instead of using `inplace=True`. Column normalisation uses `rename` rather
  than direct assignment. String stripping applied without mutation.

- `src/data_generator.py`: Replaced deprecated `np.random.seed` /
  `np.random.exponential` global API with `np.random.default_rng`. Columns now
  match the validated schema (`latitude`, `longitude`, `elevation_m`,
  `slope_pct`, etc.). Added `include_anomalies` flag for robustness testing.
  Input guard raises `ValueError` for `n < 1`.

- `src/__init__.py`: Exports public API symbols from all submodules.

- `requirements.txt`: Added `pytest>=7.4.0`, `pytest-cov>=4.1.0`,
  `openpyxl>=3.1.0`.

- `README.md`: Added badges, Quick Start, example code snippets, Sample Output
  section, scoring model table, data format table, Project Structure, and
  Running Tests section.

- `examples/basic_usage.py`: Updated to demonstrate the full pipeline including
  `run_pipeline`, `prioritise`, and `get_summary`.

### Fixed

- `score_slope`: Values above 100% now return 0 instead of raising, so the
  non-strict pipeline mode can process rows that fail validation without
  crashing.

---

## [0.1.0] - 2024-01-01

### Added

- Initial project scaffold: `src/main.py` with `SiteAssessor` (load, validate,
  preprocess, analyze, run, to_dataframe), `src/data_generator.py` with
  `generate_sample`, basic `README.md`, and `requirements.txt`.
