# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
