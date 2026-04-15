"""
Site suitability scoring and prioritization for reforestation projects.

This module provides the :class:`SiteAssessor` orchestration class, which
coordinates data loading, validation, preprocessing, scoring, and result
export through a single high-level API.

Author: github.com/achmadnaufal
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.scorer import DEFAULT_WEIGHTS, prioritise_sites, score_dataframe
from src.validators import ValidationResult, validate_dataframe

logger = logging.getLogger(__name__)


class SiteAssessor:
    """Orchestrator for end-to-end reforestation site suitability assessment.

    The assessor coordinates the full pipeline:
    load → preprocess → validate → score → prioritise → export.

    All intermediate transformations return *new* objects; the original
    DataFrames and configuration are never mutated.

    Attributes:
        config: Runtime configuration dictionary. Recognised keys:
            - ``weights`` (dict): Custom scoring weights forwarded to
              :func:`~src.scorer.score_dataframe`.
            - ``top_n`` (int): Number of top-priority sites to return from
              :meth:`prioritise`. Defaults to 10.
            - ``min_score`` (float): Minimum composite score threshold.
              Defaults to 0.0.

    Example::

        assessor = SiteAssessor()
        df = assessor.load_data("demo/sample_data.csv")
        scored = assessor.run_pipeline(df)
        print(scored[["site_id", "composite_score"]].head())
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the SiteAssessor with an optional configuration.

        Args:
            config: Optional dictionary of runtime settings. A shallow copy
                is stored so that external mutations do not affect this
                instance.
        """
        self.config: Dict[str, Any] = dict(config) if config else {}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load site data from a CSV or Excel file.

        Args:
            filepath: Absolute or relative path to the input file.
                Supported extensions: ``.csv``, ``.xlsx``, ``.xls``.

        Returns:
            Raw DataFrame with contents of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
            OSError: If the file cannot be read due to permissions or
                corruption.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: '{filepath}'.")

        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            logger.info("Loading Excel file: %s", filepath)
            return pd.read_excel(filepath)
        if suffix == ".csv":
            logger.info("Loading CSV file: %s", filepath)
            return pd.read_csv(filepath)
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Use .csv, .xlsx, or .xls."
        )

    def save_results(self, df: pd.DataFrame, filepath: str) -> None:
        """Persist a results DataFrame to CSV or Excel.

        The parent directory is created automatically if it does not exist.

        Args:
            df: DataFrame to save.
            filepath: Destination file path. Extension determines format
                (``.csv``, ``.xlsx``).

        Raises:
            ValueError: If the file extension is not supported.
            OSError: If the file cannot be written.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(filepath, index=False)
        elif suffix in (".xlsx", ".xls"):
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(
                f"Unsupported output extension '{suffix}'. Use .csv or .xlsx."
            )
        logger.info("Results saved to: %s", filepath)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise raw site data.

        Operations applied (all non-mutating):
        1. Drop rows where *every* column is null.
        2. Strip whitespace and lowercase all column names, replacing
           spaces with underscores.
        3. Strip leading/trailing whitespace from string columns.

        Args:
            df: Raw DataFrame as returned by :meth:`load_data`.

        Returns:
            New cleaned DataFrame.
        """
        cleaned = df.dropna(how="all")
        cleaned = cleaned.rename(
            columns={c: c.lower().strip().replace(" ", "_") for c in cleaned.columns}
        )

        # Strip string columns without mutating the original
        str_cols = cleaned.select_dtypes(include="object").columns.tolist()
        if str_cols:
            stripped_values = {col: cleaned[col].str.strip() for col in str_cols}
            cleaned = cleaned.assign(**stripped_values)

        return cleaned

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a preprocessed DataFrame against domain rules.

        Delegates to :func:`~src.validators.validate_dataframe`. Logs a
        warning for each advisory message and an error for each hard failure.

        Args:
            df: Preprocessed DataFrame (column names already normalised).

        Returns:
            :class:`~src.validators.ValidationResult` with accumulated
            errors and warnings.

        Raises:
            ValueError: If the DataFrame has no rows (before field-level
                checks are performed).
        """
        if df.empty:
            raise ValueError("Cannot validate an empty DataFrame.")

        result = validate_dataframe(df)

        for warning in result.warnings:
            logger.warning("Validation warning: %s", warning)
        for error in result.errors:
            logger.error("Validation error: %s", error)

        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute descriptive statistics for a preprocessed DataFrame.

        This method does not perform suitability scoring; use
        :meth:`run_pipeline` for the full assessment workflow.

        Args:
            df: DataFrame (raw or preprocessed).

        Returns:
            Dictionary with keys:
            - ``total_records`` (int)
            - ``columns`` (list[str])
            - ``missing_pct`` (dict[str, float]): Percentage missing per
              column.
            - ``summary_stats`` (dict): Output of ``DataFrame.describe()``
              for numeric columns (present only when numeric columns exist).
            - ``totals`` (dict[str, float]): Column sums for numeric columns.
            - ``means`` (dict[str, float]): Column means for numeric columns.
        """
        preprocessed = self.preprocess(df)
        result: Dict[str, Any] = {
            "total_records": len(preprocessed),
            "columns": list(preprocessed.columns),
            "missing_pct": (
                preprocessed.isnull().sum() / len(preprocessed) * 100
            ).round(1).to_dict(),
        }

        numeric_df = preprocessed.select_dtypes(include="number")
        if not numeric_df.empty:
            result["summary_stats"] = numeric_df.describe().round(3).to_dict()
            result["totals"] = numeric_df.sum().round(2).to_dict()
            result["means"] = numeric_df.mean().round(3).to_dict()

        return result

    # ------------------------------------------------------------------
    # Scoring pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        df: pd.DataFrame,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Execute the full assessment pipeline on a DataFrame.

        Steps: preprocess → validate → score → return scored DataFrame.

        Args:
            df: Raw or lightly cleaned site DataFrame.
            strict: When ``True``, raise :class:`ValueError` if validation
                produces any errors. When ``False`` (default), log errors
                and continue with the valid subset of rows.

        Returns:
            Scored DataFrame containing all original columns plus per-factor
            score columns and ``composite_score``.

        Raises:
            ValueError: If ``strict=True`` and validation errors are found,
                or if the DataFrame is empty after preprocessing.
        """
        preprocessed = self.preprocess(df)

        if preprocessed.empty:
            raise ValueError("No data remains after preprocessing.")

        validation_result = self.validate(preprocessed)

        if not validation_result.is_valid:
            if strict:
                error_summary = "; ".join(validation_result.errors[:5])
                raise ValueError(
                    f"Validation failed with {len(validation_result.errors)} error(s). "
                    f"First errors: {error_summary}"
                )
            logger.warning(
                "Proceeding despite %d validation error(s). "
                "Set strict=True to raise an exception instead.",
                len(validation_result.errors),
            )

        weights = self.config.get("weights")
        scored = score_dataframe(preprocessed, weights=weights)
        logger.info(
            "Scoring complete. %d sites scored. Mean composite score: %.2f",
            len(scored),
            scored["composite_score"].mean(),
        )
        return scored

    def prioritise(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Return a ranked shortlist of the highest-priority sites.

        Configuration keys used from ``self.config``:
        - ``top_n`` (int, default 10)
        - ``min_score`` (float, default 0.0)

        Args:
            scored_df: DataFrame produced by :meth:`run_pipeline`.

        Returns:
            Sorted subset of ``scored_df`` limited to the top candidates.
        """
        top_n: int = int(self.config.get("top_n", 10))
        min_score: float = float(self.config.get("min_score", 0.0))
        return prioritise_sites(scored_df, top_n=top_n, min_score=min_score)

    # ------------------------------------------------------------------
    # Convenience / legacy API
    # ------------------------------------------------------------------

    def run(self, filepath: str) -> Dict[str, Any]:
        """Convenience method: load a file and return descriptive statistics.

        For full suitability scoring use :meth:`run_pipeline` directly.

        Args:
            filepath: Path to a CSV or Excel data file.

        Returns:
            Analysis result dictionary as produced by :meth:`analyze`.
        """
        df = self.load_data(filepath)
        return self.analyze(df)

    def to_dataframe(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Flatten an analysis result dictionary into a two-column DataFrame.

        Nested dictionaries are expanded with dot-notation keys
        (e.g. ``summary_stats.mean``).

        Args:
            result: Dictionary as returned by :meth:`analyze`.

        Returns:
            DataFrame with columns ``metric`` and ``value``.
        """
        rows: List[Dict[str, Any]] = []
        for k, v in result.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"metric": f"{k}.{kk}", "value": vv})
            else:
                rows.append({"metric": k, "value": v})
        return pd.DataFrame(rows)

    def get_summary(self, scored_df: pd.DataFrame) -> Dict[str, Any]:
        """Produce a human-readable summary of scoring results.

        Args:
            scored_df: DataFrame produced by :meth:`run_pipeline`.

        Returns:
            Dictionary containing:
            - ``total_sites`` (int)
            - ``mean_composite_score`` (float)
            - ``score_distribution`` (dict): Counts per band
              (<40, 40–60, 60–80, >=80).
            - ``top_site`` (dict): Row with the highest composite score.
            - ``score_stats`` (dict): ``describe()`` output for
              ``composite_score``.
        """
        score_col = scored_df["composite_score"]
        bands = {
            "low_(<40)": int((score_col < 40).sum()),
            "moderate_(40-60)": int(((score_col >= 40) & (score_col < 60)).sum()),
            "good_(60-80)": int(((score_col >= 60) & (score_col < 80)).sum()),
            "excellent_(>=80)": int((score_col >= 80).sum()),
        }

        top_row = scored_df.loc[score_col.idxmax()]

        return {
            "total_sites": len(scored_df),
            "mean_composite_score": round(float(score_col.mean()), 2),
            "score_distribution": bands,
            "top_site": top_row.to_dict(),
            "score_stats": score_col.describe().round(2).to_dict(),
        }
