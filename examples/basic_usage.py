"""
Example usage of reforestation-site-assessor.

Demonstrates the full pipeline:
  1. Load demo sample data
  2. Run suitability scoring pipeline
  3. Inspect top-priority sites
  4. View a human-readable summary

Run from the project root::

    python examples/basic_usage.py
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SiteAssessor

SAMPLE_DATA = Path(__file__).parent.parent / "demo" / "sample_data.csv"


def main() -> None:
    """Execute the full assessment pipeline on the bundled sample data."""
    assessor = SiteAssessor(config={"top_n": 5, "min_score": 50.0})

    # 1. Load data
    print("Loading sample data...")
    df = assessor.load_data(str(SAMPLE_DATA))
    print(f"  Loaded {len(df)} site records with columns: {list(df.columns)}\n")

    # 2. Run full scoring pipeline
    print("Running suitability scoring pipeline...")
    scored = assessor.run_pipeline(df)
    print(f"  Scoring complete. {len(scored)} sites scored.\n")

    # 3. Top-priority sites
    print("=== Top 5 Priority Sites ===")
    top_sites = assessor.prioritise(scored)
    display_cols = ["site_id", "composite_score", "degradation_level",
                    "annual_rainfall_mm", "slope_pct"]
    print(top_sites[display_cols].to_string(index=False))

    # 4. Summary statistics
    print("\n=== Assessment Summary ===")
    summary = assessor.get_summary(scored)
    print(f"  Total sites assessed : {summary['total_sites']}")
    print(f"  Mean composite score : {summary['mean_composite_score']:.2f}")
    print("  Score distribution:")
    for band, count in summary["score_distribution"].items():
        print(f"    {band:<22}: {count} sites")

    top = summary["top_site"]
    print(f"\n  Highest priority site: {top.get('site_id', 'N/A')} "
          f"(score: {top.get('composite_score', 'N/A')})")


if __name__ == "__main__":
    main()
