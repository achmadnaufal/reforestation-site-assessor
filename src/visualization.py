"""
Lightweight text-based visualisation helpers for scored site data.

These helpers produce terminal-friendly output without requiring
matplotlib or any plotting backend -- suitable for CI logs, SSH sessions,
and notebooks. All functions return strings; none print directly.

Use them to sanity-check score distributions, eyeball top sites, or
embed summaries in generated reports.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# ASCII histogram
# ---------------------------------------------------------------------------

def ascii_histogram(
    values: Iterable[float],
    bins: int = 10,
    width: int = 40,
    *,
    value_range: tuple[float, float] | None = None,
) -> str:
    """Render an ASCII histogram of numeric values.

    The histogram uses ``#`` characters scaled to the longest bar so that
    output fits within ``width`` columns regardless of count magnitude.
    NaN values are silently ignored.

    Args:
        values: Iterable of numeric values (e.g. composite scores).
        bins: Number of equal-width bins. Must be >= 1.
        width: Maximum bar width in characters. Must be >= 1.
        value_range: Optional ``(min, max)`` range for bin edges. When
            ``None``, the min/max of ``values`` is used.

    Returns:
        Multi-line string, one line per bin. Returns an empty string if
        ``values`` is empty (after NaN removal).

    Raises:
        ValueError: If ``bins`` or ``width`` is < 1, or if ``value_range``
            is malformed (min >= max).

    Example:
        >>> print(ascii_histogram([1, 2, 2, 3, 3, 3], bins=3, width=10))
        [1.0, 1.7): #####      (1)
        [1.7, 2.3): ##########  (2)
        [2.3, 3.0]: ########### (3)
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}.")

    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return ""

    if value_range is None:
        lo, hi = float(series.min()), float(series.max())
    else:
        lo, hi = float(value_range[0]), float(value_range[1])
        if lo >= hi:
            raise ValueError(
                f"value_range min ({lo}) must be < max ({hi})."
            )

    # Degenerate case: all values equal -> one filled bar.
    if lo == hi:
        return f"[{lo:.1f}, {hi:.1f}]: {'#' * width} ({len(series)})"

    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for value in series:
        # Upper edge inclusive for the final bin only.
        if value >= hi:
            counts[-1] += 1
            continue
        if value < lo:
            continue
        idx = min(bins - 1, int((value - lo) / (hi - lo) * bins))
        counts[idx] += 1

    max_count = max(counts) if counts else 0
    scale = width / max_count if max_count > 0 else 0

    lines = []
    for i, count in enumerate(counts):
        bar = "#" * int(round(count * scale)) if count > 0 else ""
        closing = "]" if i == bins - 1 else ")"
        lines.append(
            f"[{edges[i]:.1f}, {edges[i + 1]:.1f}{closing}: {bar:<{width}} ({count})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Score-band distribution
# ---------------------------------------------------------------------------

def score_distribution_table(
    scored_df: pd.DataFrame,
    score_column: str = "composite_score",
    bands: Sequence[tuple[str, float, float]] | None = None,
) -> str:
    """Render a text table summarising how sites fall into score bands.

    Args:
        scored_df: Scored DataFrame (e.g. output of
            :func:`src.scorer.score_dataframe`).
        score_column: Name of the column holding the composite score.
        bands: Optional sequence of ``(label, lower, upper)`` tuples where
            ``lower`` is inclusive and ``upper`` is exclusive (the final band's
            ``upper`` is treated inclusively). Defaults to a four-band scheme:
            low (<40), moderate (40-60), good (60-80), excellent (80-100].

    Returns:
        Multi-line string table with columns: band, count, percentage, bar.

    Raises:
        KeyError: If ``score_column`` is not in ``scored_df``.
        ValueError: If ``scored_df`` is empty.
    """
    if scored_df.empty:
        raise ValueError("scored_df is empty.")
    if score_column not in scored_df.columns:
        raise KeyError(f"'{score_column}' column is missing from scored_df.")

    if bands is None:
        bands = (
            ("low        (<40)   ", 0.0, 40.0),
            ("moderate   (40-60) ", 40.0, 60.0),
            ("good       (60-80) ", 60.0, 80.0),
            ("excellent  (80-100)", 80.0, 100.01),  # include 100 in last band
        )

    scores = scored_df[score_column].dropna()
    total = len(scores)

    lines = [f"Score distribution across {total} site(s):"]
    for label, lower, upper in bands:
        in_band = ((scores >= lower) & (scores < upper)).sum()
        pct = (in_band / total * 100) if total else 0.0
        bar = "#" * int(round(pct / 2))  # 2% per char
        lines.append(f"  {label}: {in_band:4d}  ({pct:5.1f}%)  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-N table
# ---------------------------------------------------------------------------

def format_top_sites(
    ranked_df: pd.DataFrame,
    top_n: int = 5,
    columns: Sequence[str] | None = None,
) -> str:
    """Render the top-N sites from a ranked DataFrame as a text table.

    The DataFrame is expected to be already sorted (e.g. output of
    :func:`src.scorer.prioritise_sites` or :func:`rank_sites_topsis`).

    Args:
        ranked_df: Pre-sorted DataFrame.
        top_n: Maximum number of rows to include. Must be >= 1.
        columns: Optional explicit column list. When ``None``, uses
            ``['site_id', 'composite_score']`` if available, else falls
            back to the first four DataFrame columns.

    Returns:
        Multi-line string table, or ``"(no sites)"`` if empty.

    Raises:
        ValueError: If ``top_n`` < 1.
        KeyError: If an explicit column is missing.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")
    if ranked_df.empty:
        return "(no sites)"

    if columns is None:
        preferred = [c for c in ("site_id", "composite_score", "topsis_score", "topsis_rank")
                     if c in ranked_df.columns]
        columns = preferred if preferred else list(ranked_df.columns[:4])
    else:
        missing = [c for c in columns if c not in ranked_df.columns]
        if missing:
            raise KeyError(f"Columns not in DataFrame: {missing}.")

    subset = ranked_df[list(columns)].head(top_n).copy()
    return subset.to_string(index=False)
