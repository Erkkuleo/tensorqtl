#!/usr/bin/env python3
"""Cosinor preprocessing for circadian eQTL mapping."""
import math
import numpy as np
import pandas as pd


def parse_time_to_hours(value: str) -> float:
    """Convert a time value string to fractional hours (0-24).

    STUB: currently casts to float. To support ISO 8601 timestamps
    (e.g., "2024-01-15T14:30:00"), replace the body of this function
    with ISO 8601 parsing. All downstream code consumes the returned
    float, so no other changes are needed.

    Replacement example:
        from datetime import datetime
        dt = datetime.fromisoformat(value)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot convert '{value}' to hours. Expected a numeric value (e.g. '14.5')."
        )


def compute_cosinor(hours: pd.Series, period: float = 24.0) -> tuple:
    """Compute cosine and sine time encodings for circadian eQTL mapping.

    Args:
        hours: Series indexed by sample ID, values are fractional hours (0-24).
        period: Cycle period in hours (default 24.0 for circadian rhythm).

    Returns:
        Tuple of (cos_t, sin_t), each a named Series indexed by sample ID.
    """
    angle = 2.0 * math.pi * hours / period
    cos_t = pd.Series(np.cos(angle.values), index=hours.index, name="cos_t")
    sin_t = pd.Series(np.sin(angle.values), index=hours.index, name="sin_t")
    return cos_t, sin_t


def load_metadata(path: str, time_col: str) -> pd.Series:
    """Load sample metadata and return hour-of-day values.

    File must be tab-separated with a header; the first column is used as
    sample IDs.

    Args:
        path: Path to TSV metadata file.
        time_col: Column name containing time-of-day values.

    Returns:
        Series of float hours indexed by sample ID.

    Raises:
        ValueError: If time_col is absent, sample IDs are duplicated,
                    or any time value cannot be converted to float.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    if time_col not in df.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in metadata. "
            f"Available columns: {list(df.columns)}"
        )
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"Duplicate sample IDs in metadata: {dupes}")
    return df[time_col].map(lambda v: parse_time_to_hours(str(v)))


def load_covariates(path: str) -> pd.DataFrame:
    """Load a tensorQTL-format covariates file.

    Format: rows = covariate names, columns = sample IDs (tab-separated).

    Returns:
        DataFrame with covariate names as index, sample IDs as columns.
    """
    return pd.read_csv(path, sep="\t", index_col=0)


def append_cosinor_to_covariates(
    covariates_df: pd.DataFrame, cos_t: pd.Series, sin_t: pd.Series
) -> pd.DataFrame:
    """Append cos_t and sin_t rows to a tensorQTL covariates DataFrame.

    Args:
        covariates_df: Existing covariates (rows=covariate names, cols=samples).
        cos_t: Cosine values indexed by sample ID (must cover all samples).
        sin_t: Sine values indexed by sample ID (must cover all samples).

    Returns:
        New DataFrame with cos_t and sin_t appended as the last two rows.

    Raises:
        ValueError: If 'cos_t' or 'sin_t' are already present in the index.
    """
    for name in ("cos_t", "sin_t"):
        if name in covariates_df.index:
            raise ValueError(
                f"'{name}' already present in covariates index. "
                "Remove it before re-running cosinor_preprocess.py."
            )
    new_rows = pd.DataFrame(
        [
            cos_t.reindex(covariates_df.columns).values,
            sin_t.reindex(covariates_df.columns).values,
        ],
        index=["cos_t", "sin_t"],
        columns=covariates_df.columns,
    )
    return pd.concat([covariates_df, new_rows])


def make_interaction_df(cos_t: pd.Series) -> pd.DataFrame:
    """Create a tensorQTL interaction DataFrame for the cos_t term.

    To upgrade to a 2-DF test, change this function to return a
    2-column DataFrame: pd.DataFrame({"cos_t": cos_t, "sin_t": sin_t}).

    Returns:
        DataFrame with sample IDs as index, single column 'cos_t'.
    """
    return cos_t.to_frame(name="cos_t")
